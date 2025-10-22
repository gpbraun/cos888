// -----------------------------------------------------------------------------
// uDGP Delayed Relax-and-Cut (covering-only dualization) + optional REPAIR
// (nonconvex QCQP refit with t = ||x_i-x_j||^2 and t = d^2 + alpha on chosen k)
// -----------------------------------------------------------------------------
// Build & Run instructions are given in the message above.
// -----------------------------------------------------------------------------

#include <ilcplex/ilocplex.h>
#include <bits/stdc++.h>
ILOSTLBEGIN

static inline size_t PairKey(int i, int j) { return (size_t)i << 32 | (unsigned)j; }
static inline double sq(double x) { return x * x; }

struct Dist
{
    double d, d2;
};
struct Instance
{
    int n = 0, m = 0;
    double B = 5.0;
    std::vector<Dist> D; // sorted by d2
    double Tcap() const { return 12.0 * B * B; }
    double BigM() const
    {
        double M = Tcap();
        if (!D.empty())
            M = std::max(M, D.back().d2);
        return M + 1.0;
    }
};

static Instance read_instance()
{
    Instance ins;
    if (!(std::cin >> ins.n >> ins.m >> ins.B))
    {
        ins.n = 4;
        ins.m = 6;
        ins.B = 2.0;
        std::vector<double> dd = {1, 1, 1, std::sqrt(2.0), std::sqrt(2.0), std::sqrt(2.0)};
        for (double d : dd)
            ins.D.push_back({d, d * d});
        std::sort(ins.D.begin(), ins.D.end(), [](auto &a, auto &b)
                  { return a.d2 < b.d2; });
        return ins;
    }
    ins.D.resize(ins.m);
    for (int k = 0; k < ins.m; k++)
    {
        double d;
        std::cin >> d;
        ins.D[k] = {d, d * d};
    }
    std::sort(ins.D.begin(), ins.D.end(), [](auto &a, auto &b)
              { return a.d2 < b.d2; });
    return ins;
}

struct Params
{
    int maxCycles = 200;
    int Ktri = 40;
    int age = 4;

    double eps = 1.0;
    double mu_min = 1e-4;
    double trust0 = 1.0, trustMax = 5.0, trInc = 1.25, trDec = 0.75;
    double gapStop = 1e-6;

    bool triangles = true;
    bool verbose = false;
    bool dump = false, verify = false, repair = false;

    double tau_t = 1e-3, tau_x2 = 2e-4;
    unsigned seed = 42;
};

struct RowKey
{
    enum Kind
    {
        USE_GE,
        PAIR_GE
    } kind;
    int i, j, k;
};
static bool operator==(const RowKey &a, const RowKey &b)
{
    return a.kind == b.kind && a.i == b.i && a.j == b.j && a.k == b.k;
}
struct RowKeyHash
{
    size_t operator()(RowKey const &r) const noexcept
    {
        size_t h = std::hash<int>()((int)r.kind);
        h = h * 1315423911u + std::hash<int>()(r.i);
        h = h * 1315423911u + std::hash<int>()(r.j);
        h = h * 1315423911u + std::hash<int>()(r.k);
        return h;
    }
};

// ---------- Main RaC model ----------
struct Model
{
    const Instance &ins;
    const Params &par;
    IloEnv env;
    IloModel mdl;
    IloCplex cpx;

    IloNumVarArray X;                            // 3n coords
    std::vector<std::vector<IloNumVar>> t;       // t_ij
    std::vector<std::vector<IloNumVar>> r;       // r_ij
    std::vector<std::vector<IloBoolVarArray>> a; // a_ijk
    std::vector<IloNumVar> alpha, y;             // alpha_k, y_k
    IloObjective obj;

    Model(const Instance &ins_, const Params &par_) : ins(ins_), par(par_), env(), mdl(env), cpx(mdl), X(env), obj(IloMinimize(env, 0.0))
    {
        build_vars_();
        add_geometry_();
        add_capacity_();
        mdl.add(obj);

        cpx.setOut(par.verbose ? std::cout : env.getNullStream());
        cpx.setParam(IloCplex::Param::Threads, 1);
    }

    inline IloNumVar &Xv(int i, int d) { return X[3 * i + d]; }

    void build_vars_()
    {
        X = IloNumVarArray(env, 3 * ins.n, -ins.B, ins.B, ILOFLOAT);

        t.assign(ins.n, std::vector<IloNumVar>(ins.n));
        r.assign(ins.n, std::vector<IloNumVar>(ins.n));
        for (int i = 0; i < ins.n; i++)
            for (int j = i + 1; j < ins.n; j++)
            {
                t[i][j] = IloNumVar(env, 0.0, ins.Tcap(), ILOFLOAT);
                if (par.triangles)
                    r[i][j] = IloNumVar(env, 0.0, std::sqrt(ins.Tcap()), ILOFLOAT);
            }

        a.assign(ins.n, std::vector<IloBoolVarArray>(ins.n));
        for (int i = 0; i < ins.n; i++)
            for (int j = i + 1; j < ins.n; j++)
            {
                a[i][j] = IloBoolVarArray(env, ins.m);
                for (int k = 0; k < ins.m; k++)
                    a[i][j][k] = IloBoolVar(env);
            }

        alpha.resize(ins.m);
        y.resize(ins.m);
        for (int k = 0; k < ins.m; k++)
        {
            alpha[k] = IloNumVar(env, -ins.BigM(), ins.BigM(), ILOFLOAT);
            y[k] = IloNumVar(env, 0.0, IloInfinity, ILOFLOAT);
        }

        // symmetry anchors
        mdl.add(Xv(0, 0) == 0);
        mdl.add(Xv(0, 1) == 0);
        mdl.add(Xv(0, 2) == 0);
        if (ins.n >= 2)
        {
            mdl.add(Xv(1, 1) == 0);
            mdl.add(Xv(1, 2) == 0);
        }
        if (ins.n >= 3)
        {
            mdl.add(Xv(2, 2) == 0);
        }
    }

    void add_geometry_()
    {
        for (int i = 0; i < ins.n; i++)
            for (int j = i + 1; j < ins.n; j++)
            {
                IloExpr q(env);
                for (int d = 0; d < 3; ++d)
                {
                    IloExpr diff = Xv(i, d) - Xv(j, d);
                    q += diff * diff;
                    diff.end();
                }
                mdl.add(t[i][j] >= q);
                q.end();
                if (par.triangles)
                    mdl.add(t[i][j] >= r[i][j] * r[i][j]);
            }
        for (int k = 0; k < ins.m; k++)
        {
            mdl.add(y[k] >= alpha[k]);
            mdl.add(y[k] >= -alpha[k]);
        }
        double M = ins.BigM();
        for (int i = 0; i < ins.n; i++)
            for (int j = i + 1; j < ins.n; j++)
                for (int k = 0; k < ins.m; k++)
                {
                    double d2 = ins.D[k].d2;
                    mdl.add(t[i][j] - (d2 + alpha[k]) <= M * (1 - a[i][j][k]));
                    mdl.add(t[i][j] - (d2 + alpha[k]) >= -M * (1 - a[i][j][k]));
                }
    }

    void add_capacity_()
    {
        for (int k = 0; k < ins.m; k++)
        {
            IloExpr s(env);
            for (int i = 0; i < ins.n; i++)
                for (int j = i + 1; j < ins.n; j++)
                    s += a[i][j][k];
            mdl.add(s <= 1.0);
            s.end();
        }
        for (int i = 0; i < ins.n; i++)
            for (int j = i + 1; j < ins.n; j++)
            {
                IloExpr s(env);
                for (int k = 0; k < ins.m; k++)
                    s += a[i][j][k];
                mdl.add(s <= 1.0);
                s.end();
            }
    }

    void set_obj(const std::unordered_map<RowKey, double, RowKeyHash> &lambda, const Params &par)
    {
        if (obj.getImpl())
        {
            mdl.remove(obj);
            obj.end();
            obj = IloObjective();
        }
        IloExpr e(env);

        for (int k = 0; k < ins.m; k++)
            e += y[k];
        if (par.tau_t > 0)
            for (int i = 0; i < ins.n; i++)
                for (int j = i + 1; j < ins.n; j++)
                    e += par.tau_t * t[i][j];
        if (par.tau_x2 > 0)
            for (int i = 0; i < ins.n; i++)
                for (int d = 0; d < 3; ++d)
                    e += par.tau_x2 * X[3 * i + d] * X[3 * i + d];

        for (auto &kv : lambda)
        {
            const RowKey &rk = kv.first;
            double lam = kv.second;
            if (lam == 0.0)
                continue;
            if (rk.kind == RowKey::USE_GE)
            {
                IloExpr s(env);
                for (int i = 0; i < ins.n; i++)
                    for (int j = i + 1; j < ins.n; j++)
                        s += a[i][j][rk.k];
                e += lam * (1.0 - s);
                s.end();
            }
            else
            {
                IloExpr s(env);
                for (int K = 0; K < ins.m; K++)
                    s += a[rk.i][rk.j][K];
                e += lam * (1.0 - s);
                s.end();
            }
        }

        obj = IloMinimize(env, e);
        mdl.add(obj);
        e.end();
    }
};

// ---------- RaC driver ----------
struct DRC
{
    const Instance &ins;
    Params par;
    Model model;

    std::unordered_map<RowKey, double, RowKeyHash> lam; // GE rows
    double bestUB = std::numeric_limits<double>::infinity();
    double trust;

    DRC(const Instance &I, const Params &P) : ins(I), par(P), model(I, P), trust(P.trust0)
    {
        for (int k = 0; k < ins.m; k++)
            lam.insert({RowKey{RowKey::USE_GE, -1, -1, k}, 0.2});
        for (int i = 0; i < ins.n; i++)
            for (int j = i + 1; j < ins.n; j++)
                lam.insert({RowKey{RowKey::PAIR_GE, i, j, -1}, 0.2});
    }

    bool solveLRP(double &val)
    {
        model.set_obj(lam, par);
        if (!model.cpx.solve())
            return false;
        val = model.cpx.getObjValue();
        return true;
    }

    double greedyUB()
    {
        std::vector<std::tuple<double, int, int, int>> C;
        for (int i = 0; i < ins.n; i++)
            for (int j = i + 1; j < ins.n; j++)
            {
                double tij = model.cpx.getValue(model.t[i][j]);
                for (int k = 0; k < ins.m; k++)
                    C.emplace_back(std::fabs(tij - ins.D[k].d2), i, j, k);
            }
        std::sort(C.begin(), C.end(), [](auto &a, auto &b)
                  { return std::get<0>(a) < std::get<0>(b); });
        std::vector<int> usedK(ins.m, 0);
        std::unordered_set<size_t> usedPair;
        double obj = 0;
        int taken = 0;
        for (auto &c : C)
        {
            if (taken == ins.m)
                break;
            auto [cost, i, j, k] = c;
            size_t key = PairKey(i, j);
            if (usedK[k] || usedPair.count(key))
                continue;
            usedK[k] = 1;
            usedPair.insert(key);
            obj += cost;
            taken++;
        }
        if (taken < ins.m)
            return std::numeric_limits<double>::infinity();
        return obj;
    }

    void updateLambda(double zLRP)
    {
        std::unordered_map<RowKey, double, RowKeyHash> g;
        double g2 = 0;
        auto aVal = [&](int i, int j, int k)
        { return model.cpx.getValue(model.a[i][j][k]); };

        for (auto &kv : lam)
        {
            const RowKey &rk = kv.first;
            double sum = 0.0;
            if (rk.kind == RowKey::USE_GE)
            {
                for (int i = 0; i < ins.n; i++)
                    for (int j = i + 1; j < ins.n; j++)
                        sum += aVal(i, j, rk.k);
            }
            else
            {
                for (int K = 0; K < ins.m; K++)
                    sum += aVal(rk.i, rk.j, K);
            }
            double grad = 1.0 - sum;
            if (kv.second == 0.0 && grad < 0)
                grad = 0;
            g[rk] = grad;
            g2 += grad * grad;
        }
        if (g2 == 0)
            return;

        double zStar = bestUB;
        if (!std::isfinite(zStar))
            zStar = zLRP + 1.0;
        double mu = par.eps * (zStar - zLRP) / g2;
        if (!(mu > 0))
            mu = par.mu_min;

        double maxChange = trust;
        for (auto &kv : lam)
        {
            double step = mu * g[kv.first];
            if (step > maxChange)
                step = maxChange;
            if (step < -maxChange)
                step = -maxChange;
            kv.second = std::max(0.0, kv.second + step);
        }
    }

    // Dump + verify + optional repair
    void finalize_and_report()
    {
        // collect solution
        std::vector<double> X(3 * ins.n, 0);
        for (int i = 0; i < ins.n; i++)
            for (int d = 0; d < 3; ++d)
                X[3 * i + d] = model.cpx.getValue(model.X[3 * i + d]);
        std::vector<std::vector<double>> t(ins.n, std::vector<double>(ins.n, 0));
        for (int i = 0; i < ins.n; i++)
            for (int j = i + 1; j < ins.n; j++)
                t[i][j] = model.cpx.getValue(model.t[i][j]);
        std::vector<double> alpha(ins.m, 0), y(ins.m, 0);
        for (int k = 0; k < ins.m; k++)
        {
            alpha[k] = model.cpx.getValue(model.alpha[k]);
            y[k] = model.cpx.getValue(model.y[k]);
        }

        // chosen assignment
        std::vector<int> matchK(ins.m, -1);              // matchK[k] = pair index linearized, not used
        std::vector<std::pair<int, int>> pairOfK(ins.m); // pairOfK[k]=(i,j)
        std::vector<int> KofPair;                        // not needed, but could store
        {
            for (int i = 0; i < ins.n; i++)
                for (int j = i + 1; j < ins.n; j++)
                {
                    for (int k = 0; k < ins.m; k++)
                    {
                        double aij = model.cpx.getValue(model.a[i][j][k]);
                        if (aij > 0.5)
                        {
                            pairOfK[k] = {i, j};
                            break;
                        }
                    }
                }
        }

        if (par.dump)
        {
            std::cout << "\n=== SOLUTION DUMP ===\n";
            std::cout << "Coordinates:\n";
            for (int i = 0; i < ins.n; i++)
                std::cout << "  x[" << i << "] = (" << X[3 * i] << "," << X[3 * i + 1] << "," << X[3 * i + 2] << ")\n";

            std::cout << "Assignments (i,j)->k, t_ij, d_k^2, |t-d^2|:\n";
            for (int i = 0; i < ins.n; i++)
                for (int j = i + 1; j < ins.n; j++)
                {
                    for (int k = 0; k < ins.m; k++)
                    {
                        double aij = model.cpx.getValue(model.a[i][j][k]);
                        if (aij > 0.5)
                        {
                            double tij = t[i][j], d2 = ins.D[k].d2;
                            std::cout << "  (" << i << "," << j << ")->" << k
                                      << "  t=" << tij << "  d2=" << d2 << "  err=" << std::fabs(tij - d2) << "\n";
                        }
                    }
                }
            double sumY = 0;
            for (double v : y)
                sumY += v;
            std::cout << "Alphas & y:\n";
            for (int k = 0; k < ins.m; k++)
                std::cout << "  k=" << k << "  alpha=" << alpha[k] << "  y=" << y[k] << "\n";
            std::cout << "Sum |alpha| = " << sumY << "\n";
        }

        if (par.verify)
        {
            double maxQC = 0, maxCapUse = 0, maxCapPair = 0, maxGEuse = 0, maxGEpair = 0, maxLink = 0, maxTri = 0;
            for (int i = 0; i < ins.n; i++)
                for (int j = i + 1; j < ins.n; j++)
                {
                    double q = sq(X[3 * i] - X[3 * j]) + sq(X[3 * i + 1] - X[3 * j + 1]) + sq(X[3 * i + 2] - X[3 * j + 2]);
                    maxQC = std::max(maxQC, q - t[i][j]);
                }
            for (int k = 0; k < ins.m; k++)
            {
                double s = 0;
                for (int i = 0; i < ins.n; i++)
                    for (int j = i + 1; j < ins.n; j++)
                        s += model.cpx.getValue(model.a[i][j][k]);
                maxCapUse = std::max(maxCapUse, s - 1.0);
                maxGEuse = std::max(maxGEuse, 1.0 - s);
            }
            for (int i = 0; i < ins.n; i++)
                for (int j = i + 1; j < ins.n; j++)
                {
                    double s = 0;
                    for (int k = 0; k < ins.m; k++)
                        s += model.cpx.getValue(model.a[i][j][k]);
                    maxCapPair = std::max(maxCapPair, s - 1.0);
                    maxGEpair = std::max(maxGEpair, 1.0 - s);
                    for (int k = 0; k < ins.m; k++)
                    {
                        if (model.cpx.getValue(model.a[i][j][k]) > 0.5)
                        {
                            maxLink = std::max(maxLink, std::fabs(t[i][j] - (ins.D[k].d2 + alpha[k])));
                        }
                    }
                }
            if (par.triangles)
            {
                for (int i = 0; i < ins.n; i++)
                    for (int j = i + 1; j < ins.n; j++)
                    {
                        double dij = std::sqrt(std::max(0.0, t[i][j]));
                        for (int k = 0; k < ins.n; k++)
                            if (k != i && k != j)
                            {
                                int p1 = std::min(i, k), q1 = std::max(i, k);
                                int p2 = std::min(k, j), q2 = std::max(k, j);
                                double dik = std::sqrt(std::max(0.0, model.cpx.getValue(model.t[p1][q1])));
                                double dkj = std::sqrt(std::max(0.0, model.cpx.getValue(model.t[p2][q2])));
                                maxTri = std::max(maxTri, dij - (dik + dkj));
                            }
                    }
            }
            std::cout << "\n=== VERIFIER ===\n";
            std::cout << "Max(q(x)-t)           : " << maxQC << " (<= 0 ok)\n";
            std::cout << "Max(∑ a^k - 1)        : " << maxCapUse << " (<= 0 ok)\n";
            std::cout << "Max(∑ a_ij - 1)       : " << maxCapPair << " (<= 0 ok)\n";
            std::cout << "Max(1 - ∑ a^k)        : " << maxGEuse << " (<= 0 desired)\n";
            std::cout << "Max(1 - ∑ a_ij)       : " << maxGEpair << " (<= 0 desired)\n";
            std::cout << "Max |t - (d^2+α)|     : " << maxLink << " (≈ 0 ok)\n";
        }

        if (!par.repair)
            return;

        // ---------- REPAIR: fix a, enforce t = ||x_i-x_j||^2 and t = d^2 + alpha on chosen k ----------
        IloEnv env2;
        IloModel mdl2(env2);
        IloCplex cpx2(mdl2);
        IloNumVarArray X2(env2, 3 * ins.n, -ins.B, ins.B, ILOFLOAT);
        std::vector<std::vector<IloNumVar>> t2(ins.n, std::vector<IloNumVar>(ins.n));
        std::vector<IloNumVar> alpha2(ins.m), y2(ins.m);

        for (int i = 0; i < ins.n; i++)
            for (int j = i + 1; j < ins.n; j++)
                t2[i][j] = IloNumVar(env2, 0.0, ins.Tcap(), ILOFLOAT);
        for (int k = 0; k < ins.m; k++)
        {
            alpha2[k] = IloNumVar(env2, -ins.BigM(), ins.BigM(), ILOFLOAT);
            y2[k] = IloNumVar(env2, 0.0, IloInfinity, ILOFLOAT);
        }

        // anchors
        auto Xv2 = [&](int i, int d) -> IloNumVar &
        { return X2[3 * i + d]; };
        mdl2.add(Xv2(0, 0) == 0);
        mdl2.add(Xv2(0, 1) == 0);
        mdl2.add(Xv2(0, 2) == 0);
        if (ins.n >= 2)
        {
            mdl2.add(Xv2(1, 1) == 0);
            mdl2.add(Xv2(1, 2) == 0);
        }
        if (ins.n >= 3)
        {
            mdl2.add(Xv2(2, 2) == 0);
        }

        // equalities t = ||x_i-x_j||^2
        for (int i = 0; i < ins.n; i++)
            for (int j = i + 1; j < ins.n; j++)
            {
                IloExpr q(env2);
                for (int d = 0; d < 3; ++d)
                {
                    IloExpr diff = Xv2(i, d) - Xv2(j, d);
                    q += diff * diff;
                    diff.end();
                }
                mdl2.add(t2[i][j] == q);
                q.end();
            }

        // for chosen k on each (i,j), enforce t = d^2 + alpha; others free
        for (int i = 0; i < ins.n; i++)
            for (int j = i + 1; j < ins.n; j++)
            {
                bool found = false;
                for (int k = 0; k < ins.m; k++)
                {
                    if (model.cpx.getValue(model.a[i][j][k]) > 0.5)
                    {
                        mdl2.add(t2[i][j] == ins.D[k].d2 + alpha2[k]);
                        found = true;
                        break;
                    }
                }
                if (!found)
                {
                    // If no k chosen (shouldn't happen), keep a soft link via inequalities with big-M
                    for (int k = 0; k < ins.m; k++)
                    {
                        double M = ins.BigM();
                        mdl2.add(t2[i][j] - (ins.D[k].d2 + alpha2[k]) <= M);
                        mdl2.add(t2[i][j] - (ins.D[k].d2 + alpha2[k]) >= -M);
                    }
                }
            }

        // y >= ± alpha
        for (int k = 0; k < ins.m; k++)
        {
            mdl2.add(y2[k] >= alpha2[k]);
            mdl2.add(y2[k] >= -alpha2[k]);
        }

        // objective: minimize sum |alpha| + tiny ||x||^2 regularizer
        IloExpr obj(env2);
        for (int k = 0; k < ins.m; k++)
            obj += y2[k];
        double tau_x = 1e-8; // tiny to pin a scale if needed
        for (int i = 0; i < ins.n; i++)
            for (int d = 0; d < 3; ++d)
                obj += tau_x * X2[3 * i + d] * X2[3 * i + d];
        IloObjective OBJ = IloMinimize(env2, obj);
        obj.end();
        mdl2.add(OBJ);

        cpx2.setOut(par.verbose ? std::cout : env2.getNullStream());
        cpx2.setParam(IloCplex::Param::Threads, 1);
        // allow nonconvex search (local)
        cpx2.setParam(IloCplex::Param::OptimalityTarget, 3); // 3 = nonconvex

        bool ok = cpx2.solve();
        if (!ok)
        {
            std::cout << "\n[REPAIR] Nonconvex refit failed. (local solver couldn’t find a feasible/local optimum)\n";
            env2.end();
            return;
        }

        std::cout << "\n=== REPAIR (local QCQP) ===\n";
        std::cout << "Status: " << (cpx2.getStatus() == IloAlgorithm::Optimal ? "Optimal (local)" : "Feasible (local)") << "\n";
        std::cout << "Obj (sum |alpha| + tiny reg): " << cpx2.getObjValue() << "\n";
        std::cout << "Coordinates after repair:\n";
        for (int i = 0; i < ins.n; i++)
        {
            double x0 = cpx2.getValue(X2[3 * i + 0]), x1 = cpx2.getValue(X2[3 * i + 1]), x2 = cpx2.getValue(X2[3 * i + 2]);
            std::cout << "  x[" << i << "] = (" << x0 << "," << x1 << "," << x2 << ")\n";
        }
        // quick check of distances
        double maxErr = 0;
        for (int i = 0; i < ins.n; i++)
            for (int j = i + 1; j < ins.n; j++)
            {
                double tij = cpx2.getValue(t2[i][j]);
                double d2t = 0;
                for (int k = 0; k < ins.m; k++)
                    if (model.cpx.getValue(model.a[i][j][k]) > 0.5)
                    {
                        d2t = ins.D[k].d2;
                        break;
                    }
                maxErr = std::max(maxErr, std::fabs(tij - d2t));
            }
        std::cout << "Max |t - d^2(chosen)| after repair: " << maxErr << "\n";
        env2.end();
    }

    void run()
    {
        for (int cyc = 0; cyc < par.maxCycles; ++cyc)
        {
            double zLRP;
            if (!solveLRP(zLRP))
            {
                std::cerr << "LRP solve failed.\n";
                break;
            }

            double UB = greedyUB();
            if (UB < bestUB)
                bestUB = UB;

            double zBefore = zLRP;
            updateLambda(zLRP);
            if (!solveLRP(zLRP))
                zLRP = zBefore;

            if (zLRP + 1e-9 < zBefore)
                trust = std::min(par.trustMax, trust * par.trInc);
            else
                trust = std::max(0.1, trust * par.trDec);

            double gap = (std::isfinite(bestUB) ? (bestUB - zLRP) : std::numeric_limits<double>::infinity());
            if (std::isfinite(gap))
                gap /= std::max(1.0, std::fabs(bestUB));

            std::cout << "cyc=" << cyc << "  LRP=" << zLRP << "  UB=" << bestUB << "  gap=" << gap
                      << "  |A'|=" << lam.size() << "  Tri=" << 0 << "  Δ=" << trust << "\n";

            if (std::isfinite(bestUB) && (bestUB - zLRP) <= par.gapStop)
            {
                std::cout << "Converged (dual-primal gap).\n";
                finalize_and_report();
                break;
            }
        }
    }

    static void parse_cli(Params &p, int argc, char **argv)
    {
        auto low = [&](std::string s)
        { for(char&c:s) c=std::tolower(c); return s; };
        auto getv = [&](const std::string &a, const char *key) -> const char *
        {
            size_t eq = a.find('=');
            if (eq == std::string::npos)
                return nullptr;
            if (a.substr(0, eq) == key)
                return a.c_str() + eq + 1;
            return nullptr;
        };
        for (int i = 1; i < argc; i++)
        {
            std::string a = argv[i];
            if (a.rfind("--cycles", 0) == 0)
            {
                if (auto v = getv(a, "--cycles"))
                    p.maxCycles = std::max(1, atoi(v));
            }
            else if (a.rfind("--triangles", 0) == 0)
            {
                if (auto v = getv(a, "--triangles"))
                {
                    std::string s = low(v);
                    p.triangles = (s == "on" || s == "1" || s == "true");
                }
            }
            else if (a.rfind("--seed", 0) == 0)
            {
                if (auto v = getv(a, "--seed"))
                    p.seed = std::stoul(v);
            }
            else if (a.rfind("--ridgeT", 0) == 0)
            {
                if (auto v = getv(a, "--ridgeT"))
                    p.tau_t = std::stod(v);
            }
            else if (a.rfind("--proxX2", 0) == 0)
            {
                if (auto v = getv(a, "--proxX2"))
                    p.tau_x2 = std::stod(v);
            }
            else if (a.rfind("--trust0", 0) == 0)
            {
                if (auto v = getv(a, "--trust0"))
                    p.trust0 = std::stod(v);
            }
            else if (a == "--verbose")
            {
                p.verbose = true;
            }
            else if (a == "--dump")
            {
                p.dump = true;
            }
            else if (a == "--verify")
            {
                p.verify = true;
            }
            else if (a == "--repair")
            {
                p.repair = true;
            }
        }
    }
};

int main(int argc, char **argv)
{
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    Instance ins = read_instance();
    Params par;
    DRC::parse_cli(par, argc, argv);

    DRC solver(ins, par);
    solver.run();
    return 0;
}
