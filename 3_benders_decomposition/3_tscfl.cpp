// 3_tscfl.cpp
// COS888 – Benders decomposition for the Two-stage Capacitated Facility Location (TSCFL)
// Master MIP + Worker LP (dual) + lazy/user cut callbacks (ilobendersatsp2.cpp style)
// This version fixes dual signs AND uses hierarchical parameter API.

#include <ilcplex/ilocplex.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <stdexcept>

ILOSTLBEGIN

// ---------- small helpers ----------
static inline int idx2(int i, int j, int ncols) { return i * ncols + j; }

// ============================================================
// Instance: same layout as Python TSCFLInstance.from_txt
// File layout (space separated):
// nI nJ nK
// r[k] (k=0..nK-1)
// (q[j], g[j]) for j=0..nJ-1
// c[i,j]  (row-major over i then j) size nI*nJ
// (p[i], f[i]) for i=0..nI-1
// d[j,k]  (row-major over j then k) size nJ*nK
// ============================================================
struct TSCFLInstance
{
    int nI{0}, nJ{0}, nK{0};
    std::vector<double> f; // size nI
    std::vector<double> g; // size nJ
    std::vector<double> c; // size nI*nJ (i,j)
    std::vector<double> d; // size nJ*nK (j,k)
    std::vector<double> p; // size nI
    std::vector<double> q; // size nJ
    std::vector<double> r; // size nK

    double demandTotal() const
    {
        return std::accumulate(r.begin(), r.end(), 0.0);
    }
    double C(int i, int j) const { return c[idx2(i, j, nJ)]; }
    double D(int j, int k) const { return d[idx2(j, k, nK)]; }

    static TSCFLInstance fromTxt(const std::string &path)
    {
        std::ifstream in(path);
        if (!in)
            throw std::runtime_error("Cannot open instance: " + path);
        std::vector<double> a;
        a.reserve(1 << 20);
        double v;
        while (in >> v)
            a.push_back(v);
        if (a.size() < 3)
            throw std::runtime_error("Malformed file (header).");

        size_t pos = 0;
        TSCFLInstance inst;
        inst.nI = static_cast<int>(a[pos++]);
        inst.nJ = static_cast<int>(a[pos++]);
        inst.nK = static_cast<int>(a[pos++]);

        const int nI = inst.nI, nJ = inst.nJ, nK = inst.nK;

        // r: nK
        if (pos + nK > a.size())
            throw std::runtime_error("Malformed file (r).");
        inst.r.assign(a.begin() + pos, a.begin() + pos + nK);
        pos += nK;

        // (q,g): nJ pairs
        if (pos + 2 * nJ > a.size())
            throw std::runtime_error("Malformed file (q,g).");
        inst.q.resize(nJ);
        inst.g.resize(nJ);
        for (int j = 0; j < nJ; ++j)
        {
            inst.q[j] = a[pos++];
            inst.g[j] = a[pos++];
        }

        // c: nI*nJ
        if (pos + nI * nJ > a.size())
            throw std::runtime_error("Malformed file (c).");
        inst.c.assign(a.begin() + pos, a.begin() + pos + nI * nJ);
        pos += nI * nJ;

        // (p,f): nI pairs
        if (pos + 2 * nI > a.size())
            throw std::runtime_error("Malformed file (p,f).");
        inst.p.resize(nI);
        inst.f.resize(nI);
        for (int i = 0; i < nI; ++i)
        {
            inst.p[i] = a[pos++];
            inst.f[i] = a[pos++];
        }

        // d: nJ*nK
        if (pos + nJ * nK > a.size())
            throw std::runtime_error("Malformed file (d).");
        inst.d.assign(a.begin() + pos, a.begin() + pos + nJ * nK);
        pos += nJ * nK;

        return inst;
    }
};

// ============================================================
// Worker Dual (built once, objective reset per (x,y))
// Correct dual (primal min with ≤,≤,=,=):
//   Dual vars: alpha_i ≤ 0, beta_j ≤ 0, gamma_j free, delta_k free
//   Dual cons: alpha_i + gamma_j ≤ c_ij
//              beta_j  - gamma_j + delta_k ≤ d_jk
//   Dual obj : max  sum_i p_i x_i alpha_i + sum_j q_j y_j beta_j + sum_k r_k delta_k
// Benders cut: eta ≥ [sum_k r_k delta_k] + Σ_i (p_i alpha_i) x_i + Σ_j (q_j beta_j) y_j
// (Note alpha_i, beta_j are non-positive at optimum; slopes are negative numbers.)
// ============================================================
struct WorkerDual
{
    const TSCFLInstance &inst;
    IloEnv env;
    IloModel model;
    IloCplex cplex;
    IloNumVarArray alpha; // nI, (-inf, 0]
    IloNumVarArray beta;  // nJ, (-inf, 0]
    IloNumVarArray delta; // nK, free
    IloNumVarArray gamma; // nJ, free
    IloObjective obj;

    WorkerDual(const TSCFLInstance &I, bool log = false)
        : inst(I), env(), model(env), cplex(env),
          alpha(env, I.nI, -IloInfinity, 0.0, ILOFLOAT),
          beta(env, I.nJ, -IloInfinity, 0.0, ILOFLOAT),
          delta(env, I.nK, -IloInfinity, IloInfinity, ILOFLOAT),
          gamma(env, I.nJ, -IloInfinity, IloInfinity, ILOFLOAT),
          obj(IloMaximize(env, 0.0))
    {
        // alpha_i + gamma_j ≤ c_ij
        for (int i = 0; i < inst.nI; ++i)
            for (int j = 0; j < inst.nJ; ++j)
                model.add(alpha[i] + gamma[j] <= inst.C(i, j));
        // beta_j - gamma_j + delta_k ≤ d_jk
        for (int j = 0; j < inst.nJ; ++j)
            for (int k = 0; k < inst.nK; ++k)
                model.add(beta[j] - gamma[j] + delta[k] <= inst.D(j, k));

        model.add(obj);
        cplex.extract(model);
        cplex.setOut(env.getNullStream());
        cplex.setWarning(env.getNullStream());
        cplex.setParam(IloCplex::Param::Threads, 1); // safe during callbacks
        if (log)
            cplex.setOut(std::cout);
    }

    void setObjective(const std::vector<double> &x, const std::vector<double> &y)
    {
        IloExpr e(env);
        for (int i = 0; i < inst.nI; ++i)
            e += (inst.p[i] * x[i]) * alpha[i]; // alpha ≤ 0
        for (int j = 0; j < inst.nJ; ++j)
            e += (inst.q[j] * y[j]) * beta[j]; // beta  ≤ 0
        for (int k = 0; k < inst.nK; ++k)
            e += (inst.r[k]) * delta[k]; // delta free
        obj.setExpr(e);
        e.end();
    }

    void solve(const std::vector<double> &x, const std::vector<double> &y,
               double &theta,
               std::vector<double> &coef_x,
               std::vector<double> &coef_y,
               double &rhs)
    {
        setObjective(x, y);
        if (!cplex.solve())
            throw std::runtime_error("Worker dual failed to solve.");

        theta = cplex.getObjValue();

        coef_x.assign(inst.nI, 0.0);
        for (int i = 0; i < inst.nI; ++i)
            coef_x[i] = inst.p[i] * cplex.getValue(alpha[i]); // ≤ 0

        coef_y.assign(inst.nJ, 0.0);
        for (int j = 0; j < inst.nJ; ++j)
            coef_y[j] = inst.q[j] * cplex.getValue(beta[j]); // ≤ 0

        rhs = 0.0;
        for (int k = 0; k < inst.nK; ++k)
            rhs += inst.r[k] * cplex.getValue(delta[k]);
    }
};

// ============================================================
// Lazy (integer) callback: add violated Benders optimality cuts
// ============================================================
class LazyBendersCallbackI : public IloCplex::LazyConstraintCallbackI
{
    const TSCFLInstance &inst;
    WorkerDual &worker;
    IloBoolVarArray x;
    IloBoolVarArray y;
    IloNumVar eta;
    double eps;

public:
    LazyBendersCallbackI(IloEnv env,
                         const TSCFLInstance &_inst,
                         WorkerDual &_worker,
                         IloBoolVarArray _x,
                         IloBoolVarArray _y,
                         IloNumVar _eta,
                         double _eps)
        : IloCplex::LazyConstraintCallbackI(env), inst(_inst), worker(_worker), x(_x), y(_y), eta(_eta), eps(_eps) {}

    IloCplex::CallbackI *duplicateCallback() const override
    {
        return (new (getEnv()) LazyBendersCallbackI(*this));
    }

    void main() override
    {
        std::vector<double> xv(inst.nI), yv(inst.nJ);
        for (int i = 0; i < inst.nI; ++i)
            xv[i] = getValue(x[i]);
        for (int j = 0; j < inst.nJ; ++j)
            yv[j] = getValue(y[j]);
        double etaVal = getValue(eta);

        double theta, rhs;
        std::vector<double> coef_x, coef_y;
        worker.solve(xv, yv, theta, coef_x, coef_y, rhs);

        if (theta - etaVal > eps)
        {
            IloEnv env = getEnv();
            IloExpr lin(env);
            lin += rhs;
            for (int i = 0; i < inst.nI; ++i)
                lin += coef_x[i] * x[i];
            for (int j = 0; j < inst.nJ; ++j)
                lin += coef_y[j] * y[j];
            add(eta >= lin); // cut off incumbent
            lin.end();
        }
    }
};

// ============================================================
// User-cut (fractional LP nodes) callback: tighten relaxation
// ============================================================
class UserBendersCallbackI : public IloCplex::UserCutCallbackI
{
    const TSCFLInstance &inst;
    WorkerDual &worker;
    IloBoolVarArray x;
    IloBoolVarArray y;
    IloNumVar eta;
    double eps;

public:
    UserBendersCallbackI(IloEnv env,
                         const TSCFLInstance &_inst,
                         WorkerDual &_worker,
                         IloBoolVarArray _x,
                         IloBoolVarArray _y,
                         IloNumVar _eta,
                         double _eps)
        : IloCplex::UserCutCallbackI(env), inst(_inst), worker(_worker), x(_x), y(_y), eta(_eta), eps(_eps) {}

    IloCplex::CallbackI *duplicateCallback() const override
    {
        return (new (getEnv()) UserBendersCallbackI(*this));
    }

    void main() override
    {
        std::vector<double> xv(inst.nI), yv(inst.nJ);
        for (int i = 0; i < inst.nI; ++i)
            xv[i] = getValue(x[i]);
        for (int j = 0; j < inst.nJ; ++j)
            yv[j] = getValue(y[j]);
        double etaVal = getValue(eta);

        double theta, rhs;
        std::vector<double> coef_x, coef_y;
        worker.solve(xv, yv, theta, coef_x, coef_y, rhs);

        if (theta - etaVal > eps)
        {
            IloEnv env = getEnv();
            IloExpr lin(env);
            lin += rhs;
            for (int i = 0; i < inst.nI; ++i)
                lin += coef_x[i] * x[i];
            for (int j = 0; j < inst.nJ; ++j)
                lin += coef_y[j] * y[j];
            add(eta >= lin, IloCplex::UseCutPurge); // global user cut
            lin.end();
        }
    }
};

// ============================================================
// Flow recovery: primal LP to get u_ij and v_jk (optional)
// ============================================================
struct FlowSolution
{
    std::vector<double> u; // nI*nJ
    std::vector<double> v; // nJ*nK
    double cost{0.0};
};

FlowSolution recoverFlows(const TSCFLInstance &inst,
                          const std::vector<int> &x,
                          const std::vector<int> &y,
                          double timelimit = -1.0,
                          bool log = false)
{
    IloEnv env;
    FlowSolution res;
    try
    {
        IloModel m(env);
        IloCplex cplex(m);
        if (!log)
        {
            cplex.setOut(env.getNullStream());
            cplex.setWarning(env.getNullStream());
        }
        if (timelimit > 0)
            cplex.setParam(IloCplex::Param::TimeLimit, timelimit);

        IloNumVarArray u(env, inst.nI * inst.nJ, 0.0, IloInfinity, ILOFLOAT);
        IloNumVarArray v(env, inst.nJ * inst.nK, 0.0, IloInfinity, ILOFLOAT);

        // plant capacity
        for (int i = 0; i < inst.nI; ++i)
        {
            IloExpr e(env);
            for (int j = 0; j < inst.nJ; ++j)
                e += u[idx2(i, j, inst.nJ)];
            m.add(e <= inst.p[i] * (double)x[i]);
            e.end();
        }
        // depot capacity
        for (int j = 0; j < inst.nJ; ++j)
        {
            IloExpr e(env);
            for (int k = 0; k < inst.nK; ++k)
                e += v[idx2(j, k, inst.nK)];
            m.add(e <= inst.q[j] * (double)y[j]);
            e.end();
        }
        // depot balance
        for (int j = 0; j < inst.nJ; ++j)
        {
            IloExpr e(env);
            for (int i = 0; i < inst.nI; ++i)
                e += u[idx2(i, j, inst.nJ)];
            for (int k = 0; k < inst.nK; ++k)
                e -= v[idx2(j, k, inst.nK)];
            m.add(e == 0.0);
            e.end();
        }
        // demand
        for (int k = 0; k < inst.nK; ++k)
        {
            IloExpr e(env);
            for (int j = 0; j < inst.nJ; ++j)
                e += v[idx2(j, k, inst.nK)];
            m.add(e == inst.r[k]);
            e.end();
        }

        // objective
        IloExpr obj(env);
        for (int i = 0; i < inst.nI; ++i)
            for (int j = 0; j < inst.nJ; ++j)
                obj += inst.C(i, j) * u[idx2(i, j, inst.nJ)];
        for (int j = 0; j < inst.nJ; ++j)
            for (int k = 0; k < inst.nK; ++k)
                obj += inst.D(j, k) * v[idx2(j, k, inst.nK)];
        m.add(IloMinimize(env, obj));
        obj.end();

        if (!cplex.solve())
            throw std::runtime_error("Flow LP infeasible (unexpected).");

        res.u.assign(inst.nI * inst.nJ, 0.0);
        res.v.assign(inst.nJ * inst.nK, 0.0);
        for (int i = 0; i < inst.nI; ++i)
            for (int j = 0; j < inst.nJ; ++j)
                res.u[idx2(i, j, inst.nJ)] = cplex.getValue(u[idx2(i, j, inst.nJ)]);
        for (int j = 0; j < inst.nJ; ++j)
            for (int k = 0; k < inst.nK; ++k)
                res.v[idx2(j, k, inst.nK)] = cplex.getValue(v[idx2(j, k, inst.nK)]);
        res.cost = cplex.getObjValue();
    }
    catch (...)
    {
        env.end();
        throw;
    }
    env.end();
    return res;
}

// ============================================================
// Main
// ============================================================
int main(int argc, char **argv)
{
    // Default instance if none is provided
    const char *DEFAULT_INST = "../instances/tscfl/tscfl_11_50.txt";
    std::string path = (argc >= 2) ? argv[1] : DEFAULT_INST;

    // Keep optional time limit logic disabled (no CLI arg); leave wiring in case you enable later
    double TL = -1.0; // no time limit
    const double EPS = 1e-6;

    try
    {
        std::cout << "Using instance: " << path << "\n";
        TSCFLInstance inst = TSCFLInstance::fromTxt(path);

        IloEnv env;
        IloModel master(env);
        IloCplex cplex(master);

        // variables
        IloBoolVarArray x(env, inst.nI);
        IloBoolVarArray y(env, inst.nJ);
        for (int i = 0; i < inst.nI; ++i)
            x[i] = IloBoolVar(env);
        for (int j = 0; j < inst.nJ; ++j)
            y[j] = IloBoolVar(env);
        IloNumVar eta(env, 0.0, IloInfinity, ILOFLOAT);

        // capacity aggregations (keep subproblem feasible)
        {
            IloExpr e1(env), e2(env);
            for (int i = 0; i < inst.nI; ++i)
                e1 += inst.p[i] * x[i];
            for (int j = 0; j < inst.nJ; ++j)
                e2 += inst.q[j] * y[j];
            master.add(e1 >= inst.demandTotal());
            master.add(e2 >= inst.demandTotal());
            e1.end();
            e2.end();
        }

        // objective: sum f_i x_i + sum g_j y_j + eta
        {
            IloExpr obj(env);
            for (int i = 0; i < inst.nI; ++i)
                obj += inst.f[i] * x[i];
            for (int j = 0; j < inst.nJ; ++j)
                obj += inst.g[j] * y[j];
            obj += eta;
            master.add(IloMinimize(env, obj));
            obj.end();
        }

        // parameters (hierarchical API)
        if (TL > 0)
            cplex.setParam(IloCplex::Param::TimeLimit, TL);
        cplex.setParam(IloCplex::Param::Threads, 0); // let CPLEX choose
        // Required so legacy (lazy/user) callbacks are honored
        cplex.setParam(IloCplex::Param::MIP::Strategy::Search, IloCplex::Traditional);

        // worker dual
        WorkerDual worker(inst, /*log=*/false);

        // register callbacks
        cplex.use(new (env) LazyBendersCallbackI(env, inst, worker, x, y, eta, EPS));
        cplex.use(new (env) UserBendersCallbackI(env, inst, worker, x, y, eta, EPS));

        // solve master
        if (!cplex.solve())
        {
            std::cout << "No solution found.\n";
            env.end();
            return 2;
        }

        // extract integer x,y
        std::vector<int> xsol(inst.nI, 0), ysol(inst.nJ, 0);
        for (int i = 0; i < inst.nI; ++i)
            xsol[i] = (cplex.getValue(x[i]) > 0.5) ? 1 : 0;
        for (int j = 0; j < inst.nJ; ++j)
            ysol[j] = (cplex.getValue(y[j]) > 0.5) ? 1 : 0;

        // recover flows and detailed costs
        FlowSolution flows = recoverFlows(inst, xsol, ysol, /*timelimit=*/-1.0, /*log=*/false);

        double fixed = 0.0;
        for (int i = 0; i < inst.nI; ++i)
            fixed += inst.f[i] * xsol[i];
        for (int j = 0; j < inst.nJ; ++j)
            fixed += inst.g[j] * ysol[j];
        double total = fixed + flows.cost;

        // report
        std::cout.setf(std::ios::fixed);
        std::cout.precision(6);
        IloAlgorithm::Status st = cplex.getStatus();
        std::cout << "Status: " << (st == IloAlgorithm::Optimal ? "Optimal" : "Feasible") << "\n";
        std::cout << "MIP gap: " << cplex.getMIPRelativeGap() << "\n";
        std::cout << "Nodes  : " << cplex.getNnodes() << "\n";
        std::cout << "Time(s): " << cplex.getTime() << "\n";
        std::cout << "eta*   : " << cplex.getValue(eta) << "\n";
        std::cout << "Fixed  : " << fixed << "\n";
        std::cout << "Flow   : " << flows.cost << "\n";
        std::cout << "OBJ    : " << total << "\n";

        env.end();
        return 0;
    }
    catch (const IloException &e)
    {
        std::cerr << "CPLEX/Concert exception: " << e << "\n";
        return 3;
    }
    catch (const std::exception &e)
    {
        std::cerr << "std::exception: " << e.what() << "\n";
        return 4;
    }
    catch (...)
    {
        std::cerr << "Unknown exception.\n";
        return 5;
    }
}