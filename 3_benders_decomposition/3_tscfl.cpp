/*
COS888

TSCFL por Decomposição de Benders

Gabriel Braun, 2025
*/

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <stdexcept>

#include <ilcplex/ilocplex.h>
ILOSTLBEGIN

// =====================================================================
//  UTILS
// =====================================================================

// Acesso em matriz 2D
static inline int idx2(int i, int j, int ncols) { return i * ncols + j; }

static inline std::vector<int> range_int(int n)
{
    std::vector<int> v(n);
    std::iota(v.begin(), v.end(), 0);
    return v;
}
static inline std::vector<std::pair<int, int>> cart_prod(int nA, int nB)
{
    std::vector<std::pair<int, int>> v;
    v.reserve(static_cast<size_t>(nA) * static_cast<size_t>(nB));
    for (int i = 0; i < nA; ++i)
        for (int j = 0; j < nB; ++j)
            v.emplace_back(i, j);
    return v;
}

// =====================================================================
//  INSTÂNCIA
// =====================================================================

//
// Instância do TSCFL
//
class TSCFLInstance
{
public:
    int nI{0}; // |I| plantas
    int nJ{0}; // |J| depósitos
    int nK{0}; // |K| clientes

    std::vector<double> f; // f_i  = custo fixo da planta i
    std::vector<double> g; // g_j  = custo fixo do depósito j
    std::vector<double> c; // c_ij = custo unitário planta i -> depósito j
    std::vector<double> d; // d_jk = custo unitário depósito j -> cliente k
    std::vector<double> p; // p_i  = capacidade da planta i
    std::vector<double> q; // q_j  = capacidade do depósito j
    std::vector<double> r; // r_k  = demanda do cliente k

    inline double C(int i, int j) const { return c[idx2(i, j, nJ)]; }
    inline double &C(int i, int j) { return c[idx2(i, j, nJ)]; }
    inline double D(int j, int k) const { return d[idx2(j, k, nK)]; }
    inline double &D(int j, int k) { return d[idx2(j, k, nK)]; }

    std::vector<int> I() const { return range_int(nI); }
    std::vector<int> J() const { return range_int(nJ); }
    std::vector<int> K() const { return range_int(nK); }

    std::vector<std::pair<int, int>> IJ() const { return cart_prod(nI, nJ); }
    std::vector<std::pair<int, int>> JK() const { return cart_prod(nJ, nK); }

    static TSCFLInstance from_txt(const std::string &path)
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
        if (pos + static_cast<size_t>(nK) > a.size())
            throw std::runtime_error("Malformed file (r).");
        inst.r.assign(a.begin() + pos, a.begin() + pos + nK);
        pos += static_cast<size_t>(nK);

        // (q,g): nJ pares
        if (pos + static_cast<size_t>(2 * nJ) > a.size())
            throw std::runtime_error("Malformed file (q,g).");
        inst.q.resize(nJ);
        inst.g.resize(nJ);
        for (int j = 0; j < nJ; ++j)
        {
            inst.q[j] = a[pos++];
            inst.g[j] = a[pos++];
        }

        // c: nI*nJ
        if (pos + static_cast<size_t>(nI) * static_cast<size_t>(nJ) > a.size())
            throw std::runtime_error("Malformed file (c).");
        inst.c.assign(a.begin() + pos, a.begin() + pos + (static_cast<size_t>(nI) * static_cast<size_t>(nJ)));
        pos += static_cast<size_t>(nI) * static_cast<size_t>(nJ);

        // (p,f): nI pares
        if (pos + static_cast<size_t>(2 * nI) > a.size())
            throw std::runtime_error("Malformed file (p,f).");
        inst.p.resize(nI);
        inst.f.resize(nI);
        for (int i = 0; i < nI; ++i)
        {
            inst.p[i] = a[pos++];
            inst.f[i] = a[pos++];
        }

        // d: nJ*nK
        if (pos + static_cast<size_t>(nJ) * static_cast<size_t>(nK) > a.size())
            throw std::runtime_error("Malformed file (d).");
        inst.d.assign(a.begin() + pos, a.begin() + pos + (static_cast<size_t>(nJ) * static_cast<size_t>(nK)));
        pos += static_cast<size_t>(nJ) * static_cast<size_t>(nK);

        return inst;
    }
};

// =====================================================================
//  WORKER DUAL
// =====================================================================

//
// Worker Dual (construído uma vez; objetivo atualizado por (a,b))
//
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
        cplex.setParam(IloCplex::Param::Threads, 1); // seguro p/ callbacks
        if (log)
            cplex.setOut(std::cout);
    }

    void setObjective(const std::vector<double> &a, const std::vector<double> &b)
    {
        IloExpr e(env);
        for (int i = 0; i < inst.nI; ++i)
            e += (inst.p[i] * a[i]) * alpha[i]; // alpha ≤ 0
        for (int j = 0; j < inst.nJ; ++j)
            e += (inst.q[j] * b[j]) * beta[j]; // beta  ≤ 0
        for (int k = 0; k < inst.nK; ++k)
            e += (inst.r[k]) * delta[k]; // delta livre
        obj.setExpr(e);
        e.end();
    }

    void solve(const std::vector<double> &a,
               const std::vector<double> &b,
               double &theta,
               std::vector<double> &coef_a,
               std::vector<double> &coef_b,
               double &rhs)
    {
        setObjective(a, b);
        if (!cplex.solve())
            throw std::runtime_error("Worker dual failed to solve.");

        theta = cplex.getObjValue();

        coef_a.assign(inst.nI, 0.0);
        for (int i = 0; i < inst.nI; ++i)
            coef_a[i] = inst.p[i] * cplex.getValue(alpha[i]); // ≤ 0

        coef_b.assign(inst.nJ, 0.0);
        for (int j = 0; j < inst.nJ; ++j)
            coef_b[j] = inst.q[j] * cplex.getValue(beta[j]); // ≤ 0

        rhs = 0.0;
        for (int k = 0; k < inst.nK; ++k)
            rhs += inst.r[k] * cplex.getValue(delta[k]);
    }
};

// ============================================================
// Callbacks (lazy e user)
// ============================================================

class LazyBendersCallbackI : public IloCplex::LazyConstraintCallbackI
{
    const TSCFLInstance &inst;
    WorkerDual &worker;
    IloBoolVarArray a;
    IloBoolVarArray b;
    IloNumVar eta;
    double eps;

public:
    LazyBendersCallbackI(IloEnv env,
                         const TSCFLInstance &_inst,
                         WorkerDual &_worker,
                         IloBoolVarArray _a,
                         IloBoolVarArray _b,
                         IloNumVar _eta,
                         double _eps)
        : IloCplex::LazyConstraintCallbackI(env),
          inst(_inst), worker(_worker), a(_a), b(_b), eta(_eta), eps(_eps) {}

    IloCplex::CallbackI *duplicateCallback() const override
    {
        return (new (getEnv()) LazyBendersCallbackI(*this));
    }

    void main() override
    {
        std::vector<double> av(inst.nI), bv(inst.nJ);
        for (int i = 0; i < inst.nI; ++i)
            av[i] = getValue(a[i]);
        for (int j = 0; j < inst.nJ; ++j)
            bv[j] = getValue(b[j]);
        double etaVal = getValue(eta);

        double theta, rhs;
        std::vector<double> coef_a, coef_b;
        worker.solve(av, bv, theta, coef_a, coef_b, rhs);

        if (theta - etaVal > eps)
        {
            IloEnv env = getEnv();
            IloExpr lin(env);
            lin += rhs;
            for (int i = 0; i < inst.nI; ++i)
                lin += coef_a[i] * a[i];
            for (int j = 0; j < inst.nJ; ++j)
                lin += coef_b[j] * b[j];
            add(eta >= lin); // corta incumbente
            lin.end();
        }
    }
};

class UserBendersCallbackI : public IloCplex::UserCutCallbackI
{
    const TSCFLInstance &inst;
    WorkerDual &worker;
    IloBoolVarArray a;
    IloBoolVarArray b;
    IloNumVar eta;
    double eps;

public:
    UserBendersCallbackI(IloEnv env,
                         const TSCFLInstance &_inst,
                         WorkerDual &_worker,
                         IloBoolVarArray _a,
                         IloBoolVarArray _b,
                         IloNumVar _eta,
                         double _eps)
        : IloCplex::UserCutCallbackI(env),
          inst(_inst), worker(_worker), a(_a), b(_b), eta(_eta), eps(_eps) {}

    IloCplex::CallbackI *duplicateCallback() const override
    {
        return (new (getEnv()) UserBendersCallbackI(*this));
    }

    void main() override
    {
        std::vector<double> av(inst.nI), bv(inst.nJ);
        for (int i = 0; i < inst.nI; ++i)
            av[i] = getValue(a[i]);
        for (int j = 0; j < inst.nJ; ++j)
            bv[j] = getValue(b[j]);
        double etaVal = getValue(eta);

        double theta, rhs;
        std::vector<double> coef_a, coef_b;
        worker.solve(av, bv, theta, coef_a, coef_b, rhs);

        if (theta - etaVal > eps)
        {
            IloEnv env = getEnv();
            IloExpr lin(env);
            lin += rhs;
            for (int i = 0; i < inst.nI; ++i)
                lin += coef_a[i] * a[i];
            for (int j = 0; j < inst.nJ; ++j)
                lin += coef_b[j] * b[j];
            add(eta >= lin, IloCplex::UseCutPurge); // corte global
            lin.end();
        }
    }
};

// =====================================================================
//  SOLVER
// =====================================================================

//
// Solver: Benders Decomposition
//
class BendersTSCFL
{
public:
    const TSCFLInstance &inst;
    int time_limit;
    bool log_output;

    BendersTSCFL(const TSCFLInstance &inst_,
                 int time_limit_ = 0,
                 bool log_output_ = true)
        : inst(inst_),
          time_limit(time_limit_), log_output(log_output_) {}

    void solve()
    {
        const double EPS = 1e-6;

        IloEnv env;
        try
        {
            IloModel master(env);
            IloCplex cplex(master);

            // VARIÁVEIS
            IloBoolVarArray a(env, inst.nI);
            IloBoolVarArray b(env, inst.nJ);
            for (int i = 0; i < inst.nI; ++i)
                a[i] = IloBoolVar(env);
            for (int j = 0; j < inst.nJ; ++j)
                b[j] = IloBoolVar(env);
            IloNumVar eta(env, 0.0, IloInfinity, ILOFLOAT);

            // RESTRIÇÕES
            // capacidade agregada (garante viabilidade do subproblema)
            {
                IloExpr e1(env), e2(env);
                double demand_total = std::accumulate(inst.r.begin(), inst.r.end(), 0.0);
                for (int i = 0; i < inst.nI; ++i)
                    e1 += inst.p[i] * a[i];
                for (int j = 0; j < inst.nJ; ++j)
                    e2 += inst.q[j] * b[j];
                master.add(e1 >= demand_total);
                master.add(e2 >= demand_total);
                e1.end();
                e2.end();
            }

            // OBJETIVO
            {
                IloExpr obj(env);
                for (int i = 0; i < inst.nI; ++i)
                    obj += inst.f[i] * a[i];
                for (int j = 0; j < inst.nJ; ++j)
                    obj += inst.g[j] * b[j];
                obj += eta;
                master.add(IloMinimize(env, obj));
                obj.end();
            }

            // parâmetros
            if (time_limit > 0)
                cplex.setParam(IloCplex::Param::TimeLimit, static_cast<double>(time_limit));
            cplex.setParam(IloCplex::Param::Threads, 0);
            cplex.setParam(IloCplex::Param::MIP::Strategy::Search, IloCplex::Traditional); // p/ callbacks
            if (!log_output)
            {
                cplex.setOut(env.getNullStream());
                cplex.setWarning(env.getNullStream());
            }

            // worker dual
            WorkerDual worker(inst, /*log=*/false);

            // callbacks (lazy + user)
            cplex.use(new (env) LazyBendersCallbackI(env, inst, worker, a, b, eta, EPS));
            cplex.use(new (env) UserBendersCallbackI(env, inst, worker, a, b, eta, EPS));

            // solve master
            if (!cplex.solve())
            {
                std::cout << "No solution found.\n";
                env.end();
                return;
            }

            // relatório
            std::cout.setf(std::ios::fixed);
            std::cout.precision(6);
            IloAlgorithm::Status st = cplex.getStatus();
            std::cout << "Status   : " << (st == IloAlgorithm::Optimal ? "Optimal" : "Feasible") << "\n";
            std::cout << "Objective: " << cplex.getObjValue() << "\n";
            std::cout << "Best LB  : " << cplex.getBestObjValue() << "\n";
            std::cout << "MIP gap  : " << cplex.getMIPRelativeGap() << "\n";
            std::cout << "Nodes    : " << cplex.getNnodes() << "\n";
            std::cout << "Time(s)  : " << cplex.getTime() << "\n";
            std::cout << "eta*     : " << cplex.getValue(eta) << "\n";

            env.end();
        }
        catch (...)
        {
            env.end();
            throw;
        }
    }
};

//
// Rotina principal
//
int main()
{
    const std::string PATH = "../instances/tscfl/tscfl_15_100.txt";

    try
    {
        TSCFLInstance instance = TSCFLInstance::from_txt(PATH);

        // inst, time_limit, log
        BendersTSCFL solver(instance, 200, true);
        solver.solve();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Erro: " << e.what() << "\n";
        return 2;
    }
    return 0;
}
