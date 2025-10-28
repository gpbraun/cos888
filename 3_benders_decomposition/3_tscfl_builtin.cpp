/*
COS888

TSCFL por Benders embutido no CPLEX (Built-in)

Gabriel Braun, 2025
*/

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <cstdlib>
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
    std::vector<double> c; // c_ij = custo unitário i->j   (nI*nJ)
    std::vector<double> d; // d_jk = custo unitário j->k   (nJ*nK)
    std::vector<double> p; // p_i  = capacidade planta i
    std::vector<double> q; // q_j  = capacidade depósito j
    std::vector<double> r; // r_k  = demanda cliente k

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
//  SOLVER: Benders embutido no CPLEX
// =====================================================================

class BendersBuiltinTSCFL
{
public:
    const TSCFLInstance &inst;
    int time_limit;
    bool log_output;

    BendersBuiltinTSCFL(const TSCFLInstance &inst_,
                        int time_limit_ = 0,
                        bool log_output_ = true)
        : inst(inst_), time_limit(time_limit_), log_output(log_output_)
    {
    }

    void solve()
    {
        IloEnv env;
        try
        {
            IloModel mdl(env);

            // VARIÁVEIS
            IloBoolVarArray a(env, inst.nI); // plantas
            IloBoolVarArray b(env, inst.nJ); // depósitos
            for (int i = 0; i < inst.nI; ++i)
                a[i] = IloBoolVar(env);
            for (int j = 0; j < inst.nJ; ++j)
                b[j] = IloBoolVar(env);

            IloNumVarArray x(env, inst.nI * inst.nJ, 0.0, IloInfinity, ILOFLOAT); // fluxos i->j
            IloNumVarArray y(env, inst.nJ * inst.nK, 0.0, IloInfinity, ILOFLOAT); // fluxos j->k

            // RESTRIÇÕES
            // Capacidade das plantas
            for (int i = 0; i < inst.nI; ++i)
            {
                IloExpr lhs(env);
                for (int j = 0; j < inst.nJ; ++j)
                    lhs += x[idx2(i, j, inst.nJ)];
                mdl.add(lhs <= inst.p[i] * a[i]);
                lhs.end();
            }
            // Capacidade dos depósitos
            for (int j = 0; j < inst.nJ; ++j)
            {
                IloExpr lhs(env);
                for (int k = 0; k < inst.nK; ++k)
                    lhs += y[idx2(j, k, inst.nK)];
                mdl.add(lhs <= inst.q[j] * b[j]);
                lhs.end();
            }
            // Balanço nos depósitos
            for (int j = 0; j < inst.nJ; ++j)
            {
                IloExpr bal(env);
                for (int i = 0; i < inst.nI; ++i)
                    bal += x[idx2(i, j, inst.nJ)];
                for (int k = 0; k < inst.nK; ++k)
                    bal -= y[idx2(j, k, inst.nK)];
                mdl.add(bal == 0.0);
                bal.end();
            }
            // Atende demanda
            for (int k = 0; k < inst.nK; ++k)
            {
                IloExpr dem(env);
                for (int j = 0; j < inst.nJ; ++j)
                    dem += y[idx2(j, k, inst.nK)];
                mdl.add(dem == inst.r[k]);
                dem.end();
            }

            // OBJETIVO
            {
                IloExpr obj(env);
                for (int i = 0; i < inst.nI; ++i)
                    obj += inst.f[i] * a[i];
                for (int j = 0; j < inst.nJ; ++j)
                    obj += inst.g[j] * b[j];
                for (int i = 0; i < inst.nI; ++i)
                    for (int j = 0; j < inst.nJ; ++j)
                        obj += inst.C(i, j) * x[idx2(i, j, inst.nJ)];
                for (int j = 0; j < inst.nJ; ++j)
                    for (int k = 0; k < inst.nK; ++k)
                        obj += inst.D(j, k) * y[idx2(j, k, inst.nK)];
                mdl.add(IloMinimize(env, obj));
                obj.end();
            }

            // CPLEX + Benders
            IloCplex cplex(mdl);
            if (!log_output)
            {
                cplex.setOut(env.getNullStream());
                cplex.setWarning(env.getNullStream());
            }
            if (time_limit > 0)
                cplex.setParam(IloCplex::Param::TimeLimit, static_cast<double>(time_limit));
            cplex.setParam(IloCplex::Param::Threads, 0);
            cplex.setParam(IloCplex::Param::Benders::Strategy, IloCplex::BendersFull);

            // SOLVE
            if (!cplex.solve())
            {
                std::cout << "No solution found.\n";
                env.end();
                return;
            }

            // RELATÓRIO
            std::cout.setf(std::ios::fixed);
            std::cout.precision(6);
            IloAlgorithm::Status st = cplex.getStatus();
            std::cout << "Status   : " << (st == IloAlgorithm::Optimal ? "Optimal" : "Feasible") << "\n";
            std::cout << "Objective: " << cplex.getObjValue() << "\n";
            std::cout << "Best LB  : " << cplex.getBestObjValue() << "\n";
            std::cout << "MIP gap  : " << cplex.getMIPRelativeGap() << "\n";
            std::cout << "Nodes    : " << cplex.getNnodes() << "\n";
            std::cout << "Time(s)  : " << cplex.getTime() << "\n";

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
    const std::string PATH = "../instances/tscfl/tscfl_13_100.txt";

    try
    {
        TSCFLInstance instance = TSCFLInstance::from_txt(PATH);

        // inst, time_limit, log
        BendersBuiltinTSCFL solver(instance, 200, true);
        solver.solve();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Erro: " << e.what() << "\n";
        return 2;
    }
    return 0;
}
