// tscfl_benders_builtin.cpp
// COS888 — TSCFL solved with CPLEX **built-in Benders decomposition**
// We build the full MIP (x,y,u,v), annotate continuous flow vars (u,v)
// to subproblem 1, keep binaries (x,y) in the master, and let CPLEX do the rest.
//
// Usage:
//   ./tscfl_benders_builtin INSTANCE.txt [time_limit_seconds] [auto]
// If the 3rd arg is "auto", we let CPLEX auto-decompose (BendersFull).
// Otherwise we use our user annotations (BendersUser).

#include <ilcplex/ilocplex.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <cstdlib>
#include <stdexcept>

ILOSTLBEGIN

static inline int IDX2(int i, int j, int ncols) { return i * ncols + j; }

// =============================== Instance ===============================
struct TSCFLInstance
{
    int nI{0}, nJ{0}, nK{0};
    std::vector<double> f; // |I|
    std::vector<double> g; // |J|
    std::vector<double> c; // |I|*|J|  (i,j)
    std::vector<double> d; // |J|*|K|  (j,k)
    std::vector<double> p; // |I|
    std::vector<double> q; // |J|
    std::vector<double> r; // |K|

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
        inst.nI = (int)a[pos++];
        inst.nJ = (int)a[pos++];
        inst.nK = (int)a[pos++];

        const int nI = inst.nI, nJ = inst.nJ, nK = inst.nK;

        // r (|K|)
        if (pos + nK > a.size())
            throw std::runtime_error("Malformed file (r).");
        inst.r.assign(a.begin() + pos, a.begin() + pos + nK);
        pos += nK;

        // (q,g) for j in J
        if (pos + 2 * nJ > a.size())
            throw std::runtime_error("Malformed file (q,g).");
        inst.q.resize(nJ);
        inst.g.resize(nJ);
        for (int j = 0; j < nJ; ++j)
        {
            inst.q[j] = a[pos++];
            inst.g[j] = a[pos++];
        }

        // c (|I|*|J|)
        if (pos + nI * nJ > a.size())
            throw std::runtime_error("Malformed file (c).");
        inst.c.assign(a.begin() + pos, a.begin() + pos + nI * nJ);
        pos += nI * nJ;

        // (p,f) for i in I
        if (pos + 2 * nI > a.size())
            throw std::runtime_error("Malformed file (p,f).");
        inst.p.resize(nI);
        inst.f.resize(nI);
        for (int i = 0; i < nI; ++i)
        {
            inst.p[i] = a[pos++];
            inst.f[i] = a[pos++];
        }

        // d (|J|*|K|)
        if (pos + nJ * nK > a.size())
            throw std::runtime_error("Malformed file (d).");
        inst.d.assign(a.begin() + pos, a.begin() + pos + nJ * nK);
        pos += nJ * nK;

        return inst;
    }

    inline double C(int i, int j) const { return c[IDX2(i, j, nJ)]; }
    inline double D(int j, int k) const { return d[IDX2(j, k, nK)]; }
    inline double demandTotal() const
    {
        return std::accumulate(r.begin(), r.end(), 0.0);
    }
};

// =============================== Main ===============================
int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " INSTANCE.txt [time_limit_seconds] [auto]\n";
        return 1;
    }

    const std::string inst_path = argv[1];
    double time_limit = -1.0;
    if (argc >= 3)
        time_limit = std::atof(argv[2]);
    const bool use_auto = (argc >= 4) && (std::string(argv[3]) == "auto");

    try
    {
        TSCFLInstance inst = TSCFLInstance::fromTxt(inst_path);

        IloEnv env;
        IloModel model(env);

        // ---------- Variables ----------
        // Master (binaries)
        IloBoolVarArray x(env, inst.nI); // open plant i?
        IloBoolVarArray y(env, inst.nJ); // open depot j?
        for (int i = 0; i < inst.nI; ++i)
            x[i] = IloBoolVar(env);
        for (int j = 0; j < inst.nJ; ++j)
            y[j] = IloBoolVar(env);

        // Subproblem (flows)
        IloNumVarArray u(env, inst.nI * inst.nJ, 0.0, IloInfinity, ILOFLOAT); // u_ij
        IloNumVarArray v(env, inst.nJ * inst.nK, 0.0, IloInfinity, ILOFLOAT); // v_jk

        // ---------- Constraints ----------
        // Plant capacity: sum_j u_ij <= p_i * x_i
        for (int i = 0; i < inst.nI; ++i)
        {
            IloExpr e(env);
            for (int j = 0; j < inst.nJ; ++j)
                e += u[IDX2(i, j, inst.nJ)];
            model.add(e <= inst.p[i] * x[i]);
            e.end();
        }
        // Depot capacity: sum_k v_jk <= q_j * y_j
        for (int j = 0; j < inst.nJ; ++j)
        {
            IloExpr e(env);
            for (int k = 0; k < inst.nK; ++k)
                e += v[IDX2(j, k, inst.nK)];
            model.add(e <= inst.q[j] * y[j]);
            e.end();
        }
        // Depot balance: sum_i u_ij - sum_k v_jk = 0
        for (int j = 0; j < inst.nJ; ++j)
        {
            IloExpr e(env);
            for (int i = 0; i < inst.nI; ++i)
                e += u[IDX2(i, j, inst.nJ)];
            for (int k = 0; k < inst.nK; ++k)
                e -= v[IDX2(j, k, inst.nK)];
            model.add(e == 0.0);
            e.end();
        }
        // Demand satisfaction: sum_j v_jk = r_k
        for (int k = 0; k < inst.nK; ++k)
        {
            IloExpr e(env);
            for (int j = 0; j < inst.nJ; ++j)
                e += v[IDX2(j, k, inst.nK)];
            model.add(e == inst.r[k]);
            e.end();
        }

        // ---------- Objective ----------
        IloExpr obj(env);
        for (int i = 0; i < inst.nI; ++i)
            obj += inst.f[i] * x[i];
        for (int j = 0; j < inst.nJ; ++j)
            obj += inst.g[j] * y[j];
        for (int i = 0; i < inst.nI; ++i)
            for (int j = 0; j < inst.nJ; ++j)
                obj += inst.C(i, j) * u[IDX2(i, j, inst.nJ)];
        for (int j = 0; j < inst.nJ; ++j)
            for (int k = 0; k < inst.nK; ++k)
                obj += inst.D(j, k) * v[IDX2(j, k, inst.nK)];
        model.add(IloMinimize(env, obj));
        obj.end();

        // ---------- CPLEX & Benders ----------
        IloCplex cpx(model);

        if (time_limit > 0)
            cpx.setParam(IloCplex::Param::TimeLimit, time_limit);
        cpx.setParam(IloCplex::Param::Threads, 0); // let CPLEX choose

        if (use_auto)
        {
            // Let CPLEX find a decomposition (and write it if you want)
            cpx.setParam(IloCplex::Param::Benders::Strategy, IloCplex::BendersFull);
            // cpx.writeBendersAnnotation("benders_auto.ann");
        }
        else
        {
            // Use **user annotations**: x,y -> master (0); u,v -> subproblem 1
            cpx.setParam(IloCplex::Param::Benders::Strategy, IloCplex::BendersUser);

            IloCplex::LongAnnotation benders =
                cpx.newLongAnnotation(CPX_BENDERS_ANNOTATION, CPX_BENDERS_MASTERVALUE);

            // Explicitly put binaries in master (optional; default is master)
            for (int i = 0; i < inst.nI; ++i)
                cpx.setAnnotation(benders, x[i], CPX_BENDERS_MASTERVALUE);
            for (int j = 0; j < inst.nJ; ++j)
                cpx.setAnnotation(benders, y[j], CPX_BENDERS_MASTERVALUE);

            // Put all flow variables in subproblem #1
            const IloInt SUB1 = CPX_BENDERS_MASTERVALUE + 1;
            for (int t = 0; t < u.getSize(); ++t)
                cpx.setAnnotation(benders, u[t], SUB1);
            for (int t = 0; t < v.getSize(); ++t)
                cpx.setAnnotation(benders, v[t], SUB1);

            // If you want to inspect the partition:
            // cpx.writeBendersAnnotation("benders_user.ann");
        }

        // ---------- Solve ----------
        if (!cpx.solve())
        {
            std::cerr << "Failed to optimize.\n";
            env.end();
            return 2;
        }

        // ---------- Report ----------
        std::cout.setf(std::ios::fixed);
        std::cout.precision(6);
        std::cout << "Status      : " << cpx.getStatus() << "\n";
        std::cout << "Best bound  : " << cpx.getBestObjValue() << "\n";
        std::cout << "Objective   : " << cpx.getObjValue() << "\n";
        std::cout << "MIP gap     : " << cpx.getMIPRelativeGap() << "\n";
        std::cout << "Nodes       : " << cpx.getNnodes() << "\n";
        std::cout << "Time (s)    : " << cpx.getTime() << "\n";

        // Print openings (compact)
        std::cout << "x (plants): ";
        for (int i = 0; i < inst.nI; ++i)
            std::cout << (cpx.getValue(x[i]) > 0.5);
        std::cout << "\n";
        std::cout << "y (depots): ";
        for (int j = 0; j < inst.nJ; ++j)
            std::cout << (cpx.getValue(y[j]) > 0.5);
        std::cout << "\n";

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
        std::cerr << "Unknown exception\n";
        return 5;
    }
}
