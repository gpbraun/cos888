// -------------------------------------------------------------- -*- C++ -*-
// COS888 – Two-Stage Capacitated Facility Location (TSCFL)
// CP Optimizer model with integer flows + global constraints (COUNT / ATMOST via count)
// Uses same TSCFLInstance layout as your Benders code.
//
// Build:
//   g++ -O3 -std=c++17 tscfl_cp.cpp -o tscfl_cp -lcp -lconcert -lpthread -ldl -lm
//
// Run:
//   ./tscfl_cp ../instances/tscfl/tscfl_11_50.txt [TIME_LIMIT_SECONDS]
//
// Notes:
// * Flows A(i,j), B(j,k) are integers.
// * Linking with IloIfThen (no big-M).
// * “At most” is modeled as IloCount(array,1) <= capacity.
// * COUNT needs an IloIntVarArray: we channel Bool vars to Int vars (0/1).
// * Search phases are passed as an IloSearchPhaseArray.
//
// --------------------------------------------------------------

#include <ilcp/cp.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>
#include <climits>

static inline int idx2(int i, int j, int ncols) { return i * ncols + j; }

// ========================== Instance (same as your Benders file) ==========================
struct TSCFLInstance
{
    int nI{0}, nJ{0}, nK{0};
    std::vector<double> f; // size nI
    std::vector<double> g; // size nJ
    std::vector<double> c; // size nI*nJ (row-major i,j)
    std::vector<double> d; // size nJ*nK (row-major j,k)
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

// ========================== helpers ==========================
static inline IloInt ceil_int(double x)
{
    if (x <= 0.0)
        return 0;
    double y = std::ceil(x);
    return (y > static_cast<double>(INT_MAX)) ? INT_MAX : static_cast<IloInt>(y);
}
static inline IloInt ll_to_int(long long v)
{
    if (v < 0)
        return 0;
    return (v > static_cast<long long>(INT_MAX)) ? INT_MAX : static_cast<IloInt>(v);
}

// ========================== main ==========================
int main(int argc, const char *argv[])
{
    const char *DEFAULT_INST = "../instances/tscfl/tscfl_11_50.txt";
    std::string path = (argc >= 2) ? argv[1] : DEFAULT_INST;
    double TL = (argc >= 3) ? std::atof(argv[2]) : -1.0;

    std::cout << "Using instance: " << path << "\n";
    TSCFLInstance inst;
    try
    {
        inst = TSCFLInstance::fromTxt(path);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Reading error: " << e.what() << "\n";
        return 2;
    }

    // integer RHS / UBs
    const long long sumR_ll = static_cast<long long>(std::llround(inst.demandTotal()));
    const IloInt sumR = ll_to_int(sumR_ll);

    std::vector<IloInt> R(inst.nK);
    for (int k = 0; k < inst.nK; ++k)
        R[k] = ll_to_int(static_cast<long long>(std::llround(inst.r[k])));

    std::vector<IloInt> P(inst.nI), Q(inst.nJ);
    for (int i = 0; i < inst.nI; ++i)
        P[i] = std::min(ceil_int(inst.p[i]), sumR);
    for (int j = 0; j < inst.nJ; ++j)
        Q[j] = std::min(ceil_int(inst.q[j]), sumR);

    IloEnv env;
    try
    {
        IloModel model(env);

        // ---------- binaries ----------
        IloBoolVarArray x(env, inst.nI); // plants
        IloBoolVarArray y(env, inst.nJ); // warehouses
        for (int i = 0; i < inst.nI; ++i)
            x[i].setName((std::string("x_") + std::to_string(i)).c_str());
        for (int j = 0; j < inst.nJ; ++j)
            y[j].setName((std::string("y_") + std::to_string(j)).c_str());

        // ---------- integer flows ----------
        IloArray<IloIntVarArray> A(env, inst.nI); // plant->warehouse
        for (int i = 0; i < inst.nI; ++i)
        {
            A[i] = IloIntVarArray(env, inst.nJ);
            for (int j = 0; j < inst.nJ; ++j)
            {
                A[i][j] = IloIntVar(env, 0, P[i]);
                A[i][j].setName((std::string("A_") + std::to_string(i) + "_" + std::to_string(j)).c_str());
            }
        }
        IloArray<IloIntVarArray> B(env, inst.nJ); // warehouse->customer
        for (int j = 0; j < inst.nJ; ++j)
        {
            B[j] = IloIntVarArray(env, inst.nK);
            for (int k = 0; k < inst.nK; ++k)
            {
                B[j][k] = IloIntVar(env, 0, R[k]);
                B[j][k].setName((std::string("B_") + std::to_string(j) + "_" + std::to_string(k)).c_str());
            }
        }

        // ---------- sums ----------
        std::vector<IloIntExpr> sumA_out;
        sumA_out.reserve(inst.nI);
        std::vector<IloIntExpr> sumA_in(inst.nJ, IloIntExpr(env, 0));
        std::vector<IloIntExpr> sumB_out(inst.nJ, IloIntExpr(env, 0));
        std::vector<IloIntExpr> sumB_in;
        sumB_in.reserve(inst.nK);

        for (int i = 0; i < inst.nI; ++i)
        {
            IloIntExpr s(env, 0);
            for (int j = 0; j < inst.nJ; ++j)
            {
                s += A[i][j];
                sumA_in[j] += A[i][j];
            }
            sumA_out.push_back(s);
        }
        for (int j = 0; j < inst.nJ; ++j)
            for (int k = 0; k < inst.nK; ++k)
                sumB_out[j] += B[j][k];

        for (int k = 0; k < inst.nK; ++k)
        {
            IloIntExpr s(env, 0);
            for (int j = 0; j < inst.nJ; ++j)
                s += B[j][k];
            sumB_in.push_back(s);
        }

        // ======================= Core constraints =======================
        // Demand satisfaction
        for (int k = 0; k < inst.nK; ++k)
            model.add(sumB_in[k] == R[k]);

        // Warehouse conservation
        for (int j = 0; j < inst.nJ; ++j)
            model.add(sumA_in[j] == sumB_out[j]);

        // Plant capacity & linking
        for (int i = 0; i < inst.nI; ++i)
        {
            model.add(sumA_out[i] <= P[i]);                         // capacity
            model.add(IloIfThen(env, x[i] == 0, sumA_out[i] == 0)); // closed ⇒ no outbound
        }

        // Warehouse capacity & linking
        for (int j = 0; j < inst.nJ; ++j)
        {
            model.add(sumA_in[j] <= Q[j]); // capacity (inbound)
            model.add(IloIfThen(env, y[j] == 0, sumA_in[j] == 0));
            model.add(IloIfThen(env, y[j] == 0, sumB_out[j] == 0));
        }

        // Strong global capacity feasibility (useful pruning)
        {
            IloNumExpr plantsCap(env, 0.0), whCap(env, 0.0);
            for (int i = 0; i < inst.nI; ++i)
                plantsCap += inst.p[i] * x[i];
            for (int j = 0; j < inst.nJ; ++j)
                whCap += inst.q[j] * y[j];
            model.add(plantsCap >= static_cast<IloNum>(sumR_ll));
            model.add(whCap >= static_cast<IloNum>(sumR_ll));
            plantsCap.end();
            whCap.end();
        }

        // ====================== GLOBALS: presence + COUNT ======================
        // Presence (nonzero) booleans
        IloArray<IloBoolVarArray> Upos(env, inst.nI); // nonzero A
        for (int i = 0; i < inst.nI; ++i)
        {
            Upos[i] = IloBoolVarArray(env, inst.nJ);
            for (int j = 0; j < inst.nJ; ++j)
                Upos[i][j] = IloBoolVar(env, 0, 1);
        }
        IloArray<IloBoolVarArray> Vpos(env, inst.nJ); // nonzero B
        for (int j = 0; j < inst.nJ; ++j)
        {
            Vpos[j] = IloBoolVarArray(env, inst.nK);
            for (int k = 0; k < inst.nK; ++k)
                Vpos[j][k] = IloBoolVar(env, 0, 1);
        }

        // Reified zero/positive links (no big-M)
        for (int i = 0; i < inst.nI; ++i)
        {
            for (int j = 0; j < inst.nJ; ++j)
            {
                model.add(IloIfThen(env, Upos[i][j] == 0, A[i][j] == 0));
                if (P[i] >= 1)
                    model.add(IloIfThen(env, Upos[i][j] == 1, A[i][j] >= 1));
                // Optional: closed plant forbids positive legs out of i
                model.add(IloIfThen(env, x[i] == 0, Upos[i][j] == 0));
            }
        }
        for (int j = 0; j < inst.nJ; ++j)
        {
            for (int k = 0; k < inst.nK; ++k)
            {
                model.add(IloIfThen(env, Vpos[j][k] == 0, B[j][k] == 0));
                if (R[k] >= 1)
                    model.add(IloIfThen(env, Vpos[j][k] == 1, B[j][k] >= 1));
                // Closed warehouse forbids any positive leg:
                model.add(IloIfThen(env, y[j] == 0, Vpos[j][k] == 0));
            }
        }

        // COUNT requires IntVarArray → channel Bool→Int (0/1)
        IloArray<IloIntVarArray> UposInt(env, inst.nI);
        for (int i = 0; i < inst.nI; ++i)
        {
            UposInt[i] = IloIntVarArray(env, inst.nJ);
            for (int j = 0; j < inst.nJ; ++j)
            {
                UposInt[i][j] = IloIntVar(env, 0, 1);
                model.add(UposInt[i][j] == Upos[i][j]); // channeling
            }
        }
        IloArray<IloIntVarArray> VposInt(env, inst.nJ);
        for (int j = 0; j < inst.nJ; ++j)
        {
            VposInt[j] = IloIntVarArray(env, inst.nK);
            for (int k = 0; k < inst.nK; ++k)
            {
                VposInt[j][k] = IloIntVar(env, 0, 1);
                model.add(VposInt[j][k] == Vpos[j][k]); // channeling
            }
        }

        // Count nonzero outbound arcs from plant i
        IloIntVarArray nzA(env, inst.nI, 0, inst.nJ);
        for (int i = 0; i < inst.nI; ++i)
        {
            model.add(nzA[i] == IloCount(UposInt[i], 1)); // GCC: COUNT
            model.add(nzA[i] <= P[i]);                    // ATMOST via count
            model.add(IloIfThen(env, x[i] == 0, nzA[i] == 0));
        }

        // Count nonzero customer legs from warehouse j
        IloIntVarArray nzB(env, inst.nJ, 0, inst.nK);
        for (int j = 0; j < inst.nJ; ++j)
        {
            model.add(nzB[j] == IloCount(VposInt[j], 1)); // GCC: COUNT
            model.add(nzB[j] <= Q[j]);                    // ATMOST via count
            model.add(IloIfThen(env, y[j] == 0, nzB[j] == 0));
        }

        // ====================== Objective ======================
        IloNumExpr obj(env, 0.0);
        for (int i = 0; i < inst.nI; ++i)
            obj += inst.f[i] * x[i];
        for (int j = 0; j < inst.nJ; ++j)
            obj += inst.g[j] * y[j];
        for (int i = 0; i < inst.nI; ++i)
            for (int j = 0; j < inst.nJ; ++j)
                obj += inst.C(i, j) * A[i][j];
        for (int j = 0; j < inst.nJ; ++j)
            for (int k = 0; k < inst.nK; ++k)
                obj += inst.D(j, k) * B[j][k];
        model.add(IloMinimize(env, obj));
        obj.end();

        // ====================== Search phases (array) ======================
        IloIntVarArray binaries(env, inst.nI + inst.nJ);
        for (int i = 0; i < inst.nI; ++i)
            binaries[i] = x[i];
        for (int j = 0; j < inst.nJ; ++j)
            binaries[inst.nI + j] = y[j];

        IloIntVarArray sparsity(env);
        for (int i = 0; i < inst.nI; ++i)
            for (int j = 0; j < inst.nJ; ++j)
                sparsity.add(UposInt[i][j]);
        for (int j = 0; j < inst.nJ; ++j)
            for (int k = 0; k < inst.nK; ++k)
                sparsity.add(VposInt[j][k]);

        IloIntVarArray flows(env);
        for (int i = 0; i < inst.nI; ++i)
            for (int j = 0; j < inst.nJ; ++j)
                flows.add(A[i][j]);
        for (int j = 0; j < inst.nJ; ++j)
            for (int k = 0; k < inst.nK; ++k)
                flows.add(B[j][k]);

        IloSearchPhase sp1 = IloSearchPhase(env, binaries); // open/close first
        IloSearchPhase sp2 = IloSearchPhase(env, sparsity); // then choose used legs
        IloSearchPhase sp3 = IloSearchPhase(env, flows);    // finally route quantities
        IloSearchPhaseArray phases(env);
        phases.add(sp1);
        phases.add(sp2);
        phases.add(sp3);

        IloCP cp(model);
        if (TL > 0)
            cp.setParameter(IloCP::TimeLimit, TL);
        cp.setSearchPhases(phases);

        if (cp.solve())
        {
            cp.out() << "\n"
                     << cp.getStatus() << " — Objective = " << cp.getObjValue() << "\n";

            double fixed = 0.0, flow = 0.0;
            int openI = 0, openJ = 0;

            for (int i = 0; i < inst.nI; ++i)
            {
                int xi = cp.getValue(x[i]);
                openI += xi;
                fixed += inst.f[i] * xi;
            }
            for (int j = 0; j < inst.nJ; ++j)
            {
                int yj = cp.getValue(y[j]);
                openJ += yj;
                fixed += inst.g[j] * yj;
            }

            for (int i = 0; i < inst.nI; ++i)
                for (int j = 0; j < inst.nJ; ++j)
                    flow += inst.C(i, j) * cp.getValue(A[i][j]);
            for (int j = 0; j < inst.nJ; ++j)
                for (int k = 0; k < inst.nK; ++k)
                    flow += inst.D(j, k) * cp.getValue(B[j][k]);

            cp.out() << "Open plants: " << openI << " / " << inst.nI
                     << " | Open warehouses: " << openJ << " / " << inst.nJ << "\n";
            cp.out() << "Fixed cost : " << fixed << "\n";
            cp.out() << "Flow cost  : " << flow << "\n";
            cp.out() << "Total cost : " << (fixed + flow) << "\n";

            long long totA = 0, totB = 0;
            for (int i = 0; i < inst.nI; ++i)
                for (int j = 0; j < inst.nJ; ++j)
                    totA += cp.getValue(A[i][j]);
            for (int j = 0; j < inst.nJ; ++j)
                for (int k = 0; k < inst.nK; ++k)
                    totB += cp.getValue(B[j][k]);
            cp.out() << "Total A(i→j): " << totA
                     << " | Total B(j→k): " << totB
                     << " | Sum of demands: " << sumR_ll << "\n";
        }
        else
        {
            cp.out() << "No solution. Status: " << cp.getStatus() << "\n";
        }
    }
    catch (const IloException &e)
    {
        std::cerr << "CP Optimizer exception: " << e << "\n";
        env.end();
        return 3;
    }
    catch (...)
    {
        std::cerr << "Unknown exception.\n";
        env.end();
        return 4;
    }

    env.end();
    return 0;
}
