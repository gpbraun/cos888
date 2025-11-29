/*
COS888

Resolve o TSCFL por decomposição de Benders (callbacks CPLEX).

Gabriel Braun, 2025
*/

#pragma once

#include <ilcplex/ilocplex.h>
#include <iostream>
#include <numeric>
#include <string>
#include <chrono>
#include <memory>
#include <stdexcept>

#include "tscfl_instance.hpp"
#include "benders_workers/tscfl_benders_worker_dual.hpp"
// #include "benders_workers/tscfl_benders_worker_primal.hpp"
// #include "benders_workers/tscfl_benders_worker_net.hpp"

ILOSTLBEGIN

// =====================================================================
//  CALLBACKS (Lazy + User cuts)
// =====================================================================

class LazyBendersCallbackI : public IloCplex::LazyConstraintCallbackI
{
    const TSCFLInstance &inst;
    Worker &worker;
    IloBoolVarArray a;
    IloBoolVarArray b;
    IloNumVar eta;

public:
    LazyBendersCallbackI(
        IloEnv env,
        const TSCFLInstance &inst_,
        Worker &worker_,
        IloBoolVarArray a_,
        IloBoolVarArray b_,
        IloNumVar eta_)
        : IloCplex::LazyConstraintCallbackI(env),
          inst(inst_),
          worker(worker_),
          a(a_),
          b(b_),
          eta(eta_)
    {
    }

    IloCplex::CallbackI *duplicateCallback() const override
    {
        return (new (getEnv()) LazyBendersCallbackI(*this));
    }

    void main() override
    {
        // 1) Lê (a,b,eta) da solução corrente
        Vec av(inst.nI), bv(inst.nJ);
        for (int i = 0; i < inst.nI; ++i)
            av[i] = getValue(a[i]);
        for (int j = 0; j < inst.nJ; ++j)
            bv[j] = getValue(b[j]);
        double eta_val = getValue(eta);

        // 2) Resolve worker
        double theta, rhs;
        Vec coef_a, coef_b;
        worker.solve(av, bv, theta, coef_a, coef_b, rhs);

        // 3) Se corte violado, adiciona
        if (theta - eta_val > EPS)
        {
            IloEnv env = getEnv();
            IloExpr lin(env);

            lin += rhs;
            for (int i = 0; i < inst.nI; ++i)
                lin += coef_a[i] * a[i];
            for (int j = 0; j < inst.nJ; ++j)
                lin += coef_b[j] * b[j];

            add(eta >= lin);
            lin.end();
        }
    }
};

class UserBendersCallbackI : public IloCplex::UserCutCallbackI
{
    const TSCFLInstance &inst;
    Worker &worker;
    IloBoolVarArray a;
    IloBoolVarArray b;
    IloNumVar eta;

public:
    UserBendersCallbackI(
        IloEnv env,
        const TSCFLInstance &inst_,
        Worker &worker_,
        IloBoolVarArray a_,
        IloBoolVarArray b_,
        IloNumVar eta_)
        : IloCplex::UserCutCallbackI(env),
          inst(inst_),
          worker(worker_),
          a(a_),
          b(b_),
          eta(eta_)
    {
    }

    IloCplex::CallbackI *duplicateCallback() const override
    {
        return (new (getEnv()) UserBendersCallbackI(*this));
    }

    void main() override
    {
        // 1) Lê (a,b,eta) da solução corrente
        Vec av(inst.nI), bv(inst.nJ);
        for (int i = 0; i < inst.nI; ++i)
            av[i] = getValue(a[i]);
        for (int j = 0; j < inst.nJ; ++j)
            bv[j] = getValue(b[j]);
        double eta_val = getValue(eta);

        // 2) Resolve worker
        double theta, rhs;
        Vec coef_a, coef_b;
        worker.solve(av, bv, theta, coef_a, coef_b, rhs);

        // 3) Se corte violado, adiciona
        if (theta - eta_val > EPS)
        {
            IloEnv env = getEnv();
            IloExpr lin(env);

            lin += rhs;
            for (int i = 0; i < inst.nI; ++i)
                lin += coef_a[i] * a[i];
            for (int j = 0; j < inst.nJ; ++j)
                lin += coef_b[j] * b[j];

            add(eta >= lin, IloCplex::UseCutPurge);
            lin.end();
        }
    }
};

// =====================================================================
//  BENDERS
// =====================================================================

class TSCFLSolverBenders
{
public:
    const TSCFLInstance &inst;

    // Resultados:
    // lb   = melhor limite inferior
    // ub   = melhor solução viável
    // gap  = gap relativo entre lb e ub (do CPLEX)
    // time = tempo total de solve (segundos)
    double lb{0.0};
    double ub{0.0};
    double gap{0.0};
    double time{0.0};
    IloInt64 nodes{0};
    IloAlgorithm::Status status{IloAlgorithm::Unknown};

private:
    IloEnv env;
    IloModel master;
    IloCplex cplex;

    IloBoolVarArray a; // a_i  = abre planta i
    IloBoolVarArray b; // b_j  = abre depósito j
    IloNumVar eta;     // variável para custo de segundo estágio

    std::unique_ptr<Worker> worker;

public:
    // mode:
    // 0 -> WorkerDual  (default)
    // 1 -> WorkerPrimal
    // 2 -> WorkerNet
    explicit TSCFLSolverBenders(const TSCFLInstance &inst_, int mode = 0)
        : inst(inst_),
          env(),
          master(env),
          cplex(env),
          a(env, inst_.nI),
          b(env, inst_.nJ),
          eta(env, 0.0, IloInfinity, ILOFLOAT),
          worker(nullptr)
    {
        switch (mode)
        {
        case 0:
            worker = std::make_unique<WorkerDual>(inst_);
            break;
        case 1:
            throw std::invalid_argument("WorkerPrimal não implementado!");
            // worker = std::make_unique<WorkerPrimal>(inst_);
            // break;
        case 2:
            throw std::invalid_argument("WorkerNet não implementado!");
            // worker = std::make_unique<WorkerNet>(inst_);
            // break;
        default:
            throw std::invalid_argument("Invalid Benders worker mode (must be 0, 1, or 2).");
        }

        build_master();
        cplex.extract(master);

        // Benders com callbacks precisa de "Traditional"  search
        cplex.setParam(IloCplex::Param::MIP::Strategy::Search, IloCplex::Traditional);
        cplex.setParam(IloCplex::Param::Threads, 0);
    }

    ~TSCFLSolverBenders()
    {
        cplex.end();
        master.end();
        env.end();
    }

private:
    void build_master()
    {
        const int nI = inst.nI;
        const int nJ = inst.nJ;

        // Capacidade agregada (garante viabilidade do subproblema)
        {
            IloExpr e1(env), e2(env);
            double demand_total = std::accumulate(inst.r.begin(), inst.r.end(), 0.0);

            for (int i = 0; i < nI; ++i)
                e1 += inst.p[i] * a[i];
            for (int j = 0; j < nJ; ++j)
                e2 += inst.q[j] * b[j];

            master.add(e1 >= demand_total);
            master.add(e2 >= demand_total);
            e1.end();
            e2.end();
        }

        // OBJETIVO: custo fixo + eta
        {
            IloExpr obj(env);
            for (int i = 0; i < nI; ++i)
                obj += inst.f[i] * a[i];
            for (int j = 0; j < nJ; ++j)
                obj += inst.g[j] * b[j];
            obj += eta;
            master.add(IloMinimize(env, obj));
            obj.end();
        }
    }

public:
    bool solve(bool log_output = true, double time_limit = -1.0)
    {
        if (log_output)
        {
            cplex.setOut(env.out());
            cplex.setWarning(env.out());
        }
        else
        {
            cplex.setOut(env.getNullStream());
            cplex.setWarning(env.getNullStream());
        }

        // Parâmetros
        cplex.setParam(IloCplex::Param::MIP::Tolerances::MIPGap, MIP_GAP);

        if (time_limit > 0.0)
            cplex.setParam(IloCplex::Param::TimeLimit, time_limit);

        // Callbacks
        cplex.use(new (env) LazyBendersCallbackI(env, inst, *worker, a, b, eta));
        cplex.use(new (env) UserBendersCallbackI(env, inst, *worker, a, b, eta));

        // Solve
        auto t0 = std::chrono::steady_clock::now();
        bool ok = cplex.solve();
        auto t1 = std::chrono::steady_clock::now();

        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(t1 - t0).count();

        // Estatísticas
        status = cplex.getStatus();
        gap = cplex.getMIPRelativeGap();
        nodes = cplex.getNnodes64();
        time = static_cast<int>(elapsed);

        if (ok)
        {
            ub = cplex.getObjValue();
            lb = cplex.getBestObjValue();

            std::cout << "\n[BENDERS] Solved.\n";
            std::cout << "UB     = " << ub << "\n";
            std::cout << "LB     = " << lb << "\n";
            std::cout << "status = " << status << "\n";
            std::cout << "gap    = " << gap << "\n";
            std::cout << "nodes  = " << nodes << "\n";
            std::cout << "time   = " << time << "s\n";
            std::cout << "eta*   = " << cplex.getValue(eta) << "\n";
        }
        else
        {
            std::cerr << "\n[BENDERS] No solution. status = " << status << "\n";
            std::cerr << "nodes = " << nodes << "\n";
            std::cerr << "time  = " << time << "s\n";
        }

        return ok;
    }
};
