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
#include "benders_workers/tscfl_benders_worker_primal.hpp"
#include "benders_workers/tscfl_benders_worker_net.hpp"

ILOSTLBEGIN

// CALLBACK: Lazy Constraint
class LazyBendersCallbackI : public IloCplex::LazyConstraintCallbackI
{
    const TSCFLInstance &inst;
    Worker &worker;
    IloBoolVarArray a;
    IloBoolVarArray b;
    IloNumVar eta;

    IloNumArray a_vals;
    IloNumArray b_vals;

public:
    LazyBendersCallbackI(
        const TSCFLInstance &inst_,
        Worker &worker_,
        IloBoolVarArray a_,
        IloBoolVarArray b_,
        IloNumVar eta_)
        : IloCplex::LazyConstraintCallbackI(inst_.env),
          inst(inst_),
          worker(worker_),
          a(a_),
          b(b_),
          eta(eta_),
          a_vals(inst_.env, inst_.nI),
          b_vals(inst_.env, inst_.nJ)
    {
    }

    IloCplex::CallbackI *duplicateCallback() const override
    {
        return (new (getEnv()) LazyBendersCallbackI(*this));
    }

    void main() override
    {
        // 1) Lê (a, b) da solução corrente de uma vez
        getValues(a_vals, a);
        getValues(b_vals, b);
        double eta_val = getValue(eta);

        // 2) Resolve worker
        worker.solve(a_vals, b_vals);

        // 3) Se corte violado, adiciona lazy cut
        if (!(worker.theta - eta_val > EPS))
            return;

        add(eta >= worker.rhs + IloScalProd(worker.coef_a, a) + IloScalProd(worker.coef_b, b));
    }
};

// CALLBACK: User Cuts
class UserBendersCallbackI : public IloCplex::UserCutCallbackI
{
    const TSCFLInstance &inst;
    Worker &worker;
    IloBoolVarArray a;
    IloBoolVarArray b;
    IloNumVar eta;

    IloNumArray a_vals;
    IloNumArray b_vals;

public:
    UserBendersCallbackI(
        const TSCFLInstance &inst_,
        Worker &worker_,
        IloBoolVarArray a_,
        IloBoolVarArray b_,
        IloNumVar eta_)
        : IloCplex::UserCutCallbackI(inst_.env),
          inst(inst_),
          worker(worker_),
          a(a_),
          b(b_),
          eta(eta_),
          a_vals(inst_.env, inst_.nI),
          b_vals(inst_.env, inst_.nJ)
    {
    }

    IloCplex::CallbackI *duplicateCallback() const override
    {
        return (new (getEnv()) UserBendersCallbackI(*this));
    }

    void main() override
    {
        if (!isAfterCutLoop())
            return;

        // 1) Lê (a, b, eta) da solução LP
        getValues(a_vals, a);
        getValues(b_vals, b);
        double eta_val = getValue(eta);

        // 2) Resolve worker
        worker.solve(a_vals, b_vals);

        // 3) Se corte violado, adiciona user cut
        if (!(worker.theta - eta_val > EPS))
            return;

        add(eta >= worker.rhs + IloScalProd(worker.coef_a, a) + IloScalProd(worker.coef_b, b),
            IloCplex::UseCutPurge);
    }
};

// SOLVER TSCFL: Decomposição de Benders
class TSCFLSolverBenders
{
public:
    const TSCFLInstance &inst;

    // Resultados:
    double lb{0.0};
    double ub{0.0};
    double gap{0.0};
    double time{0.0};
    IloInt64 nodes{0};
    IloAlgorithm::Status status{IloAlgorithm::Unknown};

private:
    IloModel master;
    IloCplex cplex;

    IloBoolVarArray a; // a_i  = abre planta i
    IloBoolVarArray b; // b_j  = abre depósito j
    IloNumVar eta;     // custo de segundo estágio

    std::unique_ptr<Worker> worker;

public:
    // mode:
    // 0 -> WorkerDual  (default)
    // 1 -> WorkerPrimal
    // 2 -> WorkerNet
    explicit TSCFLSolverBenders(const TSCFLInstance &inst_, int mode = 0)
        : inst(inst_),
          master(inst_.env),
          cplex(inst_.env),
          a(inst_.env, inst_.nI),
          b(inst_.env, inst_.nJ),
          eta(inst_.env, 0.0, IloInfinity, ILOFLOAT),
          worker(nullptr)
    {
        build_master();
        cplex.extract(master);

        // Parâmetros CPLEX (mestre)
        cplex.setParam(IloCplex::Param::Threads, 1);
        cplex.setParam(IloCplex::Param::MIP::Strategy::Search, IloCplex::Traditional);

        switch (mode)
        {
        case 0:
            worker = std::make_unique<WorkerDual>(inst_);
            break;
        case 1:
            worker = std::make_unique<WorkerPrimal>(inst_);
            break;
        case 2:
            worker = std::make_unique<WorkerNet>(inst_);
            break;
        default:
            throw std::invalid_argument("Invalid Benders worker mode (must be 0, 1, or 2).");
        }
    }

    ~TSCFLSolverBenders()
    {
        cplex.end();
        master.end();
    }

private:
    void build_master()
    {
        IloEnv &env = inst.env;

        // Capacidade agregada (garante viabilidade do subproblema)
        double demand_sum = IloSum(inst.r);
        master.add(IloScalProd(inst.p, a) >= demand_sum);
        master.add(IloScalProd(inst.q, b) >= demand_sum);

        // OBJETIVO: custo fixo + eta
        IloObjective obj = IloMinimize(env, IloScalProd(inst.f, a) + IloScalProd(inst.g, b) + eta);
        master.add(obj);
    }

public:
    bool solve(bool log_output = true, double time_limit = -1.0)
    {
        IloEnv &env = inst.env;

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

        // Callbacks Benders
        cplex.use(new (env) LazyBendersCallbackI(inst, *worker, a, b, eta));
        cplex.use(new (env) UserBendersCallbackI(inst, *worker, a, b, eta));

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
