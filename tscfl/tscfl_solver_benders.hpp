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

// =====================================================================
//  CONSTANTES
// =====================================================================

static const double USERCUT_ABS_VIOL = 1;           // violação mínima absoluta
static const double USERCUT_REL_VIOL = 1e-3;        // violação mínima relativa
static const double MAX_FRAC_SUM = 10.0;            // quão fracionária pode ser a solução LP
static const int MAX_CUTS_PER_NODE = 1;             // máx. cortes por nó (user cuts)
static const IloInt64 MAX_NODE_INDEX_USER_CUTS = 0; // 0 => só no nó raiz

// =====================================================================
//  CALLBACKS
// =====================================================================

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
    IloNum eta_val;

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
          b_vals(inst_.env, inst_.nJ),
          eta_val(0)
    {
    }

    IloCplex::CallbackI *duplicateCallback() const override
    {
        return (new (getEnv()) LazyBendersCallbackI(*this));
    }

    void main() override
    {
        // 1) Lê (a, b, eta) da solução corrente
        getValues(a_vals, a);
        getValues(b_vals, b);
        eta_val = getValue(eta);

        // 2) Resolve worker
        worker.solve(a_vals, b_vals);

        // 3) Testa violação
        double viol = worker.theta - eta_val;

        if (viol <= EPS)
            return;

        // 4) Adiciona lazy cut
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
    IloNum eta_val;

    // Controle de cortes por nó
    IloInt64 lastNodeIndex;
    int cutsThisNode;

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
          b_vals(inst_.env, inst_.nJ),
          lastNodeIndex(-1),
          cutsThisNode(0)
    {
    }

    IloCplex::CallbackI *duplicateCallback() const override
    {
        return (new (getEnv()) UserBendersCallbackI(*this));
    }

    void main() override
    {
        // 0) Só depois do cut loop
        if (!isAfterCutLoop())
            return;

        // 1) Limita user cuts a nós "rasos"
        IloInt64 nodeIndex = getNnodes64();
        if (nodeIndex > MAX_NODE_INDEX_USER_CUTS)
            return;

        // 2) Atualiza controle de cortes por nó
        if (nodeIndex != lastNodeIndex)
        {
            lastNodeIndex = nodeIndex;
            cutsThisNode = 0;
        }
        if (cutsThisNode >= MAX_CUTS_PER_NODE)
            return;

        // 3) Lê (a, b, eta) da solução corrente
        getValues(a_vals, a);
        getValues(b_vals, b);
        eta_val = getValue(eta);

        // 4) Evita cortes de soluções muito fracionárias (tendem a ser fracos)
        double frac_sum = 0.0;

        for (int i = 0; i < inst.nI; ++i)
        {
            double v = a_vals[i];
            double frac = std::fabs(v - std::round(v));
            frac_sum += std::min(frac, 1.0 - frac);

            if (frac_sum > MAX_FRAC_SUM)
                return;
        }
        for (int j = 0; j < inst.nJ; ++j)
        {
            double v = b_vals[j];
            double frac = std::fabs(v - std::round(v));
            frac_sum += std::min(frac, 1.0 - frac);

            if (frac_sum > MAX_FRAC_SUM)
                return;
        }

        // 5) Resolve worker
        worker.solve(a_vals, b_vals);

        // 6) Teste de violação
        double viol = worker.theta - eta_val;

        double min_viol = std::max(
            USERCUT_ABS_VIOL, USERCUT_REL_VIOL * std::max(1.0, std::fabs(worker.theta)));

        if (viol <= min_viol)
            return;

        // 7) Adiciona user cut
        IloCplex::CutManagement cm = (nodeIndex == 0 ? IloCplex::UseCutForce : IloCplex::UseCutPurge);

        add(eta >= worker.rhs + IloScalProd(worker.coef_a, a) + IloScalProd(worker.coef_b, b), cm);

        ++cutsThisNode;
    }
};

// =====================================================================
//  SOLVER
// =====================================================================

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
