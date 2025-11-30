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

static const double USERCUT_MAX_GAP = 1e-4;     // maior gap para gerar user cuts
static const double USERCUT_ABS_VIOL = 1e-1;    // violação mínima absoluta
static const double USERCUT_REL_VIOL = 1e-4;    // violação mínima relativa
static const double MAX_FRAC_SUM = 10.0;        // quão fracionária pode ser a solução LP
static const int MAX_CUTS_PER_NODE = 1;         // máx. cortes por nó (user cuts)
static const int MAX_NODE_INDEX_USER_CUTS = 10; // 0 => só no nó raiz

static const double COREPOINT_LAMBDA = 0.5; // passo para atualizar core point externo
static const double SEPPOINT_LAMBDA = 0.5;  // peso do ponto LP no ponto de separação

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

    // Core point e ponto de separação
    IloNumArray a_core;
    IloNumArray b_core;
    IloNumArray a_sep;
    IloNumArray b_sep;
    IloBool core_initialized;

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
          eta_val(0.0),
          lastNodeIndex(-1),
          cutsThisNode(0),
          a_core(inst_.env, inst_.nI),
          b_core(inst_.env, inst_.nJ),
          a_sep(inst_.env, inst_.nI),
          b_sep(inst_.env, inst_.nJ),
          core_initialized(IloFalse)
    {
        // Inicializa core point em 0.5 (interior de [0,1]^n)
        for (int i = 0; i < inst_.nI; ++i)
            a_core[i] = 0.5;
        for (int j = 0; j < inst_.nJ; ++j)
            b_core[j] = 0.5;
        core_initialized = IloTrue;
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
        if (getMIPRelativeGap() <= USERCUT_MAX_GAP && nodeIndex > MAX_NODE_INDEX_USER_CUTS)
            return;

        // 2) Controle de cortes por nó
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

        // 4) Evita cortes para soluções muito fracionárias (tendem a ser fracos)
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

        // 5) Atualiza core point (média entre core e solução LP atual)
        //    core^{new} = (1-COREPOINT_LAMBDA)*core + COREPOINT_LAMBDA * a_vals
        if (core_initialized)
        {
            for (int i = 0; i < inst.nI; ++i)
                a_core[i] = (1.0 - COREPOINT_LAMBDA) * a_core[i] + COREPOINT_LAMBDA * a_vals[i];
            for (int j = 0; j < inst.nJ; ++j)
                b_core[j] = (1.0 - COREPOINT_LAMBDA) * b_core[j] + COREPOINT_LAMBDA * b_vals[j];
        }
        else
        {
            for (int i = 0; i < inst.nI; ++i)
                a_core[i] = a_vals[i];
            for (int j = 0; j < inst.nJ; ++j)
                b_core[j] = b_vals[j];
            core_initialized = IloTrue;
        }

        // 6) Define ponto de separação
        for (int i = 0; i < inst.nI; ++i)
            a_sep[i] = SEPPOINT_LAMBDA * a_vals[i] + (1.0 - SEPPOINT_LAMBDA) * a_core[i];
        for (int j = 0; j < inst.nJ; ++j)
            b_sep[j] = SEPPOINT_LAMBDA * b_vals[j] + (1.0 - SEPPOINT_LAMBDA) * b_core[j];

        // 7) Resolve subproblema no ponto de separação (não no ponto LP!)
        worker.solve(a_sep, b_sep);

        // 8) Teste de violação avaliado na solução LP (a_vals, b_vals, eta_val)
        double theta = worker.theta;
        double viol = theta - eta_val;

        double min_viol = std::max(USERCUT_ABS_VIOL, USERCUT_REL_VIOL * std::max(1.0, std::fabs(theta)));

        if (viol <= min_viol)
            return;

        // 9) Adiciona user cut
        IloCplex::CutManagement cm =
            (nodeIndex == 0 ? IloCplex::UseCutForce : IloCplex::UseCutPurge);

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
    double ub{IloInfinity};
    double gap{IloInfinity};
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
