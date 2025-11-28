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

#include "tscfl_instance.hpp"

ILOSTLBEGIN

// =====================================================================
//  WORKER DUAL
// =====================================================================

class WorkerDual
{
public:
    const TSCFLInstance &inst;

private:
    IloEnv env;
    IloModel model;
    IloCplex cplex;

    IloNumVarArray alpha; // nI, (-inf, 0]
    IloNumVarArray beta;  // nJ, (-inf, 0]
    IloNumVarArray delta; // nK, (-inf, +inf)
    IloNumVarArray gamma; // nJ, (-inf, +inf)
    IloObjective obj;

public:
    explicit WorkerDual(const TSCFLInstance &I, bool log_output = false)
        : inst(I),
          env(),
          model(env),
          cplex(env),
          alpha(env, I.nI, -IloInfinity, 0.0, ILOFLOAT),
          beta(env, I.nJ, -IloInfinity, 0.0, ILOFLOAT),
          delta(env, I.nK, -IloInfinity, IloInfinity, ILOFLOAT),
          gamma(env, I.nJ, -IloInfinity, IloInfinity, ILOFLOAT),
          obj(IloMaximize(env, 0.0))
    {
        const int nI = inst.nI;
        const int nJ = inst.nJ;
        const int nK = inst.nK;

        // alpha_i + gamma_j <= c_ij
        for (int i = 0; i < nI; ++i)
            for (int j = 0; j < nJ; ++j)
                model.add(alpha[i] + gamma[j] <= inst.C(i, j));

        // beta_j - gamma_j + delta_k <= d_jk
        for (int j = 0; j < nJ; ++j)
            for (int k = 0; k < nK; ++k)
                model.add(beta[j] - gamma[j] + delta[k] <= inst.D(j, k));

        model.add(obj);
        cplex.extract(model);

        // Worker dual: LP, 1 thread (seguro para callbacks)
        cplex.setParam(IloCplex::Param::Threads, 1);

        if (log_output)
        {
            cplex.setOut(std::cout);
            cplex.setWarning(std::cout);
        }
        else
        {
            cplex.setOut(env.getNullStream());
            cplex.setWarning(env.getNullStream());
        }
    }

    ~WorkerDual()
    {
        cplex.end();
        model.end();
        env.end();
    }

private:
    void set_objective(const Vec &a, const Vec &b)
    {
        IloExpr e(env);

        // alpha ≤ 0, beta ≤ 0, delta livre
        for (int i = 0; i < inst.nI; ++i)
            e += (inst.p[i] * a[i]) * alpha[i];
        for (int j = 0; j < inst.nJ; ++j)
            e += (inst.q[j] * b[j]) * beta[j];
        for (int k = 0; k < inst.nK; ++k)
            e += inst.r[k] * delta[k];

        obj.setExpr(e);
        e.end();
    }

public:
    // Resolve o dual e retorna:
    //   theta   = valor ótimo do subproblema
    //   coef_a  = coeficientes de a_i no corte
    //   coef_b  = coeficientes de b_j no corte
    //   rhs     = termo independente
    void solve(const Vec &a,
               const Vec &b,
               double &theta,
               Vec &coef_a,
               Vec &coef_b,
               double &rhs)
    {
        set_objective(a, b);

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

// =====================================================================
//  FUNÇÃO COMUM PARA CALLBACKS (Lazy + User)
// =====================================================================

template <typename BaseCallback>
class BendersCallbackT : public BaseCallback
{
protected:
    const TSCFLInstance &inst;
    WorkerDual &worker;
    IloBoolVarArray a;
    IloBoolVarArray b;
    IloNumVar eta;
    bool is_user_cut; // true = UserCut, false = LazyLazyConstraint

public:
    BendersCallbackT(
        IloEnv env,
        const TSCFLInstance &inst_,
        WorkerDual &worker_,
        IloBoolVarArray a_,
        IloBoolVarArray b_,
        IloNumVar eta_,
        bool is_user_cut_)
        : BaseCallback(env),
          inst(inst_),
          worker(worker_),
          a(a_),
          b(b_),
          eta(eta_),
          is_user_cut(is_user_cut_)
    {
    }

    IloCplex::CallbackI *duplicateCallback() const override
    {
        return (new (this->getEnv()) BendersCallbackT(*this));
    }

    void main() override
    {
        // 1) Lê solução corrente (a,b,eta)
        Vec av(inst.nI), bv(inst.nJ);
        for (int i = 0; i < inst.nI; ++i)
            av[i] = this->getValue(a[i]);
        for (int j = 0; j < inst.nJ; ++j)
            bv[j] = this->getValue(b[j]);
        double eta_val = this->getValue(eta);

        // 2) Resolve worker dual
        double theta, rhs;
        Vec coef_a, coef_b;
        worker.solve(av, bv, theta, coef_a, coef_b, rhs);

        // 3) Verifica violação e adiciona corte
        if (theta - eta_val > EPS)
        {
            IloEnv env = this->getEnv();
            IloExpr lin(env);

            lin += rhs;
            for (int i = 0; i < inst.nI; ++i)
                lin += coef_a[i] * a[i];
            for (int j = 0; j < inst.nJ; ++j)
                lin += coef_b[j] * b[j];

            if (is_user_cut)
                this->add(eta >= lin, IloCplex::UseCutPurge);
            else
                this->add(eta >= lin);

            lin.end();
        }
    }
};

using LazyBendersCallbackI = BendersCallbackT<IloCplex::LazyConstraintCallbackI>;
using UserBendersCallbackI = BendersCallbackT<IloCplex::UserCutCallbackI>;

// =====================================================================
//  BENDERS SOLVER
// =====================================================================

class TSCFLSolverBenders
{
public:
    const TSCFLInstance &inst;

    // Resultados:
    // lb   = melhor limite inferior
    // ub   = melhor solução viável
    // gap  = gap relativo entre lb e ub (do CPLEX)
    // time = tempo total (segundos)
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

    WorkerDual worker; // subproblema dual (env próprio interno)

public:
    explicit TSCFLSolverBenders(const TSCFLInstance &inst_)
        : inst(inst_),
          env(),
          master(env),
          cplex(env),
          a(env, inst_.nI),
          b(env, inst_.nJ),
          eta(env, 0.0, IloInfinity, ILOFLOAT),
          worker(inst_)
    {
        build_master();
        cplex.extract(master);

        // Benders com callbacks precisa de Traditional search
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
        // LOG
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

        // Callbacks (lazy + user cuts) usando benders_common_main
        cplex.use(new (env) LazyBendersCallbackI(env, inst, worker, a, b, eta, false));
        cplex.use(new (env) UserBendersCallbackI(env, inst, worker, a, b, eta, true));

        // Solve com cronômetro externo
        auto t0 = std::chrono::steady_clock::now();
        bool ok = cplex.solve();
        auto t1 = std::chrono::steady_clock::now();

        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(t1 - t0).count();

        // Estatísticas
        status = cplex.getStatus();
        gap = cplex.getMIPRelativeGap();
        time = static_cast<int>(elapsed);
        nodes = cplex.getNnodes64();

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
            std::cerr << "time  = " << time << " s\n";
        }

        return ok;
    }
};
