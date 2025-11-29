/*
COS888

WorkerDual: resolve o subproblema dual do TSCFL para uso na decomposição de Benders.

Gabriel Braun, 2025
*/

#pragma once

#include <ilcplex/ilocplex.h>
#include <iostream>
#include <numeric>
#include <string>
#include <chrono>

#include "tscfl_benders_worker.hpp"

ILOSTLBEGIN

// =====================================================================
//  WORKER DUAL
// =====================================================================

class WorkerDual : public Worker
{
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
    explicit WorkerDual(const TSCFLInstance &inst_, bool log_output = false)
        : Worker(inst_),
          env(),
          model(env),
          cplex(env),
          alpha(env, inst_.nI, -IloInfinity, 0.0, ILOFLOAT),
          beta(env, inst_.nJ, -IloInfinity, 0.0, ILOFLOAT),
          delta(env, inst_.nK, -IloInfinity, IloInfinity, ILOFLOAT),
          gamma(env, inst_.nJ, -IloInfinity, IloInfinity, ILOFLOAT),
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

        // Parâmetros do CPLEX
        cplex.setParam(IloCplex::Param::Threads, 1);
        cplex.setParam(IloCplex::Param::Preprocessing::Reduce, 0);
        cplex.setParam(IloCplex::Param::MIP::Tolerances::MIPGap, MIP_GAP);
        cplex.setParam(IloCplex::Param::RootAlgorithm, IloCplex::Primal);

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

    ~WorkerDual() override
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
    // Implementa a interface virtual da classe base Worker
    //   theta   = valor ótimo do subproblema
    //   coef_a  = coeficientes de a_i no corte
    //   coef_b  = coeficientes de b_j no corte
    //   rhs     = termo independente
    void solve(const Vec &a,
               const Vec &b,
               double &theta,
               Vec &coef_a,
               Vec &coef_b,
               double &rhs) override
    {
        set_objective(a, b);

        if (!cplex.solve())
            throw std::runtime_error("WorkerDual: CPLEX failed to solve dual subproblem.");

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
