/*
COS888

WorkerDual: resolve o subproblema dual do TSCFL para uso na decomposição de Benders.

Gabriel Braun, 2025
*/

#pragma once

#include <ilcplex/ilocplex.h>
#include <stdexcept>

#include "tscfl_benders_base_worker.hpp"

ILOSTLBEGIN

class WorkerDual : public Worker
{
private:
    IloModel model;
    IloCplex cplex;

    // Variáveis duais
    IloNumVarArray l1; // l1[i] = dual da capacidade da planta i   (<= 0)
    IloNumVarArray l2; // l2[j] = dual da capacidade do depósito j (<= 0)
    IloNumVarArray m1; // m1[j] = dual do balanço nos depósitos j  (livre)
    IloNumVarArray m2; // m2[k] = dual da demanda do cliente k     (livre)

    // Função objetivo
    IloObjective obj;

public:
    explicit WorkerDual(const TSCFLInstance &inst_)
        : Worker(inst_),
          model(env),
          cplex(env),
          l1(env, inst_.nI, -IloInfinity, 0.0, ILOFLOAT),
          l2(env, inst_.nJ, -IloInfinity, 0.0, ILOFLOAT),
          m1(env, inst_.nJ, -IloInfinity, IloInfinity, ILOFLOAT),
          m2(env, inst_.nK, -IloInfinity, IloInfinity, ILOFLOAT),
          obj(IloMaximize(env, 0.0))
    {
        build_base_model();
        cplex.extract(model);

        // Parâmetros do CPLEX (subproblema)
        cplex.setParam(IloCplex::Param::Threads, 1);
        cplex.setParam(IloCplex::Param::Preprocessing::Reduce, 0);
        cplex.setParam(IloCplex::Param::RootAlgorithm, IloCplex::Dual);

        // Subproblema silencioso por padrão
        cplex.setOut(env.getNullStream());
        cplex.setWarning(env.getNullStream());
    }

    ~WorkerDual() override
    {
        cplex.end();
        model.end();
    }

private:
    // Constrói o modelo base do subproblema dual
    void build_base_model()
    {
        // RESTRIÇÕES DO SUBPROBLEMA DUAL
        for (int i = 0; i < inst.nI; ++i)
            for (int j = 0; j < inst.nJ; ++j)
                model.add(l1[i] + m1[j] <= inst.c[i][j]);

        for (int j = 0; j < inst.nJ; ++j)
            for (int k = 0; k < inst.nK; ++k)
                model.add(l2[j] - m1[j] + m2[k] <= inst.d[j][k]);

        // FUNÇÃO OBJETIVO
        model.add(obj);
    }

    // Atualiza a função objetivo do subproblema em função de (a,b)
    void set_objective(const IloNumArray &a_vals, const IloNumArray &b_vals)
    {
        IloExpr obj_expr(env);

        for (int i = 0; i < inst.nI; ++i)
            obj_expr += (inst.p[i] * a_vals[i]) * l1[i];
        for (int j = 0; j < inst.nJ; ++j)
            obj_expr += (inst.q[j] * b_vals[j]) * l2[j];
        for (int k = 0; k < inst.nK; ++k)
            obj_expr += inst.r[k] * m2[k];

        obj.setExpr(obj_expr);
        obj_expr.end();
    }

public:
    // Dado (a_vals, b_vals) da solução atual do mestre, resolve o subproblema.
    // Atualiza: theta, rhs, coef_a, coef_b
    void solve(const IloNumArray &a_vals, const IloNumArray &b_vals) override
    {
        // 1) Atualiza a função objetivo
        set_objective(a_vals, b_vals);

        // 2) Resolve o LP dual
        if (!cplex.solve())
            throw std::runtime_error("WorkerDual: CPLEX failed to solve dual subproblem.");

        // 3) Calcula os coeficientes do corte
        theta = cplex.getObjValue();

        for (int i = 0; i < inst.nI; ++i)
            coef_a[i] = inst.p[i] * cplex.getValue(l1[i]); // ≤ 0

        for (int j = 0; j < inst.nJ; ++j)
            coef_b[j] = inst.q[j] * cplex.getValue(l2[j]); // ≤ 0

        rhs = 0.0;
        for (int k = 0; k < inst.nK; ++k)
            rhs += inst.r[k] * cplex.getValue(m2[k]);
    }
};
