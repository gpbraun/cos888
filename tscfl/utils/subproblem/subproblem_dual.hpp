/*
COS888

SubproblemDual: resolve o subproblema dual do TSCFL.

Gabriel Braun, 2025
*/

#pragma once

#include "utils/subproblem/subproblem.hpp"

// SOLVER DO SUBPROBLEMA: Dual.
class SubproblemDual : public Subproblem
{
private:
    IloEnv env;
    IloModel model;
    IloCplex cplex;

    // Variáveis duais
    IloNumVarArray var_l1; // l1[i] = dual da capacidade da planta i   (>= 0)
    IloNumVarArray var_l2; // l2[j] = dual da capacidade do depósito j (>= 0)
    IloNumVarArray var_m1; // m1[j] = dual do balanço nos depósitos j  (livre)
    IloNumVarArray var_m2; // m2[k] = dual da demanda do cliente k     (livre)

    // Função objetivo
    IloObjective obj;

public:
    explicit SubproblemDual(const TSCFLInstance &inst_)
        : Subproblem(inst_),
          env(),
          model(env),
          cplex(env),
          var_l1(env, inst_.nI, 0.0, IloInfinity, ILOFLOAT),
          var_l2(env, inst_.nJ, 0.0, IloInfinity, ILOFLOAT),
          var_m1(env, inst_.nJ, -IloInfinity, IloInfinity, ILOFLOAT),
          var_m2(env, inst_.nK, -IloInfinity, IloInfinity, ILOFLOAT),
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

    ~SubproblemDual() override
    {
        cplex.end();
        model.end();
        env.end();
    }

private:
    // Constrói o modelo base do subproblema dual
    void build_base_model()
    {
        // RESTRIÇÕES DO SUBPROBLEMA DUAL
        for (int i = 0; i < inst.nI; ++i)
            for (int j = 0; j < inst.nJ; ++j)
                model.add(var_m1[j] - var_l1[i] <= inst.c[i][j]);

        for (int j = 0; j < inst.nJ; ++j)
            for (int k = 0; k < inst.nK; ++k)
                model.add(var_m2[k] - var_m1[j] - var_l2[j] <= inst.d[j][k]);

        // FUNÇÃO OBJETIVO
        model.add(obj);
    }

    // Atualiza a função objetivo do subproblema em função de (a,b)
    void set_objective(const IloNumArray &a_vals, const IloNumArray &b_vals)
    {
        IloExpr obj_expr(env);

        for (int i = 0; i < inst.nI; ++i)
            obj_expr += -(inst.p[i] * a_vals[i]) * var_l1[i];
        for (int j = 0; j < inst.nJ; ++j)
            obj_expr += -(inst.q[j] * b_vals[j]) * var_l2[j];
        for (int k = 0; k < inst.nK; ++k)
            obj_expr += inst.r[k] * var_m2[k];

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
            throw std::runtime_error("SubproblemDual: falha no CPLEX.");

        // 3) Calcula os coeficientes do corte
        theta = cplex.getObjValue();

        for (int i = 0; i < inst.nI; ++i)
            coef_a[i] = -inst.p[i] * cplex.getValue(var_l1[i]);

        for (int j = 0; j < inst.nJ; ++j)
            coef_b[j] = -inst.q[j] * cplex.getValue(var_l2[j]);

        rhs = 0.0;
        for (int k = 0; k < inst.nK; ++k)
            rhs += inst.r[k] * cplex.getValue(var_m2[k]);
    }
};
