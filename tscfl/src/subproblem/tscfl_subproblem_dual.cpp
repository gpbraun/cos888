/*
COS888

tscfl_subproblem_dual.cpp

Gabriel Braun, 2025
*/

#include "tscfl_subproblem_dual.hpp"

#include <stdexcept>

SubproblemDual::SubproblemDual(const TSCFLInstance &inst_)
    : Subproblem(inst_),
      env(),
      model(env),
      cplex(env),
      var_l1(env, inst_.nI, -IloInfinity, 0.0),
      var_l2(env, inst_.nJ, -IloInfinity, 0.0),
      var_m1(env, inst_.nJ, -IloInfinity, IloInfinity),
      var_m2(env, inst_.nK, -IloInfinity, IloInfinity),
      obj(IloMaximize(env, 0.0))
{
    build_base_model();
    cplex.extract(model);

    // Parâmetros do CPLEX (subproblema)
    cplex.setParam(IloCplex::Param::Threads, 1);
    cplex.setParam(IloCplex::Param::Preprocessing::Reduce, 0);
    cplex.setParam(IloCplex::Param::RootAlgorithm, IloCplex::Primal);

    // Subproblema silencioso por padrão
    cplex.setOut(env.getNullStream());
    cplex.setWarning(env.getNullStream());
}

SubproblemDual::~SubproblemDual()
{
    cplex.end();
    model.end();
    env.end();
}

void
SubproblemDual::build_base_model()
{
    // RESTRIÇÕES DO SUBPROBLEMA DUAL
    // arcos planta -> depósito
    for (int i = 0; i < inst.nI; ++i)
        for (int j = 0; j < inst.nJ; ++j)
            model.add(var_l1[i] + var_m1[j] <= inst.c[i][j]);

    // arcos depósito -> cliente
    for (int j = 0; j < inst.nJ; ++j)
        for (int k = 0; k < inst.nK; ++k)
            model.add(-var_m1[j] + var_l2[j] + var_m2[k] <= inst.d[j][k]);

    // FUNÇÃO OBJETIVO
    model.add(obj);
}

void
SubproblemDual::update_model(const IloNumArray &a, const IloNumArray &b)
{
    IloExpr obj_expr(env);

    for (int i = 0; i < inst.nI; ++i)
        obj_expr += (inst.p[i] * a[i]) * var_l1[i];

    for (int j = 0; j < inst.nJ; ++j)
        obj_expr += (inst.q[j] * b[j]) * var_l2[j];

    for (int k = 0; k < inst.nK; ++k)
        obj_expr += inst.r[k] * var_m2[k];

    obj.setExpr(obj_expr);
    obj_expr.end();
}

void
SubproblemDual::solve(const IloNumArray &a, const IloNumArray &b)
{
    // Atualiza a função objetivo
    update_model(a, b);

    // Resolve o LP dual
    if (!cplex.solve())
        throw std::runtime_error("SubproblemDual: falha no CPLEX.");

    // Valor ótimo do dual
    theta = cplex.getObjValue();

    // Coeficientes do corte de Benders
    for (int i = 0; i < inst.nI; ++i)
        coef_a[i] = inst.p[i] * cplex.getValue(var_l1[i]);

    for (int j = 0; j < inst.nJ; ++j)
        coef_b[j] = inst.q[j] * cplex.getValue(var_l2[j]);

    rhs = 0.0;
    for (int k = 0; k < inst.nK; ++k)
        rhs += inst.r[k] * cplex.getValue(var_m2[k]);

    // Calcula o custo do problema original
    update_opt(a, b);
}
