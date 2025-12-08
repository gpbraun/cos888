/*
COS888

tscfl_subproblem_primal.cpp

Gabriel Braun, 2025
*/

#include "tscfl_subproblem_primal.hpp"

#include <stdexcept>

SubproblemPrimal::SubproblemPrimal(const TSCFLInstance &inst_)
    : Subproblem(inst_),
      env(),
      model(env),
      cplex(env),
      var_x(env, inst_.nI, inst_.nJ),
      var_y(env, inst_.nJ, inst_.nK),
      constr_l1(env, inst_.nI),
      constr_l2(env, inst_.nJ),
      constr_m1(env, inst_.nJ),
      constr_m2(env, inst_.nK),
      l1(inst_.env, inst.nI),
      l2(inst_.env, inst.nJ),
      m2(inst_.env, inst.nK)
{
    buildModel();
    cplex.extract(model);

    // Parâmetros do CPLEX (subproblema)
    cplex.setParam(IloCplex::Param::Threads, 1);
    cplex.setParam(IloCplex::Param::Preprocessing::Reduce, 0);
    cplex.setParam(IloCplex::Param::RootAlgorithm, IloCplex::Primal);

    // Subproblema silencioso por padrão
    cplex.setOut(env.getNullStream());
    cplex.setWarning(env.getNullStream());
}

SubproblemPrimal::~SubproblemPrimal()
{
    cplex.end();
    model.end();
    env.end();
}

void
SubproblemPrimal::buildModel()
{
    // RESTRIÇÕES DO SUBPROBLEMA PRIMAL
    // Capacidade das plantas
    for (IloInt i = 0; i < inst.nI; ++i)
        {
            constr_l1[i] = IloRange(env, -IloInfinity, IloSum(var_x[i]), IloInfinity);
            model.add(constr_l1[i]);
        }

    // Capacidade dos depósitos
    for (IloInt j = 0; j < inst.nJ; ++j)
        {
            constr_l2[j] = IloRange(env, -IloInfinity, IloSum(var_y[j]), IloInfinity);
            model.add(constr_l2[j]);
        }

    // Balanço nos depósitos
    for (IloInt j = 0; j < inst.nJ; ++j)
        {
            constr_m1[j] = IloRange(env, 0.0, IloSum(var_x.col(j)) - IloSum(var_y[j]), 0.0);
            model.add(constr_m1[j]);
        }

    // Demanda dos clientes
    for (IloInt k = 0; k < inst.nK; ++k)
        {
            constr_m2[k] = IloRange(env, inst.r[k], IloSum(var_y.col(k)), IloInfinity);
            model.add(constr_m2[k]);
        }

    // FUNÇÃO OBJETIVO
    IloObjective obj
        = IloMinimize(env, IloMatScalProd(inst.c, var_x) + IloMatScalProd(inst.d, var_y));
    model.add(obj);
}

void
SubproblemPrimal::updateModel()
{
    // Capacidade das plantas
    for (IloInt i = 0; i < inst.nI; ++i)
        constr_l1[i].setBounds(-IloInfinity, inst.p[i] * a[i]);

    // Capacidade dos depósitos
    for (IloInt j = 0; j < inst.nJ; ++j)
        constr_l2[j].setBounds(-IloInfinity, inst.q[j] * b[j]);
}

void
SubproblemPrimal::solve()
{
    // Atualiza as restrições dependentes de (a,b)
    updateModel();

    // Resolve o LP primal
    if (!cplex.solve())
        throw std::runtime_error("SubproblemPrimal: CPLEX failed to solve primal subproblem.");

    theta = cplex.getObjValue();

    // Extrai as variáveis duais
    cplex.getDuals(l1, constr_l1);
    cplex.getDuals(l2, constr_l2);
    cplex.getDuals(m2, constr_m2);

    // Calcula os coeficientes do corte
    for (IloInt i = 0; i < inst.nI; ++i)
        coef_a[i] = inst.p[i] * l1[i];

    for (IloInt j = 0; j < inst.nJ; ++j)
        coef_b[j] = inst.q[j] * l2[j];

    rhs = IloScalProd(inst.r, m2);

    // Calcula o custo do problema original
    opt = IloScalProd(inst.f, a) + IloScalProd(inst.g, b) + theta;
}

void
SubproblemPrimal::getFlows(IloNumMatrix &x, const IloNumMatrix &y)
{
    for (IloInt i = 0; i < inst.nI; ++i)
        cplex.getValues(x[i], var_x[i]);

    for (IloInt j = 0; j < inst.nJ; ++j)
        cplex.getValues(y[j], var_y[j]);
}
