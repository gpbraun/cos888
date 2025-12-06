/*
COS888

tscfl_subproblem_primal.cpp

Gabriel Braun, 2025
*/

#include "tscfl_subproblem_primal.hpp"

#include <stdexcept>

// ---------------------------------------------------------------------
//  Construtor / destrutor
// ---------------------------------------------------------------------

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
      constr_m2(env, inst_.nK)
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

SubproblemPrimal::~SubproblemPrimal()
{
    cplex.end();
    model.end();
    env.end();
}

// ---------------------------------------------------------------------
//  Modelo base do subproblema primal
// ---------------------------------------------------------------------

void
SubproblemPrimal::build_base_model()
{
    // Capacidade das plantas
    for (int i = 0; i < inst.nI; ++i)
        {
            constr_l1[i] = IloRange(
                env,
                -IloInfinity,     // lower bound
                IloSum(var_x[i]), // expressão
                IloInfinity
            ); // upper bound
            model.add(constr_l1[i]);
        }

    // Capacidade dos depósitos
    for (int j = 0; j < inst.nJ; ++j)
        {
            constr_l2[j] = IloRange(env, -IloInfinity, IloSum(var_y[j]), IloInfinity);
            model.add(constr_l2[j]);
        }

    // Balanço nos depósitos
    for (int j = 0; j < inst.nJ; ++j)
        {
            constr_m1[j] = IloRange(env, 0.0, IloSum(var_x.col(j)) - IloSum(var_y[j]), 0.0);
            model.add(constr_m1[j]);
        }

    // Demanda dos clientes
    for (int k = 0; k < inst.nK; ++k)
        {
            constr_m2[k] = IloRange(env, inst.r[k], IloSum(var_y.col(k)), IloInfinity);
            model.add(constr_m2[k]);
        }

    // FUNÇÃO OBJETIVO
    IloObjective obj
        = IloMinimize(env, IloMatScalProd(inst.c, var_x) + IloMatScalProd(inst.d, var_y));
    model.add(obj);
}

// ---------------------------------------------------------------------
//  Atualiza restrições dependentes de (a,b)
// ---------------------------------------------------------------------

void
SubproblemPrimal::set_constraints(const IloNumArray &a, const IloNumArray &b)
{
    // Capacidade das plantas
    for (int i = 0; i < inst.nI; ++i)
        constr_l1[i].setBounds(-IloInfinity, inst.p[i] * a[i]);

    // Capacidade dos depósitos
    for (int j = 0; j < inst.nJ; ++j)
        constr_l2[j].setBounds(-IloInfinity, inst.q[j] * b[j]);
}

// ---------------------------------------------------------------------
//  Resolve o subproblema para (a, b)
// ---------------------------------------------------------------------

IloNum
SubproblemPrimal::solve(const IloNumArray &a, const IloNumArray &b)
{
    // 1) Atualiza as restrições dependentes de (a,b)
    set_constraints(a, b);

    // 2) Resolve o LP primal
    if (!cplex.solve())
        throw std::runtime_error("SubproblemPrimal: CPLEX failed to solve primal subproblem.");

    theta = cplex.getObjValue();

    // 3) Lê as variáveis duais
    IloNumArray l1(env, inst.nI);
    IloNumArray l2(env, inst.nJ);
    IloNumArray m2(env, inst.nK);

    cplex.getDuals(l1, constr_l1);
    cplex.getDuals(l2, constr_l2);
    cplex.getDuals(m2, constr_m2);

    // 4) Calcula os coeficientes do corte
    for (int i = 0; i < inst.nI; ++i)
        coef_a[i] = inst.p[i] * l1[i];

    for (int j = 0; j < inst.nJ; ++j)
        coef_b[j] = inst.q[j] * l2[j];

    rhs = IloScalProd(inst.r, m2);

    l1.end();
    l2.end();
    m2.end();

    opt = IloScalProd(inst.f, a) + IloScalProd(inst.g, b) + theta;
    return opt;
}
