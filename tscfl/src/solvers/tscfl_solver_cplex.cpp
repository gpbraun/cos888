/*
COS888

tscfl_solver_cplex.cpp

Gabriel Braun, 2025
*/

#include "tscfl_solver_cplex.hpp"

#include <iomanip>
#include <iostream>

TSCFLSolverCplex::TSCFLSolverCplex(const TSCFLInstance &inst_)
    : TSCFLSolver(inst_),
      model(env),
      cplex(env),
      var_a(env, inst.nI),
      var_b(env, inst.nJ),
      var_x(env, inst.nI, inst.nJ),
      var_y(env, inst.nJ, inst.nK)
{
    buildModel();
    cplex.extract(model);

    // Parâmetros CPLEX
    cplex.setParam(IloCplex::Param::Threads, 1);
    cplex.setParam(IloCplex::Param::Preprocessing::Presolve, 0);
    cplex.setParam(IloCplex::Param::Preprocessing::Aggregator, 0);
    cplex.setParam(IloCplex::Param::RootAlgorithm, IloCplex::Primal);
    cplex.setParam(IloCplex::Param::Benders::Strategy, IloCplex::BendersFull);
    cplex.setParam(IloCplex::Param::MIP::Tolerances::MIPGap, MIP_GAP);

    // Nodos (0 = apenas relaxação linear)
    cplex.setParam(IloCplex::Param::MIP::Limits::Nodes, 0);
}

TSCFLSolverCplex::~TSCFLSolverCplex()
{
    cplex.end();
    model.end();
}

void
TSCFLSolverCplex::buildModel()
{
    // RESTRIÇÕES
    // Capacidade das plantas
    for (IloInt i = 0; i < inst.nI; ++i)
        model.add(IloSum(var_x[i]) <= inst.p[i] * var_a[i]);

    // Capacidade dos depósitos
    for (IloInt j = 0; j < inst.nJ; ++j)
        model.add(IloSum(var_y[j]) <= inst.q[j] * var_b[j]);

    // Balanço nos depósitos
    for (IloInt j = 0; j < inst.nJ; ++j)
        model.add(IloSum(var_x.col(j)) - IloSum(var_y[j]) == 0.0);

    // Demanda dos clientes
    for (IloInt k = 0; k < inst.nK; ++k)
        model.add(IloSum(var_y.col(k)) == inst.r[k]);

    // FUNÇÃO OBJETIVO
    IloObjective obj = IloMinimize(
        env,
        IloScalProd(inst.f, var_a) + IloScalProd(inst.g, var_b) + IloMatScalProd(inst.c, var_x)
            + IloMatScalProd(inst.d, var_y)
    );
    model.add(obj);
}

void
TSCFLSolverCplex::solve(bool log_output, double time_limit)
{
    // Controle de log
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

    if (time_limit > 0.0)
        cplex.setParam(IloCplex::Param::TimeLimit, time_limit);

    // Executa o solver
    if (cplex.solve())
        {
            lb = cplex.getBestObjValue();
            ub = cplex.getObjValue();

            cplex.getValues(a, var_a);
            cplex.getValues(b, var_b);
            updateFlows(cplex, var_x, var_y);
        }

    // Recuperação das estatísticas
    gap = cplex.getMIPRelativeGap();
    time = cplex.getTime();
    nodes = cplex.getNnodes64();
    status = cplex.getStatus();

    // Log final
    printSummary("CPLEX");
}
