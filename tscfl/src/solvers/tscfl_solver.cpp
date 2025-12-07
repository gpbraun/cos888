/*
COS888

tscfl_solver.cpp

Gabriel Braun, 2025
*/

#include "tscfl_solver.hpp"
#include "subproblem/tscfl_subproblem_primal.hpp"

#include <iomanip>
#include <iostream>

TSCFLSolver::TSCFLSolver(const TSCFLInstance &inst_)
    : env(inst_.env),
      inst(inst_),
      a(env, inst.nI),
      b(env, inst.nJ),
      x(env, inst.nI, inst.nJ),
      y(env, inst.nJ, inst.nK)
{
}

void
TSCFLSolver::printSummary(const char *tag) const
{
    // clang-format off
    std::cout << "\n\n[" << tag << "] Solver finalizado.\n\n"
              << "status = " << status << "\n"
              << std::fixed << std::setprecision(0)      << "nodes  = " << nodes << "\n"
              << std::fixed << std::setprecision(0)      << "iters  = " << iter  << "\n"
              << std::fixed << std::setprecision(1)      << "time   = " << time  << " s\n"
              << std::fixed << std::setprecision(0)      << "LB     = " << lb    << "\n"
              << std::fixed << std::setprecision(0)      << "UB     = " << ub    << "\n"
              << std::scientific << std::setprecision(2) << "gap    = " << gap   << "\n"
              << std::defaultfloat;
    // clang-format on
}

void
TSCFLSolver::updateGap()
{
    if (ub < IloInfinity && lb > EPS && ub > lb)
        {
            gap = (ub - lb) / IloAbs(ub);

            if (gap <= MIP_GAP)
                status = IloAlgorithm::Optimal;
            else
                status = IloAlgorithm::Feasible;
        }
}

void
TSCFLSolver::updateFlows(
    const IloCplex &cplex, const IloNumVarMatrix &var_x, const IloNumVarMatrix &var_y
)
{
    for (int i = 0; i < inst.nI; ++i)
        cplex.getValues(x[i], var_x[i]);

    for (int j = 0; j < inst.nJ; ++j)
        cplex.getValues(y[j], var_y[j]);
}

void
TSCFLSolver::updateFlows()
{
    SubproblemPrimal SP = SubproblemPrimal(inst);
    SP.update(a, b);
    SP.solve();
    SP.getFlows(x, y);
}
