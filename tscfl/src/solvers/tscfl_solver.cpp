/*
COS888

tscfl_solver.cpp

Gabriel Braun, 2025
*/

#include "tscfl_solver.hpp"

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
TSCFLSolver::printSummary(const char *tag) const
{
    std::cout << "\n\n"
              << "[" << tag << "] Solver finalizado.\n\n"
              << "status = " << status
              << "\n"
              // número de nodos explorados
              << std::fixed << std::setprecision(0) << "nodes  = " << nodes
              << "\n"
              // iterações
              << std::fixed << std::setprecision(0) << "iters  = " << iter
              << "\n"
              // tempo
              << std::fixed << std::setprecision(1) << "time   = " << time << " s"
              << "\n"
              // LowerBound
              << std::fixed << std::setprecision(0) << "LB     = " << lb
              << "\n"
              // UpperBound
              << std::fixed << std::setprecision(0) << "UB     = " << ub
              << "\n"
              // gap
              << std::scientific << std::setprecision(2) << "gap    = " << gap << "\n"
              << std::defaultfloat;
}
