/*
COS888

tscfl_solver_cplex.hpp

Gabriel Braun, 2025
*/

#pragma once

#include "tscfl_solver.hpp"
#include "tscfl_utils.hpp"

// SOLVER TSCFL: CPLEX
class TSCFLSolverCplex : public TSCFLSolver
{
  private:
    IloModel model;
    IloCplex cplex;

    IloBoolVarArray var_a; // var_a[i]
    IloBoolVarArray var_b; // var_b[j]
    IloNumVarMatrix var_x; // var_x[i][j]
    IloNumVarMatrix var_y; // var_y[j][k]

  public:
    explicit TSCFLSolverCplex(const TSCFLInstance &inst_);

    ~TSCFLSolverCplex();

    void solve(bool log_output = true, double time_limit = -1.0) override;

  private:
    void buildModel();
};
