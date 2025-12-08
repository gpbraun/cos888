/*
COS888

tscfl_solver_cplex.hpp

Gabriel Braun, 2025
*/

#pragma once

#include "tscfl_solver.hpp"
#include "tscfl_utils.hpp"

// SOLVER: CPLEX.
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
    IloNum lp{ 0.0 }; // Valor da relaxação linear

    explicit TSCFLSolverCplex(const TSCFLInstance &inst_);

    ~TSCFLSolverCplex();

    void solve(bool log_output = true, double time_limit = -1.0) override;

    // Resolve a relaxação linear
    // Atualiza: lp
    void solveLP(bool log_output = true, double time_limit = -1.0);

  private:
    // Constrói o modelo.
    void buildModel();
};
