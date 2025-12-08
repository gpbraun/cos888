/*
COS888

tscfl_solver_benders.hpp

Gabriel Braun, 2025
*/

#pragma once

#include <memory>

#include "subproblem/tscfl_subproblem.hpp"
#include "tscfl_solver.hpp"
#include "tscfl_utils.hpp"

// SOLVER: Decomposição de Benders
class TSCFLSolverBenders : public TSCFLSolver
{
  private:
    IloModel model;
    IloCplex cplex;
    std::unique_ptr<Subproblem> subproblem;

    IloBoolVarArray var_a; // var_a[i]
    IloBoolVarArray var_b; // var_b[j]
    IloNumVar var_eta;     // custo de segundo estágio

  public:
    explicit TSCFLSolverBenders(
        const TSCFLInstance &inst_, Subproblem::Mode sp_mode = Subproblem::Mode::NET
    );

    ~TSCFLSolverBenders();

    void solve(bool log_output = true, double time_limit = -1.0) override;

  private:
    void buildModel();
};
