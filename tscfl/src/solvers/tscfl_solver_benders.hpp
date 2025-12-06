// src/tscfl_solver_benders.hpp
#pragma once

/*
COS888

Resolve o TSCFL por decomposição de Benders (callbacks CPLEX).

Gabriel Braun, 2025
*/

#include <memory>

#include "subproblem/tscfl_subproblem.hpp"
#include "tscfl_solver.hpp"
#include "tscfl_utils.hpp"

ILOSTLBEGIN

// SOLVER TSCFL: Decomposição de Benders
class TSCFLSolverBenders : public TSCFLSolver {
   private:
    IloModel model;
    IloCplex cplex;
    std::unique_ptr<Subproblem> subproblem;

    IloBoolVarArray var_a;  // a[i]
    IloBoolVarArray var_b;  // b[j]
    IloNumVar var_eta;      // custo de segundo estágio

   public:
    explicit TSCFLSolverBenders(const TSCFLInstance& inst_,
                                Subproblem::Mode smode = Subproblem::Mode::NET);

    ~TSCFLSolverBenders();

    bool solve(bool log_output = true, double time_limit = -1.0) override;

   private:
    void build_model();
};
