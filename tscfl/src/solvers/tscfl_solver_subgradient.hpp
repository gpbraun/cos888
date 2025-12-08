/*
COS888

tscfl_solver_subgradient.hpp

Gabriel Braun, 2025
*/

#pragma once

#include "lrp/tscfl_lrp.hpp"
#include "solvers/tscfl_solver.hpp"
#include "subproblem/tscfl_subproblem.hpp"

// SOLVER: Non-Delayed Relax-and-Cut
class TSCFLSolverSubgradient : public TSCFLSolver
{
  public:
    // Parâmetros do método
    static constexpr IloNum EPSILON0 = 2.0;
    static constexpr IloInt IMPROV_EPSILON = 50;
    static constexpr IloInt EXTRA_AGE = 3;
    static constexpr IloInt MAX_NEW_CUTS = 10;
    static constexpr IloInt SOLVE_HEURISTIC_EVERY = 20;
    static constexpr IloInt MAX_NO_IMPROV = 150;
    static constexpr IloInt PRINT_EVERY = 200;

  private:
    std::unique_ptr<LRP> relaxation;
    std::unique_ptr<Subproblem> subproblem;

  public:
    explicit TSCFLSolverSubgradient(
        const TSCFLInstance &inst_,
        LRP::Mode lr_mode = LRP::Mode::CAPACITIES,
        Subproblem::Mode sp_mode = Subproblem::Mode::NET
    );

    void solve(bool log_output = true, IloNum time_limit = -1.0) override;
};
