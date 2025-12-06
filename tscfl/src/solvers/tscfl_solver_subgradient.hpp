/*
COS888

tscfl_solver_subgradient.hpp

Gabriel Braun, 2025
*/

#pragma once

#include "lrp/tscfl_lrp.hpp"
#include "solvers/tscfl_solver.hpp"
#include "subproblem/tscfl_subproblem.hpp"

class TSCFLSolverSubgradient : public TSCFLSolver
{
  public:
    // Parâmetros do método
    static constexpr IloNum EPSILON0 = 2.0;
    static constexpr IloInt IMPROV_EPSILON = 50;
    static constexpr IloInt EXTRA_AGE = 3;
    static constexpr IloInt MAX_NEW_CUTS = 10;
    static constexpr IloInt SOLVE_HEURISTIC_EVERY = 10;
    static constexpr IloInt MAX_NO_IMPROV = 250;
    static constexpr IloInt PRINT_EVERY = 10;

  private:
    std::unique_ptr<LRP> relaxation;
    std::unique_ptr<Subproblem> subproblem;

    IloNumArray a; // melhor a[i]
    IloNumArray b; // melhor b[j]

  public:
    explicit TSCFLSolverSubgradient(
        const TSCFLInstance &inst_,
        LRP::Mode rmode = LRP::Mode::CAPACITIES,
        Subproblem::Mode smode = Subproblem::Mode::NET
    );

    bool solve(bool log_output = true, IloNum time_limit = -1.0) override;
};
