/*
COS888

tscfl_solver_column_generation.hpp

Gabriel Braun, 2025
*/

#pragma once

#include <vector>

#include "subproblem/tscfl_subproblem.hpp"
#include "tscfl_solver.hpp"

class TSCFLSolverColumnGeneration : public TSCFLSolver
{
  public:
    // Parâmetros do método
    static constexpr IloInt PRINT_EVERY = 5;

  private:
    IloModel model;
    IloCplex cplex;
    std::unique_ptr<Subproblem> subproblem;

    // Variáveis do RMP
    IloNumVarArray var_a; // var_a[i]
    IloNumVarArray var_b; // var_b[j]

    // Restrições e função objetivo
    IloRangeArray constr_l1;
    IloRangeArray constr_l2;
    IloRangeArray constr_m2;
    IloArray<IloRangeArray> constrs_v;

    IloObjective obj;

    // Colunas
    struct ColumnInfo
    {
        int i; // planta
        int j; // depósito
    };
    std::vector<std::vector<ColumnInfo>> col_info; // col_info[k][t] = (i,j) do padrão t de k
    std::vector<IloNumVarArray> z;                 // z[k][t]

  public:
    explicit TSCFLSolverColumnGeneration(
        const TSCFLInstance &inst_, Subproblem::Mode smode = Subproblem::Mode::NET
    );

    ~TSCFLSolverColumnGeneration() override;

    void solve(bool log_output = true, IloNum time_limit = -1.0);

  private:
    void buildModel();

    void addColumn(int k, int i, int j);

    IloInt getNumColumns() const;
};
