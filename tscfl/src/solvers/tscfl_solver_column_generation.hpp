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
    IloNumVarArray var_a; // a[i]
    IloNumVarArray var_b; // b[j]

    // Restrições
    IloRangeArray constr_l1;
    IloRangeArray constr_l2;
    IloRangeArray constr_m2;
    IloArray<IloRangeArray> constrs_v;

    IloObjective obj;

    // Colunas (padrões) por cliente
    struct ColumnInfo
    {
        int i; // planta
        int j; // depósito
    };
    std::vector<std::vector<ColumnInfo>> col_info; // col_info[k][t] = (i,j) do padrão t de k
    std::vector<IloNumVarArray> z;                 // z[k][t]

  public:
    // Melhor solução primal inteira encontrada
    IloNumArray a;
    IloNumArray b;

    explicit TSCFLSolverColumnGeneration(
        const TSCFLInstance &inst_, Subproblem::Mode smode = Subproblem::Mode::NET
    );

    ~TSCFLSolverColumnGeneration() override;

    bool solve(bool log_output = true, IloNum time_limit = -1.0);

  private:
    void build_initial_model();

    void add_column_for_client(int k, int i, int j);

    IloInt get_num_columns() const;
};
