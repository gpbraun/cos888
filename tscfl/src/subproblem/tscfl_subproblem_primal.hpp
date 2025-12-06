// src/subproblem_primal.hpp
#pragma once

/*
COS888

SubproblemPrimal: resolve o subproblema primal do TSCFL.

Gabriel Braun, 2025
*/

#include "tscfl_subproblem.hpp"

// SOLVER DO SUBPROBLEMA: Primal.
class SubproblemPrimal : public Subproblem {
   private:
    IloEnv env;
    IloModel model;
    IloCplex cplex;

    // Variáveis de fluxo
    IloNumVarMatrix var_x;  // x[i][j]
    IloNumVarMatrix var_y;  // y[j][k]

    // Restrições
    IloRangeArray constr_l1;  // constr_l1[i]
    IloRangeArray constr_l2;  // constr_l2[j]
    IloRangeArray constr_m1;  // constr_m1[j]
    IloRangeArray constr_m2;  // constr_m2[k]

   public:
    explicit SubproblemPrimal(const TSCFLInstance& inst_);
    ~SubproblemPrimal() override;

    // Dado (a_vals, b_vals) da solução atual do mestre, resolve o subproblema.
    // Atualiza: theta, rhs, coef_a, coef_b
    void solve(const IloNumArray& a_vals, const IloNumArray& b_vals) override;

   private:
    // Constrói o modelo base do subproblema primal
    void build_base_model();

    // Atualiza o lado direito das restrições que dependem de (a,b)
    void set_constraints(const IloNumArray& a_vals, const IloNumArray& b_vals);
};
