// src/subproblem_dual.hpp
#pragma once

/*
COS888

SubproblemDual: resolve o subproblema dual do TSCFL.

Gabriel Braun, 2025
*/

#include "tscfl_subproblem.hpp"

// SOLVER DO SUBPROBLEMA: Dual.
class SubproblemDual : public Subproblem {
   private:
    IloEnv env;
    IloModel model;
    IloCplex cplex;

    // Variáveis duais
    IloNumVarArray var_l1;  // l1[i] >= 0
    IloNumVarArray var_l2;  // l2[j] >= 0
    IloNumVarArray var_m1;  // m1[j] livre
    IloNumVarArray var_m2;  // m2[k] livre

    // Função objetivo
    IloObjective obj;

   public:
    explicit SubproblemDual(const TSCFLInstance& inst_);
    ~SubproblemDual() override;

    // Dado (a_vals, b_vals) da solução atual do mestre, resolve o subproblema.
    // Atualiza: theta, rhs, coef_a, coef_b
    void solve(const IloNumArray& a_vals, const IloNumArray& b_vals) override;

   private:
    // Constrói: modelo base do subproblema dual
    void build_base_model();

    // Atualiza: função objetivo do subproblema em função de (a,b)
    void set_objective(const IloNumArray& a_vals, const IloNumArray& b_vals);
};
