/*
COS888

tscfl_subproblem_dual.hpp

Gabriel Braun, 2025
*/

#pragma once

#include "tscfl_subproblem.hpp"

// SOLVER DO SUBPROBLEMA: Dual.
class SubproblemDual : public Subproblem
{
  private:
    IloEnv env;
    IloModel model;
    IloCplex cplex;

    // Variáveis duais
    IloNumVarArray var_l1; // var_l1[i] <= 0
    IloNumVarArray var_l2; // var_l2[j] <= 0
    IloNumVarArray var_m1; // var_m1[j] livre
    IloNumVarArray var_m2; // var_m2[k] livre
    // Função objetivo
    IloObjective obj;

  public:
    explicit SubproblemDual(const TSCFLInstance &inst_);

    ~SubproblemDual() override;

    // Resolve o subproblema.
    // Atualiza: x, y, opt, theta, rhs, coef_a, coef_b
    void solve() override;

  private:
    // Constrói: modelo base do subproblema dual
    void buildModel();

    // Atualiza: função objetivo do subproblema em função de (a,b)
    void updateModel();
};
