/*
COS888

tscfl_subproblem_net.hpp

Gabriel Braun, 2025
*/

#pragma once

#include "tscfl_subproblem.hpp"

// SOLVER DO SUBPROBLEMA: Primal.
class SubproblemPrimal : public Subproblem
{
  private:
    IloEnv env;
    IloModel model;
    IloCplex cplex;

    // Variáveis
    IloNumVarMatrix var_x; // var_x[i][j]
    IloNumVarMatrix var_y; // var_y[j][k]
    // Restrições
    IloRangeArray constr_l1; // constr_l1[i]
    IloRangeArray constr_l2; // constr_l2[j]
    IloRangeArray constr_m1; // constr_m1[j]
    IloRangeArray constr_m2; // constr_m2[k]

    // Valores dos duais
    IloNumArray l1; // var_l1[i]
    IloNumArray l2; // var_l2[j]
    IloNumArray m2; // var_m2[k]

  public:
    explicit SubproblemPrimal(const TSCFLInstance &inst_);

    ~SubproblemPrimal() override;

    // Resolve o subproblema.
    // Atualiza: x, y, opt, theta, rhs, coef_a, coef_b
    void solve() override;

    // Recupera as variáveis de fluxo
    void getFlows(IloNumMatrix &x, const IloNumMatrix &y);

  private:
    // Constrói o modelo base do subproblema primal
    void buildModel();

    // Atualiza o lado direito das restrições que dependem de (a,b)
    void updateModel();
};
