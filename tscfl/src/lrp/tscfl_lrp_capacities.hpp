/*
COS888

tscfl_lrp_capacities.hpp

Gabriel Braun, 2025
*/

#pragma once

#include "lrp/tscfl_lrp.hpp"

// SOLVER DA RELAXAÇÃO LAGRANGEANA: Capacidades relaxadas
class LRPCapacity : public LRP
{
  private:
    // Multiplicadores das restrições de capacidade
    IloNumArray l1; // l1[i] >= 0: planta i
    IloNumArray l2; // l2[j] >= 0: depósito j

    // Subgradientes das restrições de capacidade
    IloNumArray g1; // g1[i] = sum_j x_ij - p_i a_i
    IloNumArray g2; // g2[j] = sum_k y_jk - q_j b_j

  public:
    explicit LRPCapacity(const TSCFLInstance &inst_);

    // Resolve o subproblema Lagrangeano para (l1, l2, u_cuts) fixos e atualiza (a, b, x, y).
    // Retorna: z_LR
    IloNum solve() override;

    // ||g||^2 = ||g1||^2 + ||g2||^2 + contribuição dos cortes
    IloNum norm2sq() const override;

    // Atualiza multiplicadores (l1, l2) e multiplicadores dos cortes
    void update_multipliers(IloNum step) override;
};
