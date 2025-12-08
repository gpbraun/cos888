/*
COS888

tscfl_lrp_balances.hpp

Gabriel Braun, 2025
*/

#pragma once

#include "lrp/tscfl_lrp.hpp"

// SOLVER DA RELAXAÇÃO LAGRANGEANA: Balanços relaxados
class LRPBalance : public LRP
{
  private:
    // Multiplicadores das restrições de balanço/demanda
    IloNumArray m1; // m1[k] (demanda dos clientes)
    IloNumArray m2; // m2[j] (balanço dos depósitos)

    // Subgradientes das restrições relaxadas
    IloNumArray g1; // g1[k] = r_k - sum_j y_jk
    IloNumArray g2; // g2[j] = sum_i x_ij - sum_k y_jk

  public:
    explicit LRPBalance(const TSCFLInstance &inst_);

    // Resolve o subproblema Lagrangeano para (m1, m2, u_cuts) fixos.
    // Atualiza: opt, a, b, x, y, g1, g2
    void solve() override;

    // Atualiza multiplicadores (m1, m2) e multiplicadores dos cortes
    void updateMultipliers(IloNum step) override;

    // ||g||^2 = ||g1||^2 + ||g2||^2 + contribuição dos cortes
    IloNum norm2sq() const override;
};
