#pragma once

/*
COS888

LRPBalance: relaxação Lagrangeana do TSCFL com relaxação das
restrições de balanço (depósitos) e demanda (clientes).

Gabriel Braun, 2025
*/

#include "lrp/tscfl_lrp.hpp"

ILOSTLBEGIN

class LRPBalance : public LRP {
   private:
    // Multiplicadores das restrições relaxadas
    IloNumArray m1;  // m1[k] — demanda dos clientes
    IloNumArray m2;  // m2[j] — balanço dos depósitos

    // Subgradientes das restrições relaxadas
    IloNumArray g1;  // g1[k] = r_k - sum_j y_jk
    IloNumArray g2;  // g2[j] = sum_i x_ij - sum_k y_jk

   public:
    explicit LRPBalance(const TSCFLInstance& inst_);

    // Resolve o subproblema Lagrangeano para (m1, m2, u_cuts) fixos.
    // Atualiza (a, b, x, y, g1, g2) e devolve z_LR.
    IloNum solve() override;

    // ||g||^2 = ||g1||^2 + ||g2||^2 + contribuição dos cortes
    IloNum norm2sq() const override;

    // Atualiza multiplicadores (m1, m2) e multiplicadores dos cortes
    void update_multipliers(IloNum step) override;
};
