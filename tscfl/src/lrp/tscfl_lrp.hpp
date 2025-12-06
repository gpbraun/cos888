/*
COS888

tscfl_lrp.hpp

Gabriel Braun, 2025
*/

#pragma once

#include <memory>

#include "tscfl_cut_manager.hpp"
#include "tscfl_instance.hpp"
#include "tscfl_utils.hpp"

ILOSTLBEGIN

// SOLVER DA RELAXAÇÃO LAGRANGEANA: Base
class LRP
{
  protected:
    IloEnv &env;
    const TSCFLInstance &inst;

    CutManager cuts;

  public:
    enum class Mode
    {
        CAPACITIES,
        BALANCES,
    };

    // Solução Lagrangeana corrente
    IloNumArray a;  // a[i]    = abre planta i (relaxação linear)
    IloNumArray b;  // b[j]    = abre depósito j (relaxação linear)
    IloNumMatrix x; // x[i][j] = fluxo planta i -> depósito j
    IloNumMatrix y; // y[j][k] = fluxo depósito j -> cliente k

    explicit LRP(const TSCFLInstance &inst_);

    virtual ~LRP() = default;

    // Factory
    static std::unique_ptr<LRP> create(const TSCFLInstance &inst, Mode mode);

    // Resolve o subproblema Lagrangeano para os multiplicadores atuais e atualiza (a, b, x, y).
    // Retorna: z_LR.
    virtual IloNum solve() = 0;

    // Retorna: ||g||^2 (quadrado da norma dos subgradientes)
    virtual IloNum norm2sq() const = 0;

    // Atualiza multiplicadores (l1,l2 ou m1,m2) e também multiplicadores dos cortes.
    virtual void update_multipliers(IloNum step) = 0;

    // Separa Flow Covers a partir da solução LR atual
    IloInt separate_flow_covers(IloInt max_new_cuts);

    IloInt separate_subset_rows(IloInt max_new_cuts);

    // Acesso ao conjunto de cortes (para o solver de subgradiente)
    CutManager &
    getCuts()
    {
        return cuts;
    }
    const CutManager &
    getCuts() const
    {
        return cuts;
    }
};
