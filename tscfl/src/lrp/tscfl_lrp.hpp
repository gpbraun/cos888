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

    Mode mode;

    // Solução Lagrangeana corrente
    IloNum opt{ 0.0 };

    IloNumArray a;  // a[i]
    IloNumArray b;  // b[j]
    IloNumMatrix x; // x[i][j]
    IloNumMatrix y; // y[j][k]

    explicit LRP(const TSCFLInstance &inst_, Mode mode_);

    virtual ~LRP() = default;

    // Factory
    static std::unique_ptr<LRP> create(const TSCFLInstance &inst, Mode mode_);

    // Resolve o subproblema Lagrangeano para os multiplicadores atuais.
    // Atualiza: opt, a, b, x, y
    virtual void solve() = 0;

    // Atualiza multiplicadores (l1,l2 ou m1,m2) e dos cortes.
    virtual void update_multipliers(IloNum step) = 0;

    // Retorna: ||g||^2 (quadrado da norma dos subgradientes)
    virtual IloNum norm2sq() const = 0;

    // Separa cortes FlowCovers a partir da solução LR atual
    IloInt separate_flow_covers(IloInt max_new_cuts);

    // Separa cortes SubsetRow a partir da solução LR atual
    IloInt separate_subset_rows(IloInt max_new_cuts);

    // Acesso ao conjunto de cortes
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
