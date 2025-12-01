/*
COS888

LagrangianRelaxationCapacity:
Relaxação Lagrangeana do TSCFL nas restrições de capacidade:

  sum_j x_ij ≤ p_i a_i   (plantas)
  sum_k y_jk ≤ q_j b_j   (depósitos)

com multiplicadores l1[i] ≥ 0, l2[j] ≥ 0.

- Reaproveita a estrutura base LagrangianRelaxation:
    * a[i], b[j], x[i][j], y[j][k]
    * conjunto de cortes FlowCoverCutSet cuts
- Resolve o subproblema Lagrangeano fechando forma:
    * roteamento cliente k → par (i,j) de menor custo reduzido
    * decisão de abertura a, b via custo reduzido

Gabriel Braun, 2025
*/

#pragma once

#include <ilcplex/ilocplex.h>
#include <algorithm>

#include "lagrangian/tscfl_lagrangian.hpp"
#include "lagrangian/tscfl_flowcuts.hpp"

ILOSTLBEGIN

class LagrangianRelaxationCapacity : public LagrangianRelaxation
{
private:
    // Multiplicadores das restrições de capacidades
    IloNumArray l1; // l1[i] >= 0: planta i
    IloNumArray l2; // l2[j] >= 0: depósito j
    // Subgradientes das restrições de capacidades
    IloNumArray g1; // g1[i] = sum_j x_ij - p_i a_i
    IloNumArray g2; // g2[j] = sum_k y_jk - q_j b_j

public:
    explicit LagrangianRelaxationCapacity(const TSCFLInstance &inst_)
        : LagrangianRelaxation(inst_),
          l1(env, inst_.nI),
          l2(env, inst_.nJ),
          g1(env, inst_.nI),
          g2(env, inst_.nJ)
    {
    }

    // --------------------------------------------------------------
    // Resolve o subproblema Lagrangeano para (l1, l2, u_cuts) fixos
    // Atualiza (a, b, x, y) e devolve z_LR
    // --------------------------------------------------------------
    IloNum solve() override
    {
        const IloInt nI = inst.nI;
        const IloInt nJ = inst.nJ;
        const IloInt nK = inst.nK;

        // 0) Zera fluxos e atualiza custos induzidos pelos cortes
        fill_zero(x);
        fill_zero(y);
        cuts.update_costs();

        // 1) Para cada cliente k: envia r_k para o par (i,j) de menor custo reduzido
        for (IloInt k = 0; k < nK; ++k)
        {
            IloNum rk = inst.r[k];
            if (rk <= EPS)
                continue;

            IloNum best_cost = IloInfinity;
            IloInt best_i = -1;
            IloInt best_j = -1;

            for (IloInt i = 0; i < nI; ++i)
            {
                for (IloInt j = 0; j < nJ; ++j)
                {
                    IloNum cost = inst.c[i][j] + inst.d[j][k] + l1[i] + l2[j] + cuts.cost_x[i][j] + cuts.cost_y[j][k];

                    if (cost < best_cost)
                    {
                        best_cost = cost;
                        best_i = i;
                        best_j = j;
                    }
                }
            }

            if (best_i < 0 || best_j < 0)
                continue; // por segurança, mas não deveria acontecer

            x[best_i][best_j] += rk;
            y[best_j][k] += rk;
        }

        // 2) Decisão de abertura via custo reduzido dos termos fixos
        for (IloInt i = 0; i < nI; ++i)
        {
            // f_i - l1_i * p_i + contribuição de cortes
            IloNum red_cost = inst.f[i] - l1[i] * inst.p[i] + cuts.cost_a[i];
            a[i] = (red_cost < 0.0 ? 1.0 : 0.0);
        }

        for (IloInt j = 0; j < nJ; ++j)
        {
            // g_j - l2_j * q_j + contribuição de cortes
            IloNum red_cost = inst.g[j] - l2[j] * inst.q[j] + cuts.cost_b[j];
            b[j] = (red_cost < 0.0 ? 1.0 : 0.0);
        }

        // 3) Subgradientes das capacidades
        for (IloInt i = 0; i < nI; ++i)
            g1[i] = IloSum(x[i]) - inst.p[i] * a[i];

        for (IloInt j = 0; j < nJ; ++j)
            g2[j] = IloSum(y[j]) - inst.q[j] * b[j];

        // 4) Retorna o valor da Lagrangeana
        return IloScalProd(inst.f, a) + IloScalProd(inst.g, b) +       // custo fixo
               IloMatScalProd(inst.c, x) + IloMatScalProd(inst.d, y) + // custo variável
               IloScalProd(l1, g1) + IloScalProd(l2, g2);              // termos lagrangeanos (l^T g)
    }

    // --------------------------------------------------------------
    // ||g||^2 = ||g1||^2 + ||g2||^2 + contribuição dos cortes
    // --------------------------------------------------------------
    IloNum norm2sq() const override
    {
        return IloScalProd(g1, g1) + IloScalProd(g2, g2) + cuts.norm2sq();
    }

    // --------------------------------------------------------------
    // Atualiza multiplicadores (l1, l2) e multiplicadores dos cortes
    // --------------------------------------------------------------
    void update_multipliers(IloNum step) override
    {
        if (step <= 0.0)
            return;

        const IloInt nI = inst.nI;
        const IloInt nJ = inst.nJ;

        for (IloInt i = 0; i < nI; ++i)
            l1[i] = std::max(0.0, l1[i] + step * g1[i]);

        for (IloInt j = 0; j < nJ; ++j)
            l2[j] = std::max(0.0, l2[j] + step * g2[j]);

        cuts.update_multipliers(step);
    }
};
