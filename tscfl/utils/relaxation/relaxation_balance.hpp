/*
COS888

Relaxação Lagrangeana do TSCFL com relaxação das restrições de balanço.

Gabriel Braun, 2025
*/

#pragma once

#include "utils/relaxation/relaxation.hpp"

// SOLVER DA RELAXAÇÃO LAGRANGEANA: Balanços relaxados
class RelaxationBalance : public Relaxation
{
private:
    IloNumArray m1; // m1[k]
    IloNumArray m2; // m2[j]
    IloNumArray g1; // g1[k] = r_k - sum_j y_jk
    IloNumArray g2; // g2[j] = sum_i x_ij - sum_k y_jk

public:
    explicit RelaxationBalance(const TSCFLInstance &inst_)
        : Relaxation(inst_),
          m1(env, inst_.nK),
          m2(env, inst_.nJ),
          g1(env, inst_.nK),
          g2(env, inst_.nJ)
    {
        fill_zero(m1);
        fill_zero(m2);
        fill_zero(g1);
        fill_zero(g2);
    }

    // --------------------------------------------------------------
    // Resolve o subproblema Lagrangeano para (m1, m2, u_cuts) fixos
    // Atualiza (a, b, x, y) e devolve z_LR
    //
    // Estrutura:
    // - para cada planta i: mochila contínua em x_ij com capacidade p_i a_i
    // - para cada depósito j: mochila contínua em y_jk com capacidade q_j b_j
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

        // 1) Subproblema das plantas (mochila contínua em x,a)
        for (IloInt i = 0; i < nI; ++i)
        {
            // Custo reduzido por unidade de fluxo em cada arco (i,j):
            // c_ij + m2_j + contribuição dos cortes
            IloNum best_red = IloInfinity;
            IloInt best_j = -1;

            for (IloInt j = 0; j < nJ; ++j)
            {
                IloNum red = inst.c[i][j] + m2[j] + cuts.cost_x[i][j];
                if (red < best_red)
                {
                    best_red = red;
                    best_j = j;
                }
            }

            // Termo fixo associado à abertura da planta i (inclui cortes)
            IloNum fixed_term = inst.f[i] + cuts.cost_a[i];

            // Caso fechado: a_i = 0, x_i. = 0
            const IloNum cost_closed = 0.0;

            // Caso aberto: a_i = 1, resolve mochila contínua
            IloNum cost_open = fixed_term;

            if (best_j >= 0 && best_red < 0.0 && inst.p[i] > EPS)
            {
                // Se há algum arco com custo reduzido negativo, preenche
                // toda a capacidade p_i nesse arco mais barato
                cost_open += inst.p[i] * best_red;
            }

            if (cost_open < cost_closed - EPS)
            {
                // Melhor abrir planta i
                a[i] = 1.0;

                if (best_j >= 0 && best_red < 0.0 && inst.p[i] > EPS)
                    x[i][best_j] = inst.p[i]; // envia toda a capacidade
            }
            else
            {
                // Melhor manter fechada
                a[i] = 0.0;
                // x[i][*] já está zerado
            }
        }

        // 2) Subproblema dos depósitos (mochila contínua em y,b)
        for (IloInt j = 0; j < nJ; ++j)
        {
            // Custo reduzido por unidade de fluxo em cada arco (j,k):
            // d_jk - m1_k - m2_j + contribuição dos cortes
            IloNum best_red = IloInfinity;
            IloInt best_k = -1;

            for (IloInt k = 0; k < nK; ++k)
            {
                IloNum red = inst.d[j][k] - m1[k] - m2[j] + cuts.cost_y[j][k];
                if (red < best_red)
                {
                    best_red = red;
                    best_k = k;
                }
            }

            // Termo fixo associado à abertura do depósito j (inclui cortes)
            IloNum fixed_term = inst.g[j] + cuts.cost_b[j];

            const IloNum cost_closed = 0.0;

            IloNum cost_open = fixed_term;

            if (best_k >= 0 && best_red < 0.0 && inst.q[j] > EPS)
            {
                // Se há arco com custo reduzido negativo, preenche toda
                // a capacidade q_j nesse cliente mais barato
                cost_open += inst.q[j] * best_red;
            }

            if (cost_open < cost_closed - EPS)
            {
                b[j] = 1.0;

                if (best_k >= 0 && best_red < 0.0 && inst.q[j] > EPS)
                    y[j][best_k] = inst.q[j];
            }
            else
            {
                b[j] = 0.0;
                // y[j][*] já está zerado
            }
        }

        // 3) Subgradientes das restrições relaxadas
        // Demanda dos clientes: r_k - sum_j y_jk
        for (IloInt k = 0; k < nK; ++k)
        {
            IloNum sum_y = 0.0;
            for (IloInt j = 0; j < nJ; ++j)
                sum_y += y[j][k];

            g1[k] = inst.r[k] - sum_y;
        }

        // Balanço nos depósitos: sum_i x_ij - sum_k y_jk
        for (IloInt j = 0; j < nJ; ++j)
        {
            IloNum sum_x = 0.0;
            for (IloInt i = 0; i < nI; ++i)
                sum_x += x[i][j];

            IloNum sum_y = IloSum(y[j]); // y[j] é um IloNumArray
            g2[j] = sum_x - sum_y;
        }

        // 4) Valor da Lagrangeana
        //
        // L(a,b,x,y; m1,m2,u_cuts) =
        //    f^T a + g^T b
        //  + c : x + d : y
        //  + m1^T g1 + m2^T g2
        //
        // (os cortes entram via costs em (a,b,x,y))
        return IloScalProd(inst.f, a) + IloScalProd(inst.g, b) +
               IloMatScalProd(inst.c, x) + IloMatScalProd(inst.d, y) +
               IloScalProd(m1, g1) + IloScalProd(m2, g2);
    }

    // --------------------------------------------------------------
    // ||g||^2 = ||g1||^2 + ||g2||^2 + contribuição dos cortes
    // --------------------------------------------------------------
    IloNum norm2sq() const override
    {
        return IloScalProd(g1, g1) + IloScalProd(g2, g2) + cuts.norm2sq();
    }

    // --------------------------------------------------------------
    // Atualiza multiplicadores (m1, m2) e multiplicadores dos cortes
    //
    // Aqui m1 e m2 são livres (restrições de igualdade), então não há
    // projeção em R_+, ao contrário de l1,l2 em RelaxationCapacity.
    // --------------------------------------------------------------
    void update_multipliers(IloNum step) override
    {
        if (step <= 0.0)
            return;

        const IloInt nK = inst.nK;
        const IloInt nJ = inst.nJ;

        for (IloInt k = 0; k < nK; ++k)
            m1[k] += step * g1[k];

        for (IloInt j = 0; j < nJ; ++j)
            m2[j] += step * g2[j];

        cuts.update_multipliers(step);
    }
};
