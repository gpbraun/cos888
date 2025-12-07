/*
COS888

tscfl_lrp_balances.cpp

Gabriel Braun, 2025
*/

#include "lrp/tscfl_lrp_balances.hpp"

ILOSTLBEGIN

LRPBalance::LRPBalance(const TSCFLInstance &inst_)
    : LRP(inst_, Mode::BALANCES),
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

void
LRPBalance::solve()
{
    const IloInt nI = inst.nI;
    const IloInt nJ = inst.nJ;
    const IloInt nK = inst.nK;

    // Zera fluxos e atualiza custos induzidos pelos cortes
    fill_zero(x);
    fill_zero(y);
    cuts.update_costs();

    // Subproblema das plantas (x, a)
    for (IloInt i = 0; i < nI; ++i)
        {
            IloNum best_red = IloInfinity;
            IloInt best_j = -1;

            for (IloInt j = 0; j < nJ; ++j)
                {
                    const IloNum red = inst.c[i][j] + m2[j] + cuts.cost_x[i][j];
                    if (red < best_red)
                        {
                            best_red = red;
                            best_j = j;
                        }
                }

            // Termo fixo da planta i (inclui cortes em a_i)
            const IloNum fixed_term = inst.f[i] + cuts.cost_a[i];

            // Caso fechado: a_i = 0, x_i· = 0 → custo 0
            const IloNum cost_closed = 0.0;

            // Caso aberto: a_i = 1
            IloNum cost_open = fixed_term;

            if (best_j >= 0 && best_red < 0.0 && inst.p[i] > EPS)
                {
                    // Envia toda a capacidade p_i no arco de menor custo reduzido
                    cost_open += inst.p[i] * best_red;
                }

            if (cost_open < cost_closed - EPS)
                {
                    // Melhor abrir planta i
                    a[i] = 1.0;
                    if (best_j >= 0 && best_red < 0.0 && inst.p[i] > EPS)
                        {
                            x[i][best_j] = inst.p[i];
                        }
                }
            else
                {
                    // Mantém planta fechada
                    a[i] = 0.0;
                    // x[i][*] já está zerado
                }
        }

    // 2) Subproblema dos depósitos (y, b)
    for (IloInt j = 0; j < nJ; ++j)
        {
            IloNum best_red = IloInfinity;
            IloInt best_k = -1;

            for (IloInt k = 0; k < nK; ++k)
                {
                    const IloNum red = inst.d[j][k] - m1[k] - m2[j] + cuts.cost_y[j][k];
                    if (red < best_red)
                        {
                            best_red = red;
                            best_k = k;
                        }
                }

            // Termo fixo do depósito j (inclui cortes em b_j)
            const IloNum fixed_term = inst.g[j] + cuts.cost_b[j];

            const IloNum cost_closed = 0.0;
            IloNum cost_open = fixed_term;

            if (best_k >= 0 && best_red < 0.0 && inst.q[j] > EPS)
                {
                    cost_open += inst.q[j] * best_red;
                }

            if (cost_open < cost_closed - EPS)
                {
                    b[j] = 1.0;
                    if (best_k >= 0 && best_red < 0.0 && inst.q[j] > EPS)
                        {
                            y[j][best_k] = inst.q[j];
                        }
                }
            else
                {
                    b[j] = 0.0;
                    // y[j][*] já está zerado
                }
        }

    // Subgradientes das restrições relaxadas
    // Demanda dos clientes
    for (IloInt k = 0; k < nK; ++k)
        {
            IloNum sum_y = 0.0;
            for (IloInt j = 0; j < nJ; ++j)
                {
                    sum_y += y[j][k];
                }
            g1[k] = inst.r[k] - sum_y;
        }

    // Balanço nos depósitos
    for (IloInt j = 0; j < nJ; ++j)
        {
            IloNum sum_x = 0.0;
            for (IloInt i = 0; i < nI; ++i)
                {
                    sum_x += x[i][j];
                }
            const IloNum sum_y = IloSum(y[j]); // y[j] é um IloNumArray
            g2[j] = sum_x - sum_y;
        }

    // Valor da Lagrangeana
    opt = IloScalProd(inst.f, a) + IloScalProd(inst.g, b) + IloMatScalProd(inst.c, x)
          + IloMatScalProd(inst.d, y) + IloScalProd(m1, g1) + IloScalProd(m2, g2);
}

// Atualiza multiplicadores (m1, m2) e multiplicadores dos cortes
// m1 e m2 são livres (restrições de igualdade)
void
LRPBalance::update_multipliers(IloNum step)
{
    if (step <= 0.0)
        return;

    const IloInt nK = inst.nK;
    const IloInt nJ = inst.nJ;

    for (IloInt k = 0; k < nK; ++k)
        {
            m1[k] += step * g1[k];
        }
    for (IloInt j = 0; j < nJ; ++j)
        {
            m2[j] += step * g2[j];
        }

    cuts.update_multipliers(step);
}

// ||g||^2 = ||g1||^2 + ||g2||^2 + contribuição dos cortes
IloNum
LRPBalance::norm2sq() const
{
    return IloScalProd(g1, g1) + IloScalProd(g2, g2) + cuts.norm2sq();
}