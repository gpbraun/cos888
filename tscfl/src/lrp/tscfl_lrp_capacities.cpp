/*
COS888

tscfl_lrp_capacities.cpp

Gabriel Braun, 2025
*/

#include "lrp/tscfl_lrp_capacities.hpp"

LRPCapacity::LRPCapacity(const TSCFLInstance &inst_)
    : LRP(inst_, Mode::CAPACITIES),
      l1(env, inst_.nI),
      l2(env, inst_.nJ),
      g1(env, inst_.nI),
      g2(env, inst_.nJ),
      best_inner_c(inst_.env, inst_.nJ),
      best_inner_i(inst_.env, inst_.nJ)
{
}

void
LRPCapacity::solve()
{
    // Zera fluxos, subgradientes e atualiza custos induzidos pelos cortes
    fillZero(x);
    fillZero(y);
    fillZero(g1);
    fillZero(g2);

    cuts.updateCosts();

    // Para cada satélite j, escolhe a melhor planta i
    for (IloInt j = 0; j < inst.nJ; ++j)
        {
            IloNum best_c = IloInfinity;
            IloInt best_i = -1;

            for (IloInt i = 0; i < inst.nI; ++i)
                {
                    IloNum c_ij = inst.c[i][j] + l1[i] + cuts.cost_x[i][j];
                    if (c_ij < best_c)
                        {
                            best_c = c_ij;
                            best_i = i;
                        }
                }

            best_inner_c[j] = best_c;
            best_inner_i[j] = best_i;
        }

    // Para cada cliente k: manda r_k via melhor j (e o i escolhido em 1a)
    for (IloInt k = 0; k < inst.nK; ++k)
        {
            IloNum rk = inst.r[k];
            if (rk <= EPS)
                continue;

            IloNum best_cost = IloInfinity;
            IloInt best_j = -1;

            for (IloInt j = 0; j < inst.nJ; ++j)
                {
                    // d_jk + l2_j + cost_y_jk depende de (j,k), não de i
                    IloNum cost_jk = best_inner_c[j] + l2[j] + inst.d[j][k] + cuts.cost_y[j][k];

                    if (cost_jk < best_cost)
                        {
                            best_cost = cost_jk;
                            best_j = j;
                        }
                }

            IloInt best_i = best_inner_i[best_j];

            // Atualiza fluxos
            x[best_i][best_j] += rk;
            y[best_j][k] += rk;

            // Acumula parte dos subgradientes
            g1[best_i] += rk;
            g2[best_j] += rk;
        }

    // Decisão de abertura via custo reduzido dos termos fixos
    for (IloInt i = 0; i < inst.nI; ++i)
        {
            // f_i - l1_i * p_i + contribuição de cortes
            IloNum f_tilde = inst.f[i] - l1[i] * inst.p[i] + cuts.cost_a[i];
            a[i] = (f_tilde < 0.0 ? 1.0 : 0.0);

            // completa subgradiente: g1[i] = sum_j x_ij - p_i * a_i
            g1[i] -= inst.p[i] * a[i];
        }

    for (IloInt j = 0; j < inst.nJ; ++j)
        {
            // g_j - l2_j * q_j + contribuição de cortes
            IloNum b_tilde = inst.g[j] - l2[j] * inst.q[j] + cuts.cost_b[j];
            b[j] = (b_tilde < 0.0 ? 1.0 : 0.0);

            // g2[j] = sum_k y_jk - q_j * b_j
            g2[j] -= inst.q[j] * b[j];
        }

    // Valor da Lagrangeana
    opt = IloScalProd(inst.f, a) + IloScalProd(inst.g, b) + IloMatScalProd(inst.c, x)
          + IloMatScalProd(inst.d, y) + IloScalProd(l1, g1) + IloScalProd(l2, g2);
}

void
LRPCapacity::updateMultipliers(IloNum step)
{
    if (step <= 0.0)
        return;

    for (IloInt i = 0; i < inst.nI; ++i)
        l1[i] = IloMax(0.0, l1[i] + step * g1[i]);

    for (IloInt j = 0; j < inst.nJ; ++j)
        l2[j] = IloMax(0.0, l2[j] + step * g2[j]);

    cuts.updateMultipliers(step);
}

IloNum
LRPCapacity::norm2sq() const
{
    return IloScalProd(g1, g1) + IloScalProd(g2, g2) + cuts.norm2sq();
}
