#include "lrp/tscfl_lrp_capacities.hpp"

ILOSTLBEGIN

// ---------------------------------------------------------------------
//  Construtor
// ---------------------------------------------------------------------

LRPCapacity::LRPCapacity(const TSCFLInstance &inst_)
    : LRP(inst_),
      l1(env, inst_.nI),
      l2(env, inst_.nJ),
      g1(env, inst_.nI),
      g2(env, inst_.nJ)
{
}

// ---------------------------------------------------------------------
//  solve
// ---------------------------------------------------------------------

IloNum
LRPCapacity::solve()
{
    // 0) Zera fluxos e atualiza custos induzidos pelos cortes
    fill_zero(x);
    fill_zero(y);
    cuts.update_costs();

    // 1) Para cada cliente k: envia r_k para o par (i,j) de menor custo reduzido
    for (IloInt k = 0; k < inst.nK; ++k)
        {
            IloNum rk = inst.r[k];
            if (rk <= EPS)
                continue;

            IloNum best_cost = IloInfinity;
            IloInt best_i = -1;
            IloInt best_j = -1;

            for (IloInt i = 0; i < inst.nI; ++i)
                {
                    for (IloInt j = 0; j < inst.nJ; ++j)
                        {
                            IloNum cost = inst.c[i][j] + inst.d[j][k] + l1[i] + l2[j]
                                          + cuts.cost_x[i][j] + cuts.cost_y[j][k];

                            if (cost < best_cost)
                                {
                                    best_cost = cost;
                                    best_i = i;
                                    best_j = j;
                                }
                        }
                }

            x[best_i][best_j] += rk;
            y[best_j][k] += rk;
        }

    // 2) Decisão de abertura via custo reduzido dos termos fixos
    for (IloInt i = 0; i < inst.nI; ++i)
        {
            // f_i - l1_i * p_i + contribuição de cortes
            IloNum f_tilde = inst.f[i] - l1[i] * inst.p[i] + cuts.cost_a[i];
            a[i] = (f_tilde < 0.0 ? 1.0 : 0.0);
        }

    for (IloInt j = 0; j < inst.nJ; ++j)
        {
            // g_j - l2_j * q_j + contribuição de cortes
            IloNum b_tilde = inst.g[j] - l2[j] * inst.q[j] + cuts.cost_b[j];
            b[j] = (b_tilde < 0.0 ? 1.0 : 0.0);
        }

    // 3) Subgradientes das capacidades
    for (IloInt i = 0; i < inst.nI; ++i)
        g1[i] = IloSum(x[i]) - inst.p[i] * a[i];

    for (IloInt j = 0; j < inst.nJ; ++j)
        g2[j] = IloSum(y[j]) - inst.q[j] * b[j];

    // 4) Retorna o valor da Lagrangeana
    return IloScalProd(inst.f, a) + IloScalProd(inst.g, b) + IloMatScalProd(inst.c, x)
           + IloMatScalProd(inst.d, y) + IloScalProd(l1, g1) + IloScalProd(l2, g2);
}

// ---------------------------------------------------------------------
//  norm2sq
// ---------------------------------------------------------------------

IloNum
LRPCapacity::norm2sq() const
{
    return IloScalProd(g1, g1) + IloScalProd(g2, g2) + cuts.norm2sq();
}

// ---------------------------------------------------------------------
//  update_multipliers
// ---------------------------------------------------------------------

void
LRPCapacity::update_multipliers(IloNum step)
{
    if (step <= 0.0)
        return;

    for (IloInt i = 0; i < inst.nI; ++i)
        l1[i] = IloMax(0.0, l1[i] + step * g1[i]);

    for (IloInt j = 0; j < inst.nJ; ++j)
        l2[j] = IloMax(0.0, l2[j] + step * g2[j]);

    cuts.update_multipliers(step);
}
