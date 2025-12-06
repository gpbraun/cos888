#include "lrp/tscfl_lrp_balances.hpp"

ILOSTLBEGIN

LRPBalance::LRPBalance(const TSCFLInstance &inst_)
    : LRP(inst_),
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
// Resolve o subproblema Lagrangeano para (m1, m2, u_cuts) fixos.
//
// Atualiza (a, b, x, y, g1, g2) e devolve z_LR.
//
// Estrutura:
//  (1) Para cada planta i: mochila contínua em x_ij com cap. p_i a_i
//      reduzido: c_ij + m2_j + cuts.cost_x[i][j]
//  (2) Para cada depósito j: mochila contínua em y_jk com cap. q_j b_j
//      reduzido: d_jk - m1_k - m2_j + cuts.cost_y[j][k]
// --------------------------------------------------------------
IloNum
LRPBalance::solve()
{
    const IloInt nI = inst.nI;
    const IloInt nJ = inst.nJ;
    const IloInt nK = inst.nK;

    // 0) Zera fluxos e atualiza custos induzidos pelos cortes
    fill_zero(x);
    fill_zero(y);
    cuts.update_costs();

    // ==========================================================
    // 1) Subproblema das plantas (x, a)
    //    Para cada i, com cap. p_i a_i:
    //
    //  Se a_i = 0  → custo = 0
    //  Se a_i = 1  → escolhe j* com menor custo reduzido
    //                 red_ij = c_ij + m2_j + cuts.cost_x[i][j]
    //               se red_{i j*} < 0, envia p_i em (i,j*)
    // ==========================================================
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

    // ==========================================================
    // 2) Subproblema dos depósitos (y, b)
    //    Para cada j, com cap. q_j b_j:
    //
    //  Se b_j = 0 → custo = 0
    //  Se b_j = 1 → escolhe k* com menor custo reduzido
    //                 red_jk = d_jk - m1_k - m2_j + cuts.cost_y[j][k]
    //               se red_{j k*} < 0, envia q_j em (j,k*)
    // ==========================================================
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

    // ==========================================================
    // 3) Subgradientes das restrições relaxadas
    //
    //    g1[k] = r_k - ∑_j y_jk
    //    g2[j] = ∑_i x_ij - ∑_k y_jk
    // ==========================================================
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

    // ==========================================================
    // 4) Valor da Lagrangeana
    //
    //    L(a,b,x,y; m1,m2,u_cuts) =
    //        f^T a + g^T b
    //      + c : x + d : y
    //      + m1^T g1 + m2^T g2
    //
    //  Obs: contribuições dos cortes entram via custos em (a,b,x,y)
    //       através de cuts.update_costs().
    // ==========================================================
    IloNum val = 0.0;
    val += IloScalProd(inst.f, a);
    val += IloScalProd(inst.g, b);
    val += IloMatScalProd(inst.c, x);
    val += IloMatScalProd(inst.d, y);
    val += IloScalProd(m1, g1);
    val += IloScalProd(m2, g2);

    return val;
}

// ||g||^2 = ||g1||^2 + ||g2||^2 + contribuição dos cortes
IloNum
LRPBalance::norm2sq() const
{
    return IloScalProd(g1, g1) + IloScalProd(g2, g2) + cuts.norm2sq();
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
