/*
COS888

tscfl_lrp.cpp

Gabriel Braun, 2025
*/

#include "lrp/tscfl_lrp.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "lrp/tscfl_lrp_balances.hpp"
#include "lrp/tscfl_lrp_capacities.hpp"

LRP::LRP(const TSCFLInstance &inst_, Mode mode_)
    : env(inst_.env),
      inst(inst_),
      cuts(inst_),
      mode(mode_),
      a(env, inst_.nI),
      b(env, inst_.nJ),
      x(env, inst_.nI, inst_.nJ),
      y(env, inst_.nJ, inst_.nK)
{
}

std::unique_ptr<LRP>
LRP::create(const TSCFLInstance &inst, Mode mode_)
{
    switch (mode_)
        {
        case Mode::CAPACITIES:
            return std::make_unique<LRPCapacity>(inst);
        case Mode::BALANCES:
            return std::make_unique<LRPBalance>(inst);
        default:
            throw std::invalid_argument("LRP::create: modo desconhecido.");
        }
}

IloInt
LRP::separate_flow_covers(IloInt max_new_cuts)
{
    if (max_new_cuts <= 0)
        return 0;

    // Máximo de covers por cada planta/depósito
    static const IloInt MAX_COVERS_PER_NODE = 3;

    std::vector<FlowCoverCut> candidates;
    candidates.reserve(static_cast<std::size_t>(3 * (inst.nI + inst.nJ)));

    // Cortes de planta (flow-cover em ∑_j x_ij ≤ p_i a_i)
    for (IloInt i = 0; i < inst.nI; ++i)
        {
            // índices j com x_ij > 0 e q_j > 0
            std::vector<IloInt> idx;
            idx.reserve(static_cast<std::size_t>(inst.nJ));
            for (IloInt j = 0; j < inst.nJ; ++j)
                {
                    if (x[i][j] > EPS && inst.q[j] > EPS)
                        idx.push_back(j);
                }
            if (idx.empty())
                continue;

            // ordena por capacidade q_j decrescente
            std::sort(
                idx.begin(),
                idx.end(),
                [&](IloInt j1, IloInt j2) { return inst.q[j1] > inst.q[j2]; }
            );

            const std::size_t m = idx.size();

            // prefixos de q_j e x_ij para reuso
            std::vector<IloNum> prefix_q(m, 0.0);
            std::vector<IloNum> prefix_x(m, 0.0);

            for (std::size_t t = 0; t < m; ++t)
                {
                    const IloInt j = idx[t];
                    prefix_q[t] = inst.q[j] + (t > 0 ? prefix_q[t - 1] : 0.0);
                    prefix_x[t] = x[i][j] + (t > 0 ? prefix_x[t - 1] : 0.0);
                }

            IloInt covers_for_i = 0;
            for (std::size_t t = 0; t < m && covers_for_i < MAX_COVERS_PER_NODE; ++t)
                {
                    const IloNum sum_q = prefix_q[t];

                    // precisa de cover: ∑_{j∈T} q_j > p_i
                    if (sum_q <= inst.p[i] + EPS)
                        continue;

                    const IloNum overflow_cap = sum_q - inst.p[i];
                    if (overflow_cap <= EPS)
                        continue;

                    // T = { idx[0], ..., idx[t] }
                    // lhs = -p_i a_i + ∑_{j∈T} x_ij
                    IloNum lhs = -inst.p[i] * a[i] + prefix_x[t];

                    // cost[j] = 1 se j ∈ T, 0 caso contrário
                    IloNumArray cost(env, inst.nJ);
                    for (IloInt j = 0; j < inst.nJ; ++j)
                        cost[j] = 0.0;
                    for (std::size_t u = 0; u <= t; ++u)
                        {
                            const IloInt j = idx[u];
                            cost[j] = 1.0;
                        }

                    // rhs = ∑_{j∉T} min(q_j, overflow_cap)
                    IloNum rhs = 0.0;
                    for (IloInt j = 0; j < inst.nJ; ++j)
                        {
                            if (cost[j] <= EPS)
                                rhs += IloMin(inst.q[j], overflow_cap);
                        }

                    const IloNum viol = lhs - rhs;
                    if (viol > EPS)
                        {
                            FlowCoverCut cut(FlowCoverCut::Family::PLANT, i, cost, rhs);
                            cut.overflow = viol; // violação do corte
                            candidates.push_back(std::move(cut));
                            ++covers_for_i;
                        }
                }
        }

    // Cortes de depósito (flow-cover em ∑_k y_jk ≤ q_j b_j)
    for (IloInt j = 0; j < inst.nJ; ++j)
        {
            // índices k com y_jk > 0 e r_k > 0
            std::vector<IloInt> idx;
            idx.reserve(static_cast<std::size_t>(inst.nK));

            for (IloInt k = 0; k < inst.nK; ++k)
                if (y[j][k] > EPS && inst.r[k] > EPS)
                    idx.push_back(k);

            if (idx.empty())
                continue;

            // ordena por demanda r_k decrescente
            std::sort(
                idx.begin(),
                idx.end(),
                [&](IloInt k1, IloInt k2) { return inst.r[k1] > inst.r[k2]; }
            );

            const std::size_t m = idx.size();

            // prefixos de r_k e y_jk para reuso
            std::vector<IloNum> prefix_r(m, 0.0);
            std::vector<IloNum> prefix_y(m, 0.0);

            for (std::size_t t = 0; t < m; ++t)
                {
                    const IloInt k = idx[t];
                    prefix_r[t] = inst.r[k] + (t > 0 ? prefix_r[t - 1] : 0.0);
                    prefix_y[t] = y[j][k] + (t > 0 ? prefix_y[t - 1] : 0.0);
                }

            IloInt covers_for_j = 0;
            for (std::size_t t = 0; t < m && covers_for_j < MAX_COVERS_PER_NODE; ++t)
                {
                    const IloNum sum_r = prefix_r[t];

                    // precisa de cover: ∑_{k∈S} r_k > q_j
                    if (sum_r <= inst.q[j] + EPS)
                        continue;

                    const IloNum overflow_cap = sum_r - inst.q[j];
                    if (overflow_cap <= EPS)
                        continue;

                    // S = { idx[0], ..., idx[t] }
                    // lhs = -q_j b_j + ∑_{k∈S} y_jk
                    IloNum lhs = -inst.q[j] * b[j] + prefix_y[t];

                    // cost[k] = 1 se k ∈ S, 0 caso contrário
                    IloNumArray cost(env, inst.nK);
                    for (IloInt k = 0; k < inst.nK; ++k)
                        cost[k] = 0.0;
                    for (std::size_t u = 0; u <= t; ++u)
                        {
                            const IloInt k = idx[u];
                            cost[k] = 1.0;
                        }

                    // rhs = ∑_{k∉S} min(r_k, overflow_cap)
                    IloNum rhs = 0.0;
                    for (IloInt k = 0; k < inst.nK; ++k)
                        {
                            if (cost[k] <= EPS)
                                rhs += IloMin(inst.r[k], overflow_cap);
                        }

                    const IloNum viol = lhs - rhs;
                    if (viol > EPS)
                        {
                            FlowCoverCut cut(FlowCoverCut::Family::DEPOT, j, cost, rhs);
                            cut.overflow = viol;
                            candidates.push_back(std::move(cut));
                            ++covers_for_j;
                        }
                }
        }

    if (candidates.empty())
        return 0;

    // Ordena por violação (overflow) decrescente
    std::sort(
        candidates.begin(),
        candidates.end(),
        [](const FlowCoverCut &c1, const FlowCoverCut &c2) { return c1.overflow > c2.overflow; }
    );

    //  Insere até max_new_cuts cortes realmente novos
    IloInt new_cuts = 0;
    for (auto &cand : candidates)
        {
            if (new_cuts >= max_new_cuts)
                break;
            if (cuts.addFlowCover(cand))
                ++new_cuts;
        }

    return new_cuts;
}

IloInt
LRP::separate_subset_rows(IloInt max_new_cuts)
{
    if (max_new_cuts <= 0)
        return 0;

    // Ordena clientes por demanda r_k decrescente
    std::vector<IloInt> order_k(static_cast<std::size_t>(inst.nK));
    for (IloInt k = 0; k < inst.nK; ++k)
        order_k[static_cast<std::size_t>(k)] = k;

    std::sort(
        order_k.begin(),
        order_k.end(),
        [&](IloInt k1, IloInt k2) { return inst.r[k1] > inst.r[k2]; }
    );

    // Vetor de candidatos
    std::vector<SubsetRowCut> candidates;
    candidates.reserve(2 * static_cast<std::size_t>(inst.nK));

    // Gera candidatos para vários subconjuntos S (prefixos)
    IloNum R_S = 0.0;
    for (IloInt t = 0; t < inst.nK; ++t)
        {
            const IloInt k_new = order_k[static_cast<std::size_t>(t)];
            R_S += inst.r[k_new];
            if (R_S <= EPS)
                continue;

            // Corte em DEPÓSITOS (b_j)
            {
                IloNumArray cost_dep(env, inst.nJ);
                IloNum lhs_dep = 0.0;
                const IloNum rhs_dep = -1.0;

                for (IloInt j = 0; j < inst.nJ; ++j)
                    {
                        const IloNum alpha_j = IloMin(inst.q[j], R_S) / R_S;
                        cost_dep[j] = alpha_j;
                        if (alpha_j > EPS)
                            lhs_dep += -alpha_j * b[j]; // lhs = -∑ α_j b_j
                    }

                const IloNum overflow_dep = lhs_dep - rhs_dep; // = 1 - ∑ α_j b_j
                if (overflow_dep > EPS)
                    {
                        SubsetRowCut cut(SubsetRowCut::Family::DEPOT, cost_dep, rhs_dep);
                        cut.overflow = overflow_dep;
                        candidates.push_back(std::move(cut));
                    }
            }

            // Corte em PLANTAS (a_i)
            {
                IloNumArray cost_plant(env, inst.nI);
                IloNum lhs_plant = 0.0;
                const IloNum rhs_plant = -1.0;

                for (IloInt i = 0; i < inst.nI; ++i)
                    {
                        const IloNum beta_i = IloMin(inst.p[i], R_S) / R_S;
                        cost_plant[i] = beta_i;
                        if (beta_i > EPS)
                            lhs_plant += -beta_i * a[i]; // lhs = -∑ β_i a_i
                    }

                const IloNum overflow_plant = lhs_plant - rhs_plant; // = 1 - ∑ β_i a_i
                if (overflow_plant > EPS)
                    {
                        SubsetRowCut cut(SubsetRowCut::Family::PLANT, cost_plant, rhs_plant);
                        cut.overflow = overflow_plant;
                        candidates.push_back(std::move(cut));
                    }
            }
        }

    if (candidates.empty())
        return 0;

    // Ordena candidatos por violação decrescente
    std::sort(
        candidates.begin(),
        candidates.end(),
        [](const SubsetRowCut &c1, const SubsetRowCut &c2) { return c1.overflow > c2.overflow; }
    );

    // Insere até max_new_cuts cortes realmente novos
    IloInt new_cuts = 0;
    for (auto &cand : candidates)
        {
            if (new_cuts >= max_new_cuts)
                break;
            if (cuts.addSubsetRow(cand))
                ++new_cuts;
        }

    return new_cuts;
}
