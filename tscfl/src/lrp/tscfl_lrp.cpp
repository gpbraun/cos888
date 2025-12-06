// src/lrp/tscfl_lrp.cpp
#include "lrp/tscfl_lrp.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "lrp/tscfl_lrp_balances.hpp"
#include "lrp/tscfl_lrp_capacities.hpp"

// ---------------------------------------------------------------------
//  Construtor
// ---------------------------------------------------------------------

LRP::LRP(const TSCFLInstance &inst_)
    : env(inst_.env),
      inst(inst_),
      cuts(inst_),
      a(env, inst_.nI),
      b(env, inst_.nJ),
      x(env, inst_.nI, inst_.nJ),
      y(env, inst_.nJ, inst_.nK)
{
}

// ---------------------------------------------------------------------
//  Factory
// ---------------------------------------------------------------------

std::unique_ptr<LRP>
LRP::create(const TSCFLInstance &inst, Mode mode)
{
    switch (mode)
        {
        case Mode::CAPACITIES:
            return std::make_unique<LRPCapacity>(inst);
        case Mode::BALANCES:
            return std::make_unique<LRPBalance>(inst);
        default:
            throw std::invalid_argument("LRP::create: modo desconhecido.");
        }
}

// ---------------------------------------------------------------------
//  Separação gulosa de flow-covers (capacidades)
// ---------------------------------------------------------------------
//
// Planta i:
//   ∑_j x_ij ≤ p_i a_i, 0 ≤ x_ij ≤ q_j
//   Cover T ⊆ J com ∑_{j∈T} q_j > p_i
//   Corte usado:
//     lhs = -p_i a_i + ∑_{j∈T} x_ij
//     rhs = ∑_{j∉T} min(q_j, overflow), overflow = ∑_{j∈T} q_j - p_i
//
// Depósito j:
//   ∑_k y_jk ≤ q_j b_j, 0 ≤ y_jk ≤ r_k
//   Cover S ⊆ K com ∑_{k∈S} r_k > q_j
//   Corte:
//     lhs = -q_j b_j + ∑_{k∈S} y_jk
//     rhs = ∑_{k∉S} min(r_k, overflow), overflow = ∑_{k∈S} r_k - q_j
//
// Nesta versão, em vez de gerar apenas UM cover T/S por nó, geramos
// vários (prefixos sucessivos) e mantemos apenas os cortes mais
// violados, até max_new_cuts.
// ---------------------------------------------------------------------
IloInt
LRP::separate_flow_covers(IloInt max_new_cuts)
{
    const IloInt nI = inst.nI;
    const IloInt nJ = inst.nJ;
    const IloInt nK = inst.nK;

    if (max_new_cuts <= 0)
        return 0;

    // Máximo de covers que tentaremos gerar por planta/depósito
    static const IloInt MAX_COVERS_PER_NODE = 3;

    std::vector<FlowCoverCut> candidates;
    candidates.reserve(static_cast<std::size_t>(3 * (nI + nJ)));

    // -------------------------------------------------------------
    // 1a) Cortes de planta (flow-cover em ∑_j x_ij ≤ p_i a_i)
    // -------------------------------------------------------------
    for (IloInt i = 0; i < nI; ++i)
        {
            // índices j com x_ij > 0 e q_j > 0
            std::vector<IloInt> idx;
            idx.reserve(static_cast<std::size_t>(nJ));
            for (IloInt j = 0; j < nJ; ++j)
                {
                    if (x[i][j] > EPS && inst.q[j] > EPS)
                        idx.push_back(j);
                }
            if (idx.empty())
                continue;

            // ordena por capacidade q_j decrescente (poderia usar x_ij também)
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
                    IloNumArray cost(env, nJ);
                    for (IloInt j = 0; j < nJ; ++j)
                        cost[j] = 0.0;
                    for (std::size_t u = 0; u <= t; ++u)
                        {
                            const IloInt j = idx[u];
                            cost[j] = 1.0;
                        }

                    // rhs = ∑_{j∉T} min(q_j, overflow_cap)
                    IloNum rhs = 0.0;
                    for (IloInt j = 0; j < nJ; ++j)
                        {
                            if (cost[j] <= EPS)
                                rhs += IloMin(inst.q[j], overflow_cap);
                        }

                    const IloNum viol = lhs - rhs;
                    if (viol > EPS)
                        {
                            FlowCoverCut cut(
                                FlowCoverCut::NodeType::PLANT, static_cast<int>(i), cost, rhs
                            );
                            cut.overflow = viol; // violação do corte
                            candidates.push_back(std::move(cut));
                            ++covers_for_i;
                        }
                }
        }

    // -------------------------------------------------------------
    // 1b) Cortes de depósito (flow-cover em ∑_k y_jk ≤ q_j b_j)
    // -------------------------------------------------------------
    for (IloInt j = 0; j < nJ; ++j)
        {
            // índices k com y_jk > 0 e r_k > 0
            std::vector<IloInt> idx;
            idx.reserve(static_cast<std::size_t>(nK));
            for (IloInt k = 0; k < nK; ++k)
                {
                    if (y[j][k] > EPS && inst.r[k] > EPS)
                        idx.push_back(k);
                }
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
                    IloNumArray cost(env, nK);
                    for (IloInt k = 0; k < nK; ++k)
                        cost[k] = 0.0;
                    for (std::size_t u = 0; u <= t; ++u)
                        {
                            const IloInt k = idx[u];
                            cost[k] = 1.0;
                        }

                    // rhs = ∑_{k∉S} min(r_k, overflow_cap)
                    IloNum rhs = 0.0;
                    for (IloInt k = 0; k < nK; ++k)
                        {
                            if (cost[k] <= EPS)
                                rhs += IloMin(inst.r[k], overflow_cap);
                        }

                    const IloNum viol = lhs - rhs;
                    if (viol > EPS)
                        {
                            FlowCoverCut cut(
                                FlowCoverCut::NodeType::DEPOT, static_cast<int>(j), cost, rhs
                            );
                            cut.overflow = viol;
                            candidates.push_back(std::move(cut));
                            ++covers_for_j;
                        }
                }
        }

    if (candidates.empty())
        return 0;

    // -------------------------------------------------------------
    // 2) Ordena por violação (overflow) decrescente
    // -------------------------------------------------------------
    std::sort(
        candidates.begin(),
        candidates.end(),
        [](const FlowCoverCut &c1, const FlowCoverCut &c2) { return c1.overflow > c2.overflow; }
    );

    // -------------------------------------------------------------
    // 3) Insere até max_new_cuts cortes realmente novos
    // -------------------------------------------------------------
    IloInt new_cuts = 0;
    for (auto &cand : candidates)
        {
            if (new_cuts >= max_new_cuts)
                break;
            if (cuts.add_flow_cover(cand))
                ++new_cuts;
        }

    return new_cuts;
}

// ---------------------------------------------------------------------
//  Separação de subset-row (vários subconjuntos S ⊆ K)
// ---------------------------------------------------------------------
//
// Para cada subconjunto S (escolhido heurísticamente como prefixos de
// clientes ordenados por demanda), definimos:
//
//   R_S = sum_{k ∈ S} r_k
//
//   α_j^{(S)} = min(q_j, R_S) / R_S   (depósitos)
//   β_i^{(S)} = min(p_i, R_S) / R_S   (plantas)
//
// e obtemos os cortes normalizados:
//
//   ∑_j α_j^{(S)} b_j ≥ 1   ⇒ lhs = -∑_j α_j^{(S)} b_j, rhs = -1
//   ∑_i β_i^{(S)} a_i ≥ 1   ⇒ lhs = -∑_i β_i^{(S)} a_i, rhs = -1
//
// overflow = lhs - rhs = 1 - ∑ α_j^{(S)} b_j   ou   1 - ∑ β_i^{(S)} a_i
//
// Assim, overflow ∈ [0, 1], o que mantém a escala do subgradiente sob
// controle.
//
// A função abaixo gera vários candidatos (para diferentes S) e mantém
// apenas os mais violados, até um limite max_new_cuts.
// ---------------------------------------------------------------------
IloInt
LRP::separate_subset_rows(IloInt max_new_cuts)
{
    const IloInt nI = inst.nI;
    const IloInt nJ = inst.nJ;
    const IloInt nK = inst.nK;

    if (max_new_cuts <= 0)
        return 0;

    if (nK <= 0)
        return 0;

    // -----------------------------------------------------------------
    // 1) Ordena clientes por demanda r_k decrescente
    // -----------------------------------------------------------------
    std::vector<IloInt> order_k(static_cast<std::size_t>(nK));
    for (IloInt k = 0; k < nK; ++k)
        order_k[static_cast<std::size_t>(k)] = k;

    std::sort(
        order_k.begin(),
        order_k.end(),
        [&](IloInt k1, IloInt k2) { return inst.r[k1] > inst.r[k2]; }
    );

    // Vamos considerar apenas alguns prefixos {k_1,...,k_t} para limitar o custo.
    // Por exemplo, no máximo PREFIX_LIMIT prefixos.
    static const IloInt PREFIX_LIMIT = 20; // ajuste fino possível

    // Vetor de candidatos
    std::vector<SubsetRowCut> candidates;
    candidates.reserve(2 * static_cast<std::size_t>(IloMin(nK, PREFIX_LIMIT)));

    // -----------------------------------------------------------------
    // 2) Gera candidatos para vários subconjuntos S (prefixos)
    // -----------------------------------------------------------------
    IloNum R_S = 0.0;
    for (IloInt t = 0; t < nK && t < PREFIX_LIMIT; ++t)
        {
            const IloInt k_new = order_k[static_cast<std::size_t>(t)];
            R_S += inst.r[k_new];
            if (R_S <= EPS)
                continue;

            // -----------------------------
            // 2a) Corte em DEPÓSITOS (b_j)
            // -----------------------------
            {
                IloNumArray coeff_dep(env, nJ);
                IloNum lhs_dep = 0.0;
                const IloNum rhs_dep = -1.0;

                for (IloInt j = 0; j < nJ; ++j)
                    {
                        const IloNum alpha_j = IloMin(inst.q[j], R_S) / R_S;
                        coeff_dep[j] = alpha_j;
                        if (alpha_j > EPS)
                            lhs_dep += -alpha_j * b[j]; // lhs = -∑ α_j b_j
                    }

                const IloNum overflow_dep = lhs_dep - rhs_dep; // = 1 - ∑ α_j b_j
                if (overflow_dep > EPS)
                    {
                        SubsetRowCut cut(SubsetRowCut::Family::DEPOT, coeff_dep, rhs_dep);
                        cut.overflow = overflow_dep;
                        candidates.push_back(std::move(cut));
                    }
            }

            // -----------------------------
            // 2b) Corte em PLANTAS (a_i)
            // -----------------------------
            {
                IloNumArray coeff_plant(env, nI);
                IloNum lhs_plant = 0.0;
                const IloNum rhs_plant = -1.0;

                for (IloInt i = 0; i < nI; ++i)
                    {
                        const IloNum beta_i = IloMin(inst.p[i], R_S) / R_S;
                        coeff_plant[i] = beta_i;
                        if (beta_i > EPS)
                            lhs_plant += -beta_i * a[i]; // lhs = -∑ β_i a_i
                    }

                const IloNum overflow_plant = lhs_plant - rhs_plant; // = 1 - ∑ β_i a_i
                if (overflow_plant > EPS)
                    {
                        SubsetRowCut cut(SubsetRowCut::Family::PLANT, coeff_plant, rhs_plant);
                        cut.overflow = overflow_plant;
                        candidates.push_back(std::move(cut));
                    }
            }
        }

    if (candidates.empty())
        return 0;

    // -----------------------------------------------------------------
    // 3) Ordena candidatos por violação decrescente
    // -----------------------------------------------------------------
    std::sort(
        candidates.begin(),
        candidates.end(),
        [](const SubsetRowCut &c1, const SubsetRowCut &c2) { return c1.overflow > c2.overflow; }
    );

    // -----------------------------------------------------------------
    // 4) Tenta inserir até max_new_cuts cortes realmente novos
    // -----------------------------------------------------------------
    IloInt new_cuts = 0;
    for (auto &cand : candidates)
        {
            if (new_cuts >= max_new_cuts)
                break;
            if (cuts.add_subset_row(cand))
                ++new_cuts;
        }

    return new_cuts;
}