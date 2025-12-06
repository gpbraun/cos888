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

LRP::LRP(const TSCFLInstance& inst_)
    : env(inst_.env),
      inst(inst_),
      cuts(inst_),
      a(env, inst_.nI),
      b(env, inst_.nJ),
      x(env, inst_.nI, inst_.nJ),
      y(env, inst_.nJ, inst_.nK) {}

// ---------------------------------------------------------------------
//  Factory
// ---------------------------------------------------------------------

std::unique_ptr<LRP> LRP::create(const TSCFLInstance& inst, Mode mode) {
    switch (mode) {
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
//   lhs = -p_i a_i + sum_{j in T} x_ij
//   rhs = sum_{j notin T} min(q_j, overflow)
// Depósito j:
//   lhs = -q_j b_j + sum_{k in S} y_jk
//   rhs = sum_{k notin S} min(r_k, overflow)
//
IloInt LRP::separate_flow_covers(IloInt max_new_cuts) {
    const IloInt nI = inst.nI;
    const IloInt nJ = inst.nJ;
    const IloInt nK = inst.nK;

    if (max_new_cuts <= 0) return 0;

    std::vector<FlowCoverCut> candidates;
    candidates.reserve(static_cast<std::size_t>(nI + nJ));

    // -------------------------------------------------------------
    // 1a) Cortes de planta
    // -------------------------------------------------------------
    for (IloInt i = 0; i < nI; ++i) {
        IloNumArray cost(env, nJ);
        IloNum sum_q = 0.0;

        for (IloInt j = 0; j < nJ; ++j) {
            if (x[i][j] > EPS) {
                cost[j] = 1.0;
                sum_q += inst.q[j];
            } else {
                cost[j] = 0.0;
            }
        }

        IloNum overflow = sum_q - inst.p[i];
        if (overflow <= EPS) continue;

        // rhs = ∑_{j : cost[j] = 1} min(q_j, overflow)
        IloNum rhs = 0.0;
        for (IloInt j = 0; j < nJ; ++j) {
            if (cost[j] > EPS) rhs += IloMin(inst.q[j], overflow);
        }

        FlowCoverCut cut(FlowCoverCut::NodeType::PLANT, static_cast<int>(i), cost, rhs);

        IloNum lhs = cut.compute_lhs(inst, x, y, a, b);
        cut.overflow = lhs - cut.rhs;

        if (cut.overflow > EPS) candidates.push_back(std::move(cut));
    }

    // -------------------------------------------------------------
    // 1b) Cortes de depósito
    // -------------------------------------------------------------
    for (IloInt j = 0; j < nJ; ++j) {
        IloNumArray cost(env, nK);
        IloNum sum_r = 0.0;

        for (IloInt k = 0; k < nK; ++k) {
            if (y[j][k] > EPS) {
                cost[k] = 1.0;
                sum_r += inst.r[k];
            } else {
                cost[k] = 0.0;
            }
        }

        IloNum overflow = sum_r - inst.q[j];
        if (overflow <= EPS) continue;

        // rhs = ∑_{k : cost[k] = 1} min(r_k, overflow)
        IloNum rhs = 0.0;
        for (IloInt k = 0; k < nK; ++k) {
            if (cost[k] > EPS) rhs += IloMin(inst.r[k], overflow);
        }

        FlowCoverCut cut(FlowCoverCut::NodeType::DEPOT, static_cast<int>(j), cost, rhs);

        IloNum lhs = cut.compute_lhs(inst, x, y, a, b);
        cut.overflow = lhs - cut.rhs;

        if (cut.overflow > EPS) candidates.push_back(std::move(cut));
    }

    if (candidates.empty()) return 0;

    // -------------------------------------------------------------
    // 2) Ordena por violação (overflow) decrescente
    // -------------------------------------------------------------
    std::sort(
        candidates.begin(), candidates.end(),
        [](const FlowCoverCut& c1, const FlowCoverCut& c2) { return c1.overflow > c2.overflow; });

    // -------------------------------------------------------------
    // 3) Insere até max_new_cuts cortes realmente novos
    // -------------------------------------------------------------
    IloInt new_cuts = 0;
    for (auto& cand : candidates) {
        if (new_cuts >= max_new_cuts) break;
        if (cuts.add_flow_cover(cand)) ++new_cuts;
    }

    return new_cuts;
}

// ---------------------------------------------------------------------
//  Separação de subset-row (demand cuts globais S = K)
// ---------------------------------------------------------------------
//
// Demanda total R = ∑_k r_k.
//
// Depósitos (família DEPOT):
//   ∑_j α_j b_j ≥ R  com  α_j = min(q_j, R)
//   ⇒  lhs = -∑_j α_j b_j, rhs = -R
//
// Plantas (família PLANT):
//   ∑_i β_i a_i ≥ R  com  β_i = min(p_i, R)
//   ⇒  lhs = -∑_i β_i a_i, rhs = -R
//
// cut.overflow = lhs - rhs = R - ∑ α_j b_j  (ou R - ∑ β_i a_i)
// Violado se overflow > EPS.
// ---------------------------------------------------------------------
IloInt LRP::separate_subset_rows(IloInt max_new_cuts) {
    const IloInt nI = inst.nI;
    const IloInt nJ = inst.nJ;
    const IloInt nK = inst.nK;

    if (max_new_cuts <= 0) return 0;

    // Demanda total R = ∑_k r_k
    IloNum R = 0.0;
    for (IloInt k = 0; k < nK; ++k) R += inst.r[k];

    if (R <= EPS) return 0;

    IloInt new_cuts = 0;

    // -------------------------------------------------------------
    // 1) Corte subset-row em depósitos (b_j)
    //     ∑_j α_j b_j ≥ R, α_j = min(q_j, R)
    // -------------------------------------------------------------
    {
        IloNumArray coeff(env, nJ);
        IloNum lhs = 0.0;

        for (IloInt j = 0; j < nJ; ++j) {
            coeff[j] = IloMin(inst.q[j], R);
            lhs += -coeff[j] * b[j];  // lhs = -∑_j α_j b_j
        }

        IloNum rhs = -R;
        IloNum overflow = lhs - rhs;  // = R - ∑_j α_j b_j

        if (overflow > EPS) {
            SubsetRowCut cut(SubsetRowCut::Family::DEPOT, coeff, rhs);
            cut.overflow = overflow;

            if (cuts.add_subset_row(cut)) {
                ++new_cuts;
                if (new_cuts >= max_new_cuts) return new_cuts;
            }
        }
    }

    // -------------------------------------------------------------
    // 2) Corte subset-row em plantas (a_i)
    //     ∑_i β_i a_i ≥ R, β_i = min(p_i, R)
    // -------------------------------------------------------------
    {
        IloNumArray coeff(env, nI);
        IloNum lhs = 0.0;

        for (IloInt i = 0; i < nI; ++i) {
            coeff[i] = IloMin(inst.p[i], R);
            lhs += -coeff[i] * a[i];  // lhs = -∑_i β_i a_i
        }

        IloNum rhs = -R;
        IloNum overflow = lhs - rhs;  // = R - ∑_i β_i a_i

        if (overflow > EPS) {
            SubsetRowCut cut(SubsetRowCut::Family::PLANT, coeff, rhs);
            cut.overflow = overflow;

            if (cuts.add_subset_row(cut)) {
                ++new_cuts;
                if (new_cuts >= max_new_cuts) return new_cuts;
            }
        }
    }

    return new_cuts;
}