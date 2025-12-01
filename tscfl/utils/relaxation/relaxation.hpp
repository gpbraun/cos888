/*
COS888

Base abstrata para relaxações lagrangeanas do TSCFL.

Gabriel Braun, 2025
*/

#pragma once

#include "utils/instance/instance.hpp"
#include "utils/relaxation/flowcuts.hpp"

// SOLVER DA RELAXAÇÃO LAGRANGEANA: Capacidades relaxadas
class Relaxation
{
protected:
    IloEnv &env;
    const TSCFLInstance &inst;

    FlowCoverCutSet cuts;

public:
    enum class Mode
    {
        CAPACITIES,
        BALANCES,
    };

    // Solução Lagrangeana corrente
    IloNumArray a;  // a[i]    = abre planta i (relaxação linear)
    IloNumArray b;  // b[j]    = abre depósito j (relaxação linear)
    IloNumMatrix x; // x[i][j] = fluxo planta i -> depósito j
    IloNumMatrix y; // y[j][k] = fluxo depósito j -> cliente k

    explicit Relaxation(const TSCFLInstance &inst_)
        : env(inst_.env),
          inst(inst_),
          cuts(inst_),
          a(env, inst_.nI),
          b(env, inst_.nJ),
          x(env, inst_.nI, inst_.nJ),
          y(env, inst_.nJ, inst_.nK)
    {
    }

    static std::unique_ptr<Relaxation> create(const TSCFLInstance &inst, Mode mode);

    virtual ~Relaxation() = default;

    // Resolve o subproblema Lagrangeano para os multiplicadores atuais e atualiza (a, b, x, y).
    // Retorna: z_LR.
    virtual IloNum solve() = 0;

    // Retorna: ||g||^2 (subgradientes + contribuição dos cortes)
    virtual IloNum norm2sq() const = 0;

    // Atualiza multiplicadores (l1,l2 ou m1,m2) e também multiplicadores dos cortes.
    virtual void update_multipliers(IloNum step) = 0;

    // Acesso ao conjunto de cortes (para separação, logs, etc.)
    FlowCoverCutSet &getCuts() { return cuts; }
    const FlowCoverCutSet &getCuts() const { return cuts; }

    // Separa Flow Covers a partir da solução LR atual
    IloInt separate_flow_covers(IloInt max_new_cuts)
    {
        const IloInt nI = inst.nI;
        const IloInt nJ = inst.nJ;
        const IloInt nK = inst.nK;

        std::vector<FlowCoverCut> candidates;
        candidates.reserve(nI + nJ);

        // 1a) Cortes de planta
        for (IloInt i = 0; i < nI; ++i)
        {
            IloNumArray cost(env, nJ);
            IloNum sum_q = 0.0;

            for (IloInt j = 0; j < nJ; ++j)
            {
                if (x[i][j] > EPS)
                {
                    cost[j] = 1.0;
                    sum_q += inst.q[j];
                }
                else
                {
                    cost[j] = 0.0;
                }
            }

            IloNum overflow = sum_q - inst.p[i];
            if (overflow <= EPS)
                continue;

            IloNum rhs = 0.0;
            for (IloInt j = 0; j < nJ; ++j)
                if (cost[j] > EPS)
                    rhs += std::min(inst.q[j], overflow);

            FlowCoverCut cut(FlowCoverCut::PLANT, i, cost, rhs);

            IloNum lhs = 0.0;
            for (IloInt j = 0; j < nJ; ++j)
                if (cut.cost[j] > EPS)
                    lhs += cut.cost[j] * x[i][j];

            lhs += -inst.p[i] * a[i];

            cut.overflow = lhs - cut.rhs;

            if (cut.overflow > EPS)
                candidates.push_back(std::move(cut));
        }

        // 1b) Cortes de depósito
        for (IloInt j = 0; j < nJ; ++j)
        {
            IloNumArray cost(env, nK);
            IloNum sum_r = 0.0;

            for (IloInt k = 0; k < nK; ++k)
            {
                if (y[j][k] > EPS)
                {
                    cost[k] = 1.0;
                    sum_r += inst.r[k];
                }
                else
                {
                    cost[k] = 0.0;
                }
            }

            IloNum overflow = sum_r - inst.q[j];
            if (overflow <= EPS)
                continue;

            IloNum rhs = 0.0;
            for (IloInt k = 0; k < nK; ++k)
                if (cost[k] > EPS)
                    rhs += std::min(inst.r[k], overflow);

            FlowCoverCut cut(FlowCoverCut::DEPOT, j, cost, rhs);

            IloNum lhs = 0.0;
            for (IloInt k = 0; k < nK; ++k)
                if (cut.cost[k] > EPS)
                    lhs += cut.cost[k] * y[j][k];

            lhs += -inst.q[j] * b[j];

            cut.overflow = lhs - cut.rhs;

            if (cut.overflow > EPS)
                candidates.push_back(std::move(cut));
        }

        if (candidates.empty())
            return 0;

        std::sort(
            candidates.begin(),
            candidates.end(),
            [](const FlowCoverCut &c1, const FlowCoverCut &c2)
            {
                return c1.overflow > c2.overflow;
            });

        IloInt new_cuts = 0;
        for (auto &cand : candidates)
        {
            if (new_cuts >= max_new_cuts)
                break;
            if (cuts.add(std::move(cand)))
                ++new_cuts;
        }
        return new_cuts;
    }
};
