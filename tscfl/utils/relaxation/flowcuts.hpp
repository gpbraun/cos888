/*
COS888

FlowCoverCut e FlowCoverCutSet para o TSCFL.

Gabriel Braun, 2025
*/

#pragma once

#include "utils/instance/instance.hpp"

class FlowCoverCut
{
public:
    enum Type
    {
        PLANT,
        DEPOT
    };
    enum Status
    {
        CA,
        PA,
        CI
    };

    Type type;
    int index;        // i (planta) ou j (depósito)
    IloNumArray cost; // coeficientes dos fluxos x[i] (planta) ou y[j] (depósito)
    IloNum rhs;
    std::size_t hash;

    Status status{CA};
    IloInt age{0};
    IloNum u{0.0};
    IloNum overflow{0.0};

    FlowCoverCut(Type t, int idx, const IloNumArray &cost_, IloNum rhs_)
        : type(t),
          index(idx),
          cost(cost_),
          rhs(rhs_),
          hash(compute_hash(t, idx, cost_))
    {
    }

private:
    static std::size_t compute_hash(Type t, int idx, const IloNumArray &cost_)
    {
        std::string key;
        key.reserve(32 + 4 * static_cast<std::size_t>(cost_.getSize()));

        key.push_back(t == PLANT ? 'P' : 'D');
        key.push_back(':');
        key += std::to_string(idx);
        key.push_back('|');

        for (IloInt j = 0; j < cost_.getSize(); ++j)
        {
            if (std::fabs(cost_[j]) > EPS)
            {
                key += std::to_string(j);
                key.push_back('#');
            }
        }
        return std::hash<std::string>{}(key);
    }
};

class FlowCoverCutSet
{
public:
    IloEnv &env;
    const TSCFLInstance &inst;

    // custos/termos agregados dos cortes
    IloNumArray cost_a;
    IloNumArray cost_b;
    IloNumMatrix cost_x;
    IloNumMatrix cost_y;

private:
    std::vector<FlowCoverCut> cuts;
    std::unordered_set<std::size_t> hashes;

public:
    FlowCoverCutSet(const TSCFLInstance &inst_)
        : env(inst_.env),
          inst(inst_),
          cost_a(env, inst_.nI),
          cost_b(env, inst_.nJ),
          cost_x(env, inst_.nI, inst_.nJ),
          cost_y(env, inst_.nJ, inst_.nK)
    {
    }

    void clear()
    {
        cuts.clear();
        hashes.clear();
    }

    std::vector<FlowCoverCut> &data() { return cuts; }
    const std::vector<FlowCoverCut> &data() const { return cuts; }

    // Insere corte se ainda não existir (baseado no hash).
    // Retorna: true se o corte foi de fato adicionado.
    bool add(FlowCoverCut &&cut)
    {
        auto res = hashes.insert(cut.hash);
        if (!res.second)
            return false; // corte duplicado

        cuts.push_back(std::move(cut));
        return true;
    }

    // Retorna: número de cortes que estão em cada status
    int count(FlowCoverCut::Status s) const
    {
        int cnt = 0;
        for (const auto &c : cuts)
            if (c.status == s)
                ++cnt;
        return cnt;
    }

    // Returna: contribuição dos cortes para ||g||^2.
    double norm2sq() const
    {
        double s = 0.0;
        for (const auto &c : cuts)
        {
            if (c.status != FlowCoverCut::CI)
                s += c.overflow * c.overflow;
        }
        return s;
    }

    // Atualiza: multiplicadores de Lagrange dos cortes.
    void update_multipliers(double step)
    {
        for (auto &c : cuts)
        {
            if (c.status != FlowCoverCut::CI)
                c.u = std::max(0.0, c.u + step * c.overflow);
        }
    }

    // Atualiza: custos adicionais.
    void update_costs()
    {
        fill_zero(cost_x);
        fill_zero(cost_y);
        fill_zero(cost_a);
        fill_zero(cost_b);

        for (const auto &cut : cuts)
        {
            if (cut.status == FlowCoverCut::CI || cut.u <= EPS)
                continue;

            if (cut.type == FlowCoverCut::PLANT)
            {
                int i = cut.index;
                for (int j = 0; j < inst.nJ && j < cut.cost.getSize(); ++j)
                    if (std::fabs(cut.cost[j]) > EPS)
                        cost_x[i][j] += cut.u * cut.cost[j];

                cost_a[i] += -cut.u * inst.p[i];
            }
            else // DEPOT
            {
                int j = cut.index;
                for (int k = 0; k < inst.nK && k < cut.cost.getSize(); ++k)
                    if (std::fabs(cut.cost[k]) > EPS)
                        cost_y[j][k] += cut.u * cut.cost[k];

                cost_b[j] += -cut.u * inst.q[j];
            }
        }
    }

    // Atualiza: violações e conjuntos CA/PA/CI.
    void update_status(
        const IloNumMatrix &x_lr,
        const IloNumMatrix &y_lr,
        const IloNumArray &a_lr,
        const IloNumArray &b_lr,
        int extra_age)
    {
        for (auto &cut : cuts)
        {
            double lhs = 0.0;

            if (cut.type == FlowCoverCut::PLANT)
            {
                int i = cut.index;
                lhs += -inst.p[i] * a_lr[i];
                for (int j = 0; j < inst.nJ && j < cut.cost.getSize(); ++j)
                    lhs += cut.cost[j] * x_lr[i][j];
            }
            else // DEPOT
            {
                int j = cut.index;
                lhs += -inst.q[j] * b_lr[j];
                for (int k = 0; k < inst.nK && k < cut.cost.getSize(); ++k)
                    lhs += cut.cost[k] * y_lr[j][k];
            }

            cut.overflow = lhs - cut.rhs;

            if (cut.overflow > EPS)
            {
                cut.status = FlowCoverCut::CA;
                cut.age = 0;
            }
            else
            {
                ++cut.age;
                if (cut.age <= extra_age && cut.u > EPS)
                {
                    cut.status = FlowCoverCut::PA;
                }
                else
                {
                    cut.status = FlowCoverCut::CI;
                    cut.u = 0.0;
                }
            }
        }
    }
};
