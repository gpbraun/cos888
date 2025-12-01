/*
COS888

Relax-and-Cut para o TSCFL
(Lagrangeano + subgradiente + heurística primal + cortes de fluxo).

- NÃO usa CPLEX para resolver subproblemas.
- Usa IloEnv apenas para IloNumArray / IloNumMatrix.
- Heurística de segundo estágio usando Worker (Dual / Primal / Net).

Gabriel Braun, 2025
*/

#pragma once

#include <ilcplex/ilocplex.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <memory>

#include "tscfl_instance.hpp"
#include "utils/tscfl_flowcuts.hpp"
#include "utils/tscfl_worker_dual.hpp"
#include "utils/tscfl_worker_primal.hpp"
#include "utils/tscfl_worker_net.hpp"

ILOSTLBEGIN

class TSCFLSolverRelaxAndCut
{
public:
    // Parâmetros do método
    static constexpr IloNum EPSILON0 = 2;               // epsilon inicial (Polyak)
    static constexpr IloInt MAX_NO_IMPROV = 50;         // iterações sem melhorar LB antes de reduzir epsilon
    static constexpr IloInt EXTRA_AGE = 10;             // vida extra de cortes em PA antes de ir pra CI
    static constexpr IloInt SOLVE_HEURISTIC_EVERY = 50; // frequência da heurística
    static constexpr IloInt MAX_NEW_CUTS_PER_ITER = 1;  // máx. novos cortes por iteração
    static constexpr IloInt PRINT_EVERY = 10;           // frequência do log

    IloEnv &env;
    const TSCFLInstance &inst;

    // Resultados globais
    IloNum lb{-IloInfinity};
    IloNum ub{IloInfinity};
    IloNum gap{IloInfinity};
    IloNum time{0.0};
    IloAlgorithm::Status status{IloAlgorithm::Unknown};

private:
    std::unique_ptr<Worker> worker; // Worker par o subproblema de fluxo mínimo
    FlowCoverCutSet cuts;           // Gerenciamento dos cortes de flow cover

    IloNum epsilon{EPSILON0}; // Parâmetro do subgradiente

    // Multiplicadores de Lagrange das capacidades
    IloNumArray l1; // l1[i] >= 0: planta i
    IloNumArray l2; // l2[j] >= 0: depósito j
    IloNumArray g1; // subgradiente em l1[i]
    IloNumArray g2; // subgradiente em l2[j]

    // Solução lagrangeana corrente
    IloNumArray a_lr;
    IloNumArray b_lr;
    IloNumMatrix x_lr;
    IloNumMatrix y_lr;

    // Melhor solução primal
    IloNumArray a_best;
    IloNumArray b_best;

    // Conjunto de cortes de flow cover

public:
    // mode (worker do subproblema de fluxo mínimo):
    // 0 -> WorkerDual
    // 1 -> WorkerPrimal
    // 2 -> WorkerNet (default)
    explicit TSCFLSolverRelaxAndCut(const TSCFLInstance &inst_, IloInt mode = 2)
        : env(inst_.env),
          inst(inst_),
          worker(nullptr),
          l1(env, inst_.nI),
          l2(env, inst_.nJ),
          g1(env, inst_.nI),
          g2(env, inst_.nJ),
          a_lr(env, inst_.nI),
          b_lr(env, inst_.nJ),
          x_lr(env, inst_.nI, inst_.nJ),
          y_lr(env, inst_.nJ, inst_.nK),
          a_best(env, inst_.nI),
          b_best(env, inst_.nJ),
          cuts(inst_)
    {
        switch (mode)
        {
        case 0:
            worker = std::make_unique<WorkerDual>(inst_);
            break;
        case 1:
            worker = std::make_unique<WorkerPrimal>(inst_);
            break;
        case 2:
            worker = std::make_unique<WorkerNet>(inst_);
            break;
        default:
            throw std::invalid_argument("Worker inválido (deve ser 0, 1, or 2).");
        }
    }

private:
    // Resolve o problema lagrangeano para (l1, l2, u_cuts) fixos
    // Retorna: z_LR
    IloNum solve_lagrangian()
    {
        const IloInt nI = inst.nI;
        const IloInt nJ = inst.nJ;
        const IloInt nK = inst.nK;

        fill_zero(x_lr);
        fill_zero(y_lr);
        cuts.update_costs();

        // Para cada cliente k: escolhe (i,j) de menor custo reduzido
        for (IloInt k = 0; k < nK; ++k)
        {
            IloNum rk = inst.r[k];
            if (rk <= EPS)
                continue;

            IloNum best_cost = IloInfinity;
            IloInt best_i = -1;
            IloInt best_j = -1;

            for (IloInt i = 0; i < nI; ++i)
                for (IloInt j = 0; j < nJ; ++j)
                {
                    IloNum cost = inst.c[i][j] + inst.d[j][k] + l1[i] + l2[j] +
                                  cuts.cost_x[i][j] + cuts.cost_y[j][k];

                    if (cost < best_cost)
                    {
                        best_cost = cost;
                        best_i = i;
                        best_j = j;
                    }
                }

            if (best_i == -1 || best_j == -1)
                continue;

            x_lr[best_i][best_j] += rk;
            y_lr[best_j][k] += rk;
        }

        // a_lr / b_lr (coeficiente reduzido < 0 => abre)
        for (IloInt i = 0; i < nI; ++i)
        {
            IloNum red_cost = inst.f[i] - l1[i] * inst.p[i] + cuts.cost_a[i];
            a_lr[i] = (red_cost < 0.0 ? 1.0 : 0.0);
        }
        for (IloInt j = 0; j < nJ; ++j)
        {
            IloNum red_cost = inst.g[j] - l2[j] * inst.q[j] + cuts.cost_b[j];
            b_lr[j] = (red_cost < 0.0 ? 1.0 : 0.0);
        }

        // Subgradientes
        for (IloInt i = 0; i < nI; ++i)
            g1[i] = IloSum(x_lr[i]) - inst.p[i] * a_lr[i];

        for (IloInt j = 0; j < nJ; ++j)
            g2[j] = IloSum(y_lr[j]) - inst.q[j] * b_lr[j];

        // Valor da lagrangeana
        IloNum z_lr = 0.0;

        // custo fixo
        z_lr += IloScalProd(inst.f, a_lr) + IloScalProd(inst.g, b_lr);
        // custo variável
        z_lr += IloMatScalProd(inst.c, x_lr) + IloMatScalProd(inst.d, y_lr);
        // termos lagrangeanos
        z_lr += IloScalProd(l1, g1) + IloScalProd(l2, g2);

        return z_lr;
    }

    // Separa Flow Covers a partir da solução LR
    void separate_flow_covers()
    {
        const IloInt nI = inst.nI;
        const IloInt nJ = inst.nJ;
        const IloInt nK = inst.nK;

        // 1) Gera candidatos de corte
        std::vector<FlowCoverCut> candidates;
        candidates.reserve(nI + nJ);

        // 1a) Cortes de planta
        for (IloInt i = 0; i < nI; ++i)
        {
            IloNumArray cost(env, nJ);

            // suporte do corte
            IloNum sum_q = 0.0;
            for (IloInt j = 0; j < nJ; ++j)
            {
                if (x_lr[i][j] > EPS)
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
            {
                if (cost[j] > EPS)
                    rhs += std::min(inst.q[j], overflow);
            }

            FlowCoverCut cut(FlowCoverCut::PLANT, i, cost, rhs);

            IloNum lhs = 0.0;
            for (IloInt j = 0; j < nJ; ++j)
            {
                if (cut.cost[j] > EPS)
                    lhs += cut.cost[j] * x_lr[i][j];
            }
            lhs += -inst.p[i] * a_lr[i];

            cut.overflow = lhs - cut.rhs;

            if (cut.overflow > EPS)
                candidates.push_back(std::move(cut));
        }

        // 1b) Cortes de depósito
        for (IloInt j = 0; j < nJ; ++j)
        {
            IloNumArray cost(env, nK);

            // suporte do corte
            IloNum sum_r = 0.0;
            for (IloInt k = 0; k < nK; ++k)
            {
                if (y_lr[j][k] > EPS)
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

            // rhs = ∑_{k∈S} min{ r_k , overflow }
            IloNum rhs = 0.0;
            for (IloInt k = 0; k < nK; ++k)
            {
                if (cost[k] > EPS)
                    rhs += std::min(inst.r[k], overflow);
            }

            FlowCoverCut cut(FlowCoverCut::DEPOT, j, cost, rhs);

            IloNum lhs = 0.0;
            for (IloInt k = 0; k < nK; ++k)
            {
                if (cut.cost[k] > EPS)
                    lhs += cut.cost[k] * y_lr[j][k];
            }
            lhs += -inst.q[j] * b_lr[j];

            cut.overflow = lhs - cut.rhs;

            if (cut.overflow > EPS)
                candidates.push_back(std::move(cut));
        }

        if (candidates.empty())
            return;

        // 2) Ordena candidatos por violação (decrescente)
        std::sort(
            candidates.begin(),
            candidates.end(),
            [](const FlowCoverCut &c1, const FlowCoverCut &c2)
            {
                return c1.overflow > c2.overflow;
            });

        // 3) Insere os mais violados no conjunto global de cortes
        IloInt new_cuts = 0;
        for (auto &cand : candidates)
        {
            if (new_cuts >= MAX_NEW_CUTS_PER_ITER)
                break;

            if (cuts.add(std::move(cand)))
                ++new_cuts;
        }
    }

    // Heurística primal (UB) usando Worker
    void solve_primal_heuristic()
    {
        IloNum demand_sum = IloSum(inst.r);
        IloNumArray a_h(env, inst.nI), b_h(env, inst.nJ);

        // 1) Determinação das plantas abertas (ordem guiada por a_lr e f/p)
        std::vector<IloInt> ordI(inst.nI);
        std::iota(ordI.begin(), ordI.end(), 0);
        std::sort(
            ordI.begin(), ordI.end(),
            [&](IloInt i, IloInt j)
            {
                if (std::fabs(a_lr[i] - a_lr[j]) > EPS)
                    return a_lr[i] > a_lr[j];

                IloNum ratio_i = inst.p[i] > EPS ? inst.f[i] / inst.p[i] : IloInfinity;
                IloNum ratio_j = inst.p[j] > EPS ? inst.f[j] / inst.p[j] : IloInfinity;
                return ratio_i < ratio_j;
            });

        IloNum capI = 0.0;
        for (IloInt pos = 0; pos < inst.nI && capI + EPS < demand_sum; ++pos)
        {
            IloInt i = ordI[pos];
            if (inst.p[i] <= EPS)
                continue;

            a_h[i] = 1.0;
            capI += inst.p[i];
        }

        // 2) Determinação dos depósitos abertos (ordem guiada por b_lr e g/q)
        std::vector<IloInt> ordJ(inst.nJ);
        std::iota(ordJ.begin(), ordJ.end(), 0);
        std::sort(
            ordJ.begin(), ordJ.end(),
            [&](IloInt j1, IloInt j2)
            {
                if (std::fabs(b_lr[j1] - b_lr[j2]) > EPS)
                    return b_lr[j1] > b_lr[j2];

                IloNum ratio1 = inst.q[j1] > EPS ? inst.g[j1] / inst.q[j1] : IloInfinity;
                IloNum ratio2 = inst.q[j2] > EPS ? inst.g[j2] / inst.q[j2] : IloInfinity;
                return ratio1 < ratio2;
            });

        IloNum capJ = 0.0;
        for (IloInt pos = 0; pos < inst.nJ && capJ + EPS < demand_sum; ++pos)
        {
            IloInt j = ordJ[pos];
            if (inst.q[j] <= EPS)
                continue;

            b_h[j] = 1.0;
            capJ += inst.q[j];
        }

        // 3) Resolução do subproblema de fluxo mínimo com Worker
        Worker &w = *worker;
        w.solve(a_h, b_h);

        IloNum ub_h = IloScalProd(inst.f, a_h) + IloScalProd(inst.g, b_h) + w.theta;
        if (ub_h + EPS < ub)
        {
            ub = ub_h;
            a_best = a_h;
            b_best = b_h;
        }
    }

public:
    // Método principal
    bool solve(bool log_output = true, IloNum time_limit = -1.0)
    {
        lb = -IloInfinity;
        ub = IloInfinity;
        gap = IloInfinity;
        status = IloAlgorithm::Unknown;
        time = 0.0;

        epsilon = EPSILON0;
        IloInt last_improv_iter = 0;
        IloNum best_lb = lb;

        cuts.clear();

        auto t0 = std::chrono::steady_clock::now();
        IloInt iter = 0;

        if (log_output)
        {
            std::cout << "[RC] Iniciando Relax-and-Cut\n";
            std::cout << "[RC] time_limit = " << time_limit
                      << ", epsilon0 = " << EPSILON0 << "\n";
            std::cout << "[RC] it   time(s)     z_LR        LB         UB"
                         "         gap    step    ||g||^2   |CA| |PA| |CI|\n";
        }

        while (true)
        {
            auto t1 = std::chrono::steady_clock::now();
            IloNum elapsed = std::chrono::duration<IloNum>(t1 - t0).count();
            if (time_limit > 0.0 && elapsed >= time_limit)
                break;

            // 1) Lagrangeano
            IloNum z_lr = solve_lagrangian();

            // 2) Cortes (separação + atualização dos conjuntos)
            separate_flow_covers();
            cuts.update_status(x_lr, y_lr, a_lr, b_lr, EXTRA_AGE);

            // 3) Atualiza LB
            if (z_lr > lb + EPS)
            {
                lb = z_lr;
                best_lb = lb;
                last_improv_iter = iter;
            }

            // 4) Heurística primal (UB)
            if (iter == 0 ||
                (iter % SOLVE_HEURISTIC_EVERY == 0) ||
                (z_lr > best_lb + EPS))
            {
                solve_primal_heuristic();
            }

            // 5) Norma do subgradiente
            IloNum norm2 = IloScalProd(g1, g1) + IloScalProd(g2, g2) + cuts.norm2();

            // 6) Passo de Polyak
            IloNum step = 0.0;
            if (ub < IloInfinity && norm2 > EPS)
                step = std::max(epsilon * (ub - z_lr) / norm2, 0.0);

            // 7) Atualiza multiplicadores (l1, l2, u)
            if (step > 0.0)
            {
                for (IloInt i = 0; i < inst.nI; ++i)
                    l1[i] = std::max(0.0, l1[i] + step * g1[i]);

                for (IloInt j = 0; j < inst.nJ; ++j)
                    l2[j] = std::max(0.0, l2[j] + step * g2[j]);

                cuts.update_multipliers(step);
            }

            // 8) Ajuste de epsilon (sem epsilon mínimo)
            if (iter - last_improv_iter >= MAX_NO_IMPROV)
            {
                epsilon *= 0.5;
                last_improv_iter = iter;
            }

            // 9) Gap atual
            if (ub < IloInfinity && lb > -IloInfinity)
            {
                gap = (ub - lb) / std::max(1.0, std::fabs(ub));
            }

            // 10) Log
            if (log_output && (iter % PRINT_EVERY == 0))
            {
                IloInt ca = cuts.count(FlowCoverCut::CA);
                IloInt pa = cuts.count(FlowCoverCut::PA);
                IloInt ci = cuts.count(FlowCoverCut::CI);

                IloNum elapsed_it =
                    std::chrono::duration<IloNum>(std::chrono::steady_clock::now() - t0).count();

                std::cout << "[RC] " << std::setw(4) << iter
                          << " " << std::setw(8) << std::fixed << std::setprecision(2) << elapsed_it
                          << " " << std::scientific << std::setprecision(6) << z_lr
                          << " " << std::scientific << std::setprecision(6) << lb
                          << " " << std::scientific << std::setprecision(6) << ub
                          << " " << std::fixed << std::setprecision(6) << gap
                          << " " << std::scientific << std::setprecision(6) << step
                          << " " << std::scientific << std::setprecision(6) << norm2
                          << " " << std::setw(4) << ca
                          << " " << std::setw(4) << pa
                          << " " << std::setw(4) << ci
                          << "\n";
            }

            // 11) Critério de parada por gap
            if (gap <= MIP_GAP && ub < IloInfinity && lb > -IloInfinity)
            {
                status = IloAlgorithm::Optimal;
                break;
            }

            ++iter;
        }

        auto t_end = std::chrono::steady_clock::now();
        time = std::chrono::duration<IloNum>(t_end - t0).count();

        if (status == IloAlgorithm::Unknown)
        {
            if (ub < IloInfinity && lb > -IloInfinity)
                status = IloAlgorithm::Feasible;
        }

        if (log_output)
        {
            std::cout << "\n[RC] Relax-and-Cut finalizado.\n";
            std::cout << "LB    = " << lb << "\n";
            std::cout << "UB    = " << ub << "\n";
            std::cout << "gap   = " << gap << "\n";
            std::cout << "status= " << status << "\n";
            std::cout << "time  = " << time << " s\n";
            if (status == IloAlgorithm::Feasible)
                std::cout << "[RC] Parada por time_limit.\n";
        }

        return (status == IloAlgorithm::Optimal);
    }
};
