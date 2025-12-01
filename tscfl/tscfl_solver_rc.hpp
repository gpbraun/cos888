/*
COS888

Relax-and-Cut para o TSCFL
(Lagrangeano + subgradiente + heurística primal + cortes de fluxo).

- NÃO usa CPLEX para resolver subproblemas.
- Usa IloEnv apenas para IloNumArray / IloNumMatrix.
- Heurística de segundo estágio usando Subproblem (Dual / Primal / Net).
- A lógica da relaxação Lagrangeana fica em LagrangianRelaxation
  (aqui usamos LagrangianRelaxationCapacity).

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
#include <numeric> // std::iota

#include "tscfl_instance.hpp"
#include "lagrangian/tscfl_flowcuts.hpp"
#include "lagrangian/tscfl_lagrangian_capacity.hpp"
#include "subproblem/tscfl_subproblem_dual.hpp"
#include "subproblem/tscfl_subproblem_primal.hpp"
#include "subproblem/tscfl_subproblem_net.hpp"

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
    // Subproblema de fluxo mínimo (para heurística primal)
    std::unique_ptr<Subproblem> subproblem;

    // Relaxação Lagrangeana (atualmente: capacidade)
    std::unique_ptr<LagrangianRelaxation> lagr;

    // Parâmetro do subgradiente
    IloNum epsilon{EPSILON0};

    // Melhor solução primal encontrada
    IloNumArray a_best;
    IloNumArray b_best;

public:
    // mode (tipo de subproblema de fluxo mínimo):
    // 0 -> SubproblemDual
    // 1 -> SubproblemPrimal
    // 2 -> SubproblemNet (default)
    explicit TSCFLSolverRelaxAndCut(const TSCFLInstance &inst_, IloInt mode = 2)
        : env(inst_.env),
          inst(inst_),
          subproblem(nullptr),
          lagr(nullptr),
          a_best(env, inst_.nI),
          b_best(env, inst_.nJ)
    {
        // Escolha do subproblema (dual / primal / net)
        switch (mode)
        {
        case 0:
            subproblem = std::make_unique<SubproblemDual>(inst_);
            break;
        case 1:
            subproblem = std::make_unique<SubproblemPrimal>(inst_);
            break;
        case 2:
            subproblem = std::make_unique<SubproblemNet>(inst_);
            break;
        default:
            throw std::invalid_argument("Subproblem inválido (deve ser 0, 1, or 2).");
        }

        // Por enquanto, sempre usamos a relaxação por capacidade
        lagr = std::make_unique<LagrangianRelaxationCapacity>(inst_);

        fill_zero(a_best);
        fill_zero(b_best);
    }

private:
    // -----------------------------------------------------------------
    // Heurística primal (UB) usando o subproblema de fluxo mínimo
    // -----------------------------------------------------------------
    void solve_primal_heuristic()
    {
        auto &LR = *lagr;
        IloNum demand_sum = IloSum(inst.r);

        IloNumArray a_h(env, inst.nI);
        IloNumArray b_h(env, inst.nJ);
        fill_zero(a_h);
        fill_zero(b_h);

        // 1) Determinação das plantas abertas (ordem guiada por a e f/p)
        std::vector<IloInt> ordI(inst.nI);
        std::iota(ordI.begin(), ordI.end(), 0);

        std::sort(
            ordI.begin(), ordI.end(),
            [&](IloInt i, IloInt j)
            {
                if (std::fabs(LR.a[i] - LR.a[j]) > EPS)
                    return LR.a[i] > LR.a[j];

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

        // 2) Determinação dos depósitos abertos (ordem guiada por b e g/q)
        std::vector<IloInt> ordJ(inst.nJ);
        std::iota(ordJ.begin(), ordJ.end(), 0);

        std::sort(
            ordJ.begin(), ordJ.end(),
            [&](IloInt j1, IloInt j2)
            {
                if (std::fabs(LR.b[j1] - LR.b[j2]) > EPS)
                    return LR.b[j1] > LR.b[j2];

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

        // 3) Resolução do subproblema de fluxo mínimo
        Subproblem &sp = *subproblem;

        try
        {
            sp.solve(a_h, b_h);
        }
        catch (...)
        {
            // se subproblema falhar, não atualiza UB
            return;
        }

        IloNum ub_h = IloScalProd(inst.f, a_h) + IloScalProd(inst.g, b_h) + sp.theta;
        if (ub_h + EPS < ub)
        {
            ub = ub_h;
            a_best = a_h;
            b_best = b_h;
        }
    }

public:
    // -----------------------------------------------------------------
    // Método principal
    // -----------------------------------------------------------------
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

        auto &LR = *lagr;
        auto &cuts = LR.getCuts();

        while (true)
        {
            auto t1 = std::chrono::steady_clock::now();
            IloNum elapsed = std::chrono::duration<IloNum>(t1 - t0).count();
            if (time_limit > 0.0 && elapsed >= time_limit)
                break;

            // 1) Resolve subproblema Lagrangeano
            IloNum z_lr = lagr->solve();

            // 2) Cortes (separação + atualização dos conjuntos)
            lagr->separate_flow_covers(MAX_NEW_CUTS_PER_ITER);
            cuts.update_status(LR.x, LR.y, LR.a, LR.b, EXTRA_AGE);

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

            // 5) Norma do subgradiente (capacidades + cortes)
            IloNum norm2 = lagr->norm2sq();

            // 6) Passo de Polyak
            IloNum step = 0.0;
            if (ub < IloInfinity && norm2 > EPS)
                step = std::max(epsilon * (ub - z_lr) / norm2, 0.0);

            // 7) Atualiza multiplicadores (capacidades + cortes)
            if (step > 0.0)
                lagr->update_multipliers(step);

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
