/*
COS888

Resolve o TSCFL por Non-Delayed Relax-and-Cut.

Gabriel Braun, 2025
*/

#pragma once

#include "utils/utils.hpp"

class TSCFLSolverSubgradient
{
protected:
    IloEnv &env;
    const TSCFLInstance &inst;

public:
    // Parâmetros do método
    static constexpr IloNum EPSILON0 = 2;
    static constexpr IloInt MAX_NO_IMPROV = 50;
    static constexpr IloInt EXTRA_AGE = 5;
    static constexpr IloInt SOLVE_HEURISTIC_EVERY = 10;
    static constexpr IloInt MAX_NEW_CUTS_PER_ITER = 1;
    static constexpr IloInt PRINT_EVERY = 10;

    // Resultados globais
    IloNum lb{0.0};
    IloNum ub{IloInfinity};
    IloNum gap{IloInfinity};
    IloNum time{0.0};
    IloInt64 iter{0};
    IloAlgorithm::Status status{IloAlgorithm::Unknown};

    // Melhor solução primal encontrada
    IloNumArray a;
    IloNumArray b;

private:
    std::unique_ptr<Relaxation> relaxation;
    std::unique_ptr<Subproblem> subproblem;

public:
    explicit TSCFLSolverSubgradient(
        const TSCFLInstance &inst_,
        Relaxation::Mode rmode = Relaxation::Mode::CAPACITIES,
        Subproblem::Mode smode = Subproblem::Mode::NET)
        : env(inst_.env),
          inst(inst_),
          relaxation(Relaxation::create(inst_, rmode)),
          subproblem(Subproblem::create(inst_, smode)),
          a(env, inst_.nI),
          b(env, inst_.nJ)
    {
    }

public:
    // Método principal
    bool solve(bool log_output = true, IloNum time_limit = -1.0)
    {
        IloInt last_improv_iter = 0;
        IloNum epsilon = EPSILON0;

        auto t0 = std::chrono::steady_clock::now();

        if (log_output)
        {
            std::cout << "[RC] Iniciando Relax-and-Cut\n";
            std::cout << "[RC] time_limit = " << time_limit
                      << ", epsilon0 = " << EPSILON0 << "\n";
            std::cout << "[RC] it   time(s)     z_LR        LB         UB"
                         "         gap    step    ||g||^2   |CA| |PA| |CI|\n";
        }

        auto &SP = *subproblem;
        auto &LR = *relaxation;
        auto &cuts = LR.getCuts();
        IloNumArray a_h(env, inst.nI), b_h(env, inst.nJ);

        while (true)
        {
            auto t1 = std::chrono::steady_clock::now();
            IloNum elapsed = std::chrono::duration<IloNum>(t1 - t0).count();
            if (time_limit > 0.0 && elapsed >= time_limit)
                break;

            // 1) Resolve subproblema Lagrangeano
            IloNum z_lr = LR.solve();

            if (z_lr > lb + EPS)
            {
                lb = z_lr;
                last_improv_iter = iter;
            }

            // 2) Gerencia os cortes
            LR.separate_flow_covers(MAX_NEW_CUTS_PER_ITER);
            cuts.update_status(LR.x, LR.y, LR.a, LR.b, EXTRA_AGE);

            // 3) Resolve a heurística primal periodicamente
            if (iter == 0 || (last_improv_iter == iter) || (iter % SOLVE_HEURISTIC_EVERY == 0))
            {
                IloNum z_h = SP.solve_primal_heuristic(LR.a, LR.b, a_h, b_h);
                if (z_h + EPS < ub)
                {
                    ub = z_h;
                    a = a_h;
                    b = b_h;
                }
            }

            if (ub < IloInfinity && lb > -IloInfinity)
                gap = (ub - lb) / std::max(1.0, std::fabs(ub));

            // 4) Atualiza o passo
            IloNum norm2 = LR.norm2sq();

            IloNum step = 0.0;
            if (norm2 > EPS)
                step = std::max(epsilon * (ub - z_lr) / norm2, 0.0);

            // 5) Atualiza os multiplicadores
            LR.update_multipliers(step);

            // 6) Atualiza o epsilon se necessário
            if (iter - last_improv_iter >= MAX_NO_IMPROV)
            {
                epsilon *= 0.5;
                last_improv_iter = iter;
            }

            // 7) Log
            if (log_output && (iter % PRINT_EVERY == 0))
            {
                IloInt nCA = cuts.count(FlowCoverCut::CA);
                IloInt nPA = cuts.count(FlowCoverCut::PA);
                IloInt nCI = cuts.count(FlowCoverCut::CI);

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
                          << " " << std::setw(4) << nCA
                          << " " << std::setw(4) << nPA
                          << " " << std::setw(4) << nCI
                          << "\n";
            }

            // 8) Critério de parada por gap
            if (gap <= MIP_GAP && ub < IloInfinity && lb > 0.0)
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
            if (ub < IloInfinity && lb > 0.0)
                status = IloAlgorithm::Feasible;
        }

        if (log_output)
        {
            std::cout << "\n[RC] Relax-and-Cut finalizado.\n";
            std::cout << "LB     = " << lb << "\n";
            std::cout << "UB     = " << ub << "\n";
            std::cout << "status = " << status << "\n";
            std::cout << "gap    = " << gap << "\n";
            std::cout << "iter   = " << iter << "\n";
            std::cout << "time   = " << time << " s\n";
            if (status == IloAlgorithm::Feasible)
                std::cout << "[RC] Parada por time_limit.\n";
        }

        return (status == IloAlgorithm::Optimal);
    }
};
