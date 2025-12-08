/*
COS888

tscfl_solver_subgradient.cpp

Gabriel Braun, 2025
*/

#include "solvers/tscfl_solver_subgradient.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>

TSCFLSolverSubgradient::TSCFLSolverSubgradient(
    const TSCFLInstance &inst_, LRP::Mode lr_mode, Subproblem::Mode sp_mode
)
    : TSCFLSolver(inst_),
      relaxation(LRP::create(inst_, lr_mode)),
      subproblem(Subproblem::create(inst_, sp_mode))
{
}

void
TSCFLSolverSubgradient::solve(bool log_output, IloNum time_limit)
{
    auto &SP = *subproblem;
    auto &LR = *relaxation;
    auto &cuts = LR.getCuts();

    IloInt last_improv_iter = 0;
    IloInt last_eps_update_iter = 0;

    IloNum epsilon = EPSILON0;

    // Log inicial
    if (log_output)
        {
            // clang-format off
            std::cout << "\n\n[SG] Iniciando Subgradiente\n\n"
                      << std::right
                      << std::setw(5)  << "iter"
                      << std::setw(10) << "n.imprv."
                      << std::setw(10) << "time(s)"
                      << std::setw(15) << "opt_LR"
                      << std::setw(15) << "LB"
                      << std::setw(15) << "UB"
                      << std::setw(12) << "gap"
                      << std::setw(12) << "step"
                      << std::setw(12) << "||g||^2"
                      << std::setw(10) << "|CA|"
                      << std::setw(10) << "|PA|"
                      << std::setw(10) << "|CI|"
                      << "\n" << std::string(140, '-') << "\n"
                      << std::defaultfloat;
            // clang-format on
        }

    IloTimer timer(env);
    timer.start();

    while (true)
        {
            // Critério de parada: tempo
            time = timer.getTime();

            if (time_limit > 0.0 && time >= time_limit)
                break;

            // Resolve o subproblema Lagrangeano
            LR.solve();

            if (LR.opt > lb + EPS)
                {
                    lb = LR.opt;
                    last_improv_iter = iter;
                }

            // Separa e gerencia os cortes
            if (LR.mode == LRP::Mode::CAPACITIES)
                {
                    LR.separate_flow_covers(MAX_NEW_CUTS);
                }
            else if (LR.mode == LRP::Mode::BALANCES)
                {
                    LR.separate_subset_rows(MAX_NEW_CUTS);
                }

            cuts.updateStatus(LR.x, LR.y, LR.a, LR.b, EXTRA_AGE);

            // Heurística primal
            if (iter == 0 || (last_improv_iter == iter) || (iter % SOLVE_HEURISTIC_EVERY == 0))
                {
                    SP.update(LR.a, LR.b, true);
                    SP.solve();

                    if (SP.opt + EPS < ub)
                        {
                            ub = SP.opt;
                            a = SP.a.copy();
                            b = SP.b.copy();
                        }
                }

            updateGap();

            // Critério de parada: iterações sem melhora
            if (iter - last_improv_iter >= MAX_NO_IMPROV)
                break;
            // Critério de parada: gap
            if (status == IloAlgorithm::Optimal)
                break;

            // Passo do subgradiente
            IloNum norm2 = LR.norm2sq();
            IloNum step = 0.0;
            if (norm2 > EPS)
                step = IloMax(epsilon * (ub - LR.opt) / norm2, 0.0);

            // Atualiza multiplicadores
            LR.updateMultipliers(step);

            // Atualiza epsilon se necessário
            if (iter - last_improv_iter >= IMPROV_EPSILON
                && iter - last_eps_update_iter >= IMPROV_EPSILON)
                {
                    epsilon *= 0.5;
                    last_eps_update_iter = iter;
                }

            // Log parcial
            if (log_output && (iter % PRINT_EVERY == 0))
                {
                    IloInt nCA = cuts.count(Cut::Status::CA);
                    IloInt nPA = cuts.count(Cut::Status::PA);
                    IloInt nCI = cuts.count(Cut::Status::CI);
                    // clang-format off
                    std::cout << std::right
                              << std::setw(5) << iter
                              << std::setw(10) << (iter - last_improv_iter)
                              << std::fixed << std::setprecision(1)
                              << std::setw(10) << time
                              << std::fixed << std::setprecision(0)
                              << std::setw(15) << LR.opt
                              << std::setw(15) << lb
                              << std::setw(15) << ub
                              << std::scientific << std::setprecision(2)
                              << std::setw(12) << gap
                              << std::setw(12) << step
                              << std::setw(12) << norm2
                              << std::fixed
                              << std::setw(10) << nCA
                              << std::setw(10) << nPA
                              << std::setw(10) << nCI
                              << "\n"
                              << std::defaultfloat;
                    // clang-format on
                }

            ++iter;
        }

    timer.stop();

    // Recupera os fluxos (x e y)
    updateFlows();

    // Log final
    printSummary("SUBGRADIENTE");
}
