/*
COS888

tscfl_solver_subgradient.cpp

Gabriel Braun, 2025
*/

#include "solvers/tscfl_solver_subgradient.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>

// ---------------------------------------------------------------------
//  Construtor
// ---------------------------------------------------------------------

TSCFLSolverSubgradient::TSCFLSolverSubgradient(
    const TSCFLInstance &inst_,
    LRP::Mode rmode,
    Subproblem::Mode smode
)
    : TSCFLSolver(inst_), // chama construtor da base
      relaxation(LRP::create(inst_, rmode)),
      subproblem(Subproblem::create(inst_, smode)),
      a(env, inst_.nI), // usa env da base
      b(env, inst_.nJ)
{
}

// ---------------------------------------------------------------------
//  Método principal
// ---------------------------------------------------------------------

bool
TSCFLSolverSubgradient::solve(bool log_output, IloNum time_limit)
{
    IloInt last_improv_iter = 0;
    IloInt last_eps_update_iter = 0;

    IloNum epsilon = EPSILON0;

    auto t0 = std::chrono::steady_clock::now();

    auto &SP = *subproblem;
    auto &LR = *relaxation;
    auto &cuts = LR.getCuts();

    IloNumArray a_h(env, inst.nI), b_h(env, inst.nJ);

    if (log_output)
        {
            std::cout << "[SG] Iniciando Subgradiente\n\n"
                      << std::right << std::setw(5) << "iter" << std::setw(10) << "no improve"
                      << std::setw(10) << "time(s)" << std::setw(15) << "opt_LR" << std::setw(15)
                      << "LB" << std::setw(15) << "UB" << std::setw(12) << "gap" << std::setw(12)
                      << "step" << std::setw(12) << "||g||^2" << std::setw(10) << "|CA|"
                      << std::setw(10) << "|PA|" << std::setw(10) << "|CI|"
                      << "\n"
                      << std::string(140, '-') << "\n"
                      << std::defaultfloat;
        }

    while (true)
        {
            // Controle de tempo
            auto t1 = std::chrono::steady_clock::now();
            IloNum elapsed = std::chrono::duration<IloNum>(t1 - t0).count();
            if (time_limit > 0.0 && elapsed >= time_limit)
                break;

            // Resolve subproblema Lagrangeano
            IloNum opt_lr = LR.solve();

            if (opt_lr > lb + EPS)
                {
                    lb = opt_lr;
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

            cuts.update_status(LR.x, LR.y, LR.a, LR.b, EXTRA_AGE);

            // Heurística primal periódica
            if (iter == 0 || (last_improv_iter == iter) || (iter % SOLVE_HEURISTIC_EVERY == 0))
                {
                    IloNum opt_h = SP.solve_primal_heuristic(LR.a, LR.b, a_h, b_h);
                    if (opt_h + EPS < ub)
                        {
                            ub = opt_h;
                            a = LR.a;
                            b = LR.b;
                        }
                }

            update_gap();

            // Passo do subgradiente
            IloNum norm2 = LR.norm2sq();
            IloNum step = 0.0;
            if (norm2 > EPS)
                step = IloMax(epsilon * (ub - opt_lr) / norm2, 0.0);

            // Atualiza multiplicadores
            LR.update_multipliers(step);

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

                    std::cout << std::right << std::setw(5) << iter << std::setw(10)
                              << (iter - last_improv_iter) << std::fixed << std::setprecision(1)
                              << std::setw(10) << elapsed << std::fixed << std::setprecision(0)
                              << std::setw(15) << opt_lr << std::setw(15) << lb << std::setw(15)
                              << ub << std::scientific << std::setprecision(2) << std::setw(12)
                              << gap << std::setw(12) << step << std::setw(12) << norm2
                              << std::fixed << std::setw(10) << nCA << std::setw(10) << nPA
                              << std::setw(10) << nCI << "\n"
                              << std::defaultfloat;
                }

            // Critério de parada por iterações sem melhora
            if (iter - last_improv_iter >= MAX_NO_IMPROV)
                {
                    break;
                }

            // Critério de parada por gap
            if (gap <= MIP_GAP && ub < IloInfinity && lb > 0.0)
                {
                    status = IloAlgorithm::Optimal;
                    break;
                }

            ++iter;
        }

    auto t_end = std::chrono::steady_clock::now();
    time = std::chrono::duration<IloNum>(t_end - t0).count();

    update_status();

    // Log final
    if (log_output)
        {
            print_summary("SUBGRADIENTE");
        }

    return (status == IloAlgorithm::Optimal);
}
