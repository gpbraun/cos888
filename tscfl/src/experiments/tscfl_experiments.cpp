/*
COS888

tscfl_experiments.cpp

Gabriel Braun, 2025
*/

#include <ilcplex/ilocplex.h>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <system_error>
#include <vector>

#include "tscfl.hpp"

int
main()
{
    // Lista de instâncias a serem rodadas
    const std::vector<std::string> instance_paths = {
        "tscfl/data/fernandes/tscfl_050_100_200_a.txt",
        "tscfl/data/fernandes/tscfl_050_100_200_b.txt",
        "tscfl/data/fernandes/tscfl_050_100_200_c.txt",
        "tscfl/data/fernandes/tscfl_050_100_200_d.txt",
        "tscfl/data/fernandes/tscfl_050_100_200_e.txt",
        "tscfl/data/fernandes/tscfl_100_200_400_a.txt",
        "tscfl/data/fernandes/tscfl_100_200_400_b.txt",
        "tscfl/data/fernandes/tscfl_100_200_400_c.txt",
        "tscfl/data/fernandes/tscfl_100_200_400_d.txt",
        "tscfl/data/fernandes/tscfl_100_200_400_e.txt",
    };

    const double time_limit = 1000.0;
    IloEnv env;
    int status = 0;
    std::string sep = "    ";

    std::error_code ec;
    std::filesystem::create_directories("out", ec);
    if (ec)
        {
            std::cerr << "Erro ao criar diretorio 'out': " << ec.message() << "\n";
            return 1;
        }
    std::ofstream out("out/tscfl_out.txt");
    if (!out)
        {
            std::cerr << "Erro ao abrir arquivo para escrita.\n";
            return 1;
        }

    try
        {
            int inst_id = 0;
            for (const auto &path : instance_paths)
                {
                    ++inst_id;
                    std::cout << "\n\n>>> Instancia #" << inst_id << " : " << path << "\n";

                    out << "====================================================\n";
                    out << "#" << inst_id << ". " << path << "\n\n";
                    out.flush();

                    // Ler instância
                    TSCFLInstance inst = TSCFLInstance::read(env, path);

                    // [CPLEX] CPLEX
                    // lp, opt, nodes, time
                    {
                        TSCFLSolverCplex solver_cplex(inst);
                        solver_cplex.solveLP(false, time_limit);
                        solver_cplex.solve(false, time_limit);

                        // clang-format off
                        out << "[CPLEX]   " << sep
                            << std::left << std::fixed << std::setprecision(0)
                            << "lp="     << solver_cplex.lp    << sep
                            << "opt="    << solver_cplex.ub    << sep
                            << "nodes="  << solver_cplex.nodes << sep
                            << std::left << std::setprecision(1)
                            << "time="   << solver_cplex.time  << sep
                            << "\n";
                        // clang-format on
                        out.flush();
                    }
                    // [BD] BENDERS
                    // lb, ub, nodes, time
                    {
                        // SUBPROBLEMA PRIMAL
                        TSCFLSolverBenders solver_bd1(inst, Subproblem::Mode::PRIMAL);
                        solver_bd1.solve(false, time_limit);

                        // clang-format off
                        out << "[BD-prim] " << sep
                            << std::fixed << std::setprecision(0)
                            << "lb="    << solver_bd1.lb    << sep
                            << "ub="    << solver_bd1.ub    << sep
                            << "nodes=" << solver_bd1.nodes << sep
                            << std::setprecision(1)
                            << "time="  << solver_bd1.time  << sep
                            << "\n";
                        // clang-format on
                        out.flush();

                        // SUBPROBLEMA DUAL
                        TSCFLSolverBenders solver_bd2(inst, Subproblem::Mode::DUAL);
                        solver_bd2.solve(false, time_limit);

                        // clang-format off
                        out << "[BD-dual] " << sep
                            << std::fixed << std::setprecision(0)
                            << "lb="    << solver_bd2.lb    << sep
                            << "ub="    << solver_bd2.ub    << sep
                            << "nodes=" << solver_bd2.nodes << sep
                            << std::setprecision(1)
                            << "time="  << solver_bd2.time  << sep
                            << "\n";
                        // clang-format on
                        out.flush();

                        // SUBPROBLEMA NET
                        TSCFLSolverBenders solver_bd3(inst, Subproblem::Mode::NET);
                        solver_bd3.solve(false, time_limit);

                        // clang-format off
                        out << "[BD-net]  " << sep
                            << std::fixed << std::setprecision(0)
                            << "lb="    << solver_bd3.lb    << sep
                            << "ub="    << solver_bd3.ub    << sep
                            << "nodes=" << solver_bd3.nodes << sep
                            << std::setprecision(1)
                            << "time="  << solver_bd3.time  << sep
                            << "\n";
                        // clang-format on
                        out.flush();
                    }
                    // [CG] GERAÇÃO DE COLUNAS
                    // lb, ub, iter, time
                    {
                        TSCFLSolverColumnGeneration solver_cg(inst, Subproblem::Mode::NET);
                        solver_cg.solve(false, time_limit);

                        // clang-format off
                        out << "[GC]      " << sep
                            << std::fixed << std::setprecision(0)
                            << "lb="   << solver_cg.lb   << sep
                            << "ub="   << solver_cg.ub   << sep
                            << "iter=" << solver_cg.iter << sep
                            << std::setprecision(1)
                            << "time=" << solver_cg.time << sep
                            << "\n";
                        // clang-format on
                        out.flush();
                    }
                    // [RC] NON-DELAYED RELAX-AND-CUT
                    // lb, ub, iter, time
                    {
                        // CAPACIDADES RELAXADAS
                        TSCFLSolverSubgradient solver_rc1(inst, LRP::Mode::CAPACITIES);
                        solver_rc1.solve(false, time_limit);

                        // clang-format off
                        out << "[RAC-cap] " << sep
                            << std::fixed << std::setprecision(0)
                            << "lb="   << solver_rc1.lb   << sep
                            << "ub="   << solver_rc1.ub   << sep
                            << "iter=" << solver_rc1.iter << sep
                            << std::setprecision(1)
                            << "time=" << solver_rc1.time << sep
                            << "\n";
                        // clang-format on
                        out.flush();

                        // BALANÇOS RELAXADOS
                        TSCFLSolverSubgradient solver_rc2(inst, LRP::Mode::BALANCES);
                        solver_rc2.solve(false, time_limit);

                        // clang-format off
                        out << "[RAC-blc] " << sep
                            << std::fixed << std::setprecision(0)
                            << "lb="   << solver_rc2.lb   << sep
                            << "ub="   << solver_rc2.ub   << sep
                            << "iter=" << solver_rc2.iter << sep
                            << std::setprecision(1)
                            << "time=" << solver_rc2.time << sep
                            << "\n";
                        // clang-format on
                        out.flush();
                    }
                    out << "\n";
                }
        }
    catch (const IloException &e)
        {
            std::cerr << "CPLEX Error: " << e.getMessage() << "\n";
            status = 1;
        }
    catch (const std::exception &e)
        {
            std::cerr << "Error: " << e.what() << "\n";
            status = 1;
        }

    env.end();
    return status;
}
