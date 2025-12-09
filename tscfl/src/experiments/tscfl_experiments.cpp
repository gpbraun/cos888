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
    int status = 0;

    // Lista de instâncias a serem rodadas
    const std::vector<std::string> instance_paths = {
        // "tscfl/data/fernandes/tscfl_050_100_200_a.txt",
        // "tscfl/data/fernandes/tscfl_050_100_200_b.txt",
        // "tscfl/data/fernandes/tscfl_050_100_200_c.txt",
        // "tscfl/data/fernandes/tscfl_050_100_200_d.txt",
        // "tscfl/data/fernandes/tscfl_050_100_200_e.txt",
        // "tscfl/data/fernandes/tscfl_100_200_400_a.txt",
        // "tscfl/data/fernandes/tscfl_100_200_400_b.txt",
        // "tscfl/data/fernandes/tscfl_100_200_400_c.txt",
        // "tscfl/data/fernandes/tscfl_100_200_400_d.txt",
        // "tscfl/data/fernandes/tscfl_100_200_400_e.txt",
    };

    IloEnv env;
    const double time_limit = 600.0;

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

    auto log_num = [](std::string_view name, IloNum v, int prec = 0)
        {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(prec);
            oss << name << "=" << v;
            return oss.str();
        };

    try
        {
            int inst_id = 0;
            for (const auto &path : instance_paths)
                {
                    ++inst_id;
                    std::cout << "\n\n>>> Instancia #" << inst_id << " : " << path << "\n";

                    out << std::string(75, '=') << "\n"
                        << "#" << inst_id << ". " << path << "\n\n";
                    out.flush();

                    // Ler instância
                    TSCFLInstance inst = TSCFLInstance::read(env, path);

                    // [CPLEX] CPLEX
                    // lp, ot, nodes, time
                    {
                        TSCFLSolverCplex solver_cplex(inst);
                        solver_cplex.solveLP(false, time_limit);
                        solver_cplex.solve(false, time_limit);

                        // clang-format off
                        out << std::left << std::setw(15) << "[  CPLEX  ]"
                            << std::setw(15) << log_num("lp", solver_cplex.lp)
                            << std::setw(15) << log_num("ot", solver_cplex.ub)
                            << std::setw(15) << log_num("nodes", solver_cplex.nodes)
                            << std::setw(15) << log_num("time", solver_cplex.time, 1)
                            << "\n";
                        // clang-format on
                        out.flush();
                    }
                    // [BNDRS] BENDERS
                    // lb, ub, nodes, time
                    {
                        // SUBPROBLEMA PRIMAL
                        TSCFLSolverBenders solver_bd1(inst, Subproblem::Mode::PRIMAL);
                        solver_bd1.solve(false, time_limit);

                        // clang-format off
                        out << std::left << std::setw(15) << "[ BNDRS-p ]"
                            << std::setw(15) << log_num("lb", solver_bd1.lb)
                            << std::setw(15) << log_num("up", solver_bd1.ub)
                            << std::setw(15) << log_num("nodes", solver_bd1.nodes)
                            << std::setw(15) << log_num("time", solver_bd1.time, 1)
                            << "\n";
                        // clang-format on
                        out.flush();

                        // SUBPROBLEMA DUAL
                        TSCFLSolverBenders solver_bd2(inst, Subproblem::Mode::DUAL);
                        solver_bd2.solve(false, time_limit);

                        // clang-format off
                        out << std::left << std::setw(15) << "[ BNDRS-d ]"
                            << std::setw(15) << log_num("lb", solver_bd2.lb)
                            << std::setw(15) << log_num("up", solver_bd2.ub)
                            << std::setw(15) << log_num("nodes", solver_bd2.nodes)
                            << std::setw(15) << log_num("time", solver_bd2.time, 1)
                            << "\n";
                        // clang-format on
                        out.flush();

                        // SUBPROBLEMA NET
                        TSCFLSolverBenders solver_bd3(inst, Subproblem::Mode::NET);
                        solver_bd3.solve(false, time_limit);

                        // clang-format off
                        out << std::left << std::setw(15) << "[ BNDRS-n ]"
                            << std::setw(15) << log_num("lb", solver_bd3.lb)
                            << std::setw(15) << log_num("up", solver_bd3.ub)
                            << std::setw(15) << log_num("nodes", solver_bd3.nodes)
                            << std::setw(15) << log_num("time", solver_bd3.time, 1)
                            << "\n";
                        // clang-format on
                        out.flush();
                    }
                    // [COLUMNS] GERAÇÃO DE COLUNAS
                    // lb, ub, iters, time
                    {
                        TSCFLSolverColumnGeneration solver_cg(inst, Subproblem::Mode::NET);
                        solver_cg.solve(false, time_limit);

                        // clang-format off
                        out << std::left << std::setw(15) << "[ COLUMNS ]"
                            << std::setw(15) << log_num("lb", solver_cg.lb)
                            << std::setw(15) << log_num("up", solver_cg.ub)
                            << std::setw(15) << log_num("iters", solver_cg.iter)
                            << std::setw(15) << log_num("time", solver_cg.time, 1)
                            << "\n";
                        // clang-format on
                        out.flush();
                    }
                    // [RAC] NON-DELAYED RELAX-AND-CUT
                    // lb, ub, iters, time
                    {
                        // CAPACIDADES RELAXADAS
                        TSCFLSolverSubgradient solver_rc1(inst, LRP::Mode::CAPACITIES);
                        solver_rc1.solve(false, time_limit);

                        // clang-format off
                        out << std::left << std::setw(15) << "[ RAC-cap ]"
                            << std::setw(15) << log_num("lb", solver_rc1.lb)
                            << std::setw(15) << log_num("up", solver_rc1.ub)
                            << std::setw(15) << log_num("iters", solver_rc1.iter)
                            << std::setw(15) << log_num("time", solver_rc1.time, 1)
                            << "\n";
                        // clang-format on
                        out.flush();

                        // BALANÇOS RELAXADOS
                        TSCFLSolverSubgradient solver_rc2(inst, LRP::Mode::BALANCES);
                        solver_rc2.solve(false, time_limit);

                        // clang-format off
                        out << std::left << std::setw(15) << "[ RAC-blc ]"
                            << std::setw(15) << log_num("lb", solver_rc2.lb)
                            << std::setw(15) << log_num("up", solver_rc2.ub)
                            << std::setw(15) << log_num("iters", solver_rc2.iter)
                            << std::setw(15) << log_num("time", solver_rc2.time, 1)
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
