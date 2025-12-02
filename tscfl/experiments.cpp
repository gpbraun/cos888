/*
COS888

Experimentos com o TSCFL.

Gabriel Braun, 2025
*/

#include <ilcplex/ilocplex.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "tscfl_solver_lp.hpp"
#include "tscfl_solver_cplex.hpp"
#include "tscfl_solver_columns.hpp"
#include "tscfl_solver_subgradient.hpp"
#include "tscfl_solver_benders.hpp"

int main()
{
    // Lista de instâncias a serem rodadas
    const std::vector<std::string> instance_paths = {
        "_instances/fernandes/tscfl_050_100_200_a.txt",
        "_instances/fernandes/tscfl_050_100_200_b.txt",
        "_instances/fernandes/tscfl_050_100_200_c.txt",
        "_instances/fernandes/tscfl_050_100_200_d.txt",
        "_instances/fernandes/tscfl_050_100_200_e.txt",
        "_instances/fernandes/tscfl_100_200_400_a.txt",
        "_instances/fernandes/tscfl_100_200_400_b.txt",
        "_instances/fernandes/tscfl_100_200_400_c.txt",
        "_instances/fernandes/tscfl_100_200_400_d.txt",
        "_instances/fernandes/tscfl_100_200_400_e.txt",
    };

    const double time_limit = 60.0;
    IloEnv env;
    int status = 0;

    std::ofstream out("experiments_cg_rc.txt");
    if (!out)
    {
        std::cerr << "Erro ao abrir arquivo experiments.txt para escrita.\n";
        return 1;
    }

    out << std::fixed << std::setprecision(2);

    try
    {
        int inst_id = 0;

        for (const auto &path : instance_paths)
        {
            ++inst_id;
            std::cout << "\n\n>>> Rodando instancia #" << inst_id
                      << " : " << path << "\n";

            out << "====================================================\n";
            out << "#" << inst_id << " " << path << "\n";
            out.flush();

            // -----------------------------------------------------------------
            // Ler instância
            // -----------------------------------------------------------------
            TSCFLInstance inst = TSCFLInstance::from_txt(env, path);

            // -----------------------------------------------------------------
            // [LP] – valor ótimo da relaxação linear
            // -----------------------------------------------------------------
            {
                TSCFLSolverLP solver_lp(inst);
                solver_lp.solve(false, time_limit);

                out << "[LP]   "
                    << std::fixed << std::setprecision(0)
                    << solver_lp.opt << "\n";
            }
            out.flush();

            // -----------------------------------------------------------------
            // [MP] – modelo inteiro resolvido diretamente no CPLEX
            // lb, ub, time
            // -----------------------------------------------------------------
            {
                TSCFLSolverCplex solver_mp(inst);
                solver_mp.solve(false, time_limit);

                out << "[MP]   "
                    << std::fixed << std::setprecision(0)
                    << solver_mp.lb << "  "
                    << solver_mp.ub << "  "
                    << solver_mp.nodes << "  "
                    << std::fixed << std::setprecision(1)
                    << solver_mp.time << "\n";
            }
            out.flush();

            // -----------------------------------------------------------------
            // [CG-*] – Geração de colunas
            // lb, ub, iter, time  (usando nodes como "iter")
            // -----------------------------------------------------------------
            {
                // Subproblema Network
                TSCFLSolverColumnGeneration solver_cg_net(inst, Subproblem::Mode::NET);
                solver_cg_net.solve(false, time_limit);
                out << "[CG-n] "
                    << std::fixed << std::setprecision(0)
                    << solver_cg_net.lb << "  "
                    << solver_cg_net.ub << "  "
                    << solver_cg_net.iter << "  "
                    << std::fixed << std::setprecision(1)
                    << solver_cg_net.time << "\n";
            }
            out.flush();

            // -----------------------------------------------------------------
            // [RC-*] – Relax-and-Cut via subgradiente
            // lb, ub, iter, time  (usando nodes como "iter")
            // -----------------------------------------------------------------
            {
                // Subproblema Network
                TSCFLSolverSubgradient solver_rc_net(inst, Relaxation::Mode::CAPACITIES, Subproblem::Mode::NET);
                solver_rc_net.solve(false, time_limit);
                out << "[RC-n] "
                    << solver_rc_net.lb << "  "
                    << solver_rc_net.ub << "  "
                    << solver_rc_net.iter << "  "
                    << solver_rc_net.time << "\n";
            }
            out.flush();

            // -----------------------------------------------------------------
            // [BD-*] – Benders
            // lb, ub, nodes, time
            // -----------------------------------------------------------------
            // {
            //     // Subproblema Primal
            //     TSCFLSolverBenders solver_bd_primal(inst, Subproblem::Mode::PRIMAL);
            //     solver_bd_primal.solve(false, time_limit);
            //     out << "[BD-p] "
            //         << std::fixed << std::setprecision(0)
            //         << solver_bd_primal.lb << "  "
            //         << solver_bd_primal.ub << "  "
            //         << solver_bd_primal.nodes << "  "
            //         << std::fixed << std::setprecision(1)
            //         << solver_bd_primal.time << "\n";

            //     // Subproblema Dual
            //     TSCFLSolverBenders solver_bd_dual(inst, Subproblem::Mode::DUAL);
            //     solver_bd_dual.solve(false, time_limit);
            //     out << "[BD-d] "
            //         << std::fixed << std::setprecision(0)
            //         << solver_bd_dual.lb << "  "
            //         << solver_bd_dual.ub << "  "
            //         << solver_bd_dual.nodes << "  "
            //         << std::fixed << std::setprecision(1)
            //         << solver_bd_dual.time << "\n";

            //     // Subproblema Network
            //     TSCFLSolverBenders solver_bd_net(inst, Subproblem::Mode::NET);
            //     solver_bd_net.solve(false, time_limit);
            //     out << "[BD-n] "
            //         << std::fixed << std::setprecision(0)
            //         << solver_bd_net.lb << "  "
            //         << solver_bd_net.ub << "  "
            //         << solver_bd_net.nodes << "  "
            //         << std::fixed << std::setprecision(1)
            //         << solver_bd_net.time << "\n";
            // }
            // out.flush();
            // out << "\n";
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
