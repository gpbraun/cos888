/*
COS888

Experimentos com o TSCFL.

Gabriel Braun, 2025
*/

#include <iostream>

// #include "tscfl_instance.hpp"
// #include "tscfl_solver_cplex.hpp"
#include "tscfl_solver_cg.hpp"
#include "tscfl_solver_benders.hpp"
#include "tscfl_solver_subgradient.hpp"

int main()
{
    const std::string path = "_instances/fernandes/tscfl_050_100_200_a.txt";

    IloEnv env;
    int status = 0;

    try
    {
        TSCFLInstance inst = TSCFLInstance::from_txt(env, path);

        // 0. CPLEX MP
        // TSCFLSolverCplex solver_cplex(inst);
        // solver_cplex.solve(true, 200.0);

        // 1. COLUNAS
        TSCFLSolverColumnGeneration solver_cg(inst);
        solver_cg.solve(true, 100.0);

        // 2. RELAX-AND-CUT
        // TSCFLSolverSubgradient solver_rc(inst);
        // solver_rc.solve(true, 100.0);

        // 3. BENDERS
        // TSCFLSolverBenders solver_benders(inst, Subproblem::Mode::NET);
        // solver_benders.solve(true, 60.0);
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
