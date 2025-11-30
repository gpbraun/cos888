/*
COS888

Experimentos com o TSCFL.

Gabriel Braun, 2025
*/

#include <iostream>

#include "tscfl_instance.hpp"
#include "tscfl_solver_cplex.hpp"
#include "tscfl_solver_benders.hpp"

int main()
{
    const std::string path = "_instances/fernandes/tscfl_050_100_200_c.txt";

    IloEnv env;
    int status = 0;

    try
    {
        TSCFLInstance inst = TSCFLInstance::from_txt(env, path);

        // 0. CPLEX MP
        // TSCFLSolverCplex solver_cplex(inst);
        // solver_cplex.solve(true, 200.0);

        // 1. BENDERS
        TSCFLSolverBenders solver_benders(inst, 2);
        solver_benders.solve(true, 100.0);
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
