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
    const std::string path = "_instances/fernandes/tscfl_050_100_200_a.txt";

    try
    {
        TSCFLInstance inst = TSCFLInstance::from_txt(path);

        // 0. CPLEX MP
        // TSCFLSolverCplex solver_cplex(inst);

        // 1. Benders
        TSCFLSolverBenders solver_benders(inst);
        solver_benders.solve(true, 600.0);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
