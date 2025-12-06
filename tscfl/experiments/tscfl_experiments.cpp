#include "tscfl.hpp"

int
main()
{
    IloEnv env;

    auto inst = TSCFLInstance::from_txt(env, "tscfl/data/fernandes/tscfl_050_100_200_a.txt");

    TSCFLSolverBenders solver_benders(inst);
    solver_benders.solve(true, 10.0);

    TSCFLSolverColumnGeneration solver_columns(inst);
    solver_columns.solve(true, 10.0);

    TSCFLSolverSubgradient solver_rac(inst, LRP::Mode::CAPACITIES);
    solver_rac.solve(true, 60.0);

    return 0;
}
