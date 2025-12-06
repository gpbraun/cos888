#include "tscfl.hpp"

int
main()
{
    IloEnv env;

    auto inst = TSCFLInstance::from_txt(env, "tscfl/data/fernandes/tscfl_050_100_200_a.txt");

    // TSCFLSolverColumnGeneration solver2 (inst);
    // solver2.solve (true, 10.0);

    TSCFLSolverSubgradient solver2(inst, LRP::Mode::BALANCES);
    solver2.solve(true, 60.0);

    return 0;
}
