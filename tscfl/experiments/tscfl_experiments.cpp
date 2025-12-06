#include "tscfl.hpp"

int
main ()
{
    IloEnv env;

    auto inst = TSCFLInstance::from_txt (
        env, "/home/braun/Developer/cos888/tscfl/data/fernandes/tscfl_050_100_200_c.txt");

    TSCFLSolverColumnGeneration solver2 (inst);
    solver2.solve (true, 10.0);

    return 0;
}
