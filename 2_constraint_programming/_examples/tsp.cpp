#include <ilcp/cp.h>
#include <iostream>

int main()
{
    IloEnv env;
    try
    {
        IloModel model(env);

        // Number of cities
        const int N = 5;

        // Distance matrix (C array)
        const int d2[N][N] = {
            {0, 2, 9, 10, 7},
            {2, 0, 6, 4, 3},
            {9, 6, 0, 8, 5},
            {10, 4, 8, 0, 10},
            {7, 3, 5, 10, 0}};

        // Decision variables: city[i] = city visited at step i (permutation of 0..N-1)
        IloIntVarArray city(env, N, 0, N - 1);
        model.add(IloAllDiff(env, city));

        // Flatten distances into a Concert array for IloElement
        IloIntArray dist(env, N * N);
        for (int i = 0, k = 0; i < N; ++i)
            for (int j = 0; j < N; ++j, ++k)
                dist[k] = d2[i][j];

        // Objective: sum of legs city[i] -> city[i+1], plus last -> first
        IloExpr totalDistance(env);
        for (int i = 0; i < N - 1; ++i)
        {
            IloIntExpr idx = city[i] * N + city[i + 1]; // row*N + col
            totalDistance += IloElement(dist, idx);
        }
        totalDistance += IloElement(dist, city[N - 1] * N + city[0]);

        model.add(IloMinimize(env, totalDistance));

        IloCP cp(model);
        // cp.setParameter(IloCP::TimeLimit, 10);  // optional
        if (cp.solve())
        {
            cp.out() << "Total Distance: " << cp.getValue(totalDistance) << std::endl;
            cp.out() << "Route: ";
            for (int i = 0; i < N; ++i)
                cp.out() << cp.getValue(city[i]) << (i + 1 < N ? " -> " : "");
            cp.out() << " -> " << cp.getValue(city[0]) << std::endl;
        }
        else
        {
            cp.out() << "No solution found.\n";
        }

        totalDistance.end();
    }
    catch (IloException &e)
    {
        std::cerr << "Concert exception caught: " << e << std::endl;
    }
    env.end();
    return 0;
}
