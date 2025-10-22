// protein_cp.cpp
//
// CP Optimizer model for the simplified Protein Folding problem.
// Fix: force-extract all fold vars y_k by adding them to the model.
//
// Build (Linux):
// g++ protein_cp.cpp -std=c++17 \
//   -I"$CPLEX/concert/include" -I"$CPLEX/cpoptimizer/include" \
//   -L"$CPLEX/concert/lib/x86-64_linux" -L"$CPLEX/cpoptimizer/lib/x86-64_linux" \
//   -lconcert -lcp -lm -lpthread -o protein_cp
//
// Run: ./protein_cp

#include <ilcp/cp.h>
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>

struct Pair
{
    int i, j;
    IloBoolVar var;
};

int main()
{
    IloEnv env;
    try
    {
        // -------------------------
        // Instance data
        // -------------------------
        const int n = 50; // amino acids indexed 1..n
        const std::vector<int> H = {2, 4, 5, 6, 11, 12, 17, 20, 21, 25, 27, 28, 30, 31, 33, 37, 44, 46};

        std::vector<char> isH(n + 1, false);
        for (int h : H)
            isH[h] = true;

        IloModel mdl(env);

        // -------------------------
        // Variables
        // -------------------------
        // y_k : fold between k and k+1  (k = 1..n-1)
        // BEST FIX: add every y_k to the model to ensure extraction.
        IloBoolVarArray y(env, n);
        for (int k = 1; k <= n - 1; ++k)
        {
            y[k] = IloBoolVar(env, ("y_" + std::to_string(k)).c_str());
            mdl.add(y[k]); // <--- force extraction so getValue(y[k]) is always valid
        }

        // x_ij for eligible hydrophobic pairs (i<j, noncontiguous, (i+j-1) even)
        std::vector<Pair> pairs;
        pairs.reserve(400);

        auto is_even = [](int v)
        { return (v & 1) == 0; };

        for (int i = 1; i <= n; ++i)
            if (isH[i])
            {
                for (int j = i + 1; j <= n; ++j)
                    if (isH[j])
                    {
                        if (j == i + 1)
                            continue; // (i) not contiguous
                        if (!is_even(i + j - 1))
                            continue; // (ii) even number in between
                        // eligible: create x_ij
                        std::string name = "x_" + std::to_string(i) + "_" + std::to_string(j);
                        IloBoolVar xij(env, name.c_str());
                        pairs.push_back({i, j, xij});

                        // (iii) exactly one fold between i and j: x_ij == y_m, m=(i+j-1)/2
                        int m = (i + j - 1) / 2;
                        mdl.add(xij == y[m]);

                        // forbid any other fold in [i, j): y_k + x_ij <= 1,  k != m
                        for (int k = i; k < j; ++k)
                            if (k != m)
                            {
                                mdl.add(y[k] + xij <= 1);
                            }
                    }
            }

        // -------------------------
        // Objective: maximize sum x_ij
        // -------------------------
        IloExpr obj(env);
        for (const auto &p : pairs)
            obj += p.var;
        mdl.add(IloMaximize(env, obj));
        obj.end();

        // -------------------------
        // Solve
        // -------------------------
        IloCP cp(mdl);
        cp.setParameter(IloCP::TimeLimit, 60);
        cp.setParameter(IloCP::LogPeriod, 1000000);
        // cp.setParameter(IloCP::SearchType, IloCP::MultiPoint); // optional

        if (cp.solve())
        {
            std::cout << "Optimal objective = " << cp.getObjValue() << "\n\n";

            // Folds
            std::cout << "FOLDS (k where y_k = 1; fold between k and k+1):\n";
            for (int k = 1; k <= n - 1; ++k)
            {
                if (cp.getValue(y[k]) > 0.5)
                    std::cout << k << " ";
            }
            std::cout << "\n\n";

            // Matched pairs
            std::cout << "MATCHED HYDROPHOBIC PAIRS (i,j) with x_ij = 1:\n";
            for (const auto &p : pairs)
            {
                if (cp.getValue(p.var) > 0.5)
                    std::cout << "(" << p.i << "," << p.j << ") ";
            }
            std::cout << "\n";
        }
        else
        {
            std::cout << "No solution found.\n";
        }

        cp.end();
    }
    catch (const IloException &e)
    {
        std::cerr << "Concert exception: " << e << "\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "Std exception: " << e.what() << "\n";
    }
    env.end();
    return 0;
}
