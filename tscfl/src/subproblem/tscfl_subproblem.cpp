/*
COS888

tscfl_subproblem.cpp

Gabriel Braun, 2025
*/

#include "tscfl_subproblem.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "tscfl_subproblem_dual.hpp"
#include "tscfl_subproblem_net.hpp"
#include "tscfl_subproblem_primal.hpp"

Subproblem::Subproblem(const TSCFLInstance &inst_)
    : inst(inst_),
      coef_a(inst_.env, inst_.nI),
      coef_b(inst_.env, inst_.nJ)
{
}

std::unique_ptr<Subproblem>
Subproblem::create(const TSCFLInstance &inst, Mode mode)
{
    switch (mode)
        {
        case Mode::DUAL:
            return std::make_unique<SubproblemDual>(inst);
        case Mode::PRIMAL:
            return std::make_unique<SubproblemPrimal>(inst);
        case Mode::NET:
            return std::make_unique<SubproblemNet>(inst);
        default:
            throw std::invalid_argument("Subproblem::create: modo desconhecido.");
        }
}

IloNum
Subproblem::solve_primal_heuristic(
    const IloNumArray &a_frac, const IloNumArray &b_frac, IloNumArray &a_int, IloNumArray &b_int
)
{
    IloNum demand_sum = IloSum(inst.r);
    fill_zero(a_int);
    fill_zero(b_int);

    // Seleção de plantas abertas
    std::vector<int> ordI(inst.nI);
    std::iota(ordI.begin(), ordI.end(), 0);

    std::sort(
        ordI.begin(),
        ordI.end(),
        [&](int i, int j)
            {
                if (std::fabs(a_frac[i] - a_frac[j]) > EPS)
                    return a_frac[i] > a_frac[j];

                double ratio_i = inst.p[i] > EPS ? inst.f[i] / inst.p[i] : IloInfinity;
                double ratio_j = inst.p[j] > EPS ? inst.f[j] / inst.p[j] : IloInfinity;
                return ratio_i < ratio_j;
            }
    );

    double capI = 0.0;
    for (int pos = 0; pos < inst.nI && capI + EPS < demand_sum; ++pos)
        {
            int i = ordI[pos];
            if (inst.p[i] <= EPS)
                continue;
            a_int[i] = 1.0;
            capI += inst.p[i];
        }

    // Seleção de depósitos abertos
    std::vector<int> ordJ(inst.nJ);
    std::iota(ordJ.begin(), ordJ.end(), 0);

    std::sort(
        ordJ.begin(),
        ordJ.end(),
        [&](int j1, int j2)
            {
                if (std::fabs(b_frac[j1] - b_frac[j2]) > EPS)
                    return b_frac[j1] > b_frac[j2];

                double ratio1 = inst.q[j1] > EPS ? inst.g[j1] / inst.q[j1] : IloInfinity;
                double ratio2 = inst.q[j2] > EPS ? inst.g[j2] / inst.q[j2] : IloInfinity;
                return ratio1 < ratio2;
            }
    );

    double capJ = 0.0;
    for (int pos = 0; pos < inst.nJ && capJ + EPS < demand_sum; ++pos)
        {
            int j = ordJ[pos];
            if (inst.q[j] <= EPS)
                continue;
            b_int[j] = 1.0;
            capJ += inst.q[j];
        }

    // Resolve o subproblema de fluxo mínimo para (a_int, b_int)
    this->solve(a_int, b_int);

    return opt;
}

IloNum
Subproblem::solve_primal_heuristic(
    const IloCplex &cpx,
    const IloNumVarArray &var_a,
    const IloNumVarArray &var_b,
    IloNumArray &a_int,
    IloNumArray &b_int
)
{
    IloNumArray a_frac(var_a.getEnv(), var_a.getSize());
    IloNumArray b_frac(var_b.getEnv(), var_b.getSize());

    cpx.getValues(a_frac, var_a);
    cpx.getValues(b_frac, var_b);

    return solve_primal_heuristic(a_frac, b_frac, a_int, b_int);
}
