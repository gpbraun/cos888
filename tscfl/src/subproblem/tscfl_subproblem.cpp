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
      a(inst_.env, inst.nI),
      b(inst_.env, inst.nJ),
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

void
Subproblem::convertToPrimal()
{
    IloNum demand_sum = IloSum(inst.r);

    // PLANTAS
    std::vector<int> ordI(inst.nI);
    std::iota(ordI.begin(), ordI.end(), 0);

    // Ordenação
    std::sort(
        ordI.begin(),
        ordI.end(),
        [&](int i, int j)
            {
                if (std::fabs(a[i] - a[j]) > EPS)
                    return a[i] > a[j];

                double ratio_i = inst.p[i] > EPS ? inst.f[i] / inst.p[i] : IloInfinity;
                double ratio_j = inst.p[j] > EPS ? inst.f[j] / inst.p[j] : IloInfinity;
                return ratio_i < ratio_j;
            }
    );
    // Construção da solução inteira
    fillZero(a);
    double capI = 0.0;
    for (int pos = 0; pos < inst.nI && capI + EPS < demand_sum; ++pos)
        {
            int i = ordI[pos];
            if (inst.p[i] <= EPS)
                continue;
            a[i] = 1.0;
            capI += inst.p[i];
        }

    // DEPÓSITOS
    std::vector<int> ordJ(inst.nJ);
    std::iota(ordJ.begin(), ordJ.end(), 0);

    // Ordenação
    std::sort(
        ordJ.begin(),
        ordJ.end(),
        [&](int j1, int j2)
            {
                if (std::fabs(b[j1] - b[j2]) > EPS)
                    return b[j1] > b[j2];

                double ratio1 = inst.q[j1] > EPS ? inst.g[j1] / inst.q[j1] : IloInfinity;
                double ratio2 = inst.q[j2] > EPS ? inst.g[j2] / inst.q[j2] : IloInfinity;
                return ratio1 < ratio2;
            }
    );
    // Construção da solução inteira
    fillZero(b);
    double capJ = 0.0;
    for (int pos = 0; pos < inst.nJ && capJ + EPS < demand_sum; ++pos)
        {
            int j = ordJ[pos];
            if (inst.q[j] <= EPS)
                continue;
            b[j] = 1.0;
            capJ += inst.q[j];
        }
}

void
Subproblem::update(const IloNumArray &a_, const IloNumArray &b_, IloBool convert_to_primal)
{
    a = a_;
    b = b_;

    if (convert_to_primal)
        convertToPrimal();
}

void
Subproblem::update(
    const IloCplex &cpx,
    const IloNumVarArray &var_a,
    const IloNumVarArray &var_b,
    IloBool convert_to_primal
)
{
    cpx.getValues(a, var_a);
    cpx.getValues(b, var_b);

    if (convert_to_primal)
        convertToPrimal();
}
