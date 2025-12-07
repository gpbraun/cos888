/*
COS888

Classe base abstrata para os solvers do subproblema de fluxo mínimo do TSCFL.

Gabriel Braun, 2025
*/

#pragma once

#include "utils/instance/instance.hpp"

// SOLVER DO SUBPROBLEMA: Base
class Subproblem
{
  protected:
    const TSCFLInstance &inst;

  public:
    enum class Mode
    {
        DUAL,
        PRIMAL,
        NET
    };

    // Saída do subproblema
    double theta{ 0.0 }; // valor ótimo do subproblema
    double rhs{ 0.0 };   // termo independente do corte
    IloNumArray coef_a;  // coeficientes multiplicando a_i
    IloNumArray coef_b;  // coeficientes multiplicando b_j

    explicit Subproblem(const TSCFLInstance &inst_)
        : inst(inst_),
          coef_a(inst_.env, inst_.nI),
          coef_b(inst_.env, inst_.nJ)
    {
    }

    static std::unique_ptr<Subproblem> create(const TSCFLInstance &inst, Mode mode);

    virtual ~Subproblem() = default;

    // Dado (a_vals, b_vals) da solução atual do mestre, resolve o subproblema.
    // Atualiza: theta, rhs, coef_a, coef_b
    virtual void solve(const IloNumArray &a_vals, const IloNumArray &b_vals) = 0;

    // Dado (a_frac, b_frac) da solução atual do mestre, encontra a_int e b_int e resolve o
    // subproblema. Atualiza: theta, rhs, coef_a, coef_b Retorna: valor da solução primal heurística
    // encontrada
    IloNum
    solveHeuristic(
        const IloNumArray &a_frac, const IloNumArray &b_frac, IloNumArray &a_int, IloNumArray &b_int
    )
    {
        const IloEnv &env = inst.env;
        IloNum demand_sum = IloSum(inst.r);

        fillZero(a_int);
        fillZero(b_int);

        // 1) Seleção de plantas abertas
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

        // 2) Seleção de depósitos abertos
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

        // 3) Resolve o subproblema de fluxo mínimo para (a_int, b_int)
        this->solve(a_int, b_int);

        return IloScalProd(inst.f, a_int) + IloScalProd(inst.g, b_int) + theta;
    }

    // Dado (var_a, var_b) da solução atual do mestre, encontra a_int e b_int e resolve o
    // subproblema. Atualiza: theta, rhs, coef_a, coef_b Retorna: valor da solução primal heurística
    // encontrada
    IloNum
    solveHeuristic(
        const IloCplex &cpx,
        const IloNumVarArray &var_a,
        const IloNumVarArray &var_b,
        IloNumArray &a_int,
        IloNumArray &b_int
    )
    {
        IloNumArray a_frac(var_a.getEnv(), var_a.getSize()),
            b_frac(var_b.getEnv(), var_b.getSize());

        cpx.getValues(a_frac, var_a);
        cpx.getValues(b_frac, var_b);

        return solveHeuristic(a_frac, b_frac, a_int, b_int);
    }
};
