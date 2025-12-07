/*
COS888

tscfl_subproblem.hpp

Gabriel Braun, 2025
*/

#pragma once

#include <memory>

#include "tscfl_instance.hpp"
#include "tscfl_utils.hpp"

// SOLVER DO SUBPROBLEMA
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
    IloNum opt{ 0.0 }; // valor ótimo do problema original

    // Coeficientes do corte de Benders
    IloNum theta{ 0.0 }; // valor ótimo do subproblema
    IloNum rhs{ 0.0 };   // termo independente do corte
    IloNumArray coef_a;  // coeficientes multiplicando a_i
    IloNumArray coef_b;  // coeficientes multiplicando b_j

    explicit Subproblem(const TSCFLInstance &inst_);

    virtual ~Subproblem() = default;

    // Factory: cria implementação concreta a partir do modo.
    static std::unique_ptr<Subproblem> create(const TSCFLInstance &inst, Mode mode);

    // Dado (a_vals, b_vals) da solução atual do mestre, resolve o subproblema.
    // Atualiza: opt, theta, rhs, coef_a, coef_b
    virtual void solve(const IloNumArray &a, const IloNumArray &b) = 0;

    // Heurística primal.
    // Dado (a_frac, b_frac), encontra (a_int e b_int) e resolve o subproblema.
    // Atualiza: opt, theta, rhs, coef_a, coef_b
    void solveHeuristic(
        const IloNumArray &a_frac, const IloNumArray &b_frac, IloNumArray &a_int, IloNumArray &b_int
    );

    // Heurística primal.
    // Dado (a_frac, b_frac), encontra (a_int e b_int) e resolve o subproblema.
    // Atualiza: opt, theta, rhs, coef_a, coef_b
    void solveHeuristic(
        const IloCplex &cpx,
        const IloNumVarArray &var_a,
        const IloNumVarArray &var_b,
        IloNumArray &a_int,
        IloNumArray &b_int
    );
};
