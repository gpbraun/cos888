// src/subproblem.hpp
#pragma once

/*
COS888

Classe base abstrata para os solvers do subproblema de fluxo mínimo do TSCFL.

Gabriel Braun, 2025
*/

#include <memory>

#include "tscfl_instance.hpp"
#include "tscfl_utils.hpp"

ILOSTLBEGIN

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

    explicit Subproblem (const TSCFLInstance &inst_);

    virtual ~Subproblem () = default;

    // Factory: cria implementação concreta a partir do modo
    static std::unique_ptr<Subproblem> create (const TSCFLInstance &inst, Mode mode);

    // Dado (a_vals, b_vals) da solução atual do mestre, resolve o subproblema.
    // Atualiza: theta, rhs, coef_a, coef_b
    virtual void solve (const IloNumArray &a_vals, const IloNumArray &b_vals) = 0;

    // Heurística primal 1:
    // Dado (a_frac, b_frac) da solução atual do mestre, encontra a_int e b_int
    // e resolve o subproblema.
    // Atualiza: theta, rhs, coef_a, coef_b
    // Retorna: valor da solução primal heurística encontrada
    IloNum solve_primal_heuristic (
        const IloNumArray &a_frac, const IloNumArray &b_frac, IloNumArray &a_int, IloNumArray &b_int
    );

    // Heurística primal 2:
    // Lê (a,b) fracionários direto do modelo (var_a, var_b) e chama a função acima.
    IloNum solve_primal_heuristic (
        const IloCplex &cpx, const IloNumVarArray &var_a, const IloNumVarArray &var_b,
        IloNumArray &a_int, IloNumArray &b_int
    );
};
