/*
COS888

Classe base abstrata para os solvers do subproblema de fluxo mínimo do TSCFL.

Gabriel Braun, 2025
*/

#pragma once

#include <ilcplex/ilocplex.h>

#include "tscfl_instance.hpp"

ILOSTLBEGIN

// SOLVER DO SUBPROBLEMA: Base
class Subproblem
{
public:
    const TSCFLInstance &inst;

    // Saída do subproblema
    double theta{0.0};  // valor ótimo do subproblema
    double rhs{0.0};    // termo independente do corte
    IloNumArray coef_a; // coeficientes multiplicando a_i
    IloNumArray coef_b; // coeficientes multiplicando b_j

    explicit Subproblem(const TSCFLInstance &inst_)
        : inst(inst_),
          coef_a(inst_.env, inst_.nI),
          coef_b(inst_.env, inst_.nJ)
    {
    }

    virtual ~Subproblem() = default;

    // Dado (a_vals, b_vals) da solução atual do mestre, resolve o subproblema.
    // Atualiza: theta, rhs, coef_a, coef_b
    virtual void solve(const IloNumArray &a_vals, const IloNumArray &b_vals) = 0;
};
