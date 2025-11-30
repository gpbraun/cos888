/*
COS888

Classe base abstrata para os Workers de Benders no TSCFL.

Gabriel Braun, 2025
*/

#pragma once

#include <ilcplex/ilocplex.h>

#include "tscfl_instance.hpp"

ILOSTLBEGIN

// SOLVER DO SUBPROBLEMA DE BENDERS: Base
class Worker
{
public:
    const TSCFLInstance &inst;

    // Saída do subproblema
    double theta{0.0};  // valor ótimo do subproblema
    double rhs{0.0};    // termo independente do corte
    IloNumArray coef_a; // coeficientes multiplicando a_i (no env do mestre)
    IloNumArray coef_b; // coeficientes multiplicando b_j

    explicit Worker(const TSCFLInstance &inst_)
        : inst(inst_),
          coef_a(inst_.env, inst_.nI),
          coef_b(inst_.env, inst_.nJ)
    {
    }

    virtual ~Worker() = default;

    // Dado (a_vals, b_vals) da solução atual do mestre, resolve o subproblema.
    // Atualiza: theta, rhs, coef_a, coef_b
    virtual void solve(const IloNumArray &a_vals, const IloNumArray &b_vals) = 0;
};
