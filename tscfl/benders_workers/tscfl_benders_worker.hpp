/*
COS888

Classe base abstrata para os Workers de Benders no TSCFL.

Gabriel Braun, 2025
*/

#pragma once

#include "tscfl_instance.hpp"

class Worker
{
public:
    const TSCFLInstance &inst;

    explicit Worker(const TSCFLInstance &inst_) : inst(inst_) {}

    virtual ~Worker() = default;

    // Interface comum a todos os Workers.
    // Dado (a, b), resolve o subproblema e monta o corte:
    //
    //   theta  = valor ótimo do subproblema (2º estágio)
    //   coef_a = coeficientes multiplicando a_i
    //   coef_b = coeficientes multiplicando b_j
    //   rhs    = termo independente
    //
    // O corte fica sempre no formato:
    //
    //   eta >= rhs + sum_i coef_a[i] * a_i + sum_j coef_b[j] * b_j
    //
    virtual void solve(
        const Vec &a,
        const Vec &b,
        double &theta,
        Vec &coef_a,
        Vec &coef_b,
        double &rhs) = 0;
};