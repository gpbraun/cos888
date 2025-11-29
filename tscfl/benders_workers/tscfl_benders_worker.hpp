/*
COS888

Classe base abstrata para os "workers" de Benders no TSCFL.

Cada Worker:
- recebe (a, b) fixos (decisões de 1º estágio),
- resolve um subproblema (dual, primal, rede, ...),
- devolve os coeficientes do corte de Benders:

    theta  = valor ótimo do subproblema
    coef_a = coeficientes de a_i
    coef_b = coeficientes de b_j
    rhs    = termo independente

Gabriel Braun, 2025
*/

#pragma once

#include "tscfl_instance.hpp"

class Worker
{
public:
    // Mantemos uma referência constante para a instância do problema.
    const TSCFLInstance &inst;

    explicit Worker(const TSCFLInstance &inst_) : inst(inst_) {}

    // Precisamos de um destrutor virtual para polimorfismo.
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