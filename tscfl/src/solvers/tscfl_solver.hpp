/*
COS888

tscfl_solver.hpp

Gabriel Braun, 2025
*/

#pragma once

#include <ilcplex/ilocplex.h>

#include "tscfl_instance.hpp"

// SOLVER TSCFL
class TSCFLSolver
{
  protected:
    const IloEnv &env;
    const TSCFLInstance &inst;

  public:
    // Resultados
    IloNumArray a;     // a[i]
    IloNumArray b;     // b[j]
    IloNumVarMatrix x; // x[i][j]
    IloNumVarMatrix y; // y[j][k]

    // Estatísticas
    IloNum lb{ 0.0 };
    IloNum ub{ IloInfinity };
    IloNum gap{ IloInfinity };
    IloNum time{ 0.0 };
    IloInt64 iter{ 0 };
    IloInt64 nodes{ 0 };
    IloAlgorithm::Status status{ IloAlgorithm::Unknown };

    explicit TSCFLSolver(const TSCFLInstance &inst_);

    virtual ~TSCFLSolver() = default;

    // Resolve a instância.
    virtual void solve(bool log_output = true, double time_limit = -1.0) = 0;

  protected:
    // Atualiza: gap (a partir de lb/ub) e status
    void updateGap();

    // Imprime: resumo padronizado
    void printSummary(const char *tag) const;
};
