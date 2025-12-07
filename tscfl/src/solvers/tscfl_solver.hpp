/*
COS888

tscfl_solver.hpp

Gabriel Braun, 2025
*/

#pragma once

#include <ilcplex/ilocplex.h>

#include "tscfl_instance.hpp"

// SOLVER
class TSCFLSolver
{
  protected:
    const IloEnv &env;
    const TSCFLInstance &inst;

  public:
    // Saída
    IloNumArray a;  // a[i]
    IloNumArray b;  // b[j]
    IloNumMatrix x; // x[i][j]
    IloNumMatrix y; // y[j][k]

    IloNum lb{ 0.0 };
    IloNum ub{ IloInfinity };

    // Estatísticas
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
    // Imprime: resumo padronizado.
    void printSummary(const char *tag) const;

    // Atualiza: gap (a partir de lb/ub) e status.
    void updateGap();

    // Atualiza: x e y a partir das variáveis do CPLEX.
    void
    updateFlows(const IloCplex &cplex, const IloNumVarMatrix &var_x, const IloNumVarMatrix &var_y);

    // Atualiza: x e y a partir dos valores de a e b usando o subproblema primal
    void updateFlows();
};
