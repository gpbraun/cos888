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
    IloEnv &env;
    const TSCFLInstance &inst;

  public:
    // Resultados
    IloNum lb;
    IloNum ub;
    IloNum gap;
    IloNum time;
    IloInt64 iter;
    IloInt64 nodes;
    IloAlgorithm::Status status;

    explicit TSCFLSolver(const TSCFLInstance &inst_);
    virtual ~TSCFLSolver() = default;

    // Interface comum
    virtual bool solve(bool log_output = true, double time_limit = -1.0) = 0;

  protected:
    // Atualiza o gap a partir de lb/ub
    void update_gap();

    // Atualiza o status se ainda estiver Unknown e tivermos uma solução viável
    void update_status();

    // Imprime resumo padronizado
    void print_summary(const char *tag) const;
};