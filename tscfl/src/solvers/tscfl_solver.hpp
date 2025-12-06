/*
COS888

tscfl_solver.hpp

Gabriel Braun, 2025
*/

#pragma once

#include <ilcplex/ilocplex.h>

#include "tscfl_instance.hpp"

// =====================================================================
//  BASE: TSCFLSolver
// =====================================================================

class TSCFLSolver
{
  protected:
    IloEnv &env;
    const TSCFLInstance &inst;

  public:
    // Resultados comuns
    IloNum lb;   // lower bound
    IloNum ub;   // upper bound
    IloNum gap;  // (ub - lb)/max(1,|ub|)
    IloNum time; // tempo total (s)
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