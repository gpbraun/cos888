// src/tscfl_solver_base.hpp
#pragma once

#include <ilcplex/ilocplex.h>

#include "tscfl_instance.hpp"

ILOSTLBEGIN

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
    IloNum lb{ 0.0 };
    IloNum ub{ IloInfinity };
    IloNum gap{ IloInfinity };
    IloNum time{ 0.0 };
    IloInt64 nodes{ 0 };
    IloAlgorithm::Status status{ IloAlgorithm::Unknown };

    explicit TSCFLSolver (const TSCFLInstance &inst_) : env (inst_.env), inst (inst_) {}

    virtual ~TSCFLSolver () = default;

    // Interface comum
    virtual bool solve (bool log_output = true, double time_limit = -1.0) = 0;

  protected:
    void
    print_summary (const char *tag) const
    {
        std::cout << "\n\n"
                  << "[" << tag << "] Solver finalizado.\n\n"
                  << "status = " << status
                  << "\n"
                  // nodes
                  << std::fixed << std::setprecision (0) << "nodes  = " << nodes
                  << "\n"
                  // tempo
                  << std::fixed << std::setprecision (1) << "time   = " << time
                  << " s\n"
                  // LB, UB
                  << std::fixed << std::setprecision (0) << "LB     = " << lb << "\n"
                  << "UB     = " << ub
                  << "\n"
                  // gap
                  << std::scientific << std::setprecision (2) << "gap    = " << gap << "\n"
                  << std::defaultfloat;
    }
};
