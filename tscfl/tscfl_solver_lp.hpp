/*
COS888

Resolve o TSCFL (relaxação LP) com CPLEX.

Gabriel Braun, 2025
*/

#pragma once

#include "utils/utils.hpp"

// SOLVER TSCFL: Relaxação LP com CPLEX
class TSCFLSolverLP
{
protected:
    IloEnv &env;
    const TSCFLInstance &inst;

private:
    IloModel model;
    IloCplex cplex;

    IloNumVarArray var_a;  // a[i]
    IloNumVarArray var_b;  // b[j]
    IloNumVarMatrix var_x; // x[i][j]
    IloNumVarMatrix var_y; // y[j][k]

public:
    // Resultados:
    IloNum opt{0.0};
    IloAlgorithm::Status status{IloAlgorithm::Unknown};

    explicit TSCFLSolverLP(const TSCFLInstance &inst_)
        : env(inst_.env),
          inst(inst_),
          model(inst_.env),
          cplex(inst_.env),
          var_a(inst_.env, inst_.nI, 0.0, 1.0),
          var_b(inst_.env, inst_.nJ, 0.0, 1.0),
          var_x(inst_.env, inst_.nI, inst_.nJ),
          var_y(inst_.env, inst_.nJ, inst_.nK)
    {
        build_model();
        cplex.extract(model);

        // Parâmetros CPLEX
        cplex.setParam(IloCplex::Param::Threads, 1);
        cplex.setParam(IloCplex::Param::Preprocessing::Presolve, 0);
        cplex.setParam(IloCplex::Param::Preprocessing::Aggregator, 0);
        cplex.setParam(IloCplex::Param::RootAlgorithm, IloCplex::Primal);
        cplex.setParam(IloCplex::Param::Benders::Strategy, IloCplex::BendersFull);
    }

    ~TSCFLSolverLP()
    {
        cplex.end();
        model.end();
    }

private:
    void build_model()
    {
        // RESTRIÇÕES
        // Capacidade das plantas
        for (int i = 0; i < inst.nI; ++i)
            model.add(IloSum(var_x[i]) <= inst.p[i] * var_a[i]);

        // Capacidade dos depósitos
        for (int j = 0; j < inst.nJ; ++j)
            model.add(IloSum(var_y[j]) <= inst.q[j] * var_b[j]);

        // Balanço nos depósitos
        for (int j = 0; j < inst.nJ; ++j)
            model.add(IloSum(var_x.col(j)) - IloSum(var_y[j]) == 0.0);

        // Demanda dos clientes
        for (int k = 0; k < inst.nK; ++k)
            model.add(IloSum(var_y.col(k)) == inst.r[k]);

        // Capacidade agregada (viabilidade extra)
        for (int i = 0; i < inst.nI; ++i)
            for (int j = 0; j < inst.nJ; ++j)
                model.add(var_x[i][j] <= inst.q[j] * var_b[j]);

        for (int j = 0; j < inst.nJ; ++j)
            for (int k = 0; k < inst.nK; ++k)
                model.add(var_y[j][k] <= inst.r[k] * var_b[j]);

        // FUNÇÃO OBJETIVO
        IloExpr obj_expr(env);

        obj_expr += IloScalProd(inst.f, var_a) + IloScalProd(inst.g, var_b);
        obj_expr += IloMatScalProd(inst.c, var_x) + IloMatScalProd(inst.d, var_y);

        IloObjective obj = IloMinimize(env, obj_expr);
        model.add(obj);
        obj_expr.end();
    }

public:
    bool solve(bool log_output = true, double time_limit = -1.0)
    {
        // Controle de log
        if (log_output)
        {
            cplex.setOut(env.out());
            cplex.setWarning(env.out());
        }
        else
        {
            cplex.setOut(env.getNullStream());
            cplex.setWarning(env.getNullStream());
        }

        if (time_limit > 0.0)
            cplex.setParam(IloCplex::Param::TimeLimit, time_limit);

        // Solve
        IloBool ok = cplex.solve();
        status = cplex.getStatus();

        // Estatísticas
        if (ok)
        {
            opt = cplex.getObjValue();
            std::cerr << "\n[LP] OPT = " << opt << "\n";
        }
        else
        {
            std::cerr << "\n[LP] Sem solução. status = " << status << "\n";
        }

        return ok;
    }
};
