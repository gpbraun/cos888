/*
COS888

Resolve o TSCFL com CPLEX.

Gabriel Braun, 2025
*/

#pragma once

#include <ilcplex/ilocplex.h>
#include <iostream>
#include <string>

#include "tscfl_instance.hpp"

ILOSTLBEGIN

// SOLVER TSCFL: CPLEX
class TSCFLSolverCplex
{
public:
    const TSCFLInstance &inst;

    // Parâmetros de resultados do solver
    double lb{0.0};
    double ub{0.0};
    double gap{0.0};
    double time{0.0};
    IloInt64 nodes{0};
    IloAlgorithm::Status status{IloAlgorithm::Unknown};

private:
    IloModel model;
    IloCplex cplex;

    IloBoolVarArray a; // a_i  = abre planta i
    IloBoolVarArray b; // b_j  = abre depósito j
    IloNumVarMatrix x; // x_ij = fluxo planta i -> depósito j
    IloNumVarMatrix y; // y_jk = fluxo depósito j -> cliente k

public:
    explicit TSCFLSolverCplex(const TSCFLInstance &inst_)
        : inst(inst_),
          model(inst_.env),
          cplex(inst_.env),
          a(inst_.env, inst_.nI),
          b(inst_.env, inst_.nJ),
          x(inst_.env, inst_.nI, inst_.nJ),
          y(inst_.env, inst_.nJ, inst_.nK)
    {
        build_model();
        cplex.extract(model);

        // Parâmetros CPLEX
        cplex.setParam(IloCplex::Param::Threads, 1);
        cplex.setParam(IloCplex::Param::Preprocessing::Reduce, 0);
        cplex.setParam(IloCplex::Param::MIP::Tolerances::MIPGap, MIP_GAP);
        cplex.setParam(IloCplex::Param::Benders::Strategy, IloCplex::BendersFull);
    }

    ~TSCFLSolverCplex()
    {
        cplex.end();
        model.end();
    }

private:
    void build_model()
    {
        IloEnv env = inst.env;

        // RESTRIÇÕES
        // Capacidade das plantas
        for (int i = 0; i < inst.nI; ++i)
            model.add(IloSum(x[i]) <= inst.p[i] * a[i]);

        // Capacidade dos depósitos
        for (int j = 0; j < inst.nJ; ++j)
            model.add(IloSum(y[j]) <= inst.q[j] * b[j]);

        // Balanço nos depósitos
        for (int j = 0; j < inst.nJ; ++j)
            model.add(IloSum(x.col(j)) - IloSum(y[j]) == 0.0);

        // Demanda dos clientes
        for (int k = 0; k < inst.nK; ++k)
            model.add(IloSum(y.col(k)) == inst.r[k]);

        // Capacidade agregada (viabilidade extra)
        for (int i = 0; i < inst.nI; ++i)
            for (int j = 0; j < inst.nJ; ++j)
                model.add(x[i][j] <= inst.q[j] * b[j]);

        for (int j = 0; j < inst.nJ; ++j)
            for (int k = 0; k < inst.nK; ++k)
                model.add(y[j][k] <= inst.r[k] * b[j]);

        // FUNÇÃO OBJETIVO
        IloExpr obj_expr(env);
        // Custos fixos
        obj_expr += IloScalProd(inst.f, a) + IloScalProd(inst.g, b);
        // Custos de fluxo
        obj_expr += IloMatProd(inst.c, x) + IloMatProd(inst.d, y);

        IloObjective obj = IloMinimize(env, obj_expr);
        model.add(obj);
        obj_expr.end();
    }

public:
    bool solve(bool log_output = true, double time_limit = -1.0)
    {
        IloEnv env = inst.env;

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
        bool ok = cplex.solve();
        status = cplex.getStatus();

        // Estatísticas
        gap = cplex.getMIPRelativeGap();
        nodes = cplex.getNnodes64();
        time = cplex.getTime();

        if (ok)
        {
            ub = cplex.getObjValue();
            lb = cplex.getBestObjValue();

            std::cout << "\n[CPLEX] Solved.\n";
            std::cout << "UB     = " << ub << "\n";
            std::cout << "LB     = " << lb << "\n";
            std::cout << "status = " << status << "\n";
            std::cout << "gap    = " << gap << "\n";
            std::cout << "nodes  = " << nodes << "\n";
            std::cout << "time   = " << time << "s\n";
        }
        else
        {
            std::cerr << "\n[CPLEX] No solution. status = " << status << "\n";
            std::cerr << "nodes = " << nodes << "\n";
            std::cerr << "time  = " << time << " s\n";
        }

        return ok;
    }
};
