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
    IloEnv env;
    IloModel model;
    IloCplex cplex;

    IloBoolVarArray a; // a_i  = abre planta i
    IloBoolVarArray b; // b_j  = abre depósito j
    IloNumVarArray x;  // x_ij = fluxo planta i -> depósito j
    IloNumVarArray y;  // y_jk = fluxo depósito j -> cliente k

    // Acessores para variáveis de fluxo.
    inline IloNumVar &X(int i, int j) { return x[idx2(i, j, inst.nJ)]; }
    inline IloNumVar &Y(int j, int k) { return y[idx2(j, k, inst.nK)]; }

public:
    explicit TSCFLSolverCplex(const TSCFLInstance &inst_)
        : inst(inst_),
          env(),
          model(env),
          cplex(env),
          a(env, inst_.nI),
          b(env, inst_.nJ),
          x(env, inst_.nI * inst_.nJ, 0.0, IloInfinity, ILOFLOAT),
          y(env, inst_.nJ * inst_.nK, 0.0, IloInfinity, ILOFLOAT)
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
        env.end();
    }

private:
    void build_model()
    {
        const int nI = inst.nI;
        const int nJ = inst.nJ;
        const int nK = inst.nK;

        // RESTRIÇÕES
        // Capacidade das plantas
        for (int i = 0; i < nI; ++i)
        {
            IloExpr expr(env);
            for (int j = 0; j < nJ; ++j)
                expr += X(i, j);
            model.add(expr <= inst.p[i] * a[i]);
            expr.end();
        }

        // Capacidade dos depósitos
        for (int j = 0; j < nJ; ++j)
        {
            IloExpr expr(env);
            for (int k = 0; k < nK; ++k)
                expr += Y(j, k);
            model.add(expr <= inst.q[j] * b[j]);
            expr.end();
        }

        // Balanço nos depósitos
        for (int j = 0; j < nJ; ++j)
        {
            IloExpr expr(env);
            for (int i = 0; i < nI; ++i)
                expr += X(i, j);
            for (int k = 0; k < nK; ++k)
                expr -= Y(j, k);
            model.add(expr == 0.0);
            expr.end();
        }

        // Demanda dos clientes
        for (int k = 0; k < nK; ++k)
        {
            IloExpr expr(env);
            for (int j = 0; j < nJ; ++j)
                expr += Y(j, k);
            model.add(expr == inst.r[k]);
            expr.end();
        }

        // Capacidade agregada (viabilidade extra)
        for (int i = 0; i < nI; ++i)
            for (int j = 0; j < nJ; ++j)
                model.add(X(i, j) <= inst.q[j] * b[j]);

        for (int j = 0; j < nJ; ++j)
            for (int k = 0; k < nK; ++k)
                model.add(Y(j, k) <= inst.r[k] * b[j]);

        // FUNÇÃO OBJETIVO
        IloExpr obj(env);

        // Custos fixos plantas
        for (int i = 0; i < nI; ++i)
            obj += inst.f[i] * a[i];

        // Custos fixos depósitos
        for (int j = 0; j < nJ; ++j)
            obj += inst.g[j] * b[j];

        // Custos de fluxo planta -> depósito
        for (int i = 0; i < nI; ++i)
            for (int j = 0; j < nJ; ++j)
                obj += inst.C(i, j) * X(i, j);

        // Custos de fluxo depósito -> cliente
        for (int j = 0; j < nJ; ++j)
            for (int k = 0; k < nK; ++k)
                obj += inst.D(j, k) * Y(j, k);

        model.add(IloMinimize(env, obj));
        obj.end();
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
        bool ok = cplex.solve();
        status = cplex.getStatus();

        // Estatísticas
        gap = cplex.getMIPRelativeGap();
        time = cplex.getTime();
        nodes = cplex.getNnodes64();

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
