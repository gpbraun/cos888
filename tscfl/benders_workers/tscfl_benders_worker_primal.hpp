/*
COS888

WorkerPrimal: resolve o subproblema primal do TSCFL para uso na decomposição de Benders.

Gabriel Braun, 2025
*/

#pragma once

#include <ilcplex/ilocplex.h>
#include <stdexcept>

#include "tscfl_benders_base_worker.hpp"

ILOSTLBEGIN

// SOLVER DO SUBPROBLEMA DE BENDERS: Primal
class WorkerPrimal : public Worker
{
private:
    IloEnv env;
    IloModel model;
    IloCplex cplex;

    // Variáveis de fluxo
    IloNumVarMatrix x; // x[i][j] = fluxo planta i -> depósito j
    IloNumVarMatrix y; // y[j][k] = fluxo depósito j -> cliente k

    // Restrições
    IloRangeArray constr_l1; // constr_l1[i] = restrição de capacidade da planta i
    IloRangeArray constr_l2; // constr_l2[j] = restrição de capacidade do depósito j
    IloRangeArray constr_m1; // constr_m1[j] = restrição de balanço nos depósitos j
    IloRangeArray constr_m2; // constr_m2[k] = restrição de demanda do cliente k

public:
    explicit WorkerPrimal(const TSCFLInstance &inst_)
        : Worker(inst_),
          env(),
          model(env),
          cplex(env),
          x(env, inst_.nI, inst_.nJ),
          y(env, inst_.nJ, inst_.nK),
          constr_l1(env, inst_.nI),
          constr_l2(env, inst_.nJ),
          constr_m1(env, inst_.nJ),
          constr_m2(env, inst_.nK)
    {
        build_base_model();
        cplex.extract(model);

        // Parâmetros do CPLEX (subproblema)
        cplex.setParam(IloCplex::Param::Threads, 1);
        cplex.setParam(IloCplex::Param::Preprocessing::Reduce, 0);
        cplex.setParam(IloCplex::Param::RootAlgorithm, IloCplex::Primal);

        // Subproblema silencioso por padrão
        cplex.setOut(env.getNullStream());
        cplex.setWarning(env.getNullStream());
    }

    ~WorkerPrimal() override
    {
        cplex.end();
        model.end();
        env.end();
    }

private:
    // Constrói o modelo base do subproblema dual
    void build_base_model()
    {
        // RESTRIÇÕES DO SUBPROBLEMA PRIMAL
        // Capacidade das plantas
        for (int i = 0; i < inst.nI; ++i)
        {
            constr_l1[i] = IloRange(env, -IloInfinity, -IloSum(x[i]), IloInfinity);
            model.add(constr_l1[i]);
        }
        // Capacidade dos depósitos
        for (int j = 0; j < inst.nJ; ++j)
        {
            constr_l2[j] = IloRange(env, -IloInfinity, -IloSum(y[j]), IloInfinity);
            model.add(constr_l2[j]);
        }
        // Balanço nos depósitos
        for (int j = 0; j < inst.nJ; ++j)
        {
            constr_m1[j] = IloRange(env, 0.0, IloSum(x.col(j)) - IloSum(y[j]), 0.0);
            model.add(constr_m1[j]);
        }
        // Demanda dos clientes
        for (int k = 0; k < inst.nK; ++k)
        {
            constr_m2[k] = IloRange(env, inst.r[k], IloSum(y.col(k)), IloInfinity);
            model.add(constr_m2[k]);
        }

        // FUNÇÃO OBJETIVO
        IloExpr obj_expr(env);

        obj_expr += IloMatScalProd(inst.c, x) + IloMatScalProd(inst.d, y);

        IloObjective obj = IloMinimize(env, obj_expr);
        model.add(obj);
        obj_expr.end();
    }

    // Atualiza o lado direito das restrições que dependem de (a,b)
    void set_constraints(const IloNumArray &a_vals, const IloNumArray &b_vals)
    {
        // Capacidade das plantas
        for (int i = 0; i < inst.nI; ++i)
            constr_l1[i].setBounds(-inst.p[i] * a_vals[i], IloInfinity);

        // Capacidade dos depósitos
        for (int j = 0; j < inst.nJ; ++j)
            constr_l2[j].setBounds(-inst.q[j] * b_vals[j], IloInfinity);
    }

public:
    // Dado (a_vals, b_vals) da solução atual do mestre, resolve o subproblema.
    // Atualiza: theta, rhs, coef_a, coef_b
    void solve(const IloNumArray &a_vals, const IloNumArray &b_vals) override
    {
        // 1) Atualiza as restrições dependentes de (a,b)
        set_constraints(a_vals, b_vals);

        // 2) Resolve o LP primal
        if (!cplex.solve())
            throw std::runtime_error("WorkerPrimal: CPLEX failed to solve primal subproblem.");

        theta = cplex.getObjValue();

        // 3) Lê as variáveis duais
        IloNumArray l1(env, inst.nI), l2(env, inst.nJ), m2(env, inst.nK);
        cplex.getDuals(l1, constr_l1);
        cplex.getDuals(l2, constr_l2);
        cplex.getDuals(m2, constr_m2);

        // 4) Calcula os coeficientes do corte
        for (int i = 0; i < inst.nI; ++i)
            coef_a[i] = -inst.p[i] * l1[i];

        for (int j = 0; j < inst.nJ; ++j)
            coef_b[j] = -inst.q[j] * l2[j];

        rhs = IloScalProd(inst.r, m2);

        l1.end();
        l2.end();
        m2.end();
    }
};
