/*
COS888

WorkerNet: resolve o subproblema primal do TSCFL usando o
otimizador de redes do CPLEX (RootAlgorithm = Network) e
constrói o corte de Benders a partir dos duais das restrições.

Gabriel Braun, 2025
*/

#pragma once

#include <ilcplex/ilocplex.h>
#include <stdexcept>

#include "tscfl_benders_base_worker.hpp"

ILOSTLBEGIN

class WorkerNet : public Worker
{
private:
    IloModel model;
    IloCplex cplex;

    // Variáveis de fluxo
    IloNumVarMatrix x; // x[i][j] = fluxo planta i -> depósito j
    IloNumVarMatrix y; // y[j][k] = fluxo depósito j -> cliente k

    // Restrições (na convenção dos duais do WorkerDual)
    IloRangeArray constr_l1; // constr_l1[i] = capacidade da planta i
    IloRangeArray constr_l2; // constr_l2[j] = capacidade do depósito j
    IloRangeArray constr_m1; // constr_m1[j] = balanço de fluxo no depósito j
    IloRangeArray constr_m2; // constr_m2[k] = demanda do cliente k

public:
    explicit WorkerNet(const TSCFLInstance &inst_)
        : Worker(inst_),
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

        // Parâmetros do CPLEX (subproblema em rede)
        cplex.setParam(IloCplex::Param::Threads, 1);
        cplex.setParam(IloCplex::Param::Preprocessing::Reduce, 0);
        cplex.setParam(IloCplex::Param::RootAlgorithm, IloCplex::Network);

        // Subproblema silencioso por padrão
        cplex.setOut(env.getNullStream());
        cplex.setWarning(env.getNullStream());
    }

    ~WorkerNet() override
    {
        cplex.end();
        model.end();
    }

private:
    // Constrói o modelo base do subproblema primal
    void build_base_model()
    {
        const int nI = inst.nI;
        const int nJ = inst.nJ;
        const int nK = inst.nK;

        // -----------------------------------------------------------------
        // RESTRIÇÕES DO SUBPROBLEMA PRIMAL
        // -----------------------------------------------------------------

        // Capacidade das plantas:
        //   sum_j x_ij <= p_i a_i
        // Reescrevemos como:
        //   -sum_j x_ij >= -p_i a_i
        // para ficar na forma expr >= LB, com LB dependente de a_i.
        for (int i = 0; i < nI; ++i)
        {
            constr_l1[i] = IloRange(env, -IloInfinity, -IloSum(x[i]), IloInfinity);
            model.add(constr_l1[i]);
        }

        // Capacidade dos depósitos:
        //   sum_k y_jk <= q_j b_j
        // Reescrito como:
        //   -sum_k y_jk >= -q_j b_j
        for (int j = 0; j < nJ; ++j)
        {
            constr_l2[j] = IloRange(env, -IloInfinity, -IloSum(y[j]), IloInfinity);
            model.add(constr_l2[j]);
        }

        // Balanço nos depósitos:
        //   sum_i x_ij - sum_k y_jk = 0
        for (int j = 0; j < nJ; ++j)
        {
            constr_m1[j] = IloRange(env, 0.0, IloSum(x.col(j)) - IloSum(y[j]), 0.0);
            model.add(constr_m1[j]);
        }

        // Demanda dos clientes:
        //   sum_j y_jk >= r_k
        for (int k = 0; k < nK; ++k)
        {
            constr_m2[k] = IloRange(env, inst.r[k], IloSum(y.col(k)), IloInfinity);
            model.add(constr_m2[k]);
        }

        // -----------------------------------------------------------------
        // FUNÇÃO OBJETIVO:
        //   min sum_{i,j} c_ij x_ij + sum_{j,k} d_jk y_jk
        // -----------------------------------------------------------------
        IloExpr obj_expr(env);
        obj_expr += IloMatProd(inst.c, x) + IloMatProd(inst.d, y);

        IloObjective obj = IloMinimize(env, obj_expr);
        model.add(obj);
        obj_expr.end();
    }

    // Atualiza o lado direito das restrições que dependem de (a,b)
    void set_constraints(const IloNumArray &a_vals, const IloNumArray &b_vals)
    {
        // Capacidade das plantas:
        //   -sum_j x_ij >= -p_i a_i
        for (int i = 0; i < inst.nI; ++i)
            constr_l1[i].setBounds(-inst.p[i] * a_vals[i], IloInfinity);

        // Capacidade dos depósitos:
        //   -sum_k y_jk >= -q_j b_j
        for (int j = 0; j < inst.nJ; ++j)
            constr_l2[j].setBounds(-inst.q[j] * b_vals[j], IloInfinity);
    }

public:
    // Dado (a_vals, b_vals) da solução atual do mestre, resolve o subproblema.
    // Atualiza: theta, rhs, coef_a, coef_b
    //
    // Corte de Benders:
    //   eta >= rhs + sum_i coef_a[i] * a_i + sum_j coef_b[j] * b_j
    //
    void solve(const IloNumArray &a_vals, const IloNumArray &b_vals) override
    {
        // 1) Atualiza as restrições que dependem de (a,b)
        set_constraints(a_vals, b_vals);

        // 2) Resolve o LP (com otimizador de rede)
        if (!cplex.solve())
            throw std::runtime_error("WorkerNet: CPLEX failed to solve network subproblem.");

        theta = cplex.getObjValue();

        // 3) Lê as variáveis duais relevantes:
        //    l1_i: duais das capacidades das plantas (constr_l1)
        //    l2_j: duais das capacidades dos depósitos (constr_l2)
        //    m2_k: duais das demandas dos clientes   (constr_m2)
        IloNumArray l1(env), l2(env), m2(env);

        cplex.getDuals(l1, constr_l1);
        cplex.getDuals(l2, constr_l2);
        cplex.getDuals(m2, constr_m2);

        // 4) Calcula os coeficientes do corte
        //
        // Plantas:
        //   -sum_j x_ij >= -p_i a_i  => RHS_i = -p_i a_i
        // Contribuição dual: sum_i RHS_i * l1_i = sum_i (-p_i a_i) l1_i
        // => coef_a[i] = d/d a_i [(-p_i a_i) l1_i] = -p_i l1_i
        for (int i = 0; i < inst.nI; ++i)
            coef_a[i] = -inst.p[i] * l1[i];

        // Depósitos:
        //   -sum_k y_jk >= -q_j b_j  => coef_b[j] = -q_j l2_j
        for (int j = 0; j < inst.nJ; ++j)
            coef_b[j] = -inst.q[j] * l2[j];

        // Demandas:
        //   sum_j y_jk >= r_k  => rhs = sum_k r_k m2_k
        rhs = IloScalProd(inst.r, m2);

        l1.end();
        l2.end();
        m2.end();
    }
};
