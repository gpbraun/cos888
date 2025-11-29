/*
COS888

WorkerPrimal: resolve o subproblema primal do TSCFL para uso na decomposição de Benders.

Gabriel Braun, 2025
*/

#pragma once

#include <ilcplex/ilocplex.h>
#include <stdexcept>

#include "tscfl_benders_worker.hpp"

ILOSTLBEGIN

class WorkerPrimal : public Worker
{
private:
    IloModel model;
    IloCplex cplex;

    // Variáveis de fluxo
    IloNumVarMatrix x; // x_ij = fluxo planta i -> depósito j
    IloNumVarMatrix y; // y_jk = fluxo depósito j -> cliente k

    // Restrições
    IloRangeArray plantCap; // capacidade das plantas
    IloRangeArray depotCap; // capacidade dos depósitos
    IloRangeArray flowBal;  // balanço de fluxo nos depósitos
    IloRangeArray demand;   // atendimento da demanda dos clientes

public:
    explicit WorkerPrimal(const TSCFLInstance &inst_)
        : Worker(inst_),
          model(env),
          cplex(env),
          x(env, inst_.nI, inst_.nJ),
          y(env, inst_.nJ, inst_.nK),
          plantCap(env, inst_.nI),
          depotCap(env, inst_.nJ),
          flowBal(env, inst_.nJ),
          demand(env, inst_.nK)
    {
        build_model();
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
    }

private:
    void build_model()
    {
        // -----------------------------------------------------------------
        // Capacidade das plantas:
        //   sum_j x_ij <= p_i a_i
        // Reescrevemos como: -sum_j x_ij >= -p_i a_i
        // para ter forma "expr >= LB", onde LB depende de a_i.
        // O LB será ajustado em set_constraints().
        // -----------------------------------------------------------------
        for (int i = 0; i < inst.nI; ++i)
        {
            IloExpr e(env);
            e -= IloSum(x[i]); // -sum_j x_ij

            plantCap[i] = IloRange(env, -IloInfinity, e, IloInfinity);
            model.add(plantCap[i]);
            e.end();
        }

        // -----------------------------------------------------------------
        // Capacidade dos depósitos:
        //   sum_k y_jk <= q_j b_j
        // Reescrevemos como: -sum_k y_jk >= -q_j b_j
        // LB ajustado em set_constraints().
        // -----------------------------------------------------------------
        for (int j = 0; j < inst.nJ; ++j)
        {
            IloExpr e(env);
            e -= IloSum(y[j]); // -sum_k y_jk

            depotCap[j] = IloRange(env, -IloInfinity, e, IloInfinity);
            model.add(depotCap[j]);
            e.end();
        }

        // -----------------------------------------------------------------
        // Balanço nos depósitos:
        //   sum_i x_ij - sum_k y_jk = 0
        // RHS fixo (=0), não depende de (a,b).
        // -----------------------------------------------------------------
        for (int j = 0; j < inst.nJ; ++j)
        {
            IloExpr e(env);
            e += IloSum(x.col(j)); // sum_i x_ij
            e -= IloSum(y[j]);     // sum_k y_jk

            flowBal[j] = IloRange(env, 0.0, e, 0.0);
            model.add(flowBal[j]);
            e.end();
        }

        // -----------------------------------------------------------------
        // Demanda dos clientes:
        //   sum_j y_jk >= r_k
        // RHS = r_k, constante, indep. de (a,b).
        // -----------------------------------------------------------------
        for (int k = 0; k < inst.nK; ++k)
        {
            IloExpr e(env);
            e += IloSum(y.col(k)); // sum_j y_jk

            demand[k] = IloRange(env, inst.r[k], e, IloInfinity);
            model.add(demand[k]);
            e.end();
        }

        // FUNÇÃO OBJETIVO
        IloExpr obj_expr(env);
        obj_expr += IloMatProd(inst.c, x) + IloMatProd(inst.d, y);
        IloObjective obj = IloMinimize(env, obj_expr);
        model.add(obj);
        obj_expr.end();
    }

    // Atualiza o lado direito das restrições que dependem de (a,b)
    void set_constraints(const IloNumArray &a_vals, const IloNumArray &b_vals)
    {
        //   -sum_j x_ij >= -p_i a_i   -> LB = -p_i a_i
        for (int i = 0; i < inst.nI; ++i)
            plantCap[i].setBounds(-inst.p[i] * a_vals[i], IloInfinity);
        //   -sum_k y_jk >= -q_j b_j   -> LB = -q_j b_j
        for (int j = 0; j < inst.nJ; ++j)
            depotCap[j].setBounds(-inst.q[j] * b_vals[j], IloInfinity);
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
        IloNumArray duPlant(env), duDepot(env), duDemand(env);

        cplex.getDuals(duPlant, plantCap);
        cplex.getDuals(duDepot, depotCap);
        cplex.getDuals(duDemand, demand);

        // 4) Calcula os coeficientes do corte
        for (int i = 0; i < inst.nI; ++i)
            coef_a[i] = -inst.p[i] * duPlant[i];

        // Depósitos:
        //   -sum_k y_jk >= -q_j b_j  => coef_b[j] = -q_j l2_j
        for (int j = 0; j < inst.nJ; ++j)
            coef_b[j] = -inst.q[j] * duDepot[j];

        // Demandas:
        //   sum_j y_jk >= r_k  => termo constante rhs = sum_k r_k m2_k
        rhs = 0.0;
        for (int k = 0; k < inst.nK; ++k)
            rhs += inst.r[k] * duDemand[k];

        duPlant.end();
        duDepot.end();
        duDemand.end();
    }
};
