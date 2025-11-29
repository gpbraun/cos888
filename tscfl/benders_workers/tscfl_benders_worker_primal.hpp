/*
COS888

WorkerPrimal: resolve o subproblema PRIMAL do TSCFL e
constrói o corte de Benders a partir dos duais das restrições.

Gabriel Braun, 2025
*/

#pragma once

#include <ilcplex/ilocplex.h>
#include <iostream>

#include "tscfl_benders_worker.hpp"

ILOSTLBEGIN

class WorkerPrimal : public Worker
{
private:
    IloEnv env;
    IloModel model;
    IloCplex cplex;

    // Variáveis de fluxo
    IloNumVarArray x; // x_ij, tamanho nI * nJ
    IloNumVarArray y; // y_jk, tamanho nJ * nK

    // Restrições (guardadas para poder pegar duais)
    IloRangeArray plantCap; // capacidade das plantas
    IloRangeArray depotCap; // capacidade dos depósitos
    IloRangeArray flowBal;  // balanço de fluxo nos depósitos
    IloRangeArray demand;   // atendimento da demanda dos clientes

    // Acessores de conveniência
    inline IloNumVar &X(int i, int j) { return x[idx2(i, j, inst.nJ)]; }
    inline IloNumVar &Y(int j, int k) { return y[idx2(j, k, inst.nK)]; }

public:
    explicit WorkerPrimal(const TSCFLInstance &inst_, bool log_output = false)
        : Worker(inst_),
          env(),
          model(env),
          cplex(env),
          x(env, inst_.nI * inst_.nJ, 0.0, IloInfinity, ILOFLOAT),
          y(env, inst_.nJ * inst_.nK, 0.0, IloInfinity, ILOFLOAT),
          plantCap(env, inst_.nI),
          depotCap(env, inst_.nJ),
          flowBal(env, inst_.nJ),
          demand(env, inst_.nK)
    {
        const int nI = inst.nI;
        const int nJ = inst.nJ;
        const int nK = inst.nK;

        // -----------------------------------------------------------------
        // Objetivo: min sum c_ij x_ij + sum d_jk y_jk
        // -----------------------------------------------------------------
        {
            IloExpr obj(env);
            for (int i = 0; i < nI; ++i)
                for (int j = 0; j < nJ; ++j)
                    obj += inst.C(i, j) * X(i, j);

            for (int j = 0; j < nJ; ++j)
                for (int k = 0; k < nK; ++k)
                    obj += inst.D(j, k) * Y(j, k);

            model.add(IloMinimize(env, obj));
            obj.end();
        }

        // -----------------------------------------------------------------
        // Capacidade das plantas:
        //   sum_j x_ij <= p_i a_i
        // Reescrevemos como: -sum_j x_ij >= -p_i a_i
        // para ficar na forma ">= RHS" (útil para Benders e duais).
        // O RHS dependerá de a_i, então só ajustamos na solve().
        // -----------------------------------------------------------------
        for (int i = 0; i < nI; ++i)
        {
            IloExpr e(env);
            for (int j = 0; j < nJ; ++j)
                e -= X(i, j); // -x_ij

            // Bounds provisórios; vamos ajustar o LB na solve()
            plantCap[i] = IloRange(env, -IloInfinity, e, IloInfinity);
            model.add(plantCap[i]);
            e.end();
        }

        // -----------------------------------------------------------------
        // Capacidade dos depósitos:
        //   sum_k y_jk <= q_j b_j
        // Reescrito como: -sum_k y_jk >= -q_j b_j
        // -----------------------------------------------------------------
        for (int j = 0; j < nJ; ++j)
        {
            IloExpr e(env);
            for (int k = 0; k < nK; ++k)
                e -= Y(j, k); // -y_jk

            depotCap[j] = IloRange(env, -IloInfinity, e, IloInfinity);
            model.add(depotCap[j]);
            e.end();
        }

        // -----------------------------------------------------------------
        // Balanço de fluxo nos depósitos:
        //   sum_i x_ij - sum_k y_jk = 0
        // RHS = 0, não depende de (a,b); duais entram mas não aparecem
        // diretamente no corte (coef. constante).
        // -----------------------------------------------------------------
        for (int j = 0; j < nJ; ++j)
        {
            IloExpr e(env);
            for (int i = 0; i < nI; ++i)
                e += X(i, j);
            for (int k = 0; k < nK; ++k)
                e -= Y(j, k);

            flowBal[j] = IloRange(env, 0.0, e, 0.0);
            model.add(flowBal[j]);
            e.end();
        }

        // -----------------------------------------------------------------
        // Demanda dos clientes:
        //   sum_j y_jk >= r_k
        // RHS = r_k (fixo, indep. de a,b)
        // -----------------------------------------------------------------
        for (int k = 0; k < nK; ++k)
        {
            IloExpr e(env);
            for (int j = 0; j < nJ; ++j)
                e += Y(j, k);

            demand[k] = IloRange(env, inst.r[k], e, IloInfinity);
            model.add(demand[k]);
            e.end();
        }

        cplex.extract(model);

        // LP puro, 1 thread, primal simplex e sem grandes reduções de presolve
        cplex.setParam(IloCplex::Param::Threads, 1);
        cplex.setParam(IloCplex::Param::RootAlgorithm, IloCplex::Primal);
        cplex.setParam(IloCplex::Param::Preprocessing::Reduce, 0);

        if (log_output)
        {
            cplex.setOut(std::cout);
            cplex.setWarning(std::cout);
        }
        else
        {
            cplex.setOut(env.getNullStream());
            cplex.setWarning(env.getNullStream());
        }
    }

    ~WorkerPrimal() override
    {
        cplex.end();
        model.end();
        env.end();
    }

public:
    // Implementa a interface virtual da classe base Worker.
    //
    // Entrada: vetores a, b fixos (da solução corrente do mestre).
    // Saída:
    //   theta  = valor ótimo do subproblema
    //   coef_a = coeficientes de a_i no corte
    //   coef_b = coeficientes de b_j no corte
    //   rhs    = termo independente do corte
    //
    // Corte:   eta >= rhs + sum_i coef_a[i] * a_i + sum_j coef_b[j] * b_j
    //
    void solve(
        const Vec &a,
        const Vec &b,
        double &theta,
        Vec &coef_a,
        Vec &coef_b,
        double &rhs) override
    {
        const int nI = inst.nI;
        const int nJ = inst.nJ;
        const int nK = inst.nK;

        // 1) Atualiza RHS das capacidades com base em (a,b):
        //
        //    -sum_j x_ij >= -p_i a_i
        //    -sum_k y_jk >= -q_j b_j
        //
        for (int i = 0; i < nI; ++i)
        {
            double lb = -inst.p[i] * a[i];
            plantCap[i].setBounds(lb, IloInfinity);
        }

        for (int j = 0; j < nJ; ++j)
        {
            double lb = -inst.q[j] * b[j];
            depotCap[j].setBounds(lb, IloInfinity);
        }

        // 2) Resolve o LP primal
        if (!cplex.solve())
            throw std::runtime_error("WorkerPrimal: CPLEX failed to solve primal subproblem.");

        theta = cplex.getObjValue();

        // 3) Lê os duais das restrições relevantes
        IloNumArray duPlant(env), duDepot(env), duDemand(env);

        cplex.getDuals(duPlant, plantCap); // π_i
        cplex.getDuals(duDepot, depotCap); // σ_j
        cplex.getDuals(duDemand, demand);  // τ_k

        // 4) Monta coeficientes do corte
        //
        //   Para plantas:  -sum_j x_ij >= -p_i a_i  => RHS = -p_i a_i
        //   Termo no dual: sum_i RHS_i * π_i = sum_i (-p_i a_i) π_i
        //   => coef_a[i] = ∂/∂a_i [(-p_i a_i) π_i] = -p_i π_i
        //
        coef_a.assign(nI, 0.0);
        for (int i = 0; i < nI; ++i)
            coef_a[i] = -inst.p[i] * duPlant[i];

        //   Para depósitos: -sum_k y_jk >= -q_j b_j  => coef_b[j] = -q_j σ_j
        coef_b.assign(nJ, 0.0);
        for (int j = 0; j < nJ; ++j)
            coef_b[j] = -inst.q[j] * duDepot[j];

        //   Para demandas:  sum_j y_jk >= r_k  => termo constante rhs = sum_k r_k τ_k
        rhs = 0.0;
        for (int k = 0; k < nK; ++k)
            rhs += inst.r[k] * duDemand[k];

        // Libera arrays temporários
        duPlant.end();
        duDepot.end();
        duDemand.end();
    }
};
