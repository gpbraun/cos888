/*
COS888

Resolve a relaxação linear do TSCFL por geração de colunas
(Dantzig–Wolfe por cliente).

Gabriel Braun, 2025
*/

#pragma once

#include <ilcplex/ilocplex.h>
#include <chrono>
#include <iostream>
#include <limits>
#include <vector>
#include <cmath>

#include "tscfl_instance.hpp"

ILOSTLBEGIN

class TSCFLSolverColumnGeneration
{
public:
    const TSCFLInstance &inst;

    // Resultados:
    double lb{-IloInfinity};   // LB da relaxação (LP)
    double ub{IloInfinity};    // melhor primal viável (IP) via heurística
    double lp_ub{IloInfinity}; // valor atual do RMP (UB p/ LP)
    double gap{IloInfinity};
    double time{0.0};
    IloAlgorithm::Status status{IloAlgorithm::Unknown};

private:
    IloModel rmp;
    IloCplex cplex;
    IloObjective obj;

    // Variáveis do RMP:
    // a[i], b[j] são relaxadas (contínuas em [0,1]), mas mantemos a nomenclatura.
    IloNumVarArray a; // a[i] = abre planta i (relaxado)
    IloNumVarArray b; // b[j] = abre depósito j (relaxado)

    // z[k][t] = coluna t do cliente k (padrão planta–depósito).
    std::vector<IloNumVarArray> z;

    // Informações das colunas: para cada z[k][t], qual (i,j) ela representa?
    struct ColumnInfo
    {
        int i;
        int j;
    };
    std::vector<std::vector<ColumnInfo>> col_info;

    // Restrições nomeadas pela variável dual associada:
    IloRangeArray constr_l1; // capacidade das plantas (dual: l1)
    IloRangeArray constr_l2; // capacidade dos depósitos (dual: l2)
    IloRangeArray constr_m1; // balanço nos depósitos (NÃO usado no RMP DW)
    IloRangeArray constr_m2; // demanda/convexidade dos clientes (dual: m2)

public:
    explicit TSCFLSolverColumnGeneration(const TSCFLInstance &inst_)
        : inst(inst_),
          rmp(inst_.env),
          cplex(inst_.env),
          obj(inst_.env),
          a(inst_.env, inst_.nI),
          b(inst_.env, inst_.nJ),
          constr_l1(inst_.env, inst_.nI),
          constr_l2(inst_.env, inst_.nJ),
          constr_m1(inst_.env), // não usado aqui
          constr_m2(inst_.env, inst_.nK)
    {
        IloEnv &env = inst.env;

        // Inicializa vetores de colunas (z) e infos
        z.resize(inst.nK);
        col_info.resize(inst.nK);
        for (int k = 0; k < inst.nK; ++k)
        {
            z[k] = IloNumVarArray(env);
            col_info[k] = std::vector<ColumnInfo>();
        }

        build_initial_rmp();
        cplex.extract(rmp);

        // Parâmetros CPLEX (LP):
        cplex.setParam(IloCplex::Param::Threads, 1);
        cplex.setParam(IloCplex::Param::Preprocessing::Reduce, 0);
        // Qualquer algoritmo de LP raiz (primal/dual) serve bem:
        // cplex.setParam(IloCplex::Param::RootAlgorithm, IloCplex::Primal);
    }

    ~TSCFLSolverColumnGeneration()
    {
        cplex.end();
        rmp.end();
    }

private:
    void build_initial_rmp()
    {
        IloEnv &env = inst.env;

        // Objetivo
        obj = IloMinimize(env, 0.0);
        rmp.add(obj);

        // Variáveis a[i] e b[j] relaxadas em [0,1]
        for (int i = 0; i < inst.nI; ++i)
            a[i] = IloNumVar(env, 0.0, 1.0, ILOFLOAT);
        for (int j = 0; j < inst.nJ; ++j)
            b[j] = IloNumVar(env, 0.0, 1.0, ILOFLOAT);

        rmp.add(a);
        rmp.add(b);

        // Custos fixos
        for (int i = 0; i < inst.nI; ++i)
            obj.setLinearCoef(a[i], inst.f[i]);
        for (int j = 0; j < inst.nJ; ++j)
            obj.setLinearCoef(b[j], inst.g[j]);

        // Restrição de capacidade das plantas: constr_l1[i]
        for (int i = 0; i < inst.nI; ++i)
        {
            IloExpr e(env);
            e -= inst.p[i] * a[i];
            constr_l1[i] = (e <= 0.0);
            rmp.add(constr_l1[i]);
            e.end();
        }

        // Restrição de capacidade dos depósitos: constr_l2[j]
        for (int j = 0; j < inst.nJ; ++j)
        {
            IloExpr e(env);
            e -= inst.q[j] * b[j];
            constr_l2[j] = (e <= 0.0);
            rmp.add(constr_l2[j]);
            e.end();
        }

        // Restrição de demanda/convexidade dos clientes: constr_m2[k]
        for (int k = 0; k < inst.nK; ++k)
        {
            IloExpr e(env);
            constr_m2[k] = (e == 1.0);
            rmp.add(constr_m2[k]);
            e.end();
        }

        // =====================================================================
        //  Construir fluxo inicial f[i][j][k] capacidade-viável (guloso)
        // =====================================================================

        // flow[i][j][k]
        std::vector<std::vector<std::vector<double>>> flow(
            inst.nI,
            std::vector<std::vector<double>>(inst.nJ,
                                             std::vector<double>(inst.nK, 0.0)));

        std::vector<double> plant_rem(inst.nI);
        std::vector<double> depot_rem(inst.nJ);

        double total_demand = 0.0;
        double total_p = 0.0;
        double total_q = 0.0;

        for (int i = 0; i < inst.nI; ++i)
        {
            plant_rem[i] = inst.p[i];
            total_p += inst.p[i];
        }
        for (int j = 0; j < inst.nJ; ++j)
        {
            depot_rem[j] = inst.q[j];
            total_q += inst.q[j];
        }
        for (int k = 0; k < inst.nK; ++k)
            total_demand += inst.r[k];

        if (total_p + EPS < total_demand || total_q + EPS < total_demand)
            throw std::runtime_error("Instância inviável: capacidade total < demanda total.");

        // Para cada cliente, distribui demanda usando qualquer (i,j) com capacidade
        for (int k = 0; k < inst.nK; ++k)
        {
            double R = inst.r[k];

            while (R > EPS)
            {
                int best_i = -1;
                int best_j = -1;
                double best_cost = std::numeric_limits<double>::infinity();

                // Escolhe um par (i,j) disponível (aqui uso o mais barato)
                for (int i = 0; i < inst.nI; ++i)
                {
                    if (plant_rem[i] <= EPS)
                        continue;

                    for (int j = 0; j < inst.nJ; ++j)
                    {
                        if (depot_rem[j] <= EPS)
                            continue;

                        double c_ij = inst.c[i][j] + inst.d[j][k];
                        if (c_ij < best_cost)
                        {
                            best_cost = c_ij;
                            best_i = i;
                            best_j = j;
                        }
                    }
                }

                if (best_i == -1 || best_j == -1)
                {
                    throw std::runtime_error(
                        "Falha ao construir fluxo inicial: sem capacidade suficiente para atender cliente.");
                }

                double delta = std::min(R, std::min(plant_rem[best_i], depot_rem[best_j]));

                flow[best_i][best_j][k] += delta;
                R -= delta;
                plant_rem[best_i] -= delta;
                depot_rem[best_j] -= delta;
            }
        }

        // =====================================================================
        //  Criar colunas iniciais z[k][t] a partir de flow[i][j][k]
        // =====================================================================
        for (int k = 0; k < inst.nK; ++k)
        {
            for (int i = 0; i < inst.nI; ++i)
            {
                for (int j = 0; j < inst.nJ; ++j)
                {
                    if (flow[i][j][k] > EPS)
                    {
                        // Criar uma coluna z_{k,(i,j)}
                        add_column_for_client(k, i, j);
                    }
                }
            }

            // Por segurança, garante que ao menos uma coluna foi criada para k
            if (z[k].getSize() == 0)
            {
                // Isso só pode acontecer se r_k ~ 0; se quiser, pode pular
                // ou criar uma coluna "dummy" aqui.
                // Para simplicidade, vamos garantir pelo menos uma coluna qualquer.
                int i0 = 0, j0 = 0;
                add_column_for_client(k, i0, j0);
            }
        }
    }

    void add_column_for_client(int k, int i, int j)
    {
        IloEnv &env = inst.env;

        double rk = inst.r[k];

        // Custo do padrão (i,j) para o cliente k:
        double cost = rk * (inst.c[i][j] + inst.d[j][k]);

        // Coluna: adiciona coeficientes na função objetivo e nas restrições
        IloNumColumn col = obj(cost);
        col += constr_l1[i](rk);  // capacidade planta i
        col += constr_l2[j](rk);  // capacidade depósito j
        col += constr_m2[k](1.0); // convexidade/demanda do cliente k

        IloNumVar z_var(col, 0.0, IloInfinity, ILOFLOAT);
        z[k].add(z_var);
        rmp.add(z_var);

        col_info[k].push_back({i, j});
    }

    // Heurística primal: constrói (a,b,x,y) viáveis a partir de z e atualiza UB.
    void run_heuristic_from_rmp()
    {
        IloEnv &env = inst.env;

        // Reconstruir fluxos x[i][j] e y[j][k] a partir de z[k][t].
        IloNumMatrix x(env, inst.nI, inst.nJ);
        IloNumMatrix y(env, inst.nJ, inst.nK);

        for (int i = 0; i < inst.nI; ++i)
            for (int j = 0; j < inst.nJ; ++j)
                x[i][j] = 0.0;

        for (int j = 0; j < inst.nJ; ++j)
            for (int k = 0; k < inst.nK; ++k)
                y[j][k] = 0.0;

        // Percorre z[k][t] e acumula fluxo = z_{k,t} * r_k
        for (int k = 0; k < inst.nK; ++k)
        {
            int ncols = z[k].getSize();
            double rk = inst.r[k];

            for (int t = 0; t < ncols; ++t)
            {
                double zval = cplex.getValue(z[k][t]);
                if (zval <= EPS)
                    continue;

                int i = col_info[k][t].i;
                int j = col_info[k][t].j;

                double flow = zval * rk;
                x[i][j] += flow;
                y[j][k] += flow;
            }
        }

        // Determina a[i] e b[j] inteiros a partir dos fluxos.
        std::vector<double> a_heur(inst.nI, 0.0);
        std::vector<double> b_heur(inst.nJ, 0.0);

        for (int i = 0; i < inst.nI; ++i)
        {
            double total_out = 0.0;
            for (int j = 0; j < inst.nJ; ++j)
                total_out += x[i][j];
            if (total_out > EPS)
                a_heur[i] = 1.0;
        }

        for (int j = 0; j < inst.nJ; ++j)
        {
            double total_out = 0.0;
            for (int k = 0; k < inst.nK; ++k)
                total_out += y[j][k];
            if (total_out > EPS)
                b_heur[j] = 1.0;
        }

        // Calcula o custo da solução heurística.
        double cost_fix = 0.0;
        for (int i = 0; i < inst.nI; ++i)
            cost_fix += inst.f[i] * a_heur[i];
        for (int j = 0; j < inst.nJ; ++j)
            cost_fix += inst.g[j] * b_heur[j];

        double cost_flow = 0.0;
        for (int i = 0; i < inst.nI; ++i)
            for (int j = 0; j < inst.nJ; ++j)
                cost_flow += inst.c[i][j] * x[i][j];
        for (int j = 0; j < inst.nJ; ++j)
            for (int k = 0; k < inst.nK; ++k)
                cost_flow += inst.d[j][k] * y[j][k];

        double ub_cand = cost_fix + cost_flow;

        if (ub_cand + 1e-6 < ub)
        {
            ub = ub_cand;
            // Se quiser, aqui poderíamos armazenar (a_heur, b_heur, x, y)
            // em membros da classe para recuperar a melhor solução primal.
        }
    }

public:
    bool solve(bool log_output = true, double time_limit = -1.0)
    {
        IloEnv &env = inst.env;

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

        const bool has_time_limit = (time_limit > 0.0);
        auto t0 = std::chrono::steady_clock::now();

        bool converged = false;
        double last_rmp_value = IloInfinity;

        while (true)
        {
            // Controle de tempo
            auto t1 = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(t1 - t0).count();

            if (has_time_limit)
            {
                double remaining = time_limit - elapsed;
                if (remaining <= 0.0)
                    break; // time limit atingido

                // Garante que cada solve do CPLEX respeite o tempo restante
                cplex.setParam(IloCplex::Param::TimeLimit, remaining);
            }

            // Resolve o RMP atual
            if (!cplex.solve())
            {
                status = cplex.getStatus();
                time = std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - t0)
                           .count();

                std::cerr << "[CG] RMP infeasible ou erro, status = "
                          << status << "\n";
                return false;
            }

            status = cplex.getStatus();
            double z_rmp = cplex.getObjValue();
            last_rmp_value = z_rmp;
            lp_ub = z_rmp;

            // Atualiza heurística primal (UB do inteiro)
            run_heuristic_from_rmp();

            // Recupera duais do RMP:
            IloNumArray l1(env, inst.nI), l2(env, inst.nJ), m2(env, inst.nK);

            cplex.getDuals(l1, constr_l1); // capacidade plantas
            cplex.getDuals(l2, constr_l2); // capacidade depósitos
            cplex.getDuals(m2, constr_m2); // demanda/convexidade clientes

            // Pricing: tenta gerar novas colunas
            bool any_new = false;

            for (int k = 0; k < inst.nK; ++k)
            {
                double rk = inst.r[k];

                double best_rc = 0.0;
                int best_i = -1;
                int best_j = -1;

                for (int i = 0; i < inst.nI; ++i)
                {
                    for (int j = 0; j < inst.nJ; ++j)
                    {
                        double rc =
                            rk * (inst.c[i][j] + inst.d[j][k]) - rk * l1[i] - rk * l2[j] - m2[k];

                        if (rc < best_rc - EPS)
                        {
                            best_rc = rc;
                            best_i = i;
                            best_j = j;
                        }
                    }
                }

                if (best_i != -1)
                {
                    add_column_for_client(k, best_i, best_j);
                    any_new = true;
                }
            }

            // Critério de parada:
            // - Se não gerou nenhuma coluna nova, atingimos o ótimo da relaxação.
            if (!any_new)
            {
                lb = z_rmp; // valor ótimo da LP
                converged = true;
                break;
            }

            // - Se não há time_limit, o loop continua até convergir.
            // - Se há time_limit, o while será interrompido lá em cima
            //   quando elapsed >= time_limit.
        }

        auto t_end = std::chrono::steady_clock::now();
        time = std::chrono::duration<double>(t_end - t0).count();

        if (converged)
        {
            status = IloAlgorithm::Optimal;
        }
        else if (status == IloAlgorithm::Unknown)
        {
            // Não convergiu (por tempo), mas temos uma solução RMP viável
            status = IloAlgorithm::Feasible;
        }

        // gap só faz sentido se tivermos LB finito e UB finito
        if (ub < IloInfinity && lb > -IloInfinity)
        {
            double denom = std::max(1.0, std::fabs(ub));
            gap = (ub - lb) / denom;
        }
        else
        {
            gap = IloInfinity;
        }

        if (log_output)
        {
            std::cout << "\n[CG] Column Generation finalizado.\n";
            std::cout << "LP*   = " << (lb > -IloInfinity ? lb : last_rmp_value)
                      << "  (LB relaxação, se convergiu)\n";
            std::cout << "UB    = " << ub << "  (melhor primal viável)\n";
            std::cout << "gap   = " << gap << "\n";
            std::cout << "status= " << status << "\n";
            std::cout << "time  = " << time << " s\n";
        }

        // Retorna true se convergiu para o ótimo da relaxação,
        // false se parou por tempo.
        return converged;
    }
};
