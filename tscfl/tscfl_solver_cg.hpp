/*
COS888

Resolve a relaxação linear do TSCFL por geração de colunas (Dantzig-Wolfe por cliente).

Gabriel Braun, 2025
*/

#pragma once

#include "utils/utils.hpp"

class TSCFLSolverColumnGeneration
{
protected:
    IloEnv &env;
    const TSCFLInstance &inst;

public:
    // Parâmetros do método
    static constexpr IloInt PRINT_EVERY = 10;

    // Resultados:
    IloNum lb{0.0};            // LB da relaxação (LP)
    IloNum ub{IloInfinity};    // melhor primal viável (IP) via heurística
    IloNum lp_ub{IloInfinity}; // valor atual do RMP (UB p/ LP)
    IloNum gap{IloInfinity};
    IloNum time{0.0};
    IloInt64 iter{0};
    IloAlgorithm::Status status{IloAlgorithm::Unknown};

    // Melhor solução primal encontrada
    IloNumArray a;
    IloNumArray b;

private:
    IloModel model;
    IloCplex cplex;
    std::unique_ptr<Subproblem> subproblem;

    // Variáveis
    IloNumVarArray var_a;
    IloNumVarArray var_b;

    // Restrições
    IloRangeArray constr_l1; // constr_l1[i] = restrição de capacidade da planta i
    IloRangeArray constr_l2; // constr_l2[j] = restrição de capacidade do depósito j
    IloRangeArray constr_m2; // constr_m2[k] = restrição de demanda do cliente k

    IloObjective obj;

    // Colunas
    struct ColumnInfo
    {
        int i;
        int j;
    };
    std::vector<std::vector<ColumnInfo>> col_info;
    std::vector<IloNumVarArray> z;

public:
    explicit TSCFLSolverColumnGeneration(
        const TSCFLInstance &inst_,
        Subproblem::Mode smode = Subproblem::Mode::NET)
        : env(inst_.env),
          inst(inst_),
          model(env),
          cplex(env),
          obj(env),
          subproblem(Subproblem::create(inst_, Subproblem::Mode::NET)),
          var_a(env, inst_.nI, 0.0, 1.0, ILOFLOAT),
          var_b(env, inst_.nJ, 0.0, 1.0, ILOFLOAT),
          constr_l1(env, inst_.nI),
          constr_l2(env, inst_.nJ),
          constr_m2(env, inst_.nK)
    {
        // Inicializa vetores de colunas e infos
        z.resize(inst.nK);
        col_info.resize(inst.nK);
        for (int k = 0; k < inst.nK; ++k)
        {
            z[k] = IloNumVarArray(env);
            col_info[k] = std::vector<ColumnInfo>();
        }

        build_initial_model();
        cplex.extract(model);

        // Parâmetros do CPLEX (RMP)
        cplex.setParam(IloCplex::Param::Threads, 1);
        cplex.setParam(IloCplex::Param::Preprocessing::Presolve, 0);
        cplex.setParam(IloCplex::Param::Preprocessing::Aggregator, 0);
        cplex.setParam(IloCplex::Param::RootAlgorithm, IloCplex::Primal);

        // RMP silencioso por padrão
        cplex.setOut(env.getNullStream());
        cplex.setWarning(env.getNullStream());
    }

    ~TSCFLSolverColumnGeneration()
    {
        cplex.end();
        model.end();
    }

private:
    void build_initial_model()
    {
        // RESTRIÇÕES
        // Restrição de capacidade das plantas: constr_l1[i]
        for (int i = 0; i < inst.nI; ++i)
        {
            IloExpr e(env);
            e -= inst.p[i] * var_a[i];
            constr_l1[i] = (e <= 0.0);
            model.add(constr_l1[i]);
            e.end();
        }

        // Restrição de capacidade dos depósitos: constr_l2[j]
        for (int j = 0; j < inst.nJ; ++j)
        {
            IloExpr e(env);
            e -= inst.q[j] * var_b[j];
            constr_l2[j] = (e <= 0.0);
            model.add(constr_l2[j]);
            e.end();
        }

        // Restrição de demanda/convexidade dos clientes: constr_m2[k]
        for (int k = 0; k < inst.nK; ++k)
        {
            IloExpr e(env);
            constr_m2[k] = (e == 1.0);
            model.add(constr_m2[k]);
            e.end();
        }

        // OBJETIVO
        obj = IloMinimize(env, IloScalProd(inst.f, var_a) + IloScalProd(inst.g, var_b));
        model.add(obj);

        // CONSTRUÇÃO DAS COLUNAS INICIAIS
        // Construir fluxo inicial f[i][j][k] capacidade-viável (guloso)
        IloNumTensor flow(env, inst.nI, inst.nJ, inst.nK);
        IloNumArray p_remaining = inst.p.copy();
        IloNumArray q_remaining = inst.q.copy();
        IloNumArray r_remaining = inst.r.copy();

        // Para cada cliente, distribui demanda usando qualquer (i,j) com capacidade
        for (int k = 0; k < inst.nK; ++k)
        {
            while (r_remaining[k] > EPS)
            {
                IloInt best_i = -1;
                IloInt best_j = -1;
                IloNum best_cost = IloInfinity;

                // Escolhe um par (i,j) disponível (aqui uso o mais barato)
                for (int i = 0; i < inst.nI; ++i)
                {
                    if (p_remaining[i] <= EPS)
                        continue;

                    for (int j = 0; j < inst.nJ; ++j)
                    {
                        if (q_remaining[j] <= EPS)
                            continue;

                        IloNum c_ij = inst.c[i][j] + inst.d[j][k];
                        if (c_ij < best_cost)
                        {
                            best_cost = c_ij;
                            best_i = i;
                            best_j = j;
                        }
                    }
                }
                IloNum delta = IloMin(r_remaining[k], IloMin(p_remaining[best_i], q_remaining[best_j]));

                flow[best_i][best_j][k] += delta;
                p_remaining[best_i] -= delta;
                q_remaining[best_j] -= delta;
                r_remaining[k] -= delta;
            }
        }
        //  Criar colunas iniciais z[k][t] a partir de flow[i][j][k]
        for (int k = 0; k < inst.nK; ++k)
        {
            for (int i = 0; i < inst.nI; ++i)
                for (int j = 0; j < inst.nJ; ++j)
                    if (flow[i][j][k] > EPS)
                        add_column_for_client(k, i, j);

            if (z[k].getSize() == 0)
                add_column_for_client(k, 0, 0);
        }
    }

    void add_column_for_client(int k, int i, int j)
    {
        col_info[k].push_back({i, j});

        // Custo do padrão (i,j) para o cliente k:
        double cost = inst.r[k] * (inst.c[i][j] + inst.d[j][k]);

        // Coluna: adiciona coeficientes na função objetivo e nas restrições
        IloNumColumn col = obj(cost);
        col += constr_l1[i](inst.r[k]) + constr_l2[j](inst.r[k]) + constr_m2[k](1.0);

        IloNumVar z_var(col, 0.0, IloInfinity, ILOFLOAT);
        z[k].add(z_var);
        model.add(z_var);
    }

public:
    bool solve(bool log_output = true, double time_limit = -1.0)
    {
        double last_model_value = IloInfinity;
        auto t0 = std::chrono::steady_clock::now();

        auto &SP = *subproblem;
        IloNumArray a_h(env, inst.nI), b_h(env, inst.nJ);
        IloNumArray l1(env, inst.nI), l2(env, inst.nJ), m2(env, inst.nK);

        if (log_output)
        {
            std::cout << "[GC] Iniciando Geração de Colunas\n";
            std::cout << std::right
                      << std::setw(5) << "it"
                      << std::setw(10) << "time(s)"
                      << std::setw(15) << "z_LR"
                      << std::setw(15) << "LB"
                      << std::setw(15) << "UB"
                      << std::setw(12) << "gap"
                      << std::setw(10) << "cols"
                      << "\n"
                      << std::string(100, '-') << "\n"
                      << std::defaultfloat;
        }

        bool converged = false;
        while (!converged)
        {
            // Controle de tempo
            auto t1 = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(t1 - t0).count();
            if (time_limit > 0.0)
            {
                if (elapsed >= time_limit)
                    break;

                cplex.setParam(IloCplex::Param::TimeLimit, time_limit - elapsed);
            }

            // Resolve o RMP atual
            if (!cplex.solve())
            {
                status = cplex.getStatus();
                std::cerr << "[CG] RMP inviável ou erro. Status = " << status << "\n";
                return false;
            }

            status = cplex.getStatus();
            double opt_model = cplex.getObjValue();
            last_model_value = opt_model;
            lp_ub = opt_model;

            // Resolve a heurística primal
            IloNum opt_h = SP.solve_primal_heuristic(cplex, var_a, var_b, a_h, b_h);
            if (opt_h + EPS < ub)
            {
                ub = opt_h;
                a = a_h;
                b = b_h;
            }

            cplex.getDuals(l1, constr_l1);
            cplex.getDuals(l2, constr_l2);
            cplex.getDuals(m2, constr_m2);

            // Pricing: tenta gerar novas colunas
            IloBool any_new = false;

            for (int k = 0; k < inst.nK; ++k)
            {
                IloNum rk = inst.r[k];
                IloNum best_rc = 0.0;
                IloInt best_i = -1;
                IloInt best_j = -1;

                for (int i = 0; i < inst.nI; ++i)
                    for (int j = 0; j < inst.nJ; ++j)
                    {
                        IloNum rc = rk * (inst.c[i][j] + inst.d[j][k]) - rk * l1[i] - rk * l2[j] - m2[k];
                        if (rc < best_rc - EPS)
                        {
                            best_rc = rc;
                            best_i = i;
                            best_j = j;
                        }
                    }
                if (best_i != -1)
                {
                    add_column_for_client(k, best_i, best_j);
                    any_new = true;
                }
            }

            // Log
            if (log_output && (iter % PRINT_EVERY == 0))
            {
                std::cout << "[SG] "
                          << std::right
                          << std::setw(5) << iter
                          // tempo
                          << std::fixed << std::setprecision(1)
                          << std::setw(10)
                          << elapsed
                          // z_LR, LB, UB
                          << std::setprecision(0)
                          << std::setw(10) << lp_ub
                          << std::setw(10) << lb
                          << std::setw(10) << ub
                          // gap
                          << std::scientific << std::setprecision(2)
                          << std::setw(12) << gap
                          << "\n"
                          << std::defaultfloat;
            }

            // Critério de parada:
            if (!any_new)
            {
                converged = true;
                lb = opt_model;
                break;
            }
        }

        auto t_end = std::chrono::steady_clock::now();
        time = std::chrono::duration<double>(t_end - t0).count();

        // Atualiza o status e o gap
        if (converged)
            status = IloAlgorithm::Optimal;

        if (ub < IloInfinity && lb > -IloInfinity)
            gap = (ub - lb) / IloMax(1.0, IloAbs(ub));

        // Log final
        if (log_output)
        {
            std::cout
                << "\n\n"
                << "[RC] Geração de colunas finalizado.\n"
                << "LP*    = " << (lb > -IloInfinity ? lb : last_model_value)
                << "LB     = " << lb << "\n"
                << "UB     = " << ub << "\n"
                << "status = " << status << "\n"
                << "gap    = " << gap << "\n"
                << "iter   = " << iter << "\n"
                // tempo
                << std::fixed << std::setprecision(1)
                << "time   = " << time << " s\n"
                << std::defaultfloat;

            std::cout << "\n[CG] Column Generation finalizado.\n";
            std::cout << "LP*   = " << (lb > -IloInfinity ? lb : last_model_value)
                      << "  (LB relaxação, se convergiu)\n";
            std::cout << "UB    = " << ub << "  (melhor primal viável)\n";
            std::cout << "gap   = " << gap << "\n";
            std::cout << "status= " << status << "\n";
            std::cout << "time  = " << time << " s\n";
        }

        return converged;
    }
};
