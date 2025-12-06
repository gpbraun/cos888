/*
COS888

tscfl_solver_column_generation.cpp

Gabriel Braun, 2025
*/

#include "solvers/tscfl_solver_column_generation.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>

// ---------------------------------------------------------------------
// Construtor / Destrutor
// ---------------------------------------------------------------------

TSCFLSolverColumnGeneration::TSCFLSolverColumnGeneration(
    const TSCFLInstance &inst_, Subproblem::Mode smode
)
    : TSCFLSolver(inst_),
      model(env),
      cplex(env),
      subproblem(Subproblem::create(inst_, smode)),
      obj(env),
      var_a(env, inst_.nI, 0.0, 1.0, ILOFLOAT),
      var_b(env, inst_.nJ, 0.0, 1.0, ILOFLOAT),
      constr_l1(env, inst_.nI),
      constr_l2(env, inst_.nJ),
      constr_m2(env, inst_.nK),
      constrs_m1(env, inst_.nJ),
      a(env, inst_.nI),
      b(env, inst_.nJ)
{
    // Inicializa vetores de colunas e infos
    z.resize(inst.nK);
    col_info.resize(inst.nK);
    for (IloInt k = 0; k < inst.nK; ++k)
        {
            z[k] = IloNumVarArray(env);
            col_info[k] = std::vector<ColumnInfo>();
        }

    // Inicializa arrays de restrições (j,k)
    for (IloInt j = 0; j < inst.nJ; ++j)
        {
            constrs_m1[j] = IloRangeArray(env, inst.nK);
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

TSCFLSolverColumnGeneration::~TSCFLSolverColumnGeneration()
{
    cplex.end();
    model.end();
}

// ---------------------------------------------------------------------
// Construção do modelo mestre restrito inicial (RMP)
// ---------------------------------------------------------------------

void
TSCFLSolverColumnGeneration::build_initial_model()
{
    // -------------------------------------------------------------
    // 1) Restrições de capacidade das plantas: constr_l1[i]
    //     -p_i a_i + sum_{k,t: col_info[k][t].i = i} r_k z_{k,t} <= 0
    // -------------------------------------------------------------
    for (IloInt i = 0; i < inst.nI; ++i)
        {
            IloExpr e(env);
            e -= inst.p[i] * var_a[i];
            constr_l1[i] = (e <= 0.0);
            model.add(constr_l1[i]);
            e.end();
        }

    // -------------------------------------------------------------
    // 2) Restrições de capacidade dos depósitos: constr_l2[j]
    //     -q_j b_j + sum_{k,t: col_info[k][t].j = j} r_k z_{k,t} <= 0
    // -------------------------------------------------------------
    for (IloInt j = 0; j < inst.nJ; ++j)
        {
            IloExpr e(env);
            e -= inst.q[j] * var_b[j];
            constr_l2[j] = (e <= 0.0);
            model.add(constr_l2[j]);
            e.end();
        }

    // -------------------------------------------------------------
    // 3) Restrições de convexidade/demanda por cliente: constr_m2[k]
    //     sum_t z_{k,t} = 1
    // -------------------------------------------------------------
    for (IloInt k = 0; k < inst.nK; ++k)
        {
            IloExpr e(env);
            constr_m2[k] = (e == 1.0);
            model.add(constr_m2[k]);
            e.end();
        }

    // -------------------------------------------------------------
    // 4) Restrição de vínculo: z_{k,t} <= b_j  (para par (j,k))
    //
    // Implementada como:
    //     -b_j + sum_{t: col_info[k][t].j = j} z_{k,t} <= 0
    // (os coeficientes de z são adicionados quando criamos colunas)
    // -------------------------------------------------------------
    for (IloInt j = 0; j < inst.nJ; ++j)
        {
            for (IloInt k = 0; k < inst.nK; ++k)
                {
                    IloExpr e(env);
                    e -= var_b[j];
                    constrs_m1[j][k] = (e <= 0.0);
                    model.add(constrs_m1[j][k]);
                    e.end();
                }
        }

    // -------------------------------------------------------------
    // 5) Objetivo: f^T a + g^T b + sum_{k,t} cost_{k,it} z_{k,t}
    //    (parte dos z é construída nas colunas)
    // -------------------------------------------------------------
    obj = IloMinimize(env, IloScalProd(inst.f, var_a) + IloScalProd(inst.g, var_b));
    model.add(obj);

    // -------------------------------------------------------------
    // 6) Construção das colunas iniciais
    //    - Geramos um fluxo capacidade-viável "flow[i][j][k]"
    //      por um esquema guloso simples.
    // -------------------------------------------------------------
    IloNumTensor flow(env, inst.nI, inst.nJ, inst.nK);
    IloNumArray p_remaining = inst.p.copy();
    IloNumArray q_remaining = inst.q.copy();
    IloNumArray r_remaining = inst.r.copy();

    for (IloInt k = 0; k < inst.nK; ++k)
        {
            while (r_remaining[k] > EPS)
                {
                    IloInt best_i = -1;
                    IloInt best_j = -1;
                    IloNum best_cost = IloInfinity;

                    // Escolhe par (i,j) viável com menor custo c_ij + d_jk
                    for (IloInt i = 0; i < inst.nI; ++i)
                        {
                            if (p_remaining[i] <= EPS)
                                continue;

                            for (IloInt j = 0; j < inst.nJ; ++j)
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

                    // Sem capacidade total suficiente: aborta distribuição restante
                    if (best_i < 0 || best_j < 0)
                        {
                            break;
                        }

                    IloNum delta
                        = IloMin(r_remaining[k], IloMin(p_remaining[best_i], q_remaining[best_j]));

                    if (delta <= EPS)
                        break;

                    flow[best_i][best_j][k] += delta;
                    p_remaining[best_i] -= delta;
                    q_remaining[best_j] -= delta;
                    r_remaining[k] -= delta;
                }
        }

    // Criar colunas iniciais z[k][t] a partir de flow[i][j][k]
    for (IloInt k = 0; k < inst.nK; ++k)
        {
            for (IloInt i = 0; i < inst.nI; ++i)
                {
                    for (IloInt j = 0; j < inst.nJ; ++j)
                        {
                            if (flow[i][j][k] > EPS)
                                {
                                    add_column_for_client(
                                        static_cast<int>(k),
                                        static_cast<int>(i),
                                        static_cast<int>(j)
                                    );
                                }
                        }
                }

            // Segurança: se nenhum padrão foi gerado para k, cria um "dummy"
            if (z[k].getSize() == 0)
                {
                    add_column_for_client(static_cast<int>(k), 0, 0);
                }
        }
}

// ---------------------------------------------------------------------
// Adiciona uma coluna (padrão) para o cliente k correspondente ao par (i,j)
// ---------------------------------------------------------------------

void
TSCFLSolverColumnGeneration::add_column_for_client(int k, int i, int j)
{
    col_info[k].push_back({ i, j });

    // Custo do padrão (i,j) para o cliente k:
    const IloNum rk = inst.r[k];
    const IloNum cost = rk * (inst.c[i][j] + inst.d[j][k]);

    // Coluna na função objetivo e nas restrições
    IloNumColumn col = obj(cost);

    // Capacidade planta i: coeficiente = r_k
    col += constr_l1[i](rk);

    // Capacidade depósito j: coeficiente = r_k
    col += constr_l2[j](rk);

    // Convexidade/demanda do cliente k: coeficiente = 1
    col += constr_m2[k](1.0);

    // Vínculo z_{k,*} <= b_j: coeficiente = 1 em (j,k)
    col += constrs_m1[j][k](1.0);

    IloNumVar z_var(col, 0.0, IloInfinity, ILOFLOAT);
    z[k].add(z_var);
    model.add(z_var);
}

// ---------------------------------------------------------------------
// Número total de colunas z[k][t]
// ---------------------------------------------------------------------

IloInt
TSCFLSolverColumnGeneration::get_num_columns() const
{
    IloInt total = 0;
    for (IloInt k = 0; k < inst.nK; ++k)
        {
            total += z[k].getSize();
        }
    return total;
}

// ---------------------------------------------------------------------
// Método principal de geração de colunas
// ---------------------------------------------------------------------

bool
TSCFLSolverColumnGeneration::solve(bool log_output, IloNum time_limit)
{
    auto &SP = *subproblem;

    IloNumArray a_lr(env, inst.nI);
    IloNumArray b_lr(env, inst.nJ);
    IloNumArray a_h(env, inst.nI);
    IloNumArray b_h(env, inst.nJ);

    IloNumArray l1(env, inst.nI);
    IloNumArray l2(env, inst.nJ);
    IloNumArray m2(env, inst.nK);
    IloNumMatrix m1(env, inst.nJ, inst.nK);

    if (log_output)
        {
            std::cout << "[CG] Iniciando Geração de Colunas\n";
            std::cout << std::right << std::setw(5) << "it" << std::setw(10) << "time(s)"
                      << std::setw(15) << "LB" << std::setw(15) << "UB" << std::setw(12) << "gap"
                      << std::setw(10) << "cols"
                      << "\n"
                      << std::string(70, '-') << "\n"
                      << std::defaultfloat;
        }

    auto t0 = std::chrono::steady_clock::now();

    while (true)
        {
            // ---------------------------------------------------------
            // Controle de tempo
            // ---------------------------------------------------------
            auto t1 = std::chrono::steady_clock::now();
            IloNum elapsed = std::chrono::duration<IloNum>(t1 - t0).count();

            if (time_limit > 0.0)
                {
                    if (elapsed >= time_limit)
                        {
                            break;
                        }
                    cplex.setParam(IloCplex::Param::TimeLimit, time_limit - elapsed);
                }

            // ---------------------------------------------------------
            // Resolve o RMP atual
            // ---------------------------------------------------------
            if (!cplex.solve())
                {
                    status = cplex.getStatus();
                    std::cerr << "[CG] RMP inviável ou erro. Status = " << status << "\n";
                    return false;
                }

            status = cplex.getStatus();
            lb = cplex.getObjValue();

            // ---------------------------------------------------------
            // Heurística primal: a partir de (a,b) fracionários do RMP
            // ---------------------------------------------------------
            cplex.getValues(var_a, a_lr);
            cplex.getValues(var_b, b_lr);

            IloNum opt_h = SP.solve_primal_heuristic(a_lr, b_lr, a_h, b_h);
            if (opt_h + EPS < ub)
                {
                    ub = opt_h;
                    a = a_h;
                    b = b_h;
                }

            update_gap();

            // ---------------------------------------------------------
            // Recupera duais para fazer pricing
            // ---------------------------------------------------------
            cplex.getDuals(l1, constr_l1);
            cplex.getDuals(l2, constr_l2);
            cplex.getDuals(m2, constr_m2);
            for (IloInt j = 0; j < inst.nJ; ++j)
                {
                    cplex.getDuals(m1[j], constrs_m1[j]);
                }

            // ---------------------------------------------------------
            // Pricing por cliente k:
            // Procurar (i,j) com custo reduzido negativo mais violado.
            //
            // rc_{k,ij} =
            //   r_k (c_ij + d_jk) - r_k(l1_i + l2_j) - m2_k - m1_{j,k}
            //
            // Se rc < 0, coluna entra.
            // ---------------------------------------------------------
            bool any_new = false;

            for (IloInt k = 0; k < inst.nK; ++k)
                {
                    const IloNum rk = inst.r[k];

                    IloNum best_rc = 0.0;
                    IloInt best_i = -1;
                    IloInt best_j = -1;

                    for (IloInt i = 0; i < inst.nI; ++i)
                        {
                            for (IloInt j = 0; j < inst.nJ; ++j)
                                {
                                    IloNum rc = rk * (inst.c[i][j] + inst.d[j][k])
                                                - rk * (l1[i] + l2[j]) - m2[k] - m1[j][k];

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
                            add_column_for_client(
                                static_cast<int>(k),
                                static_cast<int>(best_i),
                                static_cast<int>(best_j)
                            );
                            any_new = true;
                        }
                }

            // ---------------------------------------------------------
            // Log
            // ---------------------------------------------------------
            if (log_output && (iter % PRINT_EVERY == 0 || !any_new))
                {
                    std::cout << std::right << std::setw(5) << iter << std::fixed
                              << std::setprecision(1) << std::setw(10) << elapsed
                              << std::setprecision(0) << std::setw(15) << lb << std::setw(15) << ub
                              << std::scientific << std::setprecision(2) << std::setw(12) << gap
                              << std::fixed << std::setw(10) << get_num_columns() << "\n"
                              << std::defaultfloat;
                }

            // ---------------------------------------------------------
            // Critério de parada: nenhuma coluna com rc < 0
            // ---------------------------------------------------------
            if (!any_new)
                {
                    status = IloAlgorithm::Optimal;
                    break;
                }

            ++iter;
        }

    auto t_end = std::chrono::steady_clock::now();
    time = std::chrono::duration<IloNum>(t_end - t0).count();

    update_status();

    // Log final
    if (log_output)
        {
            print_summary("GERAÇÃO DE COLUNAS");
        }

    return (status == IloAlgorithm::Optimal);
}
