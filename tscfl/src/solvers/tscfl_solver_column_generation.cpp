/*
COS888

tscfl_solver_column_generation.cpp

Gabriel Braun, 2025
*/

#include "solvers/tscfl_solver_column_generation.hpp"

#include <iomanip>
#include <iostream>

TSCFLSolverColumnGeneration::TSCFLSolverColumnGeneration(
    const TSCFLInstance &inst_, Subproblem::Mode smode
)
    : TSCFLSolver(inst_),
      model(env),
      cplex(env),
      subproblem(Subproblem::create(inst_, smode)),
      obj(env),
      var_a(env, inst_.nI, 0.0, 1.0),
      var_b(env, inst_.nJ, 0.0, 1.0),
      constr_l1(env, inst_.nI),
      constr_l2(env, inst_.nJ),
      constr_m2(env, inst_.nK),
      constrs_v(env, inst_.nJ),
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
            constrs_v[j] = IloRangeArray(env, inst.nK);
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

void
TSCFLSolverColumnGeneration::build_initial_model()
{
    // RESTRIÇÕES DO RMP
    // Restrições de capacidade das plantas: constr_l1[i]
    for (IloInt i = 0; i < inst.nI; ++i)
        {
            IloExpr e(env);
            e -= inst.p[i] * var_a[i];
            constr_l1[i] = (e <= 0.0);
            model.add(constr_l1[i]);
            e.end();
        }

    // Restrições de capacidade dos depósitos: constr_l2[j]
    for (IloInt j = 0; j < inst.nJ; ++j)
        {
            IloExpr e(env);
            e -= inst.q[j] * var_b[j];
            constr_l2[j] = (e <= 0.0);
            model.add(constr_l2[j]);
            e.end();
        }

    // Restrições de convexidade/demanda por cliente: constr_m2[k]
    for (IloInt k = 0; k < inst.nK; ++k)
        {
            IloExpr e(env);
            constr_m2[k] = (e == 1.0);
            model.add(constr_m2[k]);
            e.end();
        }

    // Restrição de vínculo: z_{k,t} <= b_j  (para par (j,k))
    for (IloInt j = 0; j < inst.nJ; ++j)
        {
            for (IloInt k = 0; k < inst.nK; ++k)
                {
                    IloExpr e(env);
                    e -= var_b[j];
                    constrs_v[j][k] = (e <= 0.0);
                    model.add(constrs_v[j][k]);
                    e.end();
                }
        }

    // FUNÇÃO OBJETIVO
    obj = IloMinimize(env, IloScalProd(inst.f, var_a) + IloScalProd(inst.g, var_b));
    model.add(obj);

    // CONSTRUÇÃO DAS COLUNAS INICIAIS
    IloNumTensor flow(env, inst.nI, inst.nJ, inst.nK);
    IloNumArray p_remaining = inst.p.copy();
    IloNumArray q_remaining = inst.q.copy();
    IloNumArray r_remaining = inst.r.copy();

    for (IloInt k = 0; k < inst.nK; ++k)
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
                    break;

                IloNum delta
                    = IloMin(r_remaining[k], IloMin(p_remaining[best_i], q_remaining[best_j]));

                if (delta <= EPS)
                    break;

                flow[best_i][best_j][k] += delta;
                p_remaining[best_i] -= delta;
                q_remaining[best_j] -= delta;
                r_remaining[k] -= delta;
            }

    // Criar colunas iniciais z[k][t] a partir de flow[i][j][k]
    for (IloInt k = 0; k < inst.nK; ++k)
        for (IloInt i = 0; i < inst.nI; ++i)
            for (IloInt j = 0; j < inst.nJ; ++j)
                {
                    if (flow[i][j][k] > EPS)
                        add_column_for_client(k, i, j);
                    // Fallback:
                    if (z[k].getSize() == 0)
                        add_column_for_client(k, 0, 0);
                }
}

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
    col += constrs_v[j][k](1.0);

    IloNumVar z_var(col, 0.0, IloInfinity, ILOFLOAT);
    z[k].add(z_var);
    model.add(z_var);
}

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

bool
TSCFLSolverColumnGeneration::solve(bool log_output, IloNum time_limit)
{
    auto &SP = *subproblem;

    IloNumArray a_h(env, inst.nI), b_h(env, inst.nJ);
    IloNumArray l1(env, inst.nI), l2(env, inst.nJ), m2(env, inst.nK);
    IloNumMatrix v(env, inst.nJ, inst.nK);

    // Log inicial
    if (log_output)
        {
            std::cout << "\n\n[CG] Iniciando Geração de Colunas\n\n"
                      //
                      << std::right << std::setw(5)
                      << "it"
                      //
                      << std::setw(10)
                      << "time(s)"
                      //
                      << std::setw(15)
                      << "LB"
                      //
                      << std::setw(15)
                      << "UB"
                      //
                      << std::setw(12)
                      << "gap"
                      //
                      << std::setw(10) << "cols"
                      << "\n"
                      << std::string(70, '-') << "\n"
                      << std::defaultfloat;
        }

    IloTimer timer(env);
    timer.start();

    while (true)
        {
            // Critério de parada: tempo
            time = timer.getTime();

            if (time_limit > 0.0)
                {
                    if (time >= time_limit)
                        break;

                    cplex.setParam(IloCplex::Param::TimeLimit, time_limit - time);
                }

            // Resolve o RMP atual
            if (!cplex.solve())
                {
                    status = cplex.getStatus();
                    std::cerr << "[CG] RMP inviável ou erro. Status = " << status << "\n";
                    return false;
                }

            status = cplex.getStatus();
            lb = cplex.getObjValue();

            // Heurística primal a partir de (a,b) fracionários do RMP
            SP.solve_primal_heuristic(cplex, var_a, var_b, a_h, b_h);

            if (SP.opt + EPS < ub)
                {
                    ub = SP.opt;
                    a = a_h;
                    b = b_h;
                }

            update_gap();

            // Recuperação das variáveis duais
            cplex.getDuals(l1, constr_l1);
            cplex.getDuals(l2, constr_l2);
            cplex.getDuals(m2, constr_m2);
            for (IloInt j = 0; j < inst.nJ; ++j)
                cplex.getDuals(v[j], constrs_v[j]);

            // Pricing por cliente k
            bool any_new = false;
            for (IloInt k = 0; k < inst.nK; ++k)
                {
                    const IloNum rk = inst.r[k];

                    IloNum best_rc = 0.0;
                    IloInt best_i = -1;
                    IloInt best_j = -1;

                    for (IloInt i = 0; i < inst.nI; ++i)
                        for (IloInt j = 0; j < inst.nJ; ++j)
                            {
                                IloNum rc = rk * (inst.c[i][j] + inst.d[j][k] - l1[i] - l2[j])
                                            - m2[k] - v[j][k];

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

            // Log parcial
            if (log_output && (iter % PRINT_EVERY == 0 || !any_new))
                {
                    IloInt num_columns = get_num_columns();

                    std::cout << std::right << std::setw(5)
                              << iter
                              //
                              << std::fixed << std::setprecision(1) << std::setw(10)
                              << time
                              //
                              << std::setprecision(0) << std::setw(15)
                              << lb
                              //
                              << std::setw(15)
                              << ub
                              //
                              << std::scientific << std::setprecision(2) << std::setw(12)
                              << gap
                              //
                              << std::fixed << std::setw(10)
                              << num_columns
                              //
                              << "\n"
                              << std::defaultfloat;
                }

            // Critério de parada: nenhuma coluna com rc < 0
            if (!any_new)
                {
                    status = IloAlgorithm::Optimal;
                    break;
                }

            ++iter;
        }

    timer.stop();

    // Log final
    print_summary("GERAÇÃO DE COLUNAS");

    return (status == IloAlgorithm::Optimal || status == IloAlgorithm::Feasible);
}
