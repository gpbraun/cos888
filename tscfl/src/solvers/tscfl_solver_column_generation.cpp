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
      constrs_v(env, inst_.nJ)
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

    buildModel();
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
TSCFLSolverColumnGeneration::buildModel()
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
        for (IloInt k = 0; k < inst.nK; ++k)
            {
                IloExpr e(env);
                e -= var_b[j];
                constrs_v[j][k] = (e <= 0.0);
                model.add(constrs_v[j][k]);
                e.end();
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
                        addColumn(k, i, j);
                    // Fallback:
                    if (z[k].getSize() == 0)
                        addColumn(k, 0, 0);
                }
}

void
TSCFLSolverColumnGeneration::addColumn(int k, int i, int j)
{
    col_info[k].push_back({ i, j });

    // Custo do padrão (i,j) para o cliente k:
    const IloNum rk = inst.r[k];
    const IloNum cost = rk * (inst.c[i][j] + inst.d[j][k]);

    // Coluna na função objetivo e nas restrições
    IloNumColumn col = obj(cost);
    col += constr_l1[i](rk);
    col += constr_l2[j](rk);
    col += constr_m2[k](1.0);
    col += constrs_v[j][k](1.0);

    IloNumVar z_var(col, 0.0, IloInfinity, ILOFLOAT);
    z[k].add(z_var);
    model.add(z_var);
}

IloInt
TSCFLSolverColumnGeneration::getNumColumns() const
{
    IloInt total = 0;
    for (IloInt k = 0; k < inst.nK; ++k)
        {
            total += z[k].getSize();
        }
    return total;
}

void
TSCFLSolverColumnGeneration::solve(bool log_output, IloNum time_limit)
{
    auto &SP = *subproblem;

    IloNumArray l1(env, inst.nI), l2(env, inst.nJ), m2(env, inst.nK);
    IloNumMatrix v(env, inst.nJ, inst.nK);

    // Log inicial
    if (log_output)
        {
            // clang-format off
            std::cout << "\n\n[CG] Iniciando Geração de Colunas\n\n"
                      << std::right
                      << std::setw(5)  << "it"
                      << std::setw(10) << "time(s)"
                      << std::setw(15) << "LB"
                      << std::setw(15) << "UB"
                      << std::setw(12) << "gap"
                      << std::setw(10) << "cols"
                      << "\n" << std::string(70, '-') << "\n"
                      << std::defaultfloat;
            // clang-format on
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
                    break;
                }

            status = cplex.getStatus();
            lb = cplex.getObjValue();

            // Heurística primal a partir de (a,b) fracionários do RMP
            SP.update(cplex, var_a, var_b, true);
            SP.solve();

            if (SP.opt + EPS < ub)
                {
                    ub = SP.opt;
                    a = SP.a;
                    b = SP.b;
                }

            updateGap();

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
                                IloNum rc = rk * (inst.c[i][j] + inst.d[j][k]);
                                rc -= rk * (l1[i] + l2[j]) + m2[k] + v[j][k];

                                if (rc < best_rc - EPS)
                                    {
                                        best_rc = rc;
                                        best_i = i;
                                        best_j = j;
                                    }
                            }

                    if (best_i != -1)
                        {
                            addColumn(k, best_i, best_j);
                            any_new = true;
                        }
                }

            // Log parcial
            if (log_output && (iter % PRINT_EVERY == 0 || !any_new))
                {
                    IloInt num_columns = getNumColumns();
                    // clang-format off
                    std::cout << std::right
                              << std::setw(5) << iter
                              << std::fixed << std::setprecision(1)
                              << std::setw(10) << time
                              << std::setprecision(0)
                              << std::setw(15) << lb
                              << std::setw(15) << ub
                              << std::scientific << std::setprecision(2)
                              << std::setw(12) << gap
                              << std::fixed
                              << std::setw(10) << num_columns
                              << "\n"
                              << std::defaultfloat;
                    // clang-format on
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

    updateFlows();

    // Log final
    printSummary("GERAÇÃO DE COLUNAS");
}
