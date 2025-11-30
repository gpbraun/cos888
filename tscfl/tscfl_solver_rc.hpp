/*
COS888

Relax-and-Cut para o TSCFL
(Lagrangeano + subgradiente + heurística primal + cortes de fluxo).

- NÃO usa CPLEX para resolver subproblemas.
- Usa IloEnv apenas para IloNumArray / IloNumMatrix.
- Heurística de segundo estágio usando Worker (Dual / Primal / Net).

Gabriel Braun, 2025
*/

#pragma once

#include <ilcplex/ilocplex.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <memory>
#include <unordered_set>

#include "tscfl_instance.hpp"
#include "workers/tscfl_worker_dual.hpp"
#include "workers/tscfl_worker_primal.hpp"
#include "workers/tscfl_worker_net.hpp"

ILOSTLBEGIN

// =====================================================================
//  CONSTANTES DO RELAX-AND-CUT
// =====================================================================

static const double EPSILON0 = 2.0;  // epsilon inicial (Polyak)
static const int MAX_NO_IMPROV = 50; // iterações sem melhorar LB antes de reduzir epsilon
static const int EXTRA_AGE = 5;      // vida extra de cortes em PA antes de ir pra CI

static const int SOLVE_HEURISTIC_EVERY = 50; // frequência da heurística
static const int MAX_NEW_CUTS_PER_ITER = 10; // máx. novos cortes por iteração
static const int PRINT_EVERY = 10;           // frequência do log

// =====================================================================
//  SOLVER TSCFL: Relax-and-Cut
// =====================================================================

class TSCFLSolverRelaxAndCut
{
public:
    const TSCFLInstance &inst;

    // Resultados globais
    double lb{-IloInfinity};
    double ub{IloInfinity};
    double gap{IloInfinity};
    double time{0.0};
    IloAlgorithm::Status status{IloAlgorithm::Unknown};

private:
    static constexpr double EPSILON0 = 2.0;          // epsilon inicial (Polyak)
    static constexpr int MAX_NO_IMPROV = 50;         // iterações sem melhorar LB antes de reduzir epsilon
    static constexpr int EXTRA_AGE = 2;              // vida extra de cortes em PA antes de ir pra CI
    static constexpr int SOLVE_HEURISTIC_EVERY = 50; // frequência da heurística
    static constexpr int MAX_NEW_CUTS_PER_ITER = 1;  // máx. novos cortes por iteração
    static constexpr int PRINT_EVERY = 10;           // frequência do log

    IloEnv &env;

    // Worker para subproblema de fluxo mínimo da heurística
    std::unique_ptr<Worker> worker;

    // Demanda total
    double total_demand{0.0};

    // Parâmetro do subgradiente
    double epsilon{EPSILON0};
    int max_no_improv{MAX_NO_IMPROV};

    // Multiplicadores de Lagrange
    IloNumArray u;  // tamanho nI
    IloNumArray v;  // tamanho nJ
    IloNumArray gu; // subgradiente u[i]
    IloNumArray gv; // subgradiente v[j]

    // Solução lagrangeana corrente
    IloNumArray a_lr;          // nI
    IloNumArray b_lr;          // nJ
    IloNumMatrix x_lr;         // nI x nJ
    IloNumMatrix y_lr;         // nJ x nK
    IloNumArray plant_flow_lr; // nI
    IloNumArray depot_flow_lr; // nJ

    // Melhor solução primal (para UB)
    IloNumArray a_best; // nI
    IloNumArray b_best; // nJ

    // Custos adicionais de cortes
    IloNumMatrix cut_cost_x; // nI x nJ
    IloNumMatrix cut_cost_y; // nJ x nK
    IloNumArray cut_fix_a;   // nI
    IloNumArray cut_fix_b;   // nJ

    // =================================================================
    //  Estrutura de Flow Cover + conjuntos CA / PA / CI
    // =================================================================
    struct FlowCoverCut
    {
        enum Type
        {
            PLANT,
            DEPOT
        } type;
        int index; // i (PLANT) ou j (DEPOT)

        // Índices achatados:
        //  x_ij → idx = i * nJ + j
        //  y_jk → idx = j * nK + k
        IloIntArray idx_x;
        IloNumArray coef_x;
        IloIntArray idx_y;
        IloNumArray coef_y;

        double coef_open; // coeficiente da_i ou b_j (tipicamente -p_i ou -q_j)
        double rhs;

        double lambda;    // multiplicador λ ≥ 0
        double violation; // lhs - rhs na solução LR
        int age;          // idade desde última violação

        enum Status
        {
            CA,
            PA,
            CI
        } status;

        FlowCoverCut(IloEnv env_)
            : type(PLANT),
              index(-1),
              idx_x(env_), coef_x(env_),
              idx_y(env_), coef_y(env_),
              coef_open(0.0), rhs(0.0),
              lambda(0.0), violation(0.0), age(0),
              status(CI)
        {
        }
    };

    std::vector<FlowCoverCut> cuts;
    std::unordered_set<std::size_t> cut_hashes; // para evitar cortes duplicados

public:
    // mode (worker do subproblema de fluxo mínimo):
    // 0 -> WorkerDual
    // 1 -> WorkerPrimal
    // 2 -> WorkerNet (default)
    explicit TSCFLSolverRelaxAndCut(const TSCFLInstance &inst_, int mode = 2)
        : inst(inst_),
          env(inst_.env),
          worker(nullptr),
          u(env, inst_.nI),
          v(env, inst_.nJ),
          gu(env, inst_.nI),
          gv(env, inst_.nJ),
          a_lr(env, inst_.nI),
          b_lr(env, inst_.nJ),
          x_lr(env, inst_.nI, inst_.nJ),
          y_lr(env, inst_.nJ, inst_.nK),
          plant_flow_lr(env, inst_.nI),
          depot_flow_lr(env, inst_.nJ),
          a_best(env, inst_.nI),
          b_best(env, inst_.nJ),
          cut_cost_x(env, inst_.nI, inst_.nJ),
          cut_cost_y(env, inst_.nJ, inst_.nK),
          cut_fix_a(env, inst_.nI),
          cut_fix_b(env, inst_.nJ)
    {
        total_demand = IloSum(inst_.r);

        switch (mode)
        {
        case 0:
            worker = std::make_unique<WorkerDual>(inst_);
            break;
        case 1:
            worker = std::make_unique<WorkerPrimal>(inst_);
            break;
        case 2:
            worker = std::make_unique<WorkerNet>(inst_);
            break;
        default:
            throw std::invalid_argument("Invalid Relax-and-Cut worker mode (must be 0, 1, or 2).");
        }
    }

    ~TSCFLSolverRelaxAndCut() = default;

private:
    // -----------------------------------------------------------------
    // Helpers: hash de cortes (usa std::hash<int> + hash_combine)
    // -----------------------------------------------------------------
    static inline void hash_combine(std::size_t &seed, std::size_t value)
    {
        seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
    }

    std::size_t cut_hash(const FlowCoverCut &cut) const
    {
        std::size_t seed = 0;

        hash_combine(seed, std::hash<int>()(static_cast<int>(cut.type)));
        hash_combine(seed, std::hash<int>()(cut.index));

        if (cut.type == FlowCoverCut::PLANT)
        {
            for (IloInt t = 0; t < cut.idx_x.getSize(); ++t)
                hash_combine(seed, std::hash<int>()(cut.idx_x[t]));
        }
        else
        {
            for (IloInt t = 0; t < cut.idx_y.getSize(); ++t)
                hash_combine(seed, std::hash<int>()(cut.idx_y[t]));
        }

        return seed;
    }

    // -----------------------------------------------------------------
    // Norma 2 ao quadrado do subgradiente (u, v, λ)
    // -----------------------------------------------------------------
    double norm2_squared() const
    {
        double s = 0.0;

        for (int i = 0; i < inst.nI; ++i)
            s += gu[i] * gu[i];

        for (int j = 0; j < inst.nJ; ++j)
            s += gv[j] * gv[j];

        for (const auto &cut : cuts)
        {
            if (cut.status != FlowCoverCut::CI)
                s += cut.violation * cut.violation;
        }

        return s;
    }

    // -----------------------------------------------------------------
    // Constrói custos adicionais de cortes (a partir de λ)
    // -----------------------------------------------------------------
    void build_cut_costs()
    {
        const int nI = inst.nI;
        const int nJ = inst.nJ;
        const int nK = inst.nK;

        // zera
        for (int i = 0; i < nI; ++i)
        {
            cut_fix_a[i] = 0.0;
            for (int j = 0; j < nJ; ++j)
                cut_cost_x[i][j] = 0.0;
        }
        for (int j = 0; j < nJ; ++j)
        {
            cut_fix_b[j] = 0.0;
            for (int k = 0; k < nK; ++k)
                cut_cost_y[j][k] = 0.0;
        }

        // acumula contribuições de cortes ativos
        for (const auto &cut : cuts)
        {
            if (cut.status == FlowCoverCut::CI || cut.lambda <= EPS)
                continue;

            if (cut.type == FlowCoverCut::PLANT)
            {
                int i = cut.index;

                for (IloInt t = 0; t < cut.idx_x.getSize(); ++t)
                {
                    int idx = cut.idx_x[t];
                    int ii = idx / inst.nJ;
                    int jj = idx % inst.nJ;
                    if (ii == i)
                        cut_cost_x[ii][jj] += cut.lambda * cut.coef_x[t];
                }

                cut_fix_a[i] += cut.lambda * cut.coef_open;
            }
            else // DEPOT
            {
                int j = cut.index;

                for (IloInt t = 0; t < cut.idx_y.getSize(); ++t)
                {
                    int idx = cut.idx_y[t];
                    int jj = idx / inst.nK;
                    int kk = idx % inst.nK;
                    if (jj == j)
                        cut_cost_y[jj][kk] += cut.lambda * cut.coef_y[t];
                }

                cut_fix_b[j] += cut.lambda * cut.coef_open;
            }
        }
    }

    // -----------------------------------------------------------------
    // Resolve o problema lagrangeano para (u, v, λ) fixos
    // Retorna z_LR (cortes já incorporados em cut_cost_* / cut_fix_*).
    // -----------------------------------------------------------------
    double solve_lagrangian()
    {
        const int nI = inst.nI;
        const int nJ = inst.nJ;
        const int nK = inst.nK;

        build_cut_costs();

        // zera fluxos
        for (int i = 0; i < nI; ++i)
            for (int j = 0; j < nJ; ++j)
                x_lr[i][j] = 0.0;

        for (int j = 0; j < nJ; ++j)
            for (int k = 0; k < nK; ++k)
                y_lr[j][k] = 0.0;

        // Para cada cliente k: escolhe (i,j) de menor custo reduzido
        for (int k = 0; k < nK; ++k)
        {
            double rk = inst.r[k];
            if (rk <= EPS)
                continue;

            double best_cost = IloInfinity;
            int best_i = -1;
            int best_j = -1;

            for (int i = 0; i < nI; ++i)
            {
                for (int j = 0; j < nJ; ++j)
                {
                    double cost =
                        inst.c[i][j] + inst.d[j][k] +
                        u[i] + v[j] +
                        cut_cost_x[i][j] + cut_cost_y[j][k];

                    if (cost < best_cost)
                    {
                        best_cost = cost;
                        best_i = i;
                        best_j = j;
                    }
                }
            }

            if (best_i == -1 || best_j == -1)
                continue;

            x_lr[best_i][best_j] += rk;
            y_lr[best_j][k] += rk;
        }

        // Fluxos agregados
        for (int i = 0; i < nI; ++i)
        {
            double sum = 0.0;
            for (int j = 0; j < nJ; ++j)
                sum += x_lr[i][j];
            plant_flow_lr[i] = sum;
        }

        for (int j = 0; j < nJ; ++j)
        {
            double sum = 0.0;
            for (int k = 0; k < nK; ++k)
                sum += y_lr[j][k];
            depot_flow_lr[j] = sum;
        }

        // a_lr / b_lr (coeficiente reduzido < 0 => abre)
        for (int i = 0; i < nI; ++i)
        {
            double red_fix = inst.f[i] - u[i] * inst.p[i] + cut_fix_a[i];
            a_lr[i] = (red_fix < 0.0 ? 1.0 : 0.0);
        }
        for (int j = 0; j < nJ; ++j)
        {
            double red_fix = inst.g[j] - v[j] * inst.q[j] + cut_fix_b[j];
            b_lr[j] = (red_fix < 0.0 ? 1.0 : 0.0);
        }

        // Subgradientes (capacidade)
        for (int i = 0; i < nI; ++i)
            gu[i] = plant_flow_lr[i] - inst.p[i] * a_lr[i];

        for (int j = 0; j < nJ; ++j)
            gv[j] = depot_flow_lr[j] - inst.q[j] * b_lr[j];

        // Valor da lagrangeana
        double cost_fix = 0.0;
        for (int i = 0; i < nI; ++i)
            cost_fix += inst.f[i] * a_lr[i];
        for (int j = 0; j < nJ; ++j)
            cost_fix += inst.g[j] * b_lr[j];

        double cost_flow = 0.0;
        for (int i = 0; i < nI; ++i)
            for (int j = 0; j < nJ; ++j)
                cost_flow += inst.c[i][j] * x_lr[i][j];

        for (int j = 0; j < nJ; ++j)
            for (int k = 0; k < nK; ++k)
                cost_flow += inst.d[j][k] * y_lr[j][k];

        double lag_cap = 0.0;
        for (int i = 0; i < nI; ++i)
            lag_cap += u[i] * gu[i];
        for (int j = 0; j < nJ; ++j)
            lag_cap += v[j] * gv[j];

        // cortes já foram incorporados via cut_cost_* / cut_fix_*
        double z_lr = cost_fix + cost_flow + lag_cap;
        return z_lr;
    }

    // -----------------------------------------------------------------
    // Separa Flow Covers a partir da solução LR
    // -----------------------------------------------------------------
    void separate_flow_covers()
    {
        const int nI = inst.nI;
        const int nJ = inst.nJ;
        const int nK = inst.nK;

        int new_cuts = 0;

        // Plantas
        for (int i = 0; i < nI && new_cuts < MAX_NEW_CUTS_PER_ITER; ++i)
        {
            double cap_i = inst.p[i] * a_lr[i];
            double flow_i = plant_flow_lr[i];

            if (flow_i <= cap_i + EPS)
                continue;

            FlowCoverCut cut(env);
            cut.type = FlowCoverCut::PLANT;
            cut.index = i;
            cut.coef_open = -inst.p[i];
            cut.rhs = 0.0;
            cut.lambda = 0.0;
            cut.age = 0;
            cut.status = FlowCoverCut::CA;

            for (int j = 0; j < nJ; ++j)
            {
                if (x_lr[i][j] > EPS)
                {
                    int idx = i * nJ + j;
                    cut.idx_x.add(idx);
                    cut.coef_x.add(1.0);
                }
            }

            if (cut.idx_x.getSize() == 0)
                continue;

            std::size_t h = cut_hash(cut);
            if (!cut_hashes.insert(h).second)
                continue; // corte duplicado

            cuts.push_back(cut);
            ++new_cuts;
        }

        // Depósitos
        for (int j = 0; j < nJ && new_cuts < MAX_NEW_CUTS_PER_ITER; ++j)
        {
            double cap_j = inst.q[j] * b_lr[j];
            double flow_j = depot_flow_lr[j];

            if (flow_j <= cap_j + EPS)
                continue;

            FlowCoverCut cut(env);
            cut.type = FlowCoverCut::DEPOT;
            cut.index = j;
            cut.coef_open = -inst.q[j];
            cut.rhs = 0.0;
            cut.lambda = 0.0;
            cut.age = 0;
            cut.status = FlowCoverCut::CA;

            for (int k = 0; k < nK; ++k)
            {
                if (y_lr[j][k] > EPS)
                {
                    int idx = j * nK + k;
                    cut.idx_y.add(idx);
                    cut.coef_y.add(1.0);
                }
            }

            if (cut.idx_y.getSize() == 0)
                continue;

            std::size_t h = cut_hash(cut);
            if (!cut_hashes.insert(h).second)
                continue; // corte duplicado

            cuts.push_back(cut);
            ++new_cuts;
        }
    }

    // -----------------------------------------------------------------
    // Atualiza violações e conjuntos CA/PA/CI
    // -----------------------------------------------------------------
    void update_cut_sets()
    {
        for (auto &cut : cuts)
        {
            double lhs = 0.0;

            // x_ij
            for (IloInt t = 0; t < cut.idx_x.getSize(); ++t)
            {
                int idx = cut.idx_x[t];
                int i = idx / inst.nJ;
                int j = idx % inst.nJ;
                lhs += cut.coef_x[t] * x_lr[i][j];
            }

            // y_jk
            for (IloInt t = 0; t < cut.idx_y.getSize(); ++t)
            {
                int idx = cut.idx_y[t];
                int j = idx / inst.nK;
                int k = idx % inst.nK;
                lhs += cut.coef_y[t] * y_lr[j][k];
            }

            // termo de abertura
            if (cut.type == FlowCoverCut::PLANT)
            {
                int i = cut.index;
                lhs += cut.coef_open * a_lr[i];
            }
            else
            {
                int j = cut.index;
                lhs += cut.coef_open * b_lr[j];
            }

            cut.violation = lhs - cut.rhs;

            if (cut.violation > EPS)
            {
                cut.status = FlowCoverCut::CA;
                cut.age = 0;
            }
            else
            {
                ++cut.age;
                if (cut.age <= EXTRA_AGE && cut.lambda > EPS)
                {
                    cut.status = FlowCoverCut::PA;
                }
                else
                {
                    cut.status = FlowCoverCut::CI;
                    cut.lambda = 0.0;
                }
            }
        }
    }

    // -----------------------------------------------------------------
    // Heurística primal (UB) usando Worker (Dual / Primal / Net)
    // -----------------------------------------------------------------
    void run_primal_heuristic()
    {
        const int nI = inst.nI;
        const int nJ = inst.nJ;

        if (total_demand <= EPS || !worker)
            return;

        // 1) Plantas abertas (ordem guiada por a_lr e f/p)
        std::vector<int> ordI(nI);
        for (int i = 0; i < nI; ++i)
            ordI[i] = i;

        std::sort(ordI.begin(), ordI.end(),
                  [&](int i, int j)
                  {
                      if (std::fabs(a_lr[i] - a_lr[j]) > EPS)
                          return a_lr[i] > a_lr[j];

                      double ratio_i = inst.p[i] > EPS ? inst.f[i] / inst.p[i] : IloInfinity;
                      double ratio_j = inst.p[j] > EPS ? inst.f[j] / inst.p[j] : IloInfinity;
                      return ratio_i < ratio_j;
                  });

        std::vector<char> openI(nI, 0);
        double capI = 0.0;
        for (int pos = 0; pos < nI && capI + EPS < total_demand; ++pos)
        {
            int i = ordI[pos];
            if (inst.p[i] <= EPS)
                continue;
            openI[i] = 1;
            capI += inst.p[i];
        }
        if (capI + EPS < total_demand)
            return;

        // 2) Depósitos abertos
        std::vector<int> ordJ(nJ);
        for (int j = 0; j < nJ; ++j)
            ordJ[j] = j;

        std::sort(ordJ.begin(), ordJ.end(),
                  [&](int j1, int j2)
                  {
                      if (std::fabs(b_lr[j1] - b_lr[j2]) > EPS)
                          return b_lr[j1] > b_lr[j2];

                      double ratio1 = inst.q[j1] > EPS ? inst.g[j1] / inst.q[j1] : IloInfinity;
                      double ratio2 = inst.q[j2] > EPS ? inst.g[j2] / inst.q[j2] : IloInfinity;
                      return ratio1 < ratio2;
                  });

        std::vector<char> openJ(nJ, 0);
        double capJ = 0.0;
        for (int pos = 0; pos < nJ && capJ + EPS < total_demand; ++pos)
        {
            int j = ordJ[pos];
            if (inst.q[j] <= EPS)
                continue;
            openJ[j] = 1;
            capJ += inst.q[j];
        }
        if (capJ + EPS < total_demand)
            return;

        // 3) Monta (a_h, b_h) e chama Worker
        IloNumArray a_h(env, nI);
        IloNumArray b_h(env, nJ);

        for (int i = 0; i < nI; ++i)
            a_h[i] = openI[i] ? 1.0 : 0.0;
        for (int j = 0; j < nJ; ++j)
            b_h[j] = openJ[j] ? 1.0 : 0.0;

        double cost_fix = 0.0;
        for (int i = 0; i < nI; ++i)
            cost_fix += inst.f[i] * a_h[i];
        for (int j = 0; j < nJ; ++j)
            cost_fix += inst.g[j] * b_h[j];

        double flow_cost = 0.0;
        try
        {
            worker->solve(a_h, b_h);
            flow_cost = worker->theta;
        }
        catch (...)
        {
            return;
        }

        double ub_cand = cost_fix + flow_cost;
        if (ub_cand + EPS < ub)
        {
            ub = ub_cand;
            for (int i = 0; i < nI; ++i)
                a_best[i] = a_h[i];
            for (int j = 0; j < nJ; ++j)
                b_best[j] = b_h[j];
        }
    }

public:
    // -----------------------------------------------------------------
    //  Método principal
    // -----------------------------------------------------------------
    bool solve(bool log_output = true, double time_limit = -1.0)
    {
        lb = -IloInfinity;
        ub = IloInfinity;
        gap = IloInfinity;
        status = IloAlgorithm::Unknown;
        time = 0.0;

        epsilon = EPSILON0;
        int last_improv_iter = 0;
        double best_lb = lb;

        // multipliers e melhor solução primal começam em zero
        for (int i = 0; i < inst.nI; ++i)
        {
            u[i] = 0.0;
            gu[i] = 0.0;
            a_best[i] = 0.0;
        }
        for (int j = 0; j < inst.nJ; ++j)
        {
            v[j] = 0.0;
            gv[j] = 0.0;
            b_best[j] = 0.0;
        }

        cuts.clear();
        cut_hashes.clear();

        auto t0 = std::chrono::steady_clock::now();
        int iter = 0;

        if (log_output)
        {
            std::cout << "[RC] Iniciando Relax-and-Cut\n";
            std::cout << "[RC] time_limit = " << time_limit
                      << ", epsilon0 = " << EPSILON0 << "\n";
            std::cout << "[RC] it   time(s)     z_LR        LB         UB"
                         "         gap    step    ||g||^2   |CA| |PA| |CI|\n";
        }

        while (true)
        {
            auto t1 = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(t1 - t0).count();
            if (time_limit > 0.0 && elapsed >= time_limit)
                break;

            // 1) Lagrangeano
            double z_lr = solve_lagrangian();

            // 2) Cortes (separação + atualização dos conjuntos)
            separate_flow_covers();
            update_cut_sets();

            // 3) Atualiza LB
            if (z_lr > lb + EPS)
            {
                lb = z_lr;
                best_lb = lb;
                last_improv_iter = iter;
            }

            // 4) Heurística primal (UB)
            if (iter == 0 ||
                (iter % SOLVE_HEURISTIC_EVERY == 0) ||
                (z_lr > best_lb + EPS))
            {
                run_primal_heuristic();
            }

            // 5) Norma do subgradiente
            double norm2 = norm2_squared();

            // 6) Passo de Polyak
            double step = 0.0;
            if (ub < IloInfinity && norm2 > EPS)
            {
                step = epsilon * (ub - z_lr) / norm2;
                if (step < 0.0)
                    step = 0.0;
            }

            // 7) Atualiza multiplicadores
            if (step > 0.0)
            {
                for (int i = 0; i < inst.nI; ++i)
                    u[i] = std::max(0.0, u[i] + step * gu[i]);

                for (int j = 0; j < inst.nJ; ++j)
                    v[j] = std::max(0.0, v[j] + step * gv[j]);

                for (auto &cut : cuts)
                {
                    if (cut.status != FlowCoverCut::CI)
                        cut.lambda = std::max(0.0, cut.lambda + step * cut.violation);
                }
            }

            // 8) Ajuste de epsilon (sem epsilon mínimo)
            if (iter - last_improv_iter >= max_no_improv)
            {
                epsilon *= 0.5;
                last_improv_iter = iter;
            }

            // 9) Gap atual
            if (ub < IloInfinity && lb > -IloInfinity)
            {
                gap = (ub - lb) / std::max(1.0, std::abs(ub));
            }

            // 10) Log
            if (log_output && (iter % PRINT_EVERY == 0))
            {
                int ca = 0, pa = 0, ci = 0;
                for (const auto &cut : cuts)
                {
                    if (cut.status == FlowCoverCut::CA)
                        ++ca;
                    else if (cut.status == FlowCoverCut::PA)
                        ++pa;
                    else
                        ++ci;
                }

                double elapsed_it =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

                std::cout << "[RC] " << std::setw(4) << iter
                          << " " << std::setw(8) << std::fixed << std::setprecision(2) << elapsed_it
                          << " " << std::scientific << std::setprecision(6) << z_lr
                          << " " << std::scientific << std::setprecision(6) << lb
                          << " " << std::scientific << std::setprecision(6) << ub
                          << " " << std::fixed << std::setprecision(6) << gap
                          << " " << std::scientific << std::setprecision(6) << step
                          << " " << std::scientific << std::setprecision(6) << norm2
                          << " " << std::setw(4) << ca
                          << " " << std::setw(4) << pa
                          << " " << std::setw(4) << ci
                          << "\n";
            }

            // 11) Critério de parada por gap
            if (gap <= MIP_GAP && ub < IloInfinity && lb > -IloInfinity)
            {
                status = IloAlgorithm::Optimal;
                break;
            }

            ++iter;
        }

        auto t_end = std::chrono::steady_clock::now();
        time = std::chrono::duration<double>(t_end - t0).count();

        if (status == IloAlgorithm::Unknown)
        {
            if (ub < IloInfinity && lb > -IloInfinity)
                status = IloAlgorithm::Feasible;
        }

        if (log_output)
        {
            std::cout << "\n[RC] Relax-and-Cut finalizado.\n";
            std::cout << "LB    = " << lb << "\n";
            std::cout << "UB    = " << ub << "\n";
            std::cout << "gap   = " << gap << "\n";
            std::cout << "status= " << status << "\n";
            std::cout << "time  = " << time << " s\n";
            if (status == IloAlgorithm::Feasible)
                std::cout << "[RC] Parada por time_limit.\n";
        }

        return (status == IloAlgorithm::Optimal);
    }
};
