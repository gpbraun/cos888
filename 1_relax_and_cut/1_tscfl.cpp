/*
COS888

TSCFL com CPLEX

Gabriel Braun, 2025
*/

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <ilcplex/ilocplex.h>
ILOSTLBEGIN

// =====================================================================
//  UTILS
// =====================================================================

// Comparação com tolerância (equivalente ao np.isclose)
static inline bool is_close(double x, double y = 0.0, double tol = 1e-12)
{
    return std::abs(x - y) <= tol;
}

// Produto interno (dot)
static inline double dot(const std::vector<double> &a, const std::vector<double> &b)
{
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        s += a[i] * b[i];
    return s;
}

// Norma-2 ao quadrado
static inline double sqnorm(const std::vector<double> &a)
{
    double s = 0.0;
    for (double v : a)
        s += v * v;
    return s;
}

// intervalos e produtos cartesianos
static inline std::vector<int> range_int(int n)
{
    std::vector<int> v(n);
    std::iota(v.begin(), v.end(), 0);
    return v;
}

static inline std::vector<std::pair<int, int>> cart_prod(int nA, int nB)
{
    std::vector<std::pair<int, int>> v;
    v.reserve(static_cast<size_t>(nA) * static_cast<size_t>(nB));
    for (int i = 0; i < nA; ++i)
        for (int j = 0; j < nB; ++j)
            v.emplace_back(i, j);
    return v;
}

// Acesso em matriz 2D
static inline size_t idx2(size_t i, size_t j, size_t ncols) { return i * ncols + j; }

// =====================================================================
//  INSTÂNCIA
// =====================================================================

//
// Instância do TSCFL
//
class TSCFLInstance
{
public:
    // Tamanhos
    int nI; // |I| plantas
    int nJ; // |J| depósitos
    int nK; // |K| clientes

    std::vector<double> f; // f_i  = custo fixo da planta i
    std::vector<double> g; // g_j  = custo fixo do depósito j
    std::vector<double> c; // c_ij = custo unitário planta i -> depósito j
    std::vector<double> d; // d_jk = custo unitário depósito j -> cliente k
    std::vector<double> p; // p_i  = capacidade da planta i
    std::vector<double> q; // q_j  = capacidade do depósito j
    std::vector<double> r; // r_k  = demanda do cliente k

    // Acesso conveniente
    inline double C(int i, int j) const { return c[idx2(i, j, nJ)]; }
    inline double &C(int i, int j) { return c[idx2(i, j, nJ)]; }
    inline double D(int j, int k) const { return d[idx2(j, k, nK)]; }
    inline double &D(int j, int k) { return d[idx2(j, k, nK)]; }

    // "Propriedades" estilo Python
    std::vector<int> I() const { return range_int(nI); }
    std::vector<int> J() const { return range_int(nJ); }
    std::vector<int> K() const { return range_int(nK); }

    std::vector<std::pair<int, int>> IJ() const { return cart_prod(nI, nJ); }
    std::vector<std::pair<int, int>> JK() const { return cart_prod(nJ, nK); }

    //
    // Carrega a instância a partir de um arquivo .txt
    //
    static TSCFLInstance from_txt(const std::string &path)
    {
        std::ifstream fin(path);
        if (!fin)
            throw std::runtime_error("Falha ao abrir arquivo: " + path);

        std::vector<double> arr;
        arr.reserve(1 << 20);
        double v;
        while (fin >> v)
            arr.push_back(v);
        if (arr.size() < 3)
            throw std::runtime_error("Instância inválida (header curto).");

        int nI = static_cast<int>(arr[0]);
        int nJ = static_cast<int>(arr[1]);
        int nK = static_cast<int>(arr[2]);
        std::vector<double> data(arr.begin() + 3, arr.end());

        size_t s1 = static_cast<size_t>(nK);
        size_t s2 = s1 + static_cast<size_t>(2 * nJ);
        size_t s3 = s2 + static_cast<size_t>(nI) * static_cast<size_t>(nJ);
        size_t s4 = s3 + static_cast<size_t>(2 * nI);
        size_t s5 = s4 + static_cast<size_t>(nJ) * static_cast<size_t>(nK);

        if (data.size() < s5)
            throw std::runtime_error("Instância inválida (tamanho inconsistente).");

        // r_k
        std::vector<double> r(data.begin(), data.begin() + s1);

        // (q_j, g_j)
        std::vector<double> q(nJ), g(nJ);
        {
            size_t off = s1;
            for (int j = 0; j < nJ; ++j)
            {
                q[j] = data[off + 2 * j + 0];
                g[j] = data[off + 2 * j + 1];
            }
        }

        // c_ij
        std::vector<double> c(static_cast<size_t>(nI) * static_cast<size_t>(nJ));
        {
            size_t off = s2;
            for (int i = 0; i < nI; ++i)
                for (int j = 0; j < nJ; ++j)
                    c[idx2(i, j, nJ)] = data[off + idx2(i, j, nJ)];
        }

        // (p_i, f_i)
        std::vector<double> p(nI), f(nI);
        {
            size_t off = s3;
            for (int i = 0; i < nI; ++i)
            {
                p[i] = data[off + 2 * i + 0];
                f[i] = data[off + 2 * i + 1];
            }
        }

        // d_jk
        std::vector<double> d(static_cast<size_t>(nJ) * static_cast<size_t>(nK));
        {
            size_t off = s4;
            for (int j = 0; j < nJ; ++j)
                for (int k = 0; k < nK; ++k)
                    d[idx2(j, k, nK)] = data[off + idx2(j, k, nK)];
        }

        TSCFLInstance inst;
        inst.nI = nI;
        inst.nJ = nJ;
        inst.nK = nK;
        inst.f = std::move(f);
        inst.g = std::move(g);
        inst.c = std::move(c);
        inst.d = std::move(d);
        inst.p = std::move(p);
        inst.q = std::move(q);
        inst.r = std::move(r);
        return inst;
    }
};

// =====================================================================
//  SOLVER
// =====================================================================

//
// Solver: Non-Delayed Relax-and-Cut (NDRC)
//
class RelaxAndCutTSCFL
{
public:
    const TSCFLInstance &inst;
    double gamma;
    int dual_keep;
    double tol_stop;

    int max_iter;
    int time_limit;
    bool log_output;

    std::vector<double> lamb;
    std::vector<int> lamb_age;

    double L_best;
    double z_best;
    double gap;

    std::vector<int> a_best;
    std::vector<int> b_best;

    int iter;
    double time_sec;

    RelaxAndCutTSCFL(
        const TSCFLInstance &inst_,
        double gamma_ = 1.0,
        int dual_keep_ = 5,
        double tol_stop_ = 1e-6,
        int max_iter_ = 10000,
        int time_limit_ = 1000,
        bool log_output_ = false)
        : inst(inst_),
          gamma(gamma_), dual_keep(dual_keep_), tol_stop(tol_stop_),
          max_iter(max_iter_), time_limit(time_limit_), log_output(log_output_)
    {

        lamb.assign(inst.nJ + inst.nK, 0.0);
        lamb_age.assign(inst.nJ + inst.nK, 0);

        L_best = -std::numeric_limits<double>::infinity();
        z_best = std::numeric_limits<double>::infinity();
        gap = std::numeric_limits<double>::infinity();

        a_best.assign(inst.nI, 0);
        b_best.assign(inst.nJ, 0);

        iter = 0;
        time_sec = 0.0;
    }

private:
    // Container do resultado do subproblema Lagrangeano
    struct LRPResult
    {
        double L_val;
        std::vector<int> a_rel;      // abertura relaxada de plantas
        std::vector<int> b_rel;      // abertura relaxada de depósitos
        std::vector<double> subgrad; // subgradiente (alpha,beta)
    };

    //
    // Resolve a relaxação Lagrangeana para obter LB
    //
    LRPResult _solve_lrp(const std::vector<double> &lamb_in)
    {
        const double TOL = 1e-12;

        // Separa multiplicadores: alpha (nJ) e beta (nK)
        std::vector<double> alph(inst.nJ), beta(inst.nK);
        for (int j = 0; j < inst.nJ; ++j)
            alph[j] = lamb_in[j];
        for (int k = 0; k < inst.nK; ++k)
            beta[k] = lamb_in[inst.nJ + k];

        // Custos reduzidos
        // ctil[i,j] = c[i,j] + alpha[j]
        // dtil[j,k] = d[j,k] - alpha[j] + beta[k]
        std::vector<double> ctil(static_cast<size_t>(inst.nI) * inst.nJ);
        for (int i = 0; i < inst.nI; ++i)
            for (int j = 0; j < inst.nJ; ++j)
                ctil[idx2(i, j, inst.nJ)] = inst.C(i, j) + alph[j];

        std::vector<double> dtil(static_cast<size_t>(inst.nJ) * inst.nK);
        for (int j = 0; j < inst.nJ; ++j)
            for (int k = 0; k < inst.nK; ++k)
                dtil[idx2(j, k, inst.nK)] = inst.D(j, k) - alph[j] + beta[k];

        // Aberturas relaxadas
        std::vector<int> a_rel(inst.nI, 0);
        std::vector<int> b_rel(inst.nJ, 0);

        // Constante dual: - beta·r
        double L_val = 0.0;
        for (int k = 0; k < inst.nK; ++k)
            L_val -= beta[k] * inst.r[k];

        // Agregadores para subgradientes
        std::vector<double> sum_x_j(inst.nJ, 0.0); // soma_i x[i,j]
        std::vector<double> sum_y_j(inst.nJ, 0.0); // soma_k y[j,k]
        std::vector<double> sum_y_k(inst.nK, 0.0); // soma_j y[j,k]

        // "Greedy take": dado vetor de capacidades (ordenado), preenche até cap_total
        auto greedy_take = [&](const std::vector<double> &sorted_caps, double cap_total)
        {
            std::vector<double> take(sorted_caps.size(), 0.0);
            if (is_close(cap_total, 0.0) || sorted_caps.empty())
                return take;

            std::vector<double> cum(sorted_caps.size(), 0.0);
            std::partial_sum(sorted_caps.begin(), sorted_caps.end(), cum.begin());
            auto it = std::upper_bound(cum.begin(), cum.end(), cap_total + 1e-16);
            size_t full_cnt = static_cast<size_t>(std::distance(cum.begin(), it));

            if (full_cnt > 0)
                std::copy(sorted_caps.begin(), sorted_caps.begin() + full_cnt, take.begin());

            double rem = cap_total - (full_cnt > 0 ? cum[full_cnt - 1] : 0.0);
            if (full_cnt < sorted_caps.size() && rem > 0.0 && !is_close(rem, 0.0))
            {
                take[full_cnt] = rem;
            }
            return take;
        };

        // Resolve uma planta i
        struct PlantRes
        {
            int i;
            bool open_i;
            double contrib;
            std::vector<int> js;
            std::vector<double> take_sorted;
        };
        auto solve_plant = [&](int i) -> PlantRes
        {
            std::vector<int> js;
            js.reserve(inst.nJ);
            for (int j = 0; j < inst.nJ; ++j)
            {
                double rc = ctil[idx2(i, j, inst.nJ)];
                if (rc < 0.0 && !is_close(rc, 0.0, TOL))
                    js.push_back(j);
            }
            if (js.empty())
                return {i, false, 0.0, {}, {}};

            // ordenar por custo reduzido crescente
            std::sort(js.begin(), js.end(), [&](int a, int b)
                      { return ctil[idx2(i, a, inst.nJ)] < ctil[idx2(i, b, inst.nJ)]; });

            std::vector<double> rc(js.size()), qj(js.size());
            for (size_t t = 0; t < js.size(); ++t)
            {
                rc[t] = ctil[idx2(i, js[t], inst.nJ)];
                qj[t] = inst.q[js[t]];
            }

            auto take_sorted = greedy_take(qj, inst.p[i]);
            double var_part = 0.0;
            for (size_t t = 0; t < js.size(); ++t)
                var_part += rc[t] * take_sorted[t];

            bool open_i = std::any_of(take_sorted.begin(), take_sorted.end(),
                                      [](double z)
                                      { return !is_close(z, 0.0); }) &&
                          (inst.f[i] + var_part < 0.0);

            if (open_i)
                return {i, true, inst.f[i] + var_part, js, take_sorted};
            return {i, false, 0.0, {}, {}};
        };

        // Resolve um depósito j
        struct DepotRes
        {
            int j;
            bool open_j;
            double contrib;
            std::vector<int> ks;
            std::vector<double> take_sorted;
        };
        auto solve_depot = [&](int j) -> DepotRes
        {
            std::vector<int> ks;
            ks.reserve(inst.nK);
            for (int k = 0; k < inst.nK; ++k)
            {
                double rc = dtil[idx2(j, k, inst.nK)];
                if (rc < 0.0 && !is_close(rc, 0.0, TOL))
                    ks.push_back(k);
            }
            if (ks.empty())
                return {j, false, 0.0, {}, {}};

            std::sort(ks.begin(), ks.end(), [&](int a, int b)
                      { return dtil[idx2(j, a, inst.nK)] < dtil[idx2(j, b, inst.nK)]; });

            std::vector<double> rc(ks.size()), rk(ks.size());
            for (size_t t = 0; t < ks.size(); ++t)
            {
                rc[t] = dtil[idx2(j, ks[t], inst.nK)];
                rk[t] = inst.r[ks[t]];
            }

            auto take_sorted = greedy_take(rk, inst.q[j]);
            double var_part = 0.0;
            for (size_t t = 0; t < ks.size(); ++t)
                var_part += rc[t] * take_sorted[t];

            bool open_j = std::any_of(take_sorted.begin(), take_sorted.end(),
                                      [](double z)
                                      { return !is_close(z, 0.0); }) &&
                          (inst.g[j] + var_part < 0.0);

            if (open_j)
                return {j, true, inst.g[j] + var_part, ks, take_sorted};
            return {j, false, 0.0, {}, {}};
        };

        // Resolução das plantas em paralelo
        {
            std::vector<std::future<PlantRes>> futs;
            futs.reserve(inst.nI);
            for (int i = 0; i < inst.nI; ++i)
                futs.emplace_back(std::async(std::launch::async, solve_plant, i));

            for (auto &f : futs)
            {
                PlantRes pr = f.get();
                if (pr.open_i)
                {
                    a_rel[pr.i] = 1;
                    L_val += pr.contrib;
                    for (size_t t = 0; t < pr.js.size(); ++t)
                        sum_x_j[pr.js[t]] += pr.take_sorted[t];
                }
            }
        }

        // Resolução dos depósitos em paralelo
        {
            std::vector<std::future<DepotRes>> futs;
            futs.reserve(inst.nJ);
            for (int j = 0; j < inst.nJ; ++j)
                futs.emplace_back(std::async(std::launch::async, solve_depot, j));

            for (auto &f : futs)
            {
                DepotRes dr = f.get();
                if (dr.open_j)
                {
                    b_rel[dr.j] = 1;
                    L_val += dr.contrib;

                    const double tj = std::accumulate(dr.take_sorted.begin(), dr.take_sorted.end(), 0.0);
                    if (!is_close(tj, 0.0))
                    {
                        sum_y_j[dr.j] = tj;
                        for (size_t t = 0; t < dr.ks.size(); ++t)
                            sum_y_k[dr.ks[t]] += dr.take_sorted[t];
                    }
                }
            }
        }

        // Subgradientes
        std::vector<double> subgrad(inst.nJ + inst.nK, 0.0);
        for (int j = 0; j < inst.nJ; ++j)
            subgrad[j] = sum_x_j[j] - sum_y_j[j];
        for (int k = 0; k < inst.nK; ++k)
            subgrad[inst.nJ + k] = sum_y_k[k] - inst.r[k];

        return {L_val, std::move(a_rel), std::move(b_rel), std::move(subgrad)};
    }

    //
    // Heurística Lagrangeana para obter UB
    //
    std::optional<double> _solve_heuristic(const std::vector<int> &a_open,
                                           const std::vector<int> &b_open)
    {
        const double total_r = std::accumulate(inst.r.begin(), inst.r.end(), 0.0);

        // Copia "aberturas" sugeridas pelo LRP
        std::vector<int> a_fix = a_open;
        std::vector<int> b_fix = b_open;

        // Capacidades totais atuais
        auto cap_sum = [](const std::vector<int> &open, const std::vector<double> &cap)
        {
            double s = 0.0;
            for (size_t i = 0; i < open.size(); ++i)
                if (open[i])
                    s += cap[i];
            return s;
        };
        double cap_p = cap_sum(a_fix, inst.p);
        double cap_q = cap_sum(b_fix, inst.q);

        // Garante capacidade total: abre itens mais baratos por unidade de capacidade
        auto ensure_cap = [&](std::vector<int> &open, const std::vector<double> &cap,
                              const std::vector<double> &fixed_cost, double &curr_cap)
        {
            if (curr_cap >= total_r || is_close(curr_cap, total_r))
                return;
            std::vector<int> closed;
            for (int t = 0; t < static_cast<int>(open.size()); ++t)
                if (!open[t] && cap[t] > 0.0)
                    closed.push_back(t);

            std::sort(closed.begin(), closed.end(), [&](int a, int b)
                      {
                const double da = fixed_cost[a] / std::max(cap[a], 1e-16);
                const double db = fixed_cost[b] / std::max(cap[b], 1e-16);
                return da < db; });
            for (int t : closed)
            {
                open[t] = 1;
                curr_cap += cap[t];
                if (curr_cap > total_r || is_close(curr_cap, total_r))
                    break;
            }
        };

        ensure_cap(a_fix, inst.p, inst.f, cap_p);
        ensure_cap(b_fix, inst.q, inst.g, cap_q);

        // Se ainda faltar capacidade, abre tudo (fallback)
        if ((cap_p < total_r && !is_close(cap_p, total_r)) ||
            (cap_q < total_r && !is_close(cap_q, total_r)))
        {
            std::fill(a_fix.begin(), a_fix.end(), 1);
            std::fill(b_fix.begin(), b_fix.end(), 1);
        }

        // Subconjuntos abertos
        std::vector<int> I_open, J_open;
        for (int i = 0; i < inst.nI; ++i)
            if (a_fix[i])
                I_open.push_back(i);
        for (int j = 0; j < inst.nJ; ++j)
            if (b_fix[j])
                J_open.push_back(j);
        if (I_open.empty() || J_open.empty())
            return std::nullopt;

        // Modelo CPLEX (contínuo)
        IloEnv env;
        double obj_val = 0.0;
        try
        {
            IloModel mdl(env);
            IloCplex cplex(mdl);
            cplex.setOut(env.getNullStream());

            // Variáveis x(i,j) e y(j,k)
            std::map<std::pair<int, int>, IloNumVar> xR;
            std::map<std::pair<int, int>, IloNumVar> yR;

            for (int i : I_open)
                for (int j : J_open)
                {
                    xR[{i, j}] = IloNumVar(env, 0.0, IloInfinity, ILOFLOAT,
                                           ("x_" + std::to_string(i) + "_" + std::to_string(j)).c_str());
                    mdl.add(xR[{i, j}]);
                }

            for (int j : J_open)
                for (int k = 0; k < inst.nK; ++k)
                {
                    yR[{j, k}] = IloNumVar(env, 0.0, IloInfinity, ILOFLOAT,
                                           ("y_" + std::to_string(j) + "_" + std::to_string(k)).c_str());
                    mdl.add(yR[{j, k}]);
                }

            // capacidades: plantas
            for (int i : I_open)
            {
                IloExpr lhs(env);
                for (int j : J_open)
                    lhs += xR[{i, j}];
                mdl.add(lhs <= inst.p[i]);
                lhs.end();
            }

            // capacidades: depósitos
            for (int j : J_open)
            {
                IloExpr lhs(env);
                for (int k = 0; k < inst.nK; ++k)
                    lhs += yR[{j, k}];
                mdl.add(lhs <= inst.q[j]);
                lhs.end();
            }

            // balanço dos depósitos: ∑_i x_ij = ∑_k y_jk
            for (int j : J_open)
            {
                IloExpr lhs(env), rhs(env);
                for (int i : I_open)
                    lhs += xR[{i, j}];
                for (int k = 0; k < inst.nK; ++k)
                    rhs += yR[{j, k}];
                mdl.add(lhs == rhs);
                lhs.end();
                rhs.end();
            }

            // demandas dos clientes: ∑_j y_jk = r_k
            for (int k = 0; k < inst.nK; ++k)
            {
                IloExpr lhs(env);
                for (int j : J_open)
                    lhs += yR[{j, k}];
                mdl.add(lhs == inst.r[k]);
                lhs.end();
            }

            // objetivo: custos fixos + custos de fluxo
            double cost_fixed1 = 0.0, cost_fixed2 = 0.0;
            for (int i : I_open)
                cost_fixed1 += inst.f[i];
            for (int j : J_open)
                cost_fixed2 += inst.g[j];

            IloExpr cost(env);
            for (auto &kv : xR)
            {
                int i = kv.first.first, j = kv.first.second;
                cost += inst.C(i, j) * kv.second;
            }
            for (auto &kv : yR)
            {
                int j = kv.first.first, k = kv.first.second;
                cost += inst.D(j, k) * kv.second;
            }
            mdl.add(IloMinimize(env, cost + cost_fixed1 + cost_fixed2));
            cost.end();

            if (!cplex.solve())
            {
                env.end();
                return std::nullopt;
            }
            obj_val = cplex.getObjValue();
        }
        catch (...)
        {
            env.end();
            return std::nullopt;
        }
        env.end();
        return obj_val;
    }

public:
    //
    // Loop principal do NDRC
    //
    void solve()
    {
        using clock = std::chrono::high_resolution_clock;
        const auto t0 = clock::now();

        while (iter < max_iter)
        {
            time_sec = std::chrono::duration<double>(clock::now() - t0).count();
            if (time_sec > time_limit)
                break;

            ++iter;

            // (1) resolve o subproblema lagrangeano
            LRPResult res = _solve_lrp(lamb);
            const double L_k = res.L_val;
            const std::vector<int> &a_k = res.a_rel;
            const std::vector<int> &b_k = res.b_rel;
            std::vector<double> &subgrad_k = res.subgrad;

            // (2) manutenção de LB/UB
            bool lb_improved = (L_k > L_best);
            if (lb_improved)
                L_best = L_k;

            if (iter == 1 || lb_improved || (iter % 25 == 0))
            {
                if (auto z_try = _solve_heuristic(a_k, b_k))
                {
                    if (z_try.value() < z_best)
                    {
                        z_best = z_try.value();
                        a_best = a_k;
                        b_best = b_k;
                    }
                }
            }

            // (3) gerenciamento de CA/PA/CI
            const double TOL = 1e-12;
            std::vector<int> CA_idx, PA_idx, CI_idx;
            CA_idx.reserve(lamb.size());
            PA_idx.reserve(lamb.size());

            for (size_t i = 0; i < lamb.size(); ++i)
            {
                if (!is_close(subgrad_k[i], 0.0, TOL))
                    CA_idx.push_back((int)i);
                if (!is_close(lamb[i], 0.0, TOL))
                    PA_idx.push_back((int)i);
            }
            {
                std::vector<char> mark(lamb.size(), 0);
                for (int i : CA_idx)
                    mark[(size_t)i] = 1;
                for (int i : PA_idx)
                    mark[(size_t)i] = 1;
                for (size_t i = 0; i < mark.size(); ++i)
                    if (!mark[i])
                        CI_idx.push_back((int)i);
            }

            // zera subgrad em CI e reseta idade de CA
            for (int i : CI_idx)
                subgrad_k[(size_t)i] = 0.0;
            for (int i : CA_idx)
                lamb_age[(size_t)i] = 0;

            // envelhece PA\CA e zera após dual_keep
            {
                std::vector<char> is_CA(lamb.size(), 0);
                for (int i : CA_idx)
                    is_CA[(size_t)i] = 1;

                for (int i : PA_idx)
                {
                    if (!is_CA[(size_t)i])
                    {
                        if (++lamb_age[(size_t)i] > dual_keep)
                        {
                            lamb[(size_t)i] = 0.0;
                            lamb_age[(size_t)i] = 0;
                        }
                    }
                }
            }

            // (4) tamanho de passo e atualização do dual
            const double denom = sqnorm(subgrad_k);
            if (denom > 0.0)
            {
                const double mu = std::isfinite(z_best)
                                      ? (gamma * std::max(z_best - L_k, 0.0) / denom)
                                      : (gamma / denom);
                for (size_t i = 0; i < lamb.size(); ++i)
                    lamb[i] += mu * subgrad_k[i];
            }

            // (5) condição de encerramento
            if (std::isfinite(z_best))
            {
                gap = (z_best - L_best) / std::max(1.0, std::abs(z_best));
                if (gap <= tol_stop)
                    break;
            }

            // log do loop
            if (log_output && (iter % 20 == 0))
            {
                const double norm_g = std::sqrt(denom);
                std::cout << "[NDRC] it=" << std::setw(5) << iter
                          << "  time=" << std::setw(4) << (int)std::round(time_sec) << "s    "
                          << "LRP=" << std::setw(10) << std::fixed << std::setprecision(3) << L_k << "  "
                          << "LB=" << std::setw(10) << L_best << "  "
                          << "UB=" << std::setw(10) << (std::isfinite(z_best) ? z_best : std::numeric_limits<double>::infinity()) << "  "
                          << "||subgrad||^2=" << std::scientific << norm_g
                          << "  |CA|=" << std::dec << CA_idx.size()
                          << "  |PA|=" << PA_idx.size()
                          << "  |CI|=" << CI_idx.size()
                          << std::endl;
            }
        }
    }
};

//
// Rotina principal
//
int main()
{
    const std::string PATH = "../instances/tscfl/tscfl_11_50.txt";

    try
    {
        TSCFLInstance instance = TSCFLInstance::from_txt(PATH);

        // inst, gamma, dual_keep, tol, max_iter, time_limit, log
        RelaxAndCutTSCFL solver(instance, 1.0, 5, 1.0e-6, 10000, 200, true);
        solver.solve();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Erro: " << e.what() << "\n";
        return 2;
    }
    return 0;
}
