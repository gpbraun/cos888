/*
COS888 — TSCFL resolvido com Branch-and-Price + Geração de Colunas
Implementação apenas para fins didáticos, baseada na relaxação Lagrangeana
por blocos (plantas e satélites) e decomposição de Dantzig–Wolfe.

Gabriel Braun, 2025
*/

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <stdexcept>
#include <limits>
#include <tuple>
#include <optional>
#include <iomanip>

#include <ilcplex/ilocplex.h>
ILOSTLBEGIN

// =====================================================================
//  UTILS
// =====================================================================

// Acesso em matriz 2D
static inline int idx2(int i, int j, int ncols) { return i * ncols + j; }

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

// =====================================================================
//  INSTÂNCIA
// =====================================================================

//
// Instância do TSCFL
//
class TSCFLInstance
{
public:
    int nI{0}; // |I| plantas
    int nJ{0}; // |J| depósitos
    int nK{0}; // |K| clientes

    std::vector<double> f; // f_i  = custo fixo da planta i
    std::vector<double> g; // g_j  = custo fixo do depósito j
    std::vector<double> c; // c_ij = custo unitário planta i -> depósito j
    std::vector<double> d; // d_jk = custo unitário depósito j -> cliente k
    std::vector<double> p; // p_i  = capacidade da planta i
    std::vector<double> q; // q_j  = capacidade do depósito j
    std::vector<double> r; // r_k  = demanda do cliente k

    inline double C(int i, int j) const { return c[idx2(i, j, nJ)]; }
    inline double &C(int i, int j) { return c[idx2(i, j, nJ)]; }
    inline double D(int j, int k) const { return d[idx2(j, k, nK)]; }
    inline double &D(int j, int k) { return d[idx2(j, k, nK)]; }

    std::vector<int> I() const { return range_int(nI); }
    std::vector<int> J() const { return range_int(nJ); }
    std::vector<int> K() const { return range_int(nK); }

    std::vector<std::pair<int, int>> IJ() const { return cart_prod(nI, nJ); }
    std::vector<std::pair<int, int>> JK() const { return cart_prod(nJ, nK); }

    static TSCFLInstance from_txt(const std::string &path)
    {
        std::ifstream in(path);
        if (!in)
            throw std::runtime_error("Cannot open instance: " + path);

        std::vector<double> a;
        a.reserve(1 << 20);
        double v;
        while (in >> v)
            a.push_back(v);
        if (a.size() < 3)
            throw std::runtime_error("Malformed file (header).");

        size_t pos = 0;
        TSCFLInstance inst;
        inst.nI = static_cast<int>(a[pos++]);
        inst.nJ = static_cast<int>(a[pos++]);
        inst.nK = static_cast<int>(a[pos++]);

        const int nI = inst.nI, nJ = inst.nJ, nK = inst.nK;

        // r: nK
        if (pos + static_cast<size_t>(nK) > a.size())
            throw std::runtime_error("Malformed file (r).");
        inst.r.assign(a.begin() + pos, a.begin() + pos + nK);
        pos += static_cast<size_t>(nK);

        // (q,g): nJ pares
        if (pos + static_cast<size_t>(2 * nJ) > a.size())
            throw std::runtime_error("Malformed file (q,g).");
        inst.q.resize(nJ);
        inst.g.resize(nJ);
        for (int j = 0; j < nJ; ++j)
        {
            inst.q[j] = a[pos++];
            inst.g[j] = a[pos++];
        }

        // c: nI*nJ
        if (pos + static_cast<size_t>(nI) * static_cast<size_t>(nJ) > a.size())
            throw std::runtime_error("Malformed file (c).");
        inst.c.assign(a.begin() + pos, a.begin() + pos + (static_cast<size_t>(nI) * static_cast<size_t>(nJ)));
        pos += static_cast<size_t>(nI) * static_cast<size_t>(nJ);

        // (p,f): nI pares
        if (pos + static_cast<size_t>(2 * nI) > a.size())
            throw std::runtime_error("Malformed file (p,f).");
        inst.p.resize(nI);
        inst.f.resize(nI);
        for (int i = 0; i < nI; ++i)
        {
            inst.p[i] = a[pos++];
            inst.f[i] = a[pos++];
        }

        // d: nJ*nK
        if (pos + static_cast<size_t>(nJ) * static_cast<size_t>(nK) > a.size())
            throw std::runtime_error("Malformed file (d).");
        inst.d.assign(a.begin() + pos, a.begin() + pos + (static_cast<size_t>(nJ) * static_cast<size_t>(nK)));
        pos += static_cast<size_t>(nJ) * static_cast<size_t>(nK);

        return inst;
    }
};

// =====================================================================
//  ESTRUTURAS DO BRANCH-AND-PRICE
// =====================================================================

enum class ColumnType
{
    PLANT = 1,
    SAT = 2
};

struct Column
{
    int col_id;
    ColumnType col_type;
    int block_id;                // i se planta, j se satélite
    double cost;                 // custo original
    std::vector<double> balance; // tamanho nJ
    std::vector<double> demand;  // tamanho nK
};

struct Duals
{
    std::vector<double> pi;    // dual balanco depósitos
    std::vector<double> sigma; // dual demanda clientes
    std::vector<double> alpha; // dual convexidade plantas
    std::vector<double> nu;    // dual convexidade satélites
};

struct NodeContext
{
    std::vector<int> fixed_a; // -1 livre, 0 fixado a 0, 1 fixado a 1
    std::vector<int> fixed_b;

    NodeContext() = default;
    NodeContext(const std::vector<int> &fa, const std::vector<int> &fb)
        : fixed_a(fa), fixed_b(fb) {}
};

struct NodeLPResult
{
    double z_lp{0.0};
    std::vector<double> a_hat;
    std::vector<double> b_hat;
    Duals duals;
    double max_balance_slack{0.0};
    double max_demand_slack{0.0};
};

struct SearchNode
{
    int node_id;
    int depth;
    NodeContext ctx;
};

// =====================================================================
//  RMP (Restricted Master Problem) com CPLEX Concert
// =====================================================================

class MasterProblem
{
public:
    MasterProblem(const TSCFLInstance &inst_, double big_M_ = 1e7)
        : inst(inst_),
          big_M(big_M_),
          env(),
          model(env),
          cplex(model),
          obj(env),
          s_pos(env),
          s_neg(env),
          s_dem(env),
          balance_ct(env),
          demand_ct(env),
          conv_plant_ct(env),
          conv_sat_ct(env)
    {
        obj = IloMinimize(env, 0.0);
        model.add(obj);
        // silêncio no output do CPLEX
        cplex.setOut(env.getNullStream());
        build_base_constraints();
    }

    ~MasterProblem()
    {
        env.end();
    }

    void add_columns(const std::vector<Column> &new_columns)
    {
        if (new_columns.empty())
            return;

        int nJ = inst.nJ;
        int nK = inst.nK;

        for (const auto &col : new_columns)
        {
            // cria variável
            std::string vname = "z_" + std::to_string(col.col_id);
            IloNumVar z(env, 0.0, IloInfinity, ILOFLOAT, vname.c_str());
            z_vars.push_back(z);
            columns.push_back(col);

            // custo original no objetivo
            obj.setLinearCoef(z, col.cost);

            // convexidade
            if (col.col_type == ColumnType::PLANT)
            {
                int i = col.block_id;
                conv_plant_ct[i].setLinearCoef(z, 1.0);
            }
            else
            {
                int j = col.block_id;
                conv_sat_ct[j].setLinearCoef(z, 1.0);
            }

            // balanço nos depósitos
            for (int j = 0; j < nJ; ++j)
            {
                double coef = col.balance[j];
                if (std::fabs(coef) > 1e-12)
                    balance_ct[j].setLinearCoef(z, coef);
            }

            // demanda dos clientes
            for (int k = 0; k < nK; ++k)
            {
                double coef = col.demand[k];
                if (std::fabs(coef) > 1e-12)
                    demand_ct[k].setLinearCoef(z, coef);
            }
        }
    }

    NodeLPResult solve_lp()
    {
        if (!cplex.solve())
            throw std::runtime_error("RMP infeasible ou sem solução LP.");

        NodeLPResult res;
        res.z_lp = cplex.getObjValue();

        int nI = inst.nI;
        int nJ = inst.nJ;
        int nK = inst.nK;

        res.a_hat.assign(nI, 0.0);
        res.b_hat.assign(nJ, 0.0);

        // Reconstrói a_hat, b_hat a partir das colunas
        for (std::size_t idx = 0; idx < z_vars.size(); ++idx)
        {
            double z_val = cplex.getValue(z_vars[idx]);
            if (std::fabs(z_val) < 1e-9)
                continue;

            const Column &col = columns[idx];
            if (col.col_type == ColumnType::PLANT)
                res.a_hat[col.block_id] += z_val;
            else
                res.b_hat[col.block_id] += z_val;
        }

        // Slacks máximos de balanço: max(s_pos, s_neg)
        res.max_balance_slack = 0.0;
        for (int j = 0; j < nJ; ++j)
        {
            double sp = cplex.getValue(s_pos[j]);
            double sn = cplex.getValue(s_neg[j]);
            res.max_balance_slack = std::max(res.max_balance_slack, std::max(sp, sn));
        }

        // Slacks de demanda: s_dem
        res.max_demand_slack = 0.0;
        for (int k = 0; k < nK; ++k)
        {
            double sd = cplex.getValue(s_dem[k]);
            res.max_demand_slack = std::max(res.max_demand_slack, std::fabs(sd));
        }

        // Duais
        res.duals.pi.assign(nJ, 0.0);
        res.duals.sigma.assign(nK, 0.0);
        res.duals.alpha.assign(nI, 0.0);
        res.duals.nu.assign(nJ, 0.0);

        for (int j = 0; j < nJ; ++j)
            res.duals.pi[j] = cplex.getDual(balance_ct[j]);
        for (int k = 0; k < nK; ++k)
            res.duals.sigma[k] = cplex.getDual(demand_ct[k]);
        for (int i = 0; i < nI; ++i)
            res.duals.alpha[i] = cplex.getDual(conv_plant_ct[i]);
        for (int j = 0; j < nJ; ++j)
            res.duals.nu[j] = cplex.getDual(conv_sat_ct[j]);

        return res;
    }

    std::size_t num_columns() const { return z_vars.size(); }

private:
    const TSCFLInstance &inst;
    double big_M;

    IloEnv env;
    IloModel model;
    IloCplex cplex;
    IloObjective obj;

    std::vector<IloNumVar> z_vars;
    std::vector<Column> columns;

    IloNumVarArray s_pos;
    IloNumVarArray s_neg;
    IloNumVarArray s_dem;

    IloRangeArray balance_ct;
    IloRangeArray demand_ct;
    IloRangeArray conv_plant_ct;
    IloRangeArray conv_sat_ct;

    void build_base_constraints()
    {
        int nI = inst.nI;
        int nJ = inst.nJ;
        int nK = inst.nK;

        // Balanço nos depósitos: sum_cols balance[j]*z + s_pos[j] - s_neg[j] = 0
        s_pos = IloNumVarArray(env, nJ);
        s_neg = IloNumVarArray(env, nJ);
        balance_ct = IloRangeArray(env, nJ);

        for (int j = 0; j < nJ; ++j)
        {
            std::string npos = "s_pos_" + std::to_string(j);
            std::string nneg = "s_neg_" + std::to_string(j);

            s_pos[j] = IloNumVar(env, 0.0, IloInfinity, ILOFLOAT, npos.c_str());
            s_neg[j] = IloNumVar(env, 0.0, IloInfinity, ILOFLOAT, nneg.c_str());

            IloExpr e(env);
            e += s_pos[j] - s_neg[j];
            IloRange ct(env, 0.0, e, 0.0); // igualdade 0
            e.end();
            ct.setName(("bal_" + std::to_string(j)).c_str());
            balance_ct[j] = ct;
            model.add(balance_ct[j]);

            // penalidade no objetivo
            obj.setLinearCoef(s_pos[j], big_M);
            obj.setLinearCoef(s_neg[j], big_M);
        }

        // Demanda nos clientes: sum_cols demand[k]*z + s_dem[k] = r_k
        s_dem = IloNumVarArray(env, nK);
        demand_ct = IloRangeArray(env, nK);

        for (int k = 0; k < nK; ++k)
        {
            std::string ndem = "s_dem_" + std::to_string(k);
            s_dem[k] = IloNumVar(env, 0.0, IloInfinity, ILOFLOAT, ndem.c_str());

            IloExpr e(env);
            e += s_dem[k];
            double rhs = inst.r[k];
            IloRange ct(env, rhs, e, rhs); // igualdade = r_k
            e.end();
            ct.setName(("dem_" + std::to_string(k)).c_str());
            demand_ct[k] = ct;
            model.add(demand_ct[k]);

            obj.setLinearCoef(s_dem[k], big_M);
        }

        // Convexidade por planta i: sum_{cols de i} z <= 1
        conv_plant_ct = IloRangeArray(env, nI);
        for (int i = 0; i < nI; ++i)
        {
            IloExpr e(env); // inicialmente 0
            IloRange ct(env, -IloInfinity, e, 1.0);
            e.end();
            ct.setName(("convI_" + std::to_string(i)).c_str());
            conv_plant_ct[i] = ct;
            model.add(conv_plant_ct[i]);
        }

        // Convexidade por satélite j: sum_{cols de j} z <= 1
        conv_sat_ct = IloRangeArray(env, nJ);
        for (int j = 0; j < nJ; ++j)
        {
            IloExpr e(env);
            IloRange ct(env, -IloInfinity, e, 1.0);
            e.end();
            ct.setName(("convJ_" + std::to_string(j)).c_str());
            conv_sat_ct[j] = ct;
            model.add(conv_sat_ct[j]);
        }
    }
};

// =====================================================================
//  Branch-and-Price
// =====================================================================

class BranchAndPriceSolver
{
public:
    BranchAndPriceSolver(
        const TSCFLInstance &inst_,
        double cg_tol_ = 1e-6,
        int cg_max_iter_ = 200,
        bool log_output_ = true)
        : inst(inst_),
          cg_tol(cg_tol_),
          cg_max_iter(cg_max_iter_),
          log_output(log_output_),
          next_col_id(0)
    {
    }

    struct Result
    {
        double best_obj;
        std::vector<int> best_a;
        std::vector<int> best_b;
    };

    Result solve()
    {
        int nI = inst.nI;
        int nJ = inst.nJ;

        // Contexto raiz: tudo livre (-1)
        NodeContext root_ctx(
            std::vector<int>(nI, -1),
            std::vector<int>(nJ, -1));

        double best_obj = std::numeric_limits<double>::infinity();
        std::vector<int> best_a;
        std::vector<int> best_b;

        std::vector<SearchNode> stack;
        stack.push_back(SearchNode{1, 0, root_ctx});

        if (log_output)
            std::cout << "[BnP] starting search\n"
                      << std::endl;

        int node_counter = 1;

        while (!stack.empty())
        {
            SearchNode node = stack.back();
            stack.pop_back();
            const NodeContext &ctx = node.ctx;

            if (log_output)
            {
                int fa = 0, fb = 0;
                for (int v : ctx.fixed_a)
                    if (v != -1)
                        ++fa;
                for (int v : ctx.fixed_b)
                    if (v != -1)
                        ++fb;

                std::cout << "[BnP] Node #" << node.node_id
                          << "  depth=" << node.depth
                          << "  |fixed_a|=" << fa
                          << "  |fixed_b|=" << fb
                          << "  current_UB=";
                if (best_obj < std::numeric_limits<double>::infinity())
                {
                    std::cout << std::fixed << std::setprecision(3) << best_obj;
                }
                else
                {
                    std::cout << "inf";
                }
                std::cout << std::endl;
            }

            auto lp_opt = solve_node_lp(node, best_obj);
            if (!lp_opt.has_value())
            {
                if (log_output)
                    std::cout << "[BnP]   LP infeasible, pruning." << std::endl;
                continue;
            }

            NodeLPResult lp_res = lp_opt.value();
            double z_lp = lp_res.z_lp;

            // poda por bound
            if (z_lp >= best_obj - 1e-6)
            {
                if (log_output && best_obj < std::numeric_limits<double>::infinity())
                {
                    std::cout << "[BnP]   prune by bound: z_lp="
                              << std::fixed << std::setprecision(3) << z_lp
                              << " >= best_obj=" << best_obj << std::endl;
                }
                continue;
            }

            // verifica integralidade em a_hat, b_hat
            if (is_integral(lp_res))
            {
                if (log_output)
                    std::cout << "[BnP]   LP solution is integral." << std::endl;

                // checa slacks (viabilidade nas restrições originais)
                if (lp_res.max_balance_slack > 1e-6 || lp_res.max_demand_slack > 1e-6)
                {
                    if (log_output)
                    {
                        std::cout << "[BnP]   discarding integral LP: slacks "
                                  << "(bal=" << std::scientific << lp_res.max_balance_slack
                                  << ", dem=" << lp_res.max_demand_slack << ")" << std::endl;
                    }
                    continue;
                }

                if (z_lp < best_obj - 1e-6)
                {
                    best_obj = z_lp;
                    best_a.assign(inst.nI, 0);
                    best_b.assign(inst.nJ, 0);
                    for (int i = 0; i < inst.nI; ++i)
                        best_a[i] = static_cast<int>(std::round(lp_res.a_hat[i]));
                    for (int j = 0; j < inst.nJ; ++j)
                        best_b[j] = static_cast<int>(std::round(lp_res.b_hat[j]));

                    if (log_output)
                    {
                        std::cout << "[BnP]   new incumbent: UB="
                                  << std::fixed << std::setprecision(3) << best_obj
                                  << "  (bal_slack=" << std::scientific << lp_res.max_balance_slack
                                  << ", dem_slack=" << lp_res.max_demand_slack << ")"
                                  << std::endl;
                    }
                }
                continue;
            }

            // precisa ramificar
            bool found = false;
            char vname = '?';
            int idx = -1;
            double val = 0.0;
            std::tie(vname, idx, val) = choose_branch_var(lp_res, ctx, found);

            if (!found)
            {
                // nada claramente fracionário (provavelmente numérico) -> considera incumbente
                if (log_output)
                    std::cout << "[BnP]   no branching candidate, treating as incumbent." << std::endl;
                if (z_lp < best_obj - 1e-6)
                {
                    best_obj = z_lp;
                    best_a.assign(inst.nI, 0);
                    best_b.assign(inst.nJ, 0);
                    for (int i = 0; i < inst.nI; ++i)
                        best_a[i] = static_cast<int>(std::round(lp_res.a_hat[i]));
                    for (int j = 0; j < inst.nJ; ++j)
                        best_b[j] = static_cast<int>(std::round(lp_res.b_hat[j]));
                }
                continue;
            }

            if (log_output)
            {
                std::cout << "[BnP]   branching on " << vname << "_" << idx
                          << " = " << std::fixed << std::setprecision(3) << val << std::endl;
            }

            // filho esquerdo: fixa a/b = 0
            NodeContext ctx_left = ctx;
            if (vname == 'a')
                ctx_left.fixed_a[idx] = 0;
            else
                ctx_left.fixed_b[idx] = 0;
            node_counter++;
            SearchNode child_left{node_counter, node.depth + 1, ctx_left};

            // filho direito: fixa a/b = 1
            NodeContext ctx_right = ctx;
            if (vname == 'a')
                ctx_right.fixed_a[idx] = 1;
            else
                ctx_right.fixed_b[idx] = 1;
            node_counter++;
            SearchNode child_right{node_counter, node.depth + 1, ctx_right};

            // DFS: processa primeiro o filho "0", depois o "1" (ordem na pilha)
            stack.push_back(child_right);
            stack.push_back(child_left);
        }

        if (log_output)
        {
            std::cout << "\n[BnP] search finished." << std::endl;
            if (best_obj < std::numeric_limits<double>::infinity())
            {
                std::cout << "[BnP] best objective = " << std::fixed << std::setprecision(3)
                          << best_obj << std::endl;

                std::cout << "[BnP] best a = [";
                for (std::size_t i = 0; i < best_a.size(); ++i)
                {
                    std::cout << best_a[i];
                    if (i + 1 < best_a.size())
                        std::cout << " ";
                }
                std::cout << "]" << std::endl;

                std::cout << "[BnP] best b = [";
                for (std::size_t j = 0; j < best_b.size(); ++j)
                {
                    std::cout << best_b[j];
                    if (j + 1 < best_b.size())
                        std::cout << " ";
                }
                std::cout << "]" << std::endl;
            }
            else
            {
                std::cout << "[BnP] no feasible integer solution found." << std::endl;
            }
        }

        return Result{best_obj, best_a, best_b};
    }

private:
    const TSCFLInstance &inst;
    double cg_tol;
    int cg_max_iter;
    bool log_output;
    int next_col_id;

    // --------------------------------------------------------------
    // Colunas iniciais (dummy: abre planta/satélite sem fluxo)
    // --------------------------------------------------------------
    std::vector<Column> build_initial_columns(const NodeContext &ctx)
    {
        std::vector<Column> cols;

        int nI = inst.nI;
        int nJ = inst.nJ;
        int nK = inst.nK;

        // Colunas de plantas (sem fluxo)
        for (int i = 0; i < nI; ++i)
        {
            if (ctx.fixed_a[i] == 0)
                continue;

            Column col;
            col.col_id = next_col_id++;
            col.col_type = ColumnType::PLANT;
            col.block_id = i;
            col.cost = inst.f[i];
            col.balance.assign(nJ, 0.0);
            col.demand.assign(nK, 0.0);
            cols.push_back(col);
        }

        // Colunas de satélites (sem fluxo)
        for (int j = 0; j < nJ; ++j)
        {
            if (ctx.fixed_b[j] == 0)
                continue;

            Column col;
            col.col_id = next_col_id++;
            col.col_type = ColumnType::SAT;
            col.block_id = j;
            col.cost = inst.g[j];
            col.balance.assign(nJ, 0.0);
            col.demand.assign(nK, 0.0);
            cols.push_back(col);
        }

        return cols;
    }

    // --------------------------------------------------------------
    // Pricing: plantas
    // --------------------------------------------------------------
    std::vector<Column> price_plants(const Duals &duals, const NodeContext &ctx)
    {
        std::vector<Column> new_cols;
        int nI = inst.nI;
        int nJ = inst.nJ;
        int nK = inst.nK;

        for (int i = 0; i < nI; ++i)
        {
            if (ctx.fixed_a[i] == 0)
                continue;

            double p_i = inst.p[i];
            if (p_i <= 0.0)
                continue;

            // phi_j = c_ij - pi_j
            double phi_min = std::numeric_limits<double>::infinity();
            int j_best = -1;
            for (int j = 0; j < nJ; ++j)
            {
                double phi = inst.C(i, j) - duals.pi[j];
                if (phi < phi_min)
                {
                    phi_min = phi;
                    j_best = j;
                }
            }

            double base_cost = inst.f[i];
            double cost0 = base_cost;
            double rc0 = cost0 - duals.alpha[i];

            if (phi_min >= 0.0)
            {
                double rc = rc0;
                if (rc < -cg_tol)
                {
                    Column col;
                    col.col_id = next_col_id++;
                    col.col_type = ColumnType::PLANT;
                    col.block_id = i;
                    col.cost = cost0;
                    col.balance.assign(nJ, 0.0);
                    col.demand.assign(nK, 0.0);
                    new_cols.push_back(col);
                }
                continue;
            }

            // padrão com fluxo máximo na melhor aresta
            double x_val = p_i;
            std::vector<double> balance(nJ, 0.0);
            std::vector<double> demand(nK, 0.0);
            balance[j_best] = x_val;

            double cost = base_cost + inst.C(i, j_best) * x_val;
            double rc = cost - duals.alpha[i] - duals.pi[j_best] * x_val;

            if (rc < -cg_tol)
            {
                Column col;
                col.col_id = next_col_id++;
                col.col_type = ColumnType::PLANT;
                col.block_id = i;
                col.cost = cost;
                col.balance = std::move(balance);
                col.demand = std::move(demand);
                new_cols.push_back(col);
            }
        }

        return new_cols;
    }

    // --------------------------------------------------------------
    // Pricing: satélites
    // --------------------------------------------------------------
    std::vector<Column> price_sats(const Duals &duals, const NodeContext &ctx)
    {
        std::vector<Column> new_cols;
        int nJ = inst.nJ;
        int nK = inst.nK;

        for (int j = 0; j < nJ; ++j)
        {
            if (ctx.fixed_b[j] == 0)
                continue;

            double q_j = inst.q[j];
            if (q_j <= 0.0)
                continue;

            double base_cost = inst.g[j];

            // psi_k = d_jk + pi_j - sigma_k
            std::vector<double> psi(nK, 0.0);
            for (int k = 0; k < nK; ++k)
                psi[k] = inst.D(j, k) + duals.pi[j] - duals.sigma[k];

            double cap = q_j;
            std::vector<double> y(nK, 0.0);

            // ordem crescente de psi_k (clientes "mais negativos" primeiro)
            std::vector<int> order(nK);
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(),
                      [&](int a, int b)
                      { return psi[a] < psi[b]; });

            for (int idx = 0; idx < nK; ++idx)
            {
                int k = order[idx];
                if (psi[k] >= 0.0)
                    break;
                if (cap <= 1e-9)
                    break;

                double assign = std::min(inst.r[k], cap);
                if (assign <= 1e-9)
                    continue;

                y[k] = assign;
                cap -= assign;
            }

            double total_y = 0.0;
            for (double v : y)
                total_y += v;

            std::vector<double> balance(nJ, 0.0);
            std::vector<double> demand(nK, 0.0);
            double cost = base_cost;
            double rc = 0.0;

            if (total_y <= 1e-9)
            {
                // padrão sem fluxo
                cost = base_cost;
                demand.assign(nK, 0.0);
                balance.assign(nJ, 0.0);
                rc = cost - duals.nu[j];
            }
            else
            {
                double dot_dy = 0.0;
                for (int k = 0; k < nK; ++k)
                    dot_dy += inst.D(j, k) * y[k];

                cost = base_cost + dot_dy;

                balance[j] = -total_y;
                demand = y;

                double dot_sigma_d = 0.0;
                for (int k = 0; k < nK; ++k)
                    dot_sigma_d += duals.sigma[k] * demand[k];

                rc = cost - duals.nu[j] - duals.pi[j] * balance[j] - dot_sigma_d;
            }

            if (rc < -cg_tol)
            {
                Column col;
                col.col_id = next_col_id++;
                col.col_type = ColumnType::SAT;
                col.block_id = j;
                col.cost = cost;
                col.balance = std::move(balance);
                col.demand = std::move(demand);
                new_cols.push_back(col);
            }
        }

        return new_cols;
    }

    // --------------------------------------------------------------
    // Integrality check
    // --------------------------------------------------------------
    static bool is_integral(const NodeLPResult &lp_res, double int_tol = 1e-5)
    {
        for (double v : lp_res.a_hat)
        {
            if (v > int_tol && v < 1.0 - int_tol)
                return false;
        }
        for (double v : lp_res.b_hat)
        {
            if (v > int_tol && v < 1.0 - int_tol)
                return false;
        }
        return true;
    }

    // --------------------------------------------------------------
    // Escolhe variável de branching mais fracionária
    // --------------------------------------------------------------
    std::tuple<char, int, double> choose_branch_var(
        const NodeLPResult &lp_res,
        const NodeContext &ctx,
        bool &found) const
    {
        found = false;
        char best_type = '?';
        int best_idx = -1;
        double best_val = 0.0;
        double best_dist = 1.0; // distância a 0.5

        int nI = inst.nI;
        int nJ = inst.nJ;
        double int_tol = 1e-5;

        // a_i
        for (int i = 0; i < nI; ++i)
        {
            if (ctx.fixed_a[i] != -1)
                continue;
            double val = lp_res.a_hat[i];
            if (val <= int_tol || val >= 1.0 - int_tol)
                continue;
            double dist = std::fabs(val - 0.5);
            if (dist < best_dist)
            {
                best_dist = dist;
                best_type = 'a';
                best_idx = i;
                best_val = val;
                found = true;
            }
        }

        // b_j
        for (int j = 0; j < nJ; ++j)
        {
            if (ctx.fixed_b[j] != -1)
                continue;
            double val = lp_res.b_hat[j];
            if (val <= int_tol || val >= 1.0 - int_tol)
                continue;
            double dist = std::fabs(val - 0.5);
            if (dist < best_dist)
            {
                best_dist = dist;
                best_type = 'b';
                best_idx = j;
                best_val = val;
                found = true;
            }
        }

        return std::make_tuple(best_type, best_idx, best_val);
    }

    // --------------------------------------------------------------
    // Resolve o LP (RMP + CG) em um nó da árvore
    // --------------------------------------------------------------
    std::optional<NodeLPResult> solve_node_lp(const SearchNode &node, double current_UB)
    {
        const NodeContext &ctx = node.ctx;
        MasterProblem master(inst);

        std::vector<Column> init_cols = build_initial_columns(ctx);

        if (log_output)
        {
            int fa = 0, fb = 0;
            for (int v : ctx.fixed_a)
                if (v != -1)
                    ++fa;
            for (int v : ctx.fixed_b)
                if (v != -1)
                    ++fb;

            std::cout << "  [CG] start |fixed_a|=" << fa
                      << " |fixed_b|=" << fb
                      << " init_cols=" << init_cols.size() << std::endl;
        }

        master.add_columns(init_cols);

        NodeLPResult last_res;
        bool has_res = false;

        for (int it = 1; it <= cg_max_iter; ++it)
        {
            NodeLPResult lp_res = master.solve_lp();
            last_res = lp_res;
            has_res = true;
            double z_lp = lp_res.z_lp;

            if (log_output)
            {
                std::cout << "  [CG] it=" << std::setw(3) << it
                          << "  z_lp=" << std::fixed << std::setprecision(3) << z_lp
                          << "  |cols|=" << std::setw(4) << master.num_columns();
            }

            // Parada antecipada por bound
            if (current_UB < std::numeric_limits<double>::infinity() &&
                z_lp >= current_UB - 1e-6)
            {
                if (log_output)
                {
                    std::cout << "  (early stop: z_lp=" << z_lp
                              << " >= UB=" << current_UB << ")" << std::endl;
                }
                return lp_res;
            }

            // Pricing
            std::vector<Column> plant_cols = price_plants(lp_res.duals, ctx);
            std::vector<Column> sat_cols = price_sats(lp_res.duals, ctx);
            std::vector<Column> new_cols;
            new_cols.reserve(plant_cols.size() + sat_cols.size());
            new_cols.insert(new_cols.end(), plant_cols.begin(), plant_cols.end());
            new_cols.insert(new_cols.end(), sat_cols.begin(), sat_cols.end());

            if (log_output)
            {
                std::cout << "  new=" << std::setw(3) << new_cols.size()
                          << " (plants=" << plant_cols.size()
                          << ", sats=" << sat_cols.size() << ")" << std::endl;
            }

            if (new_cols.empty())
            {
                if (log_output)
                    std::cout << "  [CG] no column with negative reduced cost, stopping." << std::endl;
                break;
            }

            master.add_columns(new_cols);
        }

        if (!has_res)
            return std::nullopt;
        return last_res;
    }
};

// =====================================================================
//  main()
// =====================================================================

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " INSTANCE.txt\n";
        return 1;
    }

    std::string path = argv[1];

    try
    {
        TSCFLInstance inst = TSCFLInstance::from_txt(path);

        BranchAndPriceSolver solver(inst, 1e-6, 200, true);
        auto res = solver.solve();

        // res.best_obj, res.best_a, res.best_b já impressos dentro de solve()
    }
    catch (const IloException &e)
    {
        std::cerr << "CPLEX exception: " << e.getMessage() << "\n";
        return 1;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
