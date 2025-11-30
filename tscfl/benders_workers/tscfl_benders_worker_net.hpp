/*
COS888

WorkerBet: resolve o subproblema de fluxo mínimo do TSCFL para uso na decomposição de Benders.

Gabriel Braun, 2025
*/

#pragma once

#include <ilcplex/ilocplex.h>
#include <ilcplex/cplex.h>
#include <stdexcept>
#include <vector>

#include "tscfl_benders_base_worker.hpp"

ILOSTLBEGIN

class WorkerNet : public Worker
{
private:
    // Ambiente e problema da C API (CPXNET)
    CPXENVptr env_cpx{nullptr};
    CPXNETptr net{nullptr};

    // Dimensões da rede
    int nnodes{0};
    int narcs{0};

    // Índices especiais de nós
    int node_s{-1};
    int node_t{-1};

    // Índices de arcos de capacidade que geram l1 e l2
    std::vector<int> arcPlantCap; // arco s -> plant_i
    std::vector<int> arcDepotCap; // arco depotIn_j -> depotOut_j

public:
    explicit WorkerNet(const TSCFLInstance &inst_)
        : Worker(inst_),
          arcPlantCap(inst_.nI),
          arcDepotCap(inst_.nJ)
    {
        const int nI = inst.nI;
        const int nJ = inst.nJ;
        const int nK = inst.nK;

        int status = 0;

        // ---------------------------------------------------------
        // 1) Cria ambiente CPX e problema de rede
        // ---------------------------------------------------------
        env_cpx = CPXopenCPLEX(&status);
        if (env_cpx == nullptr || status)
        {
            throw std::runtime_error("WorkerNet: failed to open CPX environment.");
        }

        // desliga saída na tela para o subproblema
        CPXsetintparam(env_cpx, CPXPARAM_ScreenOutput, CPX_OFF);

        net = CPXNETcreateprob(env_cpx, &status, "tscfl_net");
        if (status)
        {
            CPXcloseCPLEX(&env_cpx);
            throw std::runtime_error("WorkerNet: failed to create CPXNET problem.");
        }

        // ---------------------------------------------------------
        // 2) Define nós e arcos da rede
        // ---------------------------------------------------------
        // Nós:
        //   0                -> s
        //   1..nI            -> plants
        //   nI+1 .. nI+nJ    -> depotIn
        //   nI+nJ+1 .. nI+2nJ-> depotOut
        //   ...              -> customers
        //   último           -> t
        //
        nnodes = 2 + nI + 2 * nJ + nK;
        node_s = 0;
        node_t = nnodes - 1;

        auto nodePlant = [nI](int i)
        { return 1 + i; };
        auto nodeDepotIn = [nI](int j)
        { return 1 + nI + j; };
        auto nodeDepotOut = [nI, nJ](int j)
        { return 1 + nI + nJ + j; };
        auto nodeCust = [nI, nJ](int k)
        { return 1 + nI + 2 * nJ + k; };

        // Contagem de arcos
        // s->plant:           nI
        // plant->depotIn:     nI*nJ
        // depotIn->depotOut:  nJ
        // depotOut->cust:     nJ*nK
        // cust->t:            nK
        narcs = nI + nI * nJ + nJ + nJ * nK + nK;

        std::vector<int> from(narcs);
        std::vector<int> to(narcs);
        std::vector<double> low(narcs, 0.0);
        std::vector<double> up(narcs, 0.0);
        std::vector<double> cost(narcs, 0.0);

        // Suprimento dos nós
        std::vector<double> supply(nnodes, 0.0);
        double demand_total = 0.0;
        for (int k = 0; k < nK; ++k)
            demand_total += inst.r[k];

        supply[node_s] = demand_total;
        supply[node_t] = -demand_total;

        int arcId = 0;

        // 2.1) Arcos s -> plant_i (capacidade p_i * a_i, custo 0)
        for (int i = 0; i < nI; ++i)
        {
            int u = node_s;
            int v = nodePlant(i);

            from[arcId] = u;
            to[arcId] = v;
            low[arcId] = 0.0;
            up[arcId] = 0.0; // será atualizado em solve()
            cost[arcId] = 0.0;

            arcPlantCap[i] = arcId;
            ++arcId;
        }

        // 2.2) Arcos plant_i -> depotIn_j (custo c_ij, cap = +inf)
        for (int i = 0; i < nI; ++i)
        {
            int u = nodePlant(i);
            for (int j = 0; j < nJ; ++j)
            {
                int v = nodeDepotIn(j);

                from[arcId] = u;
                to[arcId] = v;
                low[arcId] = 0.0;
                up[arcId] = CPX_INFBOUND;
                cost[arcId] = inst.c[i][j];

                ++arcId;
            }
        }

        // 2.3) Arcos depotIn_j -> depotOut_j (capacidade q_j * b_j, custo 0)
        for (int j = 0; j < nJ; ++j)
        {
            int u = nodeDepotIn(j);
            int v = nodeDepotOut(j);

            from[arcId] = u;
            to[arcId] = v;
            low[arcId] = 0.0;
            up[arcId] = 0.0; // será atualizado em solve()
            cost[arcId] = 0.0;

            arcDepotCap[j] = arcId;
            ++arcId;
        }

        // 2.4) Arcos depotOut_j -> cust_k (custo d_jk, cap = +inf)
        for (int j = 0; j < nJ; ++j)
        {
            int u = nodeDepotOut(j);
            for (int k = 0; k < nK; ++k)
            {
                int v = nodeCust(k);

                from[arcId] = u;
                to[arcId] = v;
                low[arcId] = 0.0;
                up[arcId] = CPX_INFBOUND;
                cost[arcId] = inst.d[j][k];

                ++arcId;
            }
        }

        // 2.5) Arcos cust_k -> t (capacidade r_k, custo 0)
        for (int k = 0; k < nK; ++k)
        {
            int u = nodeCust(k);
            int v = node_t;

            from[arcId] = u;
            to[arcId] = v;
            low[arcId] = 0.0;
            up[arcId] = inst.r[k];
            cost[arcId] = 0.0;

            ++arcId;
        }

        if (arcId != narcs)
        {
            CPXNETfreeprob(env_cpx, &net);
            CPXcloseCPLEX(&env_cpx);
            throw std::runtime_error("WorkerNet: internal error, arc count mismatch.");
        }

        // 2.6) Adiciona nós e arcos no CPXNET
        status = CPXNETaddnodes(env_cpx, net, nnodes, supply.data(), nullptr);
        if (status)
        {
            CPXNETfreeprob(env_cpx, &net);
            CPXcloseCPLEX(&env_cpx);
            throw std::runtime_error("WorkerNet: CPXNETaddnodes failed.");
        }

        status = CPXNETaddarcs(env_cpx, net,
                               narcs,
                               from.data(), to.data(),
                               low.data(), up.data(),
                               cost.data(),
                               nullptr);
        if (status)
        {
            CPXNETfreeprob(env_cpx, &net);
            CPXcloseCPLEX(&env_cpx);
            throw std::runtime_error("WorkerNet: CPXNETaddarcs failed.");
        }
    }

    ~WorkerNet() override
    {
        if (env_cpx != nullptr)
        {
            if (net != nullptr)
            {
                CPXNETfreeprob(env_cpx, &net);
            }
            CPXcloseCPLEX(&env_cpx);
        }
    }

public:
    // Resolve o subproblema para (a_vals, b_vals) e
    // atualiza: theta, coef_a, coef_b, rhs.
    void solve(const IloNumArray &a_vals, const IloNumArray &b_vals) override
    {
        const int nI = inst.nI;
        const int nJ = inst.nJ;
        const int nK = inst.nK;

        int status = 0;

        // ---------------------------------------------------------
        // 1) Atualiza capacidades dos arcos que dependem de (a,b)
        //    s->plant_i : cap = p_i * a_i
        //    depotIn_j->depotOut_j : cap = q_j * b_j
        // ---------------------------------------------------------
        int cnt = nI + nJ;
        std::vector<int> idx(cnt);
        std::vector<char> lu(cnt, 'U');
        std::vector<double> bd(cnt);

        for (int i = 0; i < nI; ++i)
        {
            idx[i] = arcPlantCap[i];
            bd[i] = inst.p[i] * a_vals[i];
        }
        for (int j = 0; j < nJ; ++j)
        {
            idx[nI + j] = arcDepotCap[j];
            bd[nI + j] = inst.q[j] * b_vals[j];
        }

        status = CPXNETchgbds(env_cpx, net, cnt, idx.data(), lu.data(), bd.data());
        if (status)
        {
            throw std::runtime_error("WorkerNet: CPXNETchgbds failed.");
        }

        // ---------------------------------------------------------
        // 2) Resolve o problema de fluxo mínimo
        // ---------------------------------------------------------
        status = CPXNETprimopt(env_cpx, net);
        if (status)
        {
            throw std::runtime_error("WorkerNet: CPXNETprimopt failed.");
        }

        double objval = 0.0;
        status = CPXNETgetobjval(env_cpx, net, &objval);
        if (status)
        {
            throw std::runtime_error("WorkerNet: CPXNETgetobjval failed.");
        }
        theta = objval;

        // ---------------------------------------------------------
        // 3) Lê potenciais de nó (pi) e custos reduzidos (dj)
        // ---------------------------------------------------------
        std::vector<double> pi(nnodes);
        std::vector<double> dj(narcs);

        status = CPXNETgetpi(env_cpx, net, pi.data(), 0, nnodes - 1);
        if (status)
        {
            throw std::runtime_error("WorkerNet: CPXNETgetpi failed.");
        }

        status = CPXNETgetdj(env_cpx, net, dj.data(), 0, narcs - 1);
        if (status)
        {
            throw std::runtime_error("WorkerNet: CPXNETgetdj failed.");
        }

        // ---------------------------------------------------------
        // 4) Mapeia duais para cortes de Benders
        //
        //   l1_i  = dj[ arcPlantCap[i] ]
        //   l2_j  = dj[ arcDepotCap[j] ]
        //   m2_k  = pi_s - pi_cust(k)
        //
        //   coef_a[i] = p_i * l1_i
        //   coef_b[j] = q_j * l2_j
        //   rhs       = sum_k r_k * m2_k
        // ---------------------------------------------------------
        for (int i = 0; i < nI; ++i)
        {
            double l1_i = dj[arcPlantCap[i]];
            coef_a[i] = inst.p[i] * l1_i;
        }

        for (int j = 0; j < nJ; ++j)
        {
            double l2_j = dj[arcDepotCap[j]];
            coef_b[j] = inst.q[j] * l2_j;
        }

        auto nodeCust = [nI, nJ](int k)
        { return 1 + nI + 2 * nJ + k; };

        double pi_s = pi[node_s];
        rhs = 0.0;
        for (int k = 0; k < nK; ++k)
        {
            int node_k = nodeCust(k);
            double m2_k = pi_s - pi[node_k];
            rhs += inst.r[k] * m2_k;
        }
    }
};
