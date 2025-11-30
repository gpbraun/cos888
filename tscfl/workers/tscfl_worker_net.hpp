/*
COS888

WorkerNet: resolve o subproblema de fluxo mínimo do TSCFL para uso na decomposição de Benders.

Gabriel Braun, 2025
*/

#pragma once

#include <ilcplex/ilocplex.h>
#include <ilcplex/cplex.h>
#include <stdexcept>
#include <vector>

#include "tscfl_worker.hpp"

ILOSTLBEGIN

// SOLVER DO SUBPROBLEMA DE BENDERS: Problema de fluxo mínimo
class WorkerNet : public Worker
{
private:
    int status;
    CPXENVptr env;
    CPXNETptr net;

    // Parâmetros da rede
    int nN;
    int nA;
    int node_s; // nó origem (super‐source)
    int node_t; // nó destino (super‐sink)

    // Índices de arcos de capacidade que geram l1 e l2
    std::vector<int> arcPlantCap; // arco s -> plant_i
    std::vector<int> arcDepotCap; // arco depotIn_j -> depotOut_j

public:
    explicit WorkerNet(const TSCFLInstance &inst_)
        : Worker(inst_),
          status(0),
          env(nullptr),
          net(nullptr),
          nN(0),
          nA(0),
          node_s(-1),
          node_t(-1),
          arcPlantCap(inst_.nI),
          arcDepotCap(inst_.nJ)
    {
        env = CPXopenCPLEX(&status);
        if (env == nullptr || status)
            throw std::runtime_error("WorkerNet: falha em CPXopenCPLEX.");

        net = CPXNETcreateprob(env, &status, "tscfl_net");
        if (status)
        {
            CPXcloseCPLEX(&env);
            env = nullptr;
            throw std::runtime_error("WorkerNet: falha em CPXNETcreateprob.");
        }

        build_base_net();

        // Subproblema silencioso por padrão
        CPXsetintparam(env, CPXPARAM_ScreenOutput, CPX_OFF);
    }

    ~WorkerNet() override
    {
        if (env != nullptr)
        {
            if (net != nullptr)
            {
                CPXNETfreeprob(env, &net);
            }
            CPXcloseCPLEX(&env);
            env = nullptr;
        }
    }

private:
    // Constrói a rede base
    void build_base_net()
    {
        const int nI = inst.nI;
        const int nJ = inst.nJ;
        const int nK = inst.nK;

        // Nós:
        //   0                 (s)
        //   1..nI             (plants)
        //   nI+1 .. nI+nJ     (depotIn)
        //   nI+nJ+1 .. nI+2nJ (depotOut)
        //   ...               (customers)
        //   último            (t)
        nN = 2 + nI + 2 * nJ + nK;
        node_s = 0;
        node_t = nN - 1;

        auto nodePlant = [nI](int i)
        { return 1 + i; };

        auto nodeDepotIn = [nI](int j)
        { return 1 + nI + j; };

        auto nodeDepotOut = [nI, nJ](int j)
        { return 1 + nI + nJ + j; };

        auto nodeCust = [nI, nJ](int k)
        { return 1 + nI + 2 * nJ + k; };

        // Contagem de arcos:
        //   s -> plant:           (nI)
        //   plant -> depotIn:     (nI*nJ)
        //   depotIn -> depotOut:  (nJ)
        //   depotOut -> customer: (nJ*nK)
        //   customer -> t:        (nK)
        nA = nI + nI * nJ + nJ + nJ * nK + nK;

        std::vector<int> from(nA);
        std::vector<int> to(nA);
        std::vector<double> lb(nA, 0.0);
        std::vector<double> ub(nA, 0.0);
        std::vector<double> cost(nA, 0.0);

        // Suprimento dos nós
        std::vector<double> supply(nN, 0.0);
        double demand_sum = IloSum(inst.r);
        supply[node_s] = demand_sum;
        supply[node_t] = -demand_sum;

        int arcId = 0;

        // 2.1) Arcos s -> plant_i (capacidade p_i * a_i, custo 0)
        for (int i = 0; i < nI; ++i)
        {
            int u = node_s;
            int v = nodePlant(i);

            from[arcId] = u;
            to[arcId] = v;
            lb[arcId] = 0.0;
            ub[arcId] = 0.0; // será atualizado em set_capacities()
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
                lb[arcId] = 0.0;
                ub[arcId] = CPX_INFBOUND;
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
            lb[arcId] = 0.0;
            ub[arcId] = 0.0; // será atualizado em set_capacities()
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
                lb[arcId] = 0.0;
                ub[arcId] = CPX_INFBOUND;
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
            lb[arcId] = 0.0;
            ub[arcId] = inst.r[k];
            cost[arcId] = 0.0;

            ++arcId;
        }

        // 2.6) Adiciona nós e arcos no CPXNET
        status = CPXNETaddnodes(env, net, nN, supply.data(), nullptr);
        if (status)
        {
            CPXNETfreeprob(env, &net);
            CPXcloseCPLEX(&env);
            env = nullptr;
            throw std::runtime_error("WorkerNet: falha em CPXNETaddnodes.");
        }

        status = CPXNETaddarcs(
            env, net, nA, from.data(), to.data(), lb.data(), ub.data(), cost.data(), nullptr);
        if (status)
        {
            CPXNETfreeprob(env, &net);
            CPXcloseCPLEX(&env);
            env = nullptr;
            throw std::runtime_error("WorkerNet: falha em CPXNETaddarcs.");
        }
    }

    // Atualiza as capacidades dos arcos que dependem de (a,b):
    void set_capacities(const IloNumArray &a_vals, const IloNumArray &b_vals)
    {
        const int nI = inst.nI;
        const int nJ = inst.nJ;

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

        status = CPXNETchgbds(env, net, cnt, idx.data(), lu.data(), bd.data());
        if (status)
            throw std::runtime_error("WorkerNet: falha em CPXNETchgbds.");
    }

public:
    // Dado (a_vals, b_vals) da solução atual do mestre, resolve o subproblema.
    // Atualiza: theta, rhs, coef_a, coef_b
    void solve(const IloNumArray &a_vals, const IloNumArray &b_vals) override
    {
        const int nI = inst.nI;
        const int nJ = inst.nJ;
        const int nK = inst.nK;

        // 1) Atualiza capacidades dos arcos dependentes de (a,b)
        set_capacities(a_vals, b_vals);

        // 2) Resolve o problema de fluxo mínimo
        status = CPXNETprimopt(env, net);
        if (status)
            throw std::runtime_error("WorkerNet: falha em CPXNETprimopt.");

        double obj_val = 0.0;
        status = CPXNETgetobjval(env, net, &obj_val);
        if (status)
            throw std::runtime_error("WorkerNet: falha em CPXNETgetobjval.");

        theta = obj_val;

        // 3) Lê potenciais de nó (pi) e custos reduzidos (dj)
        std::vector<double> pi(nN);
        std::vector<double> dj(nA);

        status = CPXNETgetpi(env, net, pi.data(), 0, nN - 1);
        if (status)
            throw std::runtime_error("WorkerNet: falha em CPXNETgetpi.");

        status = CPXNETgetdj(env, net, dj.data(), 0, nA - 1);
        if (status)
            throw std::runtime_error("WorkerNet: falha em CPXNETgetdj.");

        // 4) Mapeia duais para cortes de Benders e calcula os coeficientes do corte
        //   l1_i  = dj[ arcPlantCap[i] ]
        //   l2_j  = dj[ arcDepotCap[j] ]
        //   m2_k  = pi_s - pi_cust(k)
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
