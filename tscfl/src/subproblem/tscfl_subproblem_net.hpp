// src/subproblem_net.hpp
#pragma once

/*
COS888

SubproblemNet: resolve o subproblema de fluxo mínimo do TSCFL.

Gabriel Braun, 2025
*/

#include <ilcplex/cplex.h>

#include <vector>

#include "tscfl_subproblem.hpp"

// SOLVER DO SUBPROBLEMA: Fluxo mínimo em rede.
class SubproblemNet : public Subproblem {
   private:
    int status;
    CPXENVptr env;
    CPXNETptr net;

    // Parâmetros da rede
    int nN;
    int nA;
    int node_s;  // nó origem (super‐source)
    int node_t;  // nó destino (super‐sink)

    // Índices de arcos de capacidade que geram l1 e l2
    std::vector<int> arcPlantCap;  // arco s -> plant_i
    std::vector<int> arcDepotCap;  // arco depotIn_j -> depotOut_j

   public:
    explicit SubproblemNet(const TSCFLInstance& inst_);
    ~SubproblemNet() override;

    // Dado (a_vals, b_vals) da solução atual do mestre, resolve o subproblema.
    // Atualiza: theta, rhs, coef_a, coef_b
    void solve(const IloNumArray& a_vals, const IloNumArray& b_vals) override;

   private:
    // Constrói a rede base
    void build_base_net();

    // Atualiza as capacidades dos arcos que dependem de (a,b)
    void set_capacities(const IloNumArray& a_vals, const IloNumArray& b_vals);
};
