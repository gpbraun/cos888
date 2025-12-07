/*
COS888

tscfl_subproblem_net.hpp

Gabriel Braun, 2025
*/

#pragma once

#include <ilcplex/cplex.h>

#include <vector>

#include "tscfl_subproblem.hpp"

// SOLVER DO SUBPROBLEMA: Fluxo de custo mínimo em rede.
class SubproblemNet : public Subproblem
{
  private:
    int status;
    CPXENVptr env;
    CPXNETptr net;

    // Parâmetros da rede
    int nN{ 0 };
    int nA{ 0 };
    int node_s{ -1 }; // nó origem
    int node_t{ -1 }; // nó destino

    // Índices de arcos de capacidade que geram l1 e l2
    std::vector<int> arcPlantCap; // arco s -> plant_i
    std::vector<int> arcDepotCap; // arco depotIn_j -> depotOut_j

  public:
    explicit SubproblemNet(const TSCFLInstance &inst_);

    ~SubproblemNet() override;

    // Dado (a_vals, b_vals) da solução atual do mestre, resolve o subproblema.
    // Atualiza: theta, rhs, coef_a, coef_b
    void solve() override;

  private:
    // Constrói a rede base
    void buildNet();

    // Atualiza as capacidades dos arcos que dependem de (a,b)
    void updateNet();
};
