// include/tscfl_instance.hpp
#pragma once

#include <ilcplex/ilocplex.h>

#include <string>

#include "tscfl_utils.hpp"

ILOSTLBEGIN

// INSTÂNCIA TSCFL
class TSCFLInstance
{
  public:
    IloEnv &env;

    IloInt nI{ 0 }; // |I| plantas
    IloInt nJ{ 0 }; // |J| depósitos
    IloInt nK{ 0 }; // |K| clientes

    IloNumArray p;  // p[i]    = capacidade da planta i
    IloNumArray q;  // q[j]    = capacidade do depósito j
    IloNumArray r;  // r[k]    = demanda do cliente k
    IloNumArray f;  // f[i]    = custo fixo da planta i
    IloNumArray g;  // g[j]    = custo fixo do depósito j
    IloNumMatrix c; // c[i][j] = custo planta i -> depósito j
    IloNumMatrix d; // d[j][k] = custo depósito j -> cliente k

    TSCFLInstance(IloEnv &env_, int _nI, int _nJ, int _nK);

    // Carrega instância a partir de arquivo `.txt`.
    static TSCFLInstance from_txt(IloEnv &env, const std::string &path);
};
