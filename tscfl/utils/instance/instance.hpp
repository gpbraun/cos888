/*
COS888

Classe para Instâncias do TSCFL.

Gabriel Braun, 2025
*/

#pragma once

#include <stdexcept>
#include <fstream>
#include <iostream>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <vector>
#include <string>
#include <unordered_set>
#include <algorithm>
#include <memory>
#include <numeric>
#include <limits>

#include <ilcplex/ilocplex.h>

ILOSTLBEGIN

// =====================================================================
//  CONSTANTES GLOBAIS
// =====================================================================

inline constexpr double EPS = 1e-4;     // Tolerância numérica
inline constexpr double MIP_GAP = 1e-7; // Gap mínimo do MIP

// =====================================================================
//  UTILS
// =====================================================================

// Matriz de constantes.
class IloNumMatrix : public IloArray<IloNumArray>
{
public:
    IloNumMatrix(
        IloEnv env,
        IloInt nRows,
        IloInt nCols)
        : IloArray<IloNumArray>(env, nRows)
    {
        for (IloInt i = 0; i < nRows; ++i)
            (*this)[i] = IloNumArray(env, nCols);
    }
};

// Tensor de constantes.
class IloNumTensor : public IloArray<IloNumMatrix>
{
public:
    IloNumTensor(
        IloEnv env,
        IloInt nDim1,
        IloInt nDim2,
        IloInt nDim3)
        : IloArray<IloNumMatrix>(env, nDim1)
    {
        for (IloInt i = 0; i < nDim1; ++i)
            (*this)[i] = IloNumMatrix(env, nDim2, nDim3);
    }
};

// Matriz de variáveis (fracionária positiva).
class IloNumVarMatrix : public IloArray<IloNumVarArray>
{
public:
    IloNumVarMatrix(
        IloEnv env,
        IloInt nRows,
        IloInt nCols,
        IloNum lb = 0.0,
        IloNum ub = IloInfinity,
        IloNumVar::Type type = ILOFLOAT)
        : IloArray<IloNumVarArray>(env, nRows)
    {
        for (IloInt i = 0; i < nRows; ++i)
            (*this)[i] = IloNumVarArray(env, nCols, 0.0, IloInfinity);
    }

    IloNumVarArray col(IloInt j) const
    {
        IloEnv env = getEnv();
        IloInt nRows = getSize();

        IloNumVarArray column(env, nRows);
        for (IloInt i = 0; i < nRows; ++i)
            column[i] = (*this)[i][j];

        return column;
    }
};

// Retorna: produto escalar entre matrizes de constantes.
inline IloNum IloMatScalProd(const IloNumMatrix &c, const IloNumMatrix &d)
{
    IloNum val = 0.0;
    for (IloInt i = 0; i < d.getSize(); ++i)
        val += IloScalProd(c[i], d[i]);

    return val;
}

// Retorna: produto escalar entre uma matrix de constantes e uma matriz de variáveis.
inline IloExpr IloMatScalProd(const IloNumMatrix &c, const IloNumVarMatrix &x)
{
    IloExpr e(x.getEnv());
    for (IloInt i = 0; i < x.getSize(); ++i)
        e += IloScalProd(c[i], x[i]);

    return e;
}

// Reseta um array para zero (IloNum).
inline void fill_zero(IloNumArray &a)
{
    for (IloInt i = 0; i < a.getSize(); ++i)
        a[i] = 0.0;
}

// Reseta uma matriz para zero (IloNum).
inline void fill_zero(IloNumMatrix &M)
{
    for (IloInt i = 0; i < M.getSize(); ++i)
        fill_zero(M[i]);
}

// =====================================================================
//  INSTÂNCIA
// =====================================================================

// INSTÂNCIA TSCFL
class TSCFLInstance
{
public:
    IloEnv &env;

    IloInt nI{0}; // |I| plantas
    IloInt nJ{0}; // |J| depósitos
    IloInt nK{0}; // |K| clientes

    IloNumArray p;  // p[i]    = capacidade da planta i
    IloNumArray q;  // q[j]    = capacidade do depósito j
    IloNumArray r;  // r[k]    = demanda do cliente k
    IloNumArray f;  // f[i]    = custo fixo da planta i
    IloNumArray g;  // g[j]    = custo fixo do depósito j
    IloNumMatrix c; // c[i][j] = custo planta i -> depósito j
    IloNumMatrix d; // d[j][k] = custo depósito j -> cliente k

    explicit TSCFLInstance(IloEnv &env_, int _nI, int _nJ, int _nK)
        : env(env_),
          nI(_nI),
          nJ(_nJ),
          nK(_nK),
          p(env_, nI),
          q(env_, nJ),
          r(env_, nK),
          f(env_, nI),
          g(env_, nJ),
          c(env_, nI, nJ),
          d(env_, nJ, nK)
    {
    }

    // Carrega instância a partir de arquivo `.txt`.
    static TSCFLInstance from_txt(IloEnv &env, const std::string &path)
    {
        std::ifstream in(path);
        if (!in)
            throw std::runtime_error("Erro de leitura: " + path);

        std::vector<double> a;
        a.reserve(1 << 20);
        double v;
        while (in >> v)
            a.push_back(v);

        const int len = static_cast<int>(a.size());
        if (len < 3)
            throw std::runtime_error("Erro no arquivo da instância (header).");

        int pos = 0;
        IloInt nI = static_cast<IloInt>(a[pos++]);
        IloInt nJ = static_cast<IloInt>(a[pos++]);
        IloInt nK = static_cast<IloInt>(a[pos++]);

        TSCFLInstance inst(env, nI, nJ, nK);

        // r: nK
        if (pos + nK > len)
            throw std::runtime_error("Erro no arquivo da instância (r).");

        for (int k = 0; k < nK; ++k)
            inst.r[k] = a[pos++];

        // (q, g): nJ pares
        if (pos + 2 * nJ > len)
            throw std::runtime_error("Erro no arquivo da instância (q,g).");

        for (int j = 0; j < nJ; ++j)
        {
            inst.q[j] = a[pos++];
            inst.g[j] = a[pos++];
        }

        // c: nI * nJ
        const int nIJ = nI * nJ;
        if (pos + nIJ > len)
            throw std::runtime_error("Erro no arquivo da instância (c).");

        for (int i = 0; i < nI; ++i)
        {
            for (int j = 0; j < nJ; ++j)
                inst.c[i][j] = a[pos++];
        }

        // (p, f): nI pares
        if (pos + 2 * nI > len)
            throw std::runtime_error("Erro no arquivo da instância (p,f).");

        for (int i = 0; i < nI; ++i)
        {
            inst.p[i] = a[pos++];
            inst.f[i] = a[pos++];
        }

        // d: nJ * nK
        const int nJK = nJ * nK;
        if (pos + nJK > len)
            throw std::runtime_error("Erro no arquivo da instância (d).");

        for (int j = 0; j < nJ; ++j)
            for (int k = 0; k < nK; ++k)
                inst.d[j][k] = a[pos++];

        return inst;
    }
};
