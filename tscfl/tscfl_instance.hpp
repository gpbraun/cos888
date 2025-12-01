/*
COS888

Classe para Instâncias do TSCFL.

Gabriel Braun, 2025
*/

#pragma once

#include <ilcplex/ilocplex.h>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

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

// Matriz de variáveis.
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
            (*this)[i] = IloNumVarArray(env, nCols, 0.0, IloInfinity, ILOFLOAT);
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

// Reseta um array para zero.
inline void fill_zero(IloNumArray &a)
{
    IloInt n = a.getSize();
    for (IloInt i = 0; i < n; ++i)
        a[i] = 0.0;
}

// Reseta uma matriz para zero.
inline void fill_zero(IloNumMatrix &M)
{
    IloInt n = M.getSize();
    for (IloInt i = 0; i < n; ++i)
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

    int nI{0}; // |I| plantas
    int nJ{0}; // |J| depósitos
    int nK{0}; // |K| clientes

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
        int nI = static_cast<int>(a[pos++]);
        int nJ = static_cast<int>(a[pos++]);
        int nK = static_cast<int>(a[pos++]);

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
