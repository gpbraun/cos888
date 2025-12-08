/*
COS888

tscfl_instance.cpp

Gabriel Braun, 2025
*/

#include "tscfl_instance.hpp"

#include <fstream>
#include <stdexcept>
#include <vector>

TSCFLInstance::TSCFLInstance(IloEnv &env_, int _nI, int _nJ, int _nK)
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

TSCFLInstance
TSCFLInstance::read(IloEnv &env, const std::string &path)
{
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("Erro de leitura: " + path);

    std::vector<IloNum> a;
    a.reserve(1 << 20);

    IloNum v;
    while (in >> v)
        a.push_back(v);

    const IloInt len = a.size();
    if (len < 3)
        throw std::runtime_error("Erro no arquivo da instância (header).");

    IloInt pos = 0;
    IloInt nI = a[pos++];
    IloInt nJ = a[pos++];
    IloInt nK = a[pos++];

    TSCFLInstance inst(env, nI, nJ, nK);

    // r: nK
    if (pos + nK > len)
        throw std::runtime_error("Erro no arquivo da instância (r).");

    for (IloInt k = 0; k < nK; ++k)
        inst.r[k] = a[pos++];

    // (q, g): nJ pares
    if (pos + 2 * nJ > len)
        throw std::runtime_error("Erro no arquivo da instância (q,g).");

    for (IloInt j = 0; j < nJ; ++j)
        {
            inst.q[j] = a[pos++];
            inst.g[j] = a[pos++];
        }

    // c: nI * nJ
    const IloInt nIJ = nI * nJ;
    if (pos + nIJ > len)
        throw std::runtime_error("Erro no arquivo da instância (c).");

    for (IloInt i = 0; i < nI; ++i)
        for (int j = 0; j < nJ; ++j)
            inst.c[i][j] = a[pos++];

    // (p, f): nI pares
    if (pos + 2 * nI > len)
        throw std::runtime_error("Erro no arquivo da instância (p,f).");

    for (IloInt i = 0; i < nI; ++i)
        {
            inst.p[i] = a[pos++];
            inst.f[i] = a[pos++];
        }

    // d: nJ * nK
    const IloInt nJK = nJ * nK;
    if (pos + nJK > len)
        throw std::runtime_error("Erro no arquivo da instância (d).");

    for (IloInt j = 0; j < nJ; ++j)
        for (IloInt k = 0; k < nK; ++k)
            inst.d[j][k] = a[pos++];

    return inst;
}
