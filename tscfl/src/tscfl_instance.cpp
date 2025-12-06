// src/tscfl_instance.cpp
#include "tscfl_instance.hpp"

#include <fstream>
#include <stdexcept>
#include <vector>

TSCFLInstance::TSCFLInstance(IloEnv& env_, int _nI, int _nJ, int _nK)
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
      d(env_, nJ, nK) {}

TSCFLInstance TSCFLInstance::from_txt(IloEnv& env, const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Erro de leitura: " + path);

    std::vector<double> a;
    a.reserve(1 << 20);

    double v;
    while (in >> v) a.push_back(v);

    const int len = static_cast<int>(a.size());
    if (len < 3) throw std::runtime_error("Erro no arquivo da instância (header).");

    int pos = 0;
    IloInt nI = static_cast<IloInt>(a[pos++]);
    IloInt nJ = static_cast<IloInt>(a[pos++]);
    IloInt nK = static_cast<IloInt>(a[pos++]);

    TSCFLInstance inst(env, nI, nJ, nK);

    // r: nK
    if (pos + nK > len) throw std::runtime_error("Erro no arquivo da instância (r).");

    for (int k = 0; k < nK; ++k) inst.r[k] = a[pos++];

    // (q, g): nJ pares
    if (pos + 2 * nJ > len)
        throw std::runtime_error("Erro no arquivo da instância (q,g).");

    for (int j = 0; j < nJ; ++j) {
        inst.q[j] = a[pos++];
        inst.g[j] = a[pos++];
    }

    // c: nI * nJ
    const int nIJ = nI * nJ;
    if (pos + nIJ > len) throw std::runtime_error("Erro no arquivo da instância (c).");

    for (int i = 0; i < nI; ++i)
        for (int j = 0; j < nJ; ++j) inst.c[i][j] = a[pos++];

    // (p, f): nI pares
    if (pos + 2 * nI > len)
        throw std::runtime_error("Erro no arquivo da instância (p,f).");

    for (int i = 0; i < nI; ++i) {
        inst.p[i] = a[pos++];
        inst.f[i] = a[pos++];
    }

    // d: nJ * nK
    const int nJK = nJ * nK;
    if (pos + nJK > len) throw std::runtime_error("Erro no arquivo da instância (d).");

    for (int j = 0; j < nJ; ++j)
        for (int k = 0; k < nK; ++k) inst.d[j][k] = a[pos++];

    return inst;
}
