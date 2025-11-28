/*
COS888

Classe para Instâncias do TSCFL.

Gabriel Braun, 2025
*/

#pragma once

#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// =====================================================================
//  UTILS
// =====================================================================

using Vec = std::vector<double>;
using IntVec = std::vector<int>;
using IntPair = std::pair<int, int>;
using IntPairVec = std::vector<IntPair>;

// Acesso em matriz 2D armazenada em vetor 1D.
inline int idx2(int i, int j, int ncols)
{
    return i * ncols + j;
}

// Range [0, n).
inline IntVec range_int(int n)
{
    IntVec v(n);
    std::iota(v.begin(), v.end(), 0);
    return v;
}

// Produto cartesiano entre dois conjuntos {0,...,nA-1} × {0,...,nB-1}.
inline IntPairVec product(int nA, int nB)
{
    IntPairVec v;
    v.reserve(nA * nB);
    for (int i = 0; i < nA; ++i)
        for (int j = 0; j < nB; ++j)
            v.emplace_back(i, j);
    return v;
}

// Produto escalar entre dois vetores.
inline double dot(const Vec &a, const Vec &b)
{
    const int n = static_cast<int>(std::min(a.size(), b.size()));
    double s = 0.0;
    for (int i = 0; i < n; ++i)
        s += a[i] * b[i];
    return s;
}

// Norma-2 ao quadrado.
inline double sqnorm(const Vec &a)
{
    double s = 0.0;
    for (double v : a)
        s += v * v;
    return s;
}

// =====================================================================
//  INSTÂNCIA DO TSCFL
// =====================================================================

class TSCFLInstance
{
public:
    int nI{0}; // |I| plantas
    int nJ{0}; // |J| depósitos
    int nK{0}; // |K| clientes

    Vec f; // f_i  = custo fixo da planta i
    Vec g; // g_j  = custo fixo do depósito j
    Vec p; // p_i  = capacidade da planta i
    Vec q; // q_j  = capacidade do depósito j
    Vec r; // r_k  = demanda do cliente k
    Vec c; // c_ij = custo planta i -> depósito j
    Vec d; // d_jk = custo depósito j -> cliente k

    // Acessores de matriz para c_ij e d_jk
    inline double C(int i, int j) const { return c[idx2(i, j, nJ)]; }
    inline double &C(int i, int j) { return c[idx2(i, j, nJ)]; }

    inline double D(int j, int k) const { return d[idx2(j, k, nK)]; }
    inline double &D(int j, int k) { return d[idx2(j, k, nK)]; }

    // Conjuntos de índices
    inline IntVec I() const { return range_int(nI); }
    inline IntVec J() const { return range_int(nJ); }
    inline IntVec K() const { return range_int(nK); }

    inline IntPairVec IJ() const { return product(nI, nJ); }
    inline IntPairVec JK() const { return product(nJ, nK); }

    // Carrega instância a partir de arquivo .txt
    static TSCFLInstance from_txt(const std::string &path)
    {
        std::ifstream in(path);
        if (!in)
            throw std::runtime_error("Cannot open instance: " + path);

        Vec a;
        a.reserve(1 << 20);
        double v;
        while (in >> v)
            a.push_back(v);

        const int len = static_cast<int>(a.size());
        if (len < 3)
            throw std::runtime_error("Malformed file (header too short).");

        int pos = 0;
        TSCFLInstance inst;
        inst.nI = static_cast<int>(a[pos++]);
        inst.nJ = static_cast<int>(a[pos++]);
        inst.nK = static_cast<int>(a[pos++]);

        const int nI = inst.nI;
        const int nJ = inst.nJ;
        const int nK = inst.nK;

        // r: nK
        if (pos + nK > len)
            throw std::runtime_error("Malformed file (r).");
        inst.r.resize(nK);
        for (int k = 0; k < nK; ++k)
            inst.r[k] = a[pos++];

        // (q, g): nJ pares
        if (pos + 2 * nJ > len)
            throw std::runtime_error("Malformed file (q,g).");
        inst.q.resize(nJ);
        inst.g.resize(nJ);
        for (int j = 0; j < nJ; ++j)
        {
            inst.q[j] = a[pos++];
            inst.g[j] = a[pos++];
        }

        // c: nI * nJ
        const int nIJ = nI * nJ;
        if (pos + nIJ > len)
            throw std::runtime_error("Malformed file (c).");
        inst.c.resize(nIJ);
        for (int i = 0; i < nI; ++i)
            for (int j = 0; j < nJ; ++j)
                inst.C(i, j) = a[pos++];

        // (p, f): nI pares
        if (pos + 2 * nI > len)
            throw std::runtime_error("Malformed file (p,f).");
        inst.p.resize(nI);
        inst.f.resize(nI);
        for (int i = 0; i < nI; ++i)
        {
            inst.p[i] = a[pos++];
            inst.f[i] = a[pos++];
        }

        // d: nJ * nK
        const int nJK = nJ * nK;
        if (pos + nJK > len)
            throw std::runtime_error("Malformed file (d).");
        inst.d.resize(nJK);
        for (int j = 0; j < nJ; ++j)
            for (int k = 0; k < nK; ++k)
                inst.D(j, k) = a[pos++];

        return inst;
    }
};
