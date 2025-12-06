// include/utils.hpp
#pragma once

#include <ilcplex/ilocplex.h>

ILOSTLBEGIN

// =====================================================================
//  CONSTANTES GLOBAIS
// =====================================================================

inline constexpr double EPS = 1e-4;     // Tolerância numérica
inline constexpr double MIP_GAP = 1e-7; // Gap mínimo do MIP

// =====================================================================
//  TIPOS AUXILIARES
// =====================================================================

// Matriz de constantes.
class IloNumMatrix : public IloArray<IloNumArray>
{
  public:
    IloNumMatrix(IloEnv env, IloInt nRows, IloInt nCols);
};

// Tensor de constantes.
class IloNumTensor : public IloArray<IloNumMatrix>
{
  public:
    IloNumTensor(IloEnv env, IloInt nDim1, IloInt nDim2, IloInt nDim3);
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
        IloNumVar::Type type = ILOFLOAT
    );

    IloNumVarArray col(IloInt j) const;
};

// =====================================================================
//  FUNÇÕES AUXILIARES
// =====================================================================

// Produto escalar entre matrizes de constantes.
IloNum IloMatScalProd(const IloNumMatrix &c, const IloNumMatrix &d);

// Produto escalar entre matriz de constantes e matriz de variáveis.
IloExpr IloMatScalProd(const IloNumMatrix &c, const IloNumVarMatrix &x);

// Reseta um array para zero (IloNum).
void fill_zero(IloNumArray &a);

// Reseta uma matriz para zero (IloNum).
void fill_zero(IloNumMatrix &M);
