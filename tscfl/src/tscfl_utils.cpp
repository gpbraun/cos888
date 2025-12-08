/*
COS888

tscfl_utils.cpp

Gabriel Braun, 2025
*/

#include "tscfl_utils.hpp"

IloNumMatrix::IloNumMatrix(IloEnv env, IloInt nRows, IloInt nCols)
    : IloArray<IloNumArray>(env, nRows)
{
    for (IloInt i = 0; i < nRows; ++i)
        (*this)[i] = IloNumArray(env, nCols);
}

IloNumTensor::IloNumTensor(IloEnv env, IloInt nDim1, IloInt nDim2, IloInt nDim3)
    : IloArray<IloNumMatrix>(env, nDim1)
{
    for (IloInt i = 0; i < nDim1; ++i)
        (*this)[i] = IloNumMatrix(env, nDim2, nDim3);
}

IloNumVarMatrix::IloNumVarMatrix(
    IloEnv env, IloInt nRows, IloInt nCols, IloNum lb, IloNum ub, IloNumVar::Type type
)
    : IloArray<IloNumVarArray>(env, nRows)
{
    for (IloInt i = 0; i < nRows; ++i)
        (*this)[i] = IloNumVarArray(env, nCols, lb, ub, type);
}

IloNumVarArray
IloNumVarMatrix::col(IloInt j) const
{
    IloEnv env = getEnv();
    IloInt nRows = getSize();

    IloNumVarArray column(env, nRows);
    for (IloInt i = 0; i < nRows; ++i)
        column[i] = (*this)[i][j];

    return column;
}

IloNum
IloMatScalProd(const IloNumMatrix &c, const IloNumMatrix &d)
{
    IloNum val = 0.0;
    for (IloInt i = 0; i < d.getSize(); ++i)
        val += IloScalProd(c[i], d[i]);
    return val;
}

IloExpr
IloMatScalProd(const IloNumVarMatrix &x, const IloNumMatrix &c)
{
    IloExpr e(x.getEnv());
    for (IloInt i = 0; i < x.getSize(); ++i)
        e += IloScalProd(c[i], x[i]);
    return e;
}

IloExpr
IloMatScalProd(const IloNumMatrix &c, const IloNumVarMatrix &x)
{
    IloExpr e(x.getEnv());
    for (IloInt i = 0; i < x.getSize(); ++i)
        e += IloScalProd(c[i], x[i]);
    return e;
}

void
fillZero(IloNumArray &a)
{
    for (IloInt i = 0; i < a.getSize(); ++i)
        a[i] = 0.0;
}

void
fillZero(IloNumMatrix &a)
{
    for (IloInt i = 0; i < a.getSize(); ++i)
        fillZero(a[i]);
}
