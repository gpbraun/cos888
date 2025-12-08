/*
COS888

tscfl_cut_manager.cpp

Gabriel Braun, 2025
*/

#include <cmath>
#include <functional>
#include <string>

#include "lrp/tscfl_cut_manager.hpp"

Cut::Cut(IloNum rhs_, std::size_t hash_)
    : rhs(rhs_),
      hash(hash_)
{
}

FlowCoverCut::FlowCoverCut(Family family_, int index_, const IloNumArray &cost_, IloNum rhs_)
    : Cut(rhs_, computeHash(family_, index_, cost_)),
      family(family_),
      index(index_),
      cost(cost_)
{
    const IloInt n = cost.getSize();
    support.reserve(static_cast<std::size_t>(n));

    for (IloInt t = 0; t < n; ++t)
        if (std::fabs(cost[t]) > EPS)
            support.push_back(t);
}

std::size_t
FlowCoverCut::computeHash(Family family_, int idx_, const IloNumArray &cost_)
{
    std::string key;
    key.reserve(32 + 4 * static_cast<std::size_t>(cost_.getSize()));

    // 'F' = FlowCover, 'P'/'D' = Plant/Depot
    key.push_back('F');
    key.push_back(family_ == Family::PLANT ? 'P' : 'D');
    key.push_back(':');
    key += std::to_string(idx_);
    key.push_back('|');

    for (IloInt j = 0; j < cost_.getSize(); ++j)
        if (std::fabs(cost_[j]) > EPS)
            {
                key += std::to_string(j);
                key.push_back('#');
            }

    return std::hash<std::string>{}(key);
}

IloNum
FlowCoverCut::calculateLHS(
    const TSCFLInstance &inst,
    const IloNumMatrix &x,
    const IloNumMatrix &y,
    const IloNumArray &a,
    const IloNumArray &b
) const
{
    IloNum lhs = 0.0;

    if (family == Family::PLANT)
        {
            const int i = index;
            lhs += -inst.p[i] * a[i];
            for (IloInt j : support)
                lhs += cost[j] * x[i][j];
        }
    else
        { // DEPOT
            const int j = index;
            lhs += -inst.q[j] * b[j];
            for (IloInt k : support)
                lhs += cost[k] * y[j][k];
        }

    return lhs;
}

void
FlowCoverCut::addToCosts(
    const TSCFLInstance &inst,
    IloNumArray &cost_a,
    IloNumArray &cost_b,
    IloNumMatrix &cost_x,
    IloNumMatrix &cost_y
) const
{
    if (u <= EPS)
        return;

    if (family == Family::PLANT)
        {
            const IloInt i = index;
            for (IloInt j : support)
                cost_x[i][j] += u * cost[j];

            cost_a[i] += -u * inst.p[i];
        }
    else
        { // DEPOT
            const IloInt j = index;
            for (IloInt k : support)
                cost_y[j][k] += u * cost[k];

            cost_b[j] += -u * inst.q[j];
        }
}

SubsetRowCut::SubsetRowCut(Family family_, const IloNumArray &cost_, IloNum rhs_)
    : Cut(rhs_, computeHash(family_, cost_)),
      family(family_),
      cost(cost_)
{
    const IloInt n = cost.getSize();
    support.reserve(static_cast<std::size_t>(n));

    for (IloInt t = 0; t < n; ++t)
        if (std::fabs(cost[t]) > EPS)
            support.push_back(t);
}

std::size_t
SubsetRowCut::computeHash(Family family_, const IloNumArray &cost_)
{
    std::string key;
    key.reserve(32 + 8 * static_cast<std::size_t>(cost_.getSize()));

    // 'S' = SubsetRow, 'P'/'D' = Plant/Depot
    key.push_back('S');
    key.push_back(family_ == Family::PLANT ? 'P' : 'D');
    key.push_back('|');

    for (IloInt j = 0; j < cost_.getSize(); ++j)
        if (std::fabs(cost_[j]) > EPS)
            {
                key += std::to_string(j);
                key.push_back('=');

                // quantização para evitar ruído numérico
                const double scaled = std::round(cost_[j] * 1e6);
                key += std::to_string(static_cast<long long>(scaled));
                key.push_back('#');
            }

    return std::hash<std::string>{}(key);
}

IloNum
SubsetRowCut::calculateLHS(
    const TSCFLInstance &,
    const IloNumMatrix &,
    const IloNumMatrix &,
    const IloNumArray &a,
    const IloNumArray &b
) const
{
    IloNum lhs = 0.0;

    if (family == Family::PLANT)
        {
            for (IloInt i : support)
                lhs += -cost[i] * a[i];
        }
    else // DEPOT
        {
            for (IloInt j : support)
                lhs += -cost[j] * b[j];
        }

    return lhs;
}

void
SubsetRowCut::addToCosts(
    const TSCFLInstance &, IloNumArray &cost_a, IloNumArray &cost_b, IloNumMatrix &, IloNumMatrix &
) const
{
    if (u <= EPS)
        return;

    if (family == Family::PLANT)
        {
            for (IloInt i : support)
                cost_a[i] += -u * cost[i];
        }
    else // DEPOT
        {
            for (IloInt j : support)
                cost_b[j] += -u * cost[j];
        }
}

CutManager::CutManager(const TSCFLInstance &inst_)
    : env(inst_.env),
      inst(inst_),
      cost_a(env, inst_.nI),
      cost_b(env, inst_.nJ),
      cost_x(env, inst_.nI, inst_.nJ),
      cost_y(env, inst_.nJ, inst_.nK)
{
}

void
CutManager::clear()
{
    cuts.clear();
    hashes.clear();
}

IloBool
CutManager::add(std::unique_ptr<Cut> cut)
{
    auto res = hashes.insert(cut->hash);
    if (!res.second)
        return false; // corte duplicado

    cuts.push_back(std::move(cut));
    return true;
}

IloInt
CutManager::count(Cut::Status s) const
{
    int cnt = 0;
    for (const auto &c : cuts)
        if (c->status == s)
            ++cnt;

    return cnt;
}

IloNum
CutManager::norm2sq() const
{
    IloNum s = 0.0;
    for (const auto &c : cuts)
        if (c->status != Cut::Status::CI)
            s += c->overflow * c->overflow;

    return s;
}

void
CutManager::updateMultipliers(IloNum step)
{
    for (auto &c : cuts)
        if (c->status != Cut::Status::CI)
            c->u = IloMax(0.0, c->u + step * c->overflow);
}

void
CutManager::updateCosts()
{
    fillZero(cost_x);
    fillZero(cost_y);
    fillZero(cost_a);
    fillZero(cost_b);

    for (const auto &c : cuts)
        {
            if (c->status == Cut::Status::CI || c->u <= EPS)
                continue;

            c->addToCosts(inst, cost_a, cost_b, cost_x, cost_y);
        }
}

void
CutManager::updateStatus(
    const IloNumMatrix &x,
    const IloNumMatrix &y,
    const IloNumArray &a,
    const IloNumArray &b,
    IloInt extra_age
)
{
    for (auto &c_ptr : cuts)
        {
            Cut &c = *c_ptr;
            IloNum lhs = c.calculateLHS(inst, x, y, a, b);
            c.overflow = lhs - c.rhs;

            if (c.overflow > EPS)
                {
                    c.status = Cut::Status::CA;
                    c.age = 0;
                }
            else if (c.u > EPS)
                {
                    c.status = Cut::Status::PA;
                    c.age = 0;
                }
            else
                {
                    ++c.age;
                    if (c.age >= extra_age)
                        {
                            c.status = Cut::Status::CI;
                            c.u = 0.0;
                        }
                }
        }
}

IloBool
CutManager::addFlowCover(const FlowCoverCut &cut)
{
    auto ptr = std::make_unique<FlowCoverCut>(cut);
    return add(std::move(ptr));
}
IloBool
CutManager::addFlowCover(
    FlowCoverCut::Family family_, IloInt index_, const IloNumArray &cost_, IloNum rhs_
)
{
    auto ptr = std::make_unique<FlowCoverCut>(family_, index_, cost_, rhs_);
    return add(std::move(ptr));
}
IloBool
CutManager::addSubsetRow(const SubsetRowCut &cut)
{
    auto ptr = std::make_unique<SubsetRowCut>(cut);
    return add(std::move(ptr));
}
IloBool
CutManager::addSubsetRow(SubsetRowCut::Family family_, const IloNumArray &cost_, IloNum rhs_)
{
    auto ptr = std::make_unique<SubsetRowCut>(family_, cost_, rhs_);
    return add(std::move(ptr));
}