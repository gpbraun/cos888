#include "lrp/tscfl_cut_manager.hpp"

#include <cmath>
#include <functional>
#include <string>

ILOSTLBEGIN

// =====================================================================
//  Cut – implementação
// =====================================================================

Cut::Cut (IloNum rhs_, std::size_t hash_) : rhs (rhs_), hash (hash_) {}

// =====================================================================
//  FlowCoverCut – implementação
// =====================================================================

FlowCoverCut::FlowCoverCut (NodeType node_type, int index, const IloNumArray &cost, IloNum rhs)
    : Cut (rhs, compute_hash (node_type, index, cost)), node_type_ (node_type), index_ (index),
      cost_ (cost)
{
    const IloInt n = cost_.getSize ();
    support_.reserve (static_cast<std::size_t> (n));
    for (IloInt t = 0; t < n; ++t)
        {
            if (std::fabs (cost_[t]) > EPS)
                {
                    support_.push_back (t);
                }
        }
}

std::size_t
FlowCoverCut::compute_hash (NodeType node_type, int idx, const IloNumArray &cost)
{
    std::string key;
    key.reserve (32 + 4 * static_cast<std::size_t> (cost.getSize ()));

    // 'F' = FlowCover, 'P'/'D' = Plant/Depot
    key.push_back ('F');
    key.push_back (node_type == NodeType::PLANT ? 'P' : 'D');
    key.push_back (':');
    key += std::to_string (idx);
    key.push_back ('|');

    for (IloInt j = 0; j < cost.getSize (); ++j)
        {
            if (std::fabs (cost[j]) > EPS)
                {
                    key += std::to_string (j);
                    key.push_back ('#');
                }
        }

    return std::hash<std::string>{}(key);
}

IloNum
FlowCoverCut::compute_lhs (const TSCFLInstance &inst, const IloNumMatrix &x_lr,
                           const IloNumMatrix &y_lr, const IloNumArray &a_lr,
                           const IloNumArray &b_lr) const
{
    IloNum lhs = 0.0;

    if (node_type_ == NodeType::PLANT)
        {
            const int i = index_;
            lhs += -inst.p[i] * a_lr[i];
            for (IloInt j : support_)
                {
                    lhs += cost_[j] * x_lr[i][j];
                }
        }
    else
        { // DEPOT
            const int j = index_;
            lhs += -inst.q[j] * b_lr[j];
            for (IloInt k : support_)
                {
                    lhs += cost_[k] * y_lr[j][k];
                }
        }

    return lhs;
}

void
FlowCoverCut::add_to_costs (const TSCFLInstance &inst, IloNumArray &cost_a, IloNumArray &cost_b,
                            IloNumMatrix &cost_x, IloNumMatrix &cost_y) const
{
    if (u <= EPS)
        return;

    if (node_type_ == NodeType::PLANT)
        {
            const int i = index_;
            for (IloInt j : support_)
                {
                    cost_x[i][j] += u * cost_[j];
                }
            cost_a[i] += -u * inst.p[i];
        }
    else
        { // DEPOT
            const int j = index_;
            for (IloInt k : support_)
                {
                    cost_y[j][k] += u * cost_[k];
                }
            cost_b[j] += -u * inst.q[j];
        }
}

// =====================================================================
//  SubsetRowCut – implementação
// =====================================================================

SubsetRowCut::SubsetRowCut (Family family, const IloNumArray &coeff, IloNum rhs)
    : Cut (rhs, compute_hash (family, coeff)), family_ (family), coeff_ (coeff)
{
    const IloInt n = coeff_.getSize ();
    support_.reserve (static_cast<std::size_t> (n));
    for (IloInt t = 0; t < n; ++t)
        {
            if (std::fabs (coeff_[t]) > EPS)
                {
                    support_.push_back (t);
                }
        }
}

std::size_t
SubsetRowCut::compute_hash (Family family, const IloNumArray &coeff)
{
    std::string key;
    key.reserve (32 + 8 * static_cast<std::size_t> (coeff.getSize ()));

    // 'S' = SubsetRow, 'P'/'D' = Plant/Depot
    key.push_back ('S');
    key.push_back (family == Family::PLANT ? 'P' : 'D');
    key.push_back ('|');

    for (IloInt j = 0; j < coeff.getSize (); ++j)
        {
            if (std::fabs (coeff[j]) > EPS)
                {
                    key += std::to_string (j);
                    key.push_back ('=');

                    // quantização para evitar ruído numérico
                    const double scaled = std::round (coeff[j] * 1e6); // 6 casas
                    key += std::to_string (static_cast<long long> (scaled));
                    key.push_back ('#');
                }
        }

    return std::hash<std::string>{}(key);
}

IloNum
SubsetRowCut::compute_lhs (const TSCFLInstance &, const IloNumMatrix &, const IloNumMatrix &,
                           const IloNumArray &a_lr, const IloNumArray &b_lr) const
{
    IloNum lhs = 0.0;

    if (family_ == Family::PLANT)
        {
            for (IloInt i : support_)
                {
                    lhs += -coeff_[i] * a_lr[i];
                }
        }
    else
        { // DEPOT
            for (IloInt j : support_)
                {
                    lhs += -coeff_[j] * b_lr[j];
                }
        }

    return lhs;
}

void
SubsetRowCut::add_to_costs (const TSCFLInstance &, IloNumArray &cost_a, IloNumArray &cost_b,
                            IloNumMatrix &, IloNumMatrix &) const
{
    if (u <= EPS)
        return;

    if (family_ == Family::PLANT)
        {
            for (IloInt i : support_)
                {
                    cost_a[i] += -u * coeff_[i];
                }
        }
    else
        { // DEPOT
            for (IloInt j : support_)
                {
                    cost_b[j] += -u * coeff_[j];
                }
        }
}

// =====================================================================
//  CutManager – implementação
// =====================================================================

CutManager::CutManager (const TSCFLInstance &inst_)
    : env (inst_.env), inst (inst_), cost_a (env, inst_.nI), cost_b (env, inst_.nJ),
      cost_x (env, inst_.nI, inst_.nJ), cost_y (env, inst_.nJ, inst_.nK)
{
}

void
CutManager::clear ()
{
    cuts.clear ();
    hashes.clear ();
}

bool
CutManager::add (std::unique_ptr<Cut> cut)
{
    auto res = hashes.insert (cut->hash);
    if (!res.second)
        return false; // corte duplicado
    cuts.push_back (std::move (cut));
    return true;
}

// Conveniências para FlowCover
bool
CutManager::add_flow_cover (const FlowCoverCut &cut)
{
    auto ptr = std::make_unique<FlowCoverCut> (cut);
    return add (std::move (ptr));
}

bool
CutManager::add_flow_cover (FlowCoverCut::NodeType node_type, int index, const IloNumArray &cost,
                            IloNum rhs)
{
    auto ptr = std::make_unique<FlowCoverCut> (node_type, index, cost, rhs);
    return add (std::move (ptr));
}

// Conveniências para SubsetRow
bool
CutManager::add_subset_row (const SubsetRowCut &cut)
{
    auto ptr = std::make_unique<SubsetRowCut> (cut);
    return add (std::move (ptr));
}

bool
CutManager::add_subset_row (SubsetRowCut::Family family, const IloNumArray &coeff, IloNum rhs)
{
    auto ptr = std::make_unique<SubsetRowCut> (family, coeff, rhs);
    return add (std::move (ptr));
}

int
CutManager::count (Cut::Status s) const
{
    int cnt = 0;
    for (const auto &c : cuts)
        if (c->status == s)
            ++cnt;
    return cnt;
}

IloNum
CutManager::norm2sq () const
{
    IloNum s = 0.0;
    for (const auto &c : cuts)
        {
            if (c->status != Cut::Status::CI)
                s += c->overflow * c->overflow;
        }
    return s;
}

void
CutManager::update_multipliers (IloNum step)
{
    if (step <= 0.0)
        return;

    for (auto &c : cuts)
        {
            if (c->status != Cut::Status::CI)
                {
                    c->u = IloMax (0.0, c->u + step * c->overflow);
                }
        }
}

void
CutManager::update_costs ()
{
    fill_zero (cost_x);
    fill_zero (cost_y);
    fill_zero (cost_a);
    fill_zero (cost_b);

    for (const auto &c : cuts)
        {
            if (c->status == Cut::Status::CI || c->u <= EPS)
                continue;
            c->add_to_costs (inst, cost_a, cost_b, cost_x, cost_y);
        }
}

void
CutManager::update_status (const IloNumMatrix &x_lr, const IloNumMatrix &y_lr,
                           const IloNumArray &a_lr, const IloNumArray &b_lr, int extra_age)
{
    for (auto &c_ptr : cuts)
        {
            Cut &c = *c_ptr;
            IloNum lhs = c.compute_lhs (inst, x_lr, y_lr, a_lr, b_lr);
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
