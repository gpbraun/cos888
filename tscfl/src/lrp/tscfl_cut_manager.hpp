/*
COS888

tscfl_cut_manager.hpp

Gabriel Braun, 2025
*/

#pragma once

#include <memory>
#include <unordered_set>
#include <vector>

#include "tscfl_instance.hpp"
#include "tscfl_utils.hpp"

// CORTE: Base
class Cut
{
  public:
    enum class Status
    {
        CA,
        PA,
        CI
    };

    Status status{ Status::CA };
    IloInt age{ 0 };        // idade
    IloNum u{ 0.0 };        // multiplicador de Lagrange
    IloNum overflow{ 0.0 }; // violação (LHS - RHS)
    IloNum rhs{ 0.0 };      // lado direito
    std::size_t hash{ 0u }; // hash para detectar duplicatas

    Cut(IloNum rhs_, std::size_t hash_);

    virtual ~Cut() = default;

    // Retorna: LHS do corte em uma solução (x,y,a,b)
    virtual IloNum calculateLHS(
        const TSCFLInstance &inst,
        const IloNumMatrix &x,
        const IloNumMatrix &y,
        const IloNumArray &a,
        const IloNumArray &b
    ) const
        = 0;

    // Contribuição do corte para os custos agregados
    virtual void addToCosts(
        const TSCFLInstance &inst,
        IloNumArray &cost_a,
        IloNumArray &cost_b,
        IloNumMatrix &cost_x,
        IloNumMatrix &cost_y
    ) const
        = 0;
};

// CORTE: FlowCover
class FlowCoverCut : public Cut
{
  public:
    enum class Family
    {
        PLANT,
        DEPOT
    };

    Family family;
    int index;
    IloNumArray cost;
    std::vector<IloInt> support;

    FlowCoverCut(Family family_, int index_, const IloNumArray &cost_, IloNum rhs_);

    IloNum calculateLHS(
        const TSCFLInstance &inst,
        const IloNumMatrix &x,
        const IloNumMatrix &y,
        const IloNumArray &a,
        const IloNumArray &b
    ) const override;

    void addToCosts(
        const TSCFLInstance &inst,
        IloNumArray &cost_a,
        IloNumArray &cost_b,
        IloNumMatrix &cost_x,
        IloNumMatrix &cost_y
    ) const override;

  private:
    static std::size_t computeHash(Family family_, int idx_, const IloNumArray &cost_);
};

// CORTE: SubsetRow
class SubsetRowCut : public Cut
{
  public:
    enum class Family
    {
        PLANT,
        DEPOT
    };

    Family family;
    IloNumArray cost;
    std::vector<IloInt> support;

    SubsetRowCut(Family family_, const IloNumArray &cost_, IloNum rhs_);

    IloNum calculateLHS(
        const TSCFLInstance &inst,
        const IloNumMatrix &x,
        const IloNumMatrix &y,
        const IloNumArray &a,
        const IloNumArray &b
    ) const override;

    void addToCosts(
        const TSCFLInstance &inst,
        IloNumArray &cost_a,
        IloNumArray &cost_b,
        IloNumMatrix &cost_x,
        IloNumMatrix &cost_y
    ) const override;

  private:
    static std::size_t computeHash(Family family_, const IloNumArray &coeff_);
};

// GERENCIADOR DE CORTES
class CutManager
{
  public:
    IloEnv &env;
    const TSCFLInstance &inst;

    // custos agregados pelos cortes
    IloNumArray cost_a;
    IloNumArray cost_b;
    IloNumMatrix cost_x;
    IloNumMatrix cost_y;

  private:
    std::vector<std::unique_ptr<Cut>> cuts;
    std::unordered_set<std::size_t> hashes;

  public:
    explicit CutManager(const TSCFLInstance &inst_);

    void clear();

    std::vector<std::unique_ptr<Cut>> &
    data()
    {
        return cuts;
    }
    const std::vector<std::unique_ptr<Cut>> &
    data() const
    {
        return cuts;
    }

    // Adiciona: corte se ainda não existir (baseado no hash).
    // Retorna: true se o corte foi de fato adicionado.
    IloBool add(std::unique_ptr<Cut> cut);

    // Retorna: número de cortes em cada status
    IloInt count(Cut::Status s) const;

    // Retorna: contribuição dos cortes para ||g||^2
    IloNum norm2sq() const;

    // Atualiza: multiplicadores dos cortes (subgradiente)
    void updateMultipliers(IloNum step);

    // Atualiza: cost_a, cost_b, cost_x, cost_y
    void updateCosts();

    // Atualiza: violação e status (CA/PA/CI)
    void updateStatus(
        const IloNumMatrix &x,
        const IloNumMatrix &y,
        const IloNumArray &a,
        const IloNumArray &b,
        IloInt extra_age
    );

    // Adiciona: FlowCover
    IloBool addFlowCover(const FlowCoverCut &cut);
    IloBool addFlowCover(
        FlowCoverCut::Family family_, IloInt index_, const IloNumArray &cost_, IloNum rhs_
    );
    // Adiciona: SubsetRow
    IloBool addSubsetRow(const SubsetRowCut &cut);
    IloBool addSubsetRow(SubsetRowCut::Family family_, const IloNumArray &cost_, IloNum rhs_);
};
