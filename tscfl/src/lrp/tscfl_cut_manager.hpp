#pragma once

/*
COS888

Cut genérico, FlowCoverCut, SubsetRowCut e CutManager para o TSCFL.

Gabriel Braun, 2025
*/

#include <memory>
#include <unordered_set>
#include <vector>

#include "tscfl_instance.hpp"
#include "tscfl_utils.hpp"

ILOSTLBEGIN

// =====================================================================
//  Cut: base genérica
// =====================================================================

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
    IloInt age{ 0 };
    IloNum u{ 0.0 };        // multiplicador de Lagrange
    IloNum overflow{ 0.0 }; // violação (LHS - RHS)
    IloNum rhs{ 0.0 };      // lado direito
    std::size_t hash{ 0u }; // assinatura para detectar duplicados

    Cut(IloNum rhs_, std::size_t hash_);
    virtual ~Cut() = default;

    // LHS do corte em uma solução (x,y,a,b)
    virtual IloNum compute_lhs(
        const TSCFLInstance &inst,
        const IloNumMatrix &x_lr,
        const IloNumMatrix &y_lr,
        const IloNumArray &a_lr,
        const IloNumArray &b_lr
    ) const
        = 0;

    // Contribuição do corte para os custos agregados
    virtual void add_to_costs(
        const TSCFLInstance &inst,
        IloNumArray &cost_a,
        IloNumArray &cost_b,
        IloNumMatrix &cost_x,
        IloNumMatrix &cost_y
    ) const
        = 0;
};

// =====================================================================
//  FlowCoverCut: corte de flow-cover (planta/deposito)
// =====================================================================

class FlowCoverCut : public Cut
{
  public:
    enum class NodeType
    {
        PLANT,
        DEPOT
    };

  private:
    NodeType node_type_;
    int index_;                   // i (planta) ou j (depósito)
    IloNumArray cost_;            // coeficientes em x[i][·] ou y[j][·]
    std::vector<IloInt> support_; // índices com cost_ != 0

    static std::size_t compute_hash(NodeType node_type, int idx, const IloNumArray &cost);

  public:
    FlowCoverCut(NodeType node_type, int index, const IloNumArray &cost, IloNum rhs);

    NodeType
    node_type() const noexcept
    {
        return node_type_;
    }
    int
    index() const noexcept
    {
        return index_;
    }
    const IloNumArray &
    cost() const noexcept
    {
        return cost_;
    }

    IloNum compute_lhs(
        const TSCFLInstance &inst,
        const IloNumMatrix &x_lr,
        const IloNumMatrix &y_lr,
        const IloNumArray &a_lr,
        const IloNumArray &b_lr
    ) const override;

    void add_to_costs(
        const TSCFLInstance &inst,
        IloNumArray &cost_a,
        IloNumArray &cost_b,
        IloNumMatrix &cost_x,
        IloNumMatrix &cost_y
    ) const override;
};

// =====================================================================
//  SubsetRowCut: cortes de demanda agregada (subset-row)
// =====================================================================

class SubsetRowCut : public Cut
{
  public:
    enum class Family
    {
        PLANT,
        DEPOT
    };

  private:
    Family family_;
    IloNumArray coeff_;
    std::vector<IloInt> support_;

    static std::size_t compute_hash(Family family, const IloNumArray &coeff);

  public:
    SubsetRowCut(Family family, const IloNumArray &coeff, IloNum rhs);

    Family
    family() const noexcept
    {
        return family_;
    }
    const IloNumArray &
    coeff() const noexcept
    {
        return coeff_;
    }

    IloNum compute_lhs(
        const TSCFLInstance &inst,
        const IloNumMatrix &x_lr,
        const IloNumMatrix &y_lr,
        const IloNumArray &a_lr,
        const IloNumArray &b_lr
    ) const override;

    void add_to_costs(
        const TSCFLInstance &inst,
        IloNumArray &cost_a,
        IloNumArray &cost_b,
        IloNumMatrix &cost_x,
        IloNumMatrix &cost_y
    ) const override;
};

// =====================================================================
//  CutManager: gerenciador genérico de cortes
// =====================================================================

class CutManager
{
  public:
    IloEnv &env;
    const TSCFLInstance &inst;

    // custos agregados pelos cortes (recalculados a cada iteração)
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

    // Insere corte se ainda não existir (baseado no hash).
    // Retorna: true se o corte foi de fato adicionado.
    bool add(std::unique_ptr<Cut> cut);

    // Conveniências para FlowCover
    bool add_flow_cover(const FlowCoverCut &cut);
    bool add_flow_cover(
        FlowCoverCut::NodeType node_type, int index, const IloNumArray &cost, IloNum rhs
    );

    // Conveniências para SubsetRow
    bool add_subset_row(const SubsetRowCut &cut);
    bool add_subset_row(SubsetRowCut::Family family, const IloNumArray &coeff, IloNum rhs);

    // Retorna: número de cortes em cada status
    int count(Cut::Status s) const;

    // Contribuição dos cortes para ||g||^2
    IloNum norm2sq() const;

    // Atualiza multiplicadores dos cortes (subgradiente)
    void update_multipliers(IloNum step);

    // Recalcula custos agregados cost_a, cost_b, cost_x, cost_y
    void update_costs();

    // Atualiza violação e status CA/PA/CI, com aging
    void update_status(
        const IloNumMatrix &x_lr,
        const IloNumMatrix &y_lr,
        const IloNumArray &a_lr,
        const IloNumArray &b_lr,
        int extra_age
    );
};
