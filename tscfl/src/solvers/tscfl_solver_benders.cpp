/*
COS888

tscfl_solver_benders.hpp

Gabriel Braun, 2025
*/

#include "tscfl_solver_benders.hpp"

#include <iomanip>
#include <iostream>

namespace
{
// CUTS: Lazy
class LazyBendersCallbackI : public IloCplex::LazyConstraintCallbackI
{
  private:
    const IloEnv &env;
    const TSCFLInstance &inst;
    Subproblem &subproblem;

    const IloBoolVarArray &var_a;
    const IloBoolVarArray &var_b;
    const IloNumVar &var_eta;

    IloNumArray a;
    IloNumArray b;
    IloNum eta{ 0.0 };

  public:
    LazyBendersCallbackI(
        const TSCFLInstance &inst_,
        Subproblem &subproblem_,
        const IloBoolVarArray &var_a_,
        const IloBoolVarArray &var_b_,
        const IloNumVar &var_eta_
    )
        : IloCplex::LazyConstraintCallbackI(inst_.env),
          env(inst_.env),
          inst(inst_),
          subproblem(subproblem_),
          var_a(var_a_),
          var_b(var_b_),
          var_eta(var_eta_),
          a(env, inst_.nI),
          b(env, inst_.nJ)
    {
    }

    IloCplex::CallbackI *
    duplicateCallback() const override
    {
        return (new (getEnv()) LazyBendersCallbackI(*this));
    }

    void
    main() override
    {
        // Lê (a, b, eta) da solução corrente
        getValues(a, var_a);
        getValues(b, var_b);
        eta = getValue(var_eta);

        // Resolve subproblema
        subproblem.update(a, b);
        subproblem.solve();

        // Testa violação
        IloNum viol = subproblem.theta - eta;
        if (viol <= EPS)
            return;

        // Adiciona lazy cut
        add(var_eta >= subproblem.rhs + IloScalProd(subproblem.coef_a, var_a)
                           + IloScalProd(subproblem.coef_b, var_b));
    }
};

// CUTS: User
class UserBendersCallbackI : public IloCplex::UserCutCallbackI
{
  public:
    // Parâmetros do callback
    static constexpr IloNum MAX_GAP = 1e-4;
    static constexpr IloNum ABS_VIOL = 1e-1;
    static constexpr IloNum REL_VIOL = 1e-4;
    static constexpr IloNum MAX_FRAC_SUM = 10.0;
    static constexpr IloInt MAX_NODE_CUTS = 1;
    static constexpr IloInt MAX_NODE_INDEX = 10;
    static constexpr IloNum OMEGA_CORE = 0.5;
    static constexpr IloNum OMEGA_SET = 0.5;

  private:
    const IloEnv &env;
    const TSCFLInstance &inst;
    Subproblem &subproblem;

    const IloBoolVarArray &var_a;
    const IloBoolVarArray &var_b;
    const IloNumVar &var_eta;

    IloNumArray a;
    IloNumArray b;
    IloNum eta{ 0.0 };

    // Core point e ponto de separação
    IloNumArray a_core;
    IloNumArray b_core;
    IloNumArray a_sep;
    IloNumArray b_sep;
    IloBool core_initialized{ IloFalse };

    // Controle de cortes por nó
    IloInt cuts_this_node{ 0 };
    IloInt64 last_node_index{ -1 };

  public:
    UserBendersCallbackI(
        const TSCFLInstance &inst_,
        Subproblem &subproblem_,
        const IloBoolVarArray &var_a_,
        const IloBoolVarArray &var_b_,
        const IloNumVar &var_eta_
    )
        : IloCplex::UserCutCallbackI(inst_.env),
          env(inst_.env),
          inst(inst_),
          var_a(var_a_),
          var_b(var_b_),
          var_eta(var_eta_),
          subproblem(subproblem_),
          a(env, inst_.nI),
          b(env, inst_.nJ),
          a_core(env, inst_.nI),
          b_core(env, inst_.nJ),
          a_sep(env, inst_.nI),
          b_sep(env, inst_.nJ)
    {
    }

    IloCplex::CallbackI *
    duplicateCallback() const override
    {
        return (new (getEnv()) UserBendersCallbackI(*this));
    }

    void
    main() override
    {
        // Só depois do cut loop
        if (!isAfterCutLoop())
            return;

        // Limita user cuts a nós "rasos"
        IloInt64 nodeIndex = getNnodes64();
        if (getMIPRelativeGap() <= MAX_GAP && nodeIndex > MAX_NODE_INDEX)
            return;

        // Controle de cortes por nó
        if (nodeIndex != last_node_index)
            {
                last_node_index = nodeIndex;
                cuts_this_node = 0;
            }
        if (cuts_this_node >= MAX_NODE_CUTS)
            return;

        // Lê (a, b, eta) da solução corrente
        getValues(a, var_a);
        getValues(b, var_b);
        eta = getValue(var_eta);

        // Evita cortes para soluções muito fracionárias
        IloNum frac_sum = 0.0;

        for (IloInt i = 0; i < inst.nI; ++i)
            {
                IloNum v = a[i];
                IloNum frac = IloAbs(v - IloRound(v));
                frac_sum += IloMin(frac, 1.0 - frac);
                if (frac_sum > MAX_FRAC_SUM)
                    return;
            }
        for (IloInt j = 0; j < inst.nJ; ++j)
            {
                IloNum v = b[j];
                IloNum frac = IloAbs(v - IloRound(v));
                frac_sum += IloMin(frac, 1.0 - frac);
                if (frac_sum > MAX_FRAC_SUM)
                    return;
            }

        // Atualiza core point
        if (core_initialized)
            {
                for (IloInt i = 0; i < inst.nI; ++i)
                    a_core[i] = (1.0 - OMEGA_CORE) * a_core[i] + OMEGA_CORE * a[i];
                for (IloInt j = 0; j < inst.nJ; ++j)
                    b_core[j] = (1.0 - OMEGA_CORE) * b_core[j] + OMEGA_CORE * b[j];
            }
        else
            {
                for (IloInt i = 0; i < inst.nI; ++i)
                    a_core[i] = a[i];
                for (IloInt j = 0; j < inst.nJ; ++j)
                    b_core[j] = b[j];

                core_initialized = IloTrue;
            }

        // Ponto de separação
        for (IloInt i = 0; i < inst.nI; ++i)
            a_sep[i] = OMEGA_SET * a[i] + (1.0 - OMEGA_SET) * a_core[i];
        for (IloInt j = 0; j < inst.nJ; ++j)
            b_sep[j] = OMEGA_SET * b[j] + (1.0 - OMEGA_SET) * b_core[j];

        // Resolve subproblema no ponto de separação
        subproblem.update(a_sep, b_sep);
        subproblem.solve();

        // Teste de violação avaliado na solução LP
        IloNum theta = subproblem.theta;
        IloNum viol = theta - eta;

        IloNum min_viol = IloMax(ABS_VIOL, REL_VIOL * IloMax(1.0, IloAbs(theta)));
        if (viol <= min_viol)
            return;

        // Adiciona user cut
        IloCplex::CutManagement cut_mgmt
            = (nodeIndex == 0) ? IloCplex::UseCutForce : IloCplex::UseCutPurge;

        add(var_eta >= subproblem.rhs + IloScalProd(subproblem.coef_a, var_a)
                           + IloScalProd(subproblem.coef_b, var_b),
            cut_mgmt);

        ++cuts_this_node;
    }
};
} // namespace

TSCFLSolverBenders::TSCFLSolverBenders(const TSCFLInstance &inst_, Subproblem::Mode sp_mode)
    : TSCFLSolver(inst_),
      model(env),
      cplex(env),
      subproblem(Subproblem::create(inst_, sp_mode)),
      var_a(env, inst.nI),
      var_b(env, inst.nJ),
      var_eta(env, 0.0, IloInfinity)
{
    buildModel();
    cplex.extract(model);

    // Parâmetros CPLEX (mestre)
    cplex.setParam(IloCplex::Param::Threads, 1);
    cplex.setParam(IloCplex::Param::Preprocessing::Presolve, 0);
    cplex.setParam(IloCplex::Param::Preprocessing::Aggregator, 0);
    cplex.setParam(IloCplex::Param::MIP::Strategy::Search, IloCplex::Traditional);
    cplex.setParam(IloCplex::Param::MIP::Tolerances::MIPGap, MIP_GAP);
}

TSCFLSolverBenders::~TSCFLSolverBenders()
{
    cplex.end();
    model.end();
}

void
TSCFLSolverBenders::buildModel()
{
    // Capacidade agregada (garante viabilidade do subproblema)
    double demand_sum = IloSum(inst.r);
    model.add(IloScalProd(inst.p, var_a) >= demand_sum);
    model.add(IloScalProd(inst.q, var_b) >= demand_sum);

    // OBJETIVO
    IloObjective obj
        = IloMinimize(env, IloScalProd(inst.f, var_a) + IloScalProd(inst.g, var_b) + var_eta);

    model.add(obj);
}

void
TSCFLSolverBenders::solve(bool log_output, double time_limit)
{
    // Controle de log
    if (log_output)
        {
            cplex.setOut(env.out());
            cplex.setWarning(env.out());
        }
    else
        {
            cplex.setOut(env.getNullStream());
            cplex.setWarning(env.getNullStream());
        }

    if (time_limit > 0.0)
        cplex.setParam(IloCplex::Param::TimeLimit, time_limit);

    // Callbacks Benders
    cplex.use(new (env) LazyBendersCallbackI(inst, *subproblem, var_a, var_b, var_eta));
    cplex.use(new (env) UserBendersCallbackI(inst, *subproblem, var_a, var_b, var_eta));

    // Executa o solver
    if (cplex.solve())
        {
            lb = cplex.getBestObjValue();
            ub = cplex.getObjValue();

            cplex.getValues(a, var_a);
            cplex.getValues(b, var_b);

            // Recupera os fluxos (x e y)
            updateFlows();
        }

    // Recuperação das estatísticas
    gap = cplex.getMIPRelativeGap();
    time = cplex.getTime();
    nodes = cplex.getNnodes64();
    status = cplex.getStatus();

    // Log final
    printSummary("BENDERS");
}
