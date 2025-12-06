// src/tscfl_solver_benders.cpp
#include "tscfl_solver_benders.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>

// =====================================================================
//  CALLBACKS BENDERS (apenas neste arquivo)
// =====================================================================
namespace {

// ---------------------------------------------------------------------
//  Lazy constraints: cortes clássicos de Benders (ótimos)
// ---------------------------------------------------------------------
class LazyBendersCallbackI : public IloCplex::LazyConstraintCallbackI {
   protected:
    IloEnv& env;
    const TSCFLInstance& inst;

   private:
    Subproblem& subproblem;

    IloBoolVarArray& var_a;
    IloBoolVarArray& var_b;
    IloNumVar& var_eta;

    IloNumArray a;
    IloNumArray b;
    IloNum eta{0.0};

   public:
    LazyBendersCallbackI(const TSCFLInstance& inst_, Subproblem& subproblem_,
                         IloBoolVarArray& var_a_, IloBoolVarArray& var_b_,
                         IloNumVar& var_eta_)
        : IloCplex::LazyConstraintCallbackI(inst_.env),
          env(inst_.env),
          inst(inst_),
          subproblem(subproblem_),
          var_a(var_a_),
          var_b(var_b_),
          var_eta(var_eta_),
          a(env, inst_.nI),
          b(env, inst_.nJ) {}

    IloCplex::CallbackI* duplicateCallback() const override {
        return (new (getEnv()) LazyBendersCallbackI(*this));
    }

    void main() override {
        // 1) Lê (a, b, eta) da solução corrente
        getValues(a, var_a);
        getValues(b, var_b);
        eta = getValue(var_eta);

        // 2) Resolve subproblema
        subproblem.solve(a, b);

        // 3) Testa violação
        double viol = subproblem.theta - eta;
        if (viol <= EPS) return;

        // 4) Adiciona lazy cut
        add(var_eta >= subproblem.rhs + IloScalProd(subproblem.coef_a, var_a) +
                           IloScalProd(subproblem.coef_b, var_b));
    }
};

// ---------------------------------------------------------------------
//  User cuts: Magnanti-Wong + heurísticas
// ---------------------------------------------------------------------
class UserBendersCallbackI : public IloCplex::UserCutCallbackI {
   public:
    // Parâmetros do callback
    static constexpr IloNum MAX_GAP = 1e-4;
    static constexpr IloNum ABS_VIOL = 1e-1;
    static constexpr IloNum REL_VIOL = 1e-4;
    static constexpr IloNum MAX_FRAC_SUM = 10.0;
    static constexpr IloInt MAX_CUTS_PER_NODE = 1;
    static constexpr IloInt MAX_NODE_INDEX_USER_CUTS = 10;
    static constexpr IloNum COREPOINT_LAMBDA = 0.5;
    static constexpr IloNum SEPPOINT_LAMBDA = 0.5;

   protected:
    IloEnv& env;
    const TSCFLInstance& inst;

   private:
    IloBoolVarArray& var_a;
    IloBoolVarArray& var_b;
    IloNumVar& var_eta;
    Subproblem& subproblem;

    IloNumArray a;
    IloNumArray b;
    IloNum eta{0.0};

    // Controle de cortes por nó
    IloInt64 lastNodeIndex;
    int cutsThisNode;

    // Core point e ponto de separação
    IloNumArray a_core;
    IloNumArray b_core;
    IloNumArray a_sep;
    IloNumArray b_sep;
    IloBool core_initialized{IloFalse};

   public:
    UserBendersCallbackI(const TSCFLInstance& inst_, Subproblem& subproblem_,
                         IloBoolVarArray& var_a_, IloBoolVarArray& var_b_,
                         IloNumVar& var_eta_)
        : IloCplex::UserCutCallbackI(inst_.env),
          env(inst_.env),
          inst(inst_),
          var_a(var_a_),
          var_b(var_b_),
          var_eta(var_eta_),
          subproblem(subproblem_),
          a(env, inst_.nI),
          b(env, inst_.nJ),
          lastNodeIndex(-1),
          cutsThisNode(0),
          a_core(env, inst_.nI),
          b_core(env, inst_.nJ),
          a_sep(env, inst_.nI),
          b_sep(env, inst_.nJ) {}

    IloCplex::CallbackI* duplicateCallback() const override {
        return (new (getEnv()) UserBendersCallbackI(*this));
    }

    void main() override {
        // Só depois do cut loop
        if (!isAfterCutLoop()) return;

        // Limita user cuts a nós "rasos"
        IloInt64 nodeIndex = getNnodes64();
        if (getMIPRelativeGap() <= MAX_GAP && nodeIndex > MAX_NODE_INDEX_USER_CUTS)
            return;

        // Controle de cortes por nó
        if (nodeIndex != lastNodeIndex) {
            lastNodeIndex = nodeIndex;
            cutsThisNode = 0;
        }
        if (cutsThisNode >= MAX_CUTS_PER_NODE) return;

        // Lê (a, b, eta) da solução corrente
        getValues(a, var_a);
        getValues(b, var_b);
        eta = getValue(var_eta);

        // Evita cortes para soluções muito fracionárias
        double frac_sum = 0.0;

        for (int i = 0; i < inst.nI; ++i) {
            double v = a[i];
            double frac = IloAbs(v - std::round(v));
            frac_sum += IloMin(frac, 1.0 - frac);
            if (frac_sum > MAX_FRAC_SUM) return;
        }
        for (int j = 0; j < inst.nJ; ++j) {
            double v = b[j];
            double frac = IloAbs(v - std::round(v));
            frac_sum += IloMin(frac, 1.0 - frac);
            if (frac_sum > MAX_FRAC_SUM) return;
        }

        // Atualiza core point
        if (core_initialized) {
            for (int i = 0; i < inst.nI; ++i)
                a_core[i] =
                    (1.0 - COREPOINT_LAMBDA) * a_core[i] + COREPOINT_LAMBDA * a[i];
            for (int j = 0; j < inst.nJ; ++j)
                b_core[j] =
                    (1.0 - COREPOINT_LAMBDA) * b_core[j] + COREPOINT_LAMBDA * b[j];
        } else {
            for (int i = 0; i < inst.nI; ++i) a_core[i] = a[i];
            for (int j = 0; j < inst.nJ; ++j) b_core[j] = b[j];
            core_initialized = IloTrue;
        }

        // Ponto de separação
        for (int i = 0; i < inst.nI; ++i)
            a_sep[i] = SEPPOINT_LAMBDA * a[i] + (1.0 - SEPPOINT_LAMBDA) * a_core[i];
        for (int j = 0; j < inst.nJ; ++j)
            b_sep[j] = SEPPOINT_LAMBDA * b[j] + (1.0 - SEPPOINT_LAMBDA) * b_core[j];

        // Resolve subproblema no ponto de separação
        subproblem.solve(a_sep, b_sep);

        // Teste de violação avaliado na solução LP
        double theta = subproblem.theta;
        double viol = theta - eta;

        double min_viol = IloMax(ABS_VIOL, REL_VIOL * IloMax(1.0, IloAbs(theta)));
        if (viol <= min_viol) return;

        // Adiciona user cut
        add(var_eta >= subproblem.rhs + IloScalProd(subproblem.coef_a, var_a) +
                           IloScalProd(subproblem.coef_b, var_b),
            nodeIndex == 0 ? IloCplex::UseCutForce : IloCplex::UseCutPurge);

        ++cutsThisNode;
    }
};

}  // namespace

// =====================================================================
//  SOLVER BENDERS
// =====================================================================

TSCFLSolverBenders::TSCFLSolverBenders(const TSCFLInstance& inst_, Subproblem::Mode smode)
    : TSCFLSolver(inst_),
      model(env),
      cplex(env),
      subproblem(Subproblem::create(inst_, smode)),
      var_a(env, inst.nI),
      var_b(env, inst.nJ),
      var_eta(env, 0.0, IloInfinity) {
    build_model();
    cplex.extract(model);

    // Parâmetros CPLEX (mestre)
    cplex.setParam(IloCplex::Param::Threads, 1);
    cplex.setParam(IloCplex::Param::Preprocessing::Presolve, 0);
    cplex.setParam(IloCplex::Param::Preprocessing::Aggregator, 0);
    cplex.setParam(IloCplex::Param::MIP::Strategy::Search, IloCplex::Traditional);
    cplex.setParam(IloCplex::Param::MIP::Tolerances::MIPGap, MIP_GAP);
}

TSCFLSolverBenders::~TSCFLSolverBenders() {
    cplex.end();
    model.end();
}

void TSCFLSolverBenders::build_model() {
    // Capacidade agregada (garante viabilidade do subproblema)
    double demand_sum = IloSum(inst.r);
    model.add(IloScalProd(inst.p, var_a) >= demand_sum);
    model.add(IloScalProd(inst.q, var_b) >= demand_sum);

    // OBJETIVO: custo fixo + eta
    IloObjective obj = IloMinimize(
        env, IloScalProd(inst.f, var_a) + IloScalProd(inst.g, var_b) + var_eta);
    model.add(obj);
}

bool TSCFLSolverBenders::solve(bool log_output, double time_limit) {
    // Controle de log
    if (log_output) {
        cplex.setOut(env.out());
        cplex.setWarning(env.out());
    } else {
        cplex.setOut(env.getNullStream());
        cplex.setWarning(env.getNullStream());
    }

    if (time_limit > 0.0) cplex.setParam(IloCplex::Param::TimeLimit, time_limit);

    // Callbacks Benders
    cplex.use(new (env) LazyBendersCallbackI(inst, *subproblem, var_a, var_b, var_eta));
    cplex.use(new (env) UserBendersCallbackI(inst, *subproblem, var_a, var_b, var_eta));

    // Solve
    IloBool ok = cplex.solve();
    status = cplex.getStatus();

    // Estatísticas
    gap = cplex.getMIPRelativeGap();
    nodes = cplex.getNnodes64();
    time = cplex.getTime();

    if (ok) {
        ub = cplex.getObjValue();
        lb = cplex.getBestObjValue();

        print_summary("BENDERS");
        // valor de eta*
        std::cout << std::fixed << std::setprecision(0)
                  << "eta*   = " << cplex.getValue(var_eta) << "\n"
                  << std::defaultfloat;
    } else {
        std::cerr << "\n[BENDERS] Sem solução. status = " << status << "\n";
        std::cerr << "nodes = " << nodes << "\n";
        std::cerr << "time  = " << time << " s\n";
    }

    return ok;
}
