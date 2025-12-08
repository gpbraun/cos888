/*
COS888

tscfl_subproblem.hpp

Gabriel Braun, 2025
*/

#pragma once

#include <memory>

#include "tscfl_instance.hpp"
#include "tscfl_utils.hpp"

// SOLVER DO SUBPROBLEMA
class Subproblem
{
  protected:
    const TSCFLInstance &inst;

  public:
    enum class Mode
    {
        DUAL,
        PRIMAL,
        NET
    };

    IloNumArray a; // a[i]
    IloNumArray b; // b[j]

    // Solução do subproblema
    IloNum opt{ 0.0 }; // valor ótimo do problema original

    // Coeficientes do corte de Benders
    IloNum theta{ 0.0 }; // valor ótimo do subproblema
    IloNum rhs{ 0.0 };   // termo independente do corte
    IloNumArray coef_a;  // coeficientes multiplicando a_i
    IloNumArray coef_b;  // coeficientes multiplicando b_j

    explicit Subproblem(const TSCFLInstance &inst_);

    virtual ~Subproblem() = default;

    // Factory: cria implementação concreta a partir do modo.
    static std::unique_ptr<Subproblem> create(const TSCFLInstance &inst, Mode mode);

    // Atualiza: a e b.
    void update(const IloNumArray &a_, const IloNumArray &b_, IloBool convert_to_primal = false);

    // Atualiza: a e b a partir das variáveis do modelo.
    void update(
        const IloCplex &cpx,
        const IloNumVarArray &var_a,
        const IloNumVarArray &var_b,
        IloBool convert_to_primal = false
    );

    // Resolve o subproblema.
    // Atualiza: x, y, opt, theta, rhs, coef_a, coef_b
    virtual void solve() = 0;

  private:
    // Atualiza: a e b para respeitarem o problema original usando uma heurística.
    void convertToPrimal();
};
