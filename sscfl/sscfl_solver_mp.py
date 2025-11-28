"""
COS888

CFL com CPLEX

Gabriel Braun, 2025
"""

from docplex.mp.model import Model
from sscfl_instance import SSCFLInstance


def solve_instance_mp(inst: SSCFLInstance) -> None:
    """
    Resolve a instância SSCFL (single-source) usando CPLEX.
    """
    mdl = Model(name="SSCFL", log_output=True)

    # VARIÁVEIS
    #   a_i  = decisão: abre instalação i
    #   x_ij = decisão: instalação i -> cliente j
    a = mdl.binary_var_dict(inst.I, name="a")
    x = mdl.binary_var_dict(inst.IJ, name="x")

    # RESTRIÇÕES
    # cada cliente é atendido por exatamente uma instalação
    mdl.add_constraints_(mdl.sum(x[i, j] for i in inst.I) == 1 for j in inst.J)

    # capacidade das instalações
    mdl.add_constraints_(
        mdl.sum(inst.r[j] * x[i, j] for j in inst.J) <= inst.p[i] * a[i] for i in inst.I
    )

    # vinculação: se instalação está fechada, ninguém pode ser atendido por ela
    mdl.add_constraints_((x[i, j] <= a[i]) for i, j in inst.IJ)

    # OBJETIVO
    cost_fixed = mdl.sum(inst.f[i] * a[i] for i in inst.I)
    cost_flow = mdl.sum(inst.c[i, j] * x[i, j] for i, j in inst.IJ)

    mdl.minimize(cost_fixed + cost_flow)

    # SOLVE
    solution = mdl.solve()

    if solution:
        print(f"\nSolved.\n")
        print(f"objective  = {solution.objective_value:.2f}")
        print(f"best bound = {solution.solve_details.best_bound:.2f}")

        print(f"\n{solution.solve_details}")

    return solution


def main():
    """
    Rotina principal
    """
    instance = SSCFLInstance.load("holmberg/sscfl_h_40")

    solve_instance_mp(instance, time_limit=100)

    return


if __name__ == "__main__":
    main()
