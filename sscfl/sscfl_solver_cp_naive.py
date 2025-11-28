"""
COS888

CFL com CPLEX

Gabriel Braun, 2025
"""

from docplex.cp.model import CpoModel
from sscfl_instance import SSCFLInstance


def solve_instance_cp_naive(
    inst,
    log_output: bool = True,
    time_limit: float | None = None,
    workers: int | None = None,
):
    """
    Resolve a instância SSCFL (single-source) usando CP Optimizer (ingênuo).
    """
    mdl = CpoModel(name="SSCFL_CP_Naive")

    # VARIÁVEIS
    # a_i  = decisão: abre planta i
    # x_ij = decisão: planta i -> cliente j
    a = mdl.binary_var_dict(inst.I, name="a")
    x = mdl.binary_var_dict(inst.IJ, name="x")

    # RESTRIÇÕES
    # cada cliente é atendido por exatamente uma instalação
    mdl.add(mdl.sum(x[i, j] for i in inst.I) == 1 for j in inst.J)

    # capacidade das instalações
    mdl.add(
        mdl.sum(inst.r[j] * x[i, j] for j in inst.J) <= inst.p[i] * a[i] for i in inst.I
    )

    # vinculação: se instalação está fechada, ninguém pode ser atendido por ela
    mdl.add(x[i, j] <= a[i] for i, j in inst.IJ)

    # OBJETIVO
    cost_fixed = mdl.sum(inst.f[i] * a[i] for i in inst.I)
    cost_flow = mdl.sum(inst.c[i, j] * inst.r[j] * x[i, j] for (i, j) in inst.IJ)

    mdl.minimize(cost_fixed + cost_flow)

    # SOLVE
    mdl.solve(
        LogVerbosity=("Terse" if log_output else "Quiet"),
        TimeLimit=time_limit,
        Workers=workers,
    )


def main():
    """
    Rotina principal
    """
    instance = SSCFLInstance.load("holmberg/sscfl_h_40")

    solve_instance_cp_naive(instance, time_limit=100)

    return


if __name__ == "__main__":
    main()
