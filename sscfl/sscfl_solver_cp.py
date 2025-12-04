"""
COS888

CFL com CPLEX

Gabriel Braun, 2025
"""

import numpy as np
from docplex.cp.model import CpoModel
from sscfl_instance import SSCFLInstance


def solve_instance_cp(
    inst,
    log_output: bool = True,
    time_limit: float | None = None,
    workers: int | None = None,
):
    """
    Resolve a instância SSCFL usando CP Optimizer (ingênuo).
    """
    mdl = CpoModel(name="SSCFL_CP")

    # VARIÁVEIS
    #   a_i  = decisão: abre planta i
    #   Y_j = associação: cliente j -> planta i
    a = mdl.binary_var_list(inst.nI, name="a")
    Y = mdl.integer_var_list(inst.nJ, 0, inst.nI - 1, name="Y")

    # VARIÁVEIS AUXILIARES
    #   L_i = decisão: abre instalação i
    #   N_j = associação: instalação i -> cliente j
    #   U_j = #used containers (from pack)
    L = mdl.integer_var_list(inst.nI, 0, inst.r.sum(), name="L")
    N = mdl.integer_var_list(inst.nI, 0, inst.nJ, name="N")

    # FILTRO: Só permite plantas que suportam a demanda do cliente
    for j in inst.J:
        feas_i = [i for i in inst.I if inst.p[i] >= inst.r[j]]
        if len(feas_i) < inst.nI:  # add only if it prunes
            mdl.add(
                mdl.allowed_assignments([Y[j]], mdl.tuple_set((i,) for i in feas_i))
            )

    # GLOBAL CONSTRAINTS
    # Bin packing das demandas (r) nas plantas (pela variável Y)
    mdl.add(mdl.pack(L, Y, inst.r))
    # Conta quantos clientes cada planta recebe
    mdl.add(mdl.distribute(N, Y, values=inst.I))

    # CONSTRAINTS
    # capacidade
    mdl.add(L[i] <= inst.p[i] for i in inst.I)
    # Open <-> used: closed ⇒ L=0 & N=0 ; used ⇒ open
    mdl.add(mdl.if_then(a[i] == 0, L[i] == 0) for i in inst.I)
    mdl.add(mdl.if_then(a[i] == 0, N[i] == 0) for i in inst.I)
    mdl.add(mdl.if_then(N[i] >= 1, a[i] == 1) for i in inst.I)

    # QUEBRA DE SIMETRIA
    # Cria uma chave para ordenar os gupos idênticos
    keys = np.column_stack((inst.p, inst.f, inst.c)).astype(int, copy=False)
    _, grp = np.unique(keys, axis=0, return_inverse=True)

    # Para cada grupo idêntico, forçar ordem lexicográfica em [O, L, N]
    ng = grp.max() + 1
    for g in range(ng):
        idxs = np.where(grp == g)[0]
        if idxs.size > 1:
            idxs.sort()
            for i1, i2 in zip(idxs[:-1], idxs[1:]):
                mdl.add(mdl.lexicographic([a[i2], L[i2], N[i2]], [a[i1], L[i1], N[i1]]))

    # OBJETIVO
    cost_fixed = mdl.sum(inst.f[i] * a[i] for i in inst.I)
    cost_flow = mdl.sum(mdl.element(inst.c[:, j], Y[j]) for j in inst.J)

    mdl.minimize(cost_fixed + cost_flow)

    # SOLVE
    return mdl.solve(
        LogVerbosity=("Terse" if log_output else "Quiet"),
        TimeLimit=time_limit,
        Workers=workers,
    )


def main():
    """
    Rotina principal
    """
    instance = SSCFLInstance.load("holmberg/sscfl_h_40")

    solve_instance_cp(instance, time_limit=100)

    return


if __name__ == "__main__":
    main()
