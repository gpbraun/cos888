"""
COS888

CFL com CPLEX

Gabriel Braun, 2025
"""

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
from docplex.cp.model import CpoModel


@dataclass(frozen=True)
class SSCFLInstance:
    """
    Instância do SSCFL
    """

    nI: int  # |I| plantas
    nJ: int  # |J| clientes

    f: np.ndarray  # f_i  = custo fixo da planta i
    c: np.ndarray  # c_ij = custo unitário planta i -> cliente j
    p: np.ndarray  # p_i  = capacidade da planta i
    r: np.ndarray  # r_j  = demanda do cliente j

    @property
    def I(self) -> list[int]:
        return list(range(self.nI))

    @property
    def J(self) -> list[int]:
        return list(range(self.nJ))

    @property
    def IJ(self) -> list[tuple[int]]:
        return list(product(self.I, self.J))

    @classmethod
    def from_txt(cls, path: str) -> "SSCFLInstance":
        """
        Retorna: Instância SSCFL a partir de um arquivo .txt.
        """
        arr = np.fromstring(Path(path).read_text(), sep=" ", dtype=float).astype(int)

        nI, nJ = arr[:2]
        data = arr[2:]

        s1 = 2 * nI
        s2 = s1 + nJ
        s3 = s2 + nI * nJ

        pf = data[:s1].reshape(nI, 2)
        p = pf[:, 0]
        f = pf[:, 1]

        r = data[s1:s2]
        c = data[s2:s3].reshape(nI, nJ)

        return cls(nI=nI, nJ=nJ, f=f, c=c, p=p, r=r)


def solve_instance_cp(
    inst,
    log_output: bool = True,
    time_limit: float | None = None,
    workers: int | None = None,
):
    """
    Resolve a instância SSCFL (single-source) usando CP Optimizer (ingênuo).
    """
    mdl = CpoModel(name="SSCFL_CP")

    # VARIÁVEIS
    #   a_i  = decisão: abre planta i
    #   Y_j = associação: cliente j -> planta i
    a = mdl.binary_var_dict(inst.I, name="a")
    Y = mdl.integer_var_dict(inst.J, 0, inst.nI - 1, name="Y")

    # auxiliares:
    #   L_i = decisão: abre instalação i
    #   N_j = associação: instalação i -> cliente j
    #   U_j = #used containers (from pack)
    L = mdl.integer_var_dict(inst.I, 0, inst.r.sum(), name="L")
    N = mdl.integer_var_list(inst.nI, 0, inst.nJ, name="N")
    U = mdl.integer_var(0, inst.nI, name="U")

    # -------------------- Domain filtering on Y --------------------
    # Only allow facilities that can host the customer's demand
    for j in inst.J:
        feas_i = [i for i in inst.I if inst.p[i] >= inst.r[j]]
        if len(feas_i) < inst.nI:  # add only if it prunes
            mdl.add(
                mdl.allowed_assignments([Y[j]], mdl.tuple_set((i,) for i in feas_i))
            )

    # GLOBAL CONSTRAINTS
    # Bin packing of customer demands r into facilities via Y
    mdl.add(mdl.pack([L[i] for i in inst.I], [Y[j] for j in inst.J], inst.r, used=U))

    # Count how many customers each facility receives
    mdl.add(mdl.distribute(N, [Y[j] for j in inst.J], values=inst.I))

    # -------------------- Capacity & open linking --------------------
    # Capacity
    for i in inst.I:
        mdl.add(L[i] <= int(inst.p[i]))

    # Open <-> used: closed ⇒ L=0 & N=0 ; used ⇒ open
    for i in inst.I:
        mdl.add(mdl.if_then(a[i] == 0, L[i] == 0))
        mdl.add(mdl.if_then(a[i] == 0, N[i] == 0))
        mdl.add(mdl.if_then(N[i] >= 1, a[i] == 1))

    # Covering LB on number of opens (quick greedy cover)
    need, acc = 0, 0
    for cap in sorted(inst.p.astype(int), reverse=True):
        need += 1
        acc += cap
        if acc >= inst.r.sum():
            break
    mdl.add(mdl.sum(a[i] for i in inst.I) >= need)

    # quebra de simetria
    # Build a 2D key: [p | f | c-row], then group identical rows
    keys = np.column_stack((inst.p, inst.f, inst.c)).astype(int, copy=False)
    _, grp = np.unique(keys, axis=0, return_inverse=True)

    # Within each identical group, enforce non-increasing lex order on [O, L, N]
    ng = int(grp.max()) + 1
    for g in range(ng):
        idxs = np.where(grp == g)[0]
        if idxs.size > 1:
            idxs.sort()
            for i1, i2 in zip(idxs[:-1], idxs[1:]):
                # [O[b], L[b], N[b]] <=_lex [O[a], L[a], N[a]]
                mdl.add(mdl.lexicographic([a[i2], L[i2], N[i2]], [a[i1], L[i1], N[i1]]))

    # objetivo: custo fixo + custo de atribuição
    cost_fixed = mdl.sum(inst.f[i] * a[i] for i in inst.I)
    cost_stage = mdl.sum(mdl.element(inst.c[:, j], Y[j]) for j in inst.J)

    mdl.minimize(cost_fixed + cost_stage)

    # solve
    mdl.solve(
        LogVerbosity=("Terse" if log_output else "Quiet"),
        TimeLimit=time_limit,
        Workers=workers,
    )


def main():
    """
    Rotina principal
    """
    PATH = "instances/sscfl/holmberg/sscfl_h_40.txt"

    instance = SSCFLInstance.from_txt(PATH)

    solve_instance_cp(instance, time_limit=100)

    return


if __name__ == "__main__":
    main()
