"""
COS888

CFL com CPLEX

Gabriel Braun, 2025
"""

from collections import defaultdict
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

    f: np.ndarray  # f_i = custo fixo da planta i
    c: np.ndarray  # c_ij = custo unitário planta i -> cliente j
    p: np.ndarray  # p_i = capacidade da planta i
    r: np.ndarray  # r_j = demanda do cliente j

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
    SSCFL via CP Optimizer with global constraints + symmetry breaking.

    Variables
      Y[j] in {0..nI-1}  : facility assigned to customer j
      O[i] in {0,1}      : facility i open
      L[i] in [0..sum r] : total load at facility i   (maintained by pack)
      N[i] in [0..nJ]    : #customers at facility i   (maintained by distribute)

    Globals
      pack(L, Y, r, used=U)           -> maintain facility loads and #used bins
      distribute(N, Y, values=inst.I) -> maintain per-facility cardinalities
      allowed_assignments(Y[j], feas) -> forbid infeasible facility choices (capacity filter)

    Symmetry breaking
      For groups of *identical facilities* (same p, f, and cost row),
      enforce lexicographic order: [O, L, N] non-increasing within the group.
    """
    mdl = CpoModel(name="SSCFL_CP")

    # -------------------- Decision vars --------------------
    Y = mdl.integer_var_dict(
        inst.J, 0, inst.nI - 1, name="Y"
    )  # assignment (customer -> facility)
    O = mdl.binary_var_dict(inst.I, name="O")  # open flags

    Rtot = int(inst.r.sum())
    L = mdl.integer_var_dict(
        inst.I, 0, Rtot, name="L"
    )  # facility load (pack-maintained)
    N = mdl.integer_var_list(
        inst.nI, 0, inst.nJ, name="N"
    )  # facility card (distribute-maintained)
    U = mdl.integer_var(0, inst.nI, name="U")  # #used containers (from pack)

    # -------------------- Domain filtering on Y --------------------
    # Only allow facilities that can host the customer's demand
    for j in inst.J:
        feas_i = [i for i in inst.I if inst.p[i] >= inst.r[j]]
        if len(feas_i) < inst.nI:  # add only if it prunes
            mdl.add(
                mdl.allowed_assignments([Y[j]], mdl.tuple_set((i,) for i in feas_i))
            )

    # -------------------- Global constraints --------------------
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
        mdl.add(mdl.if_then(O[i] == 0, L[i] == 0))
        mdl.add(mdl.if_then(O[i] == 0, N[i] == 0))
        mdl.add(mdl.if_then(N[i] >= 1, O[i] == 1))

    # Redundant but strong: sum loads equals total demand
    mdl.add(mdl.sum(L[i] for i in inst.I) == Rtot)

    # Covering LB on number of opens (quick greedy cover)
    need, acc = 0, 0
    for cap in sorted(inst.p.astype(int), reverse=True):
        need += 1
        acc += cap
        if acc >= Rtot:
            break
    mdl.add(mdl.sum(O[i] for i in inst.I) >= need)

    # -------------------- Symmetry breaking (identical facilities) --------------------
    # Group facilities with same (capacity, fixed cost, full cost row)
    def _row_signature(i: int):
        # Use tuple of ints/floats from the cost row; NumPy is fine to iterate
        row = tuple(int(v) if float(v).is_integer() else float(v) for v in inst.c[i, :])
        return (int(inst.p[i]), float(inst.f[i]), row)

    groups = defaultdict(list)
    for i in inst.I:
        groups[_row_signature(i)].append(i)

    # Within each identical group, enforce non-increasing lex order on [O, L, N]
    # lexicographic(x, y) enforces x <=_lex y; to get non-increasing, order pairs reversed.
    for idxs in groups.values():
        if len(idxs) > 1:
            sidx = sorted(idxs)
            for t in range(len(sidx) - 1):
                i1, i2 = sidx[t], sidx[t + 1]
                # [O[i2], L[i2], N[i2]] <=_lex [O[i1], L[i1], N[i1]]
                mdl.add(mdl.lexicographic([O[i2], L[i2], N[i2]], [O[i1], L[i1], N[i1]]))

    # -------------------- Objective --------------------
    # Assignment cost via element(c[:,j], Y[j]) and fixed costs
    assign_cost = mdl.sum(mdl.element(inst.c[:, j], Y[j]) for j in inst.J)
    fixed_cost = mdl.sum(inst.f[i] * O[i] for i in inst.I)
    mdl.minimize(fixed_cost + assign_cost)

    # -------------------- Solve --------------------
    sol = mdl.solve(
        LogVerbosity=("Terse" if log_output else "Quiet"),
        TimeLimit=time_limit,
        Workers=workers,
    )


def main():
    """
    Rotina principal
    """
    # PATH = "instances/sscfl/holmberg/sscfl_h_40.txt"
    PATH = "instances/sscfl/yang/sscfl_y_a1.txt"

    instance = SSCFLInstance.from_txt(PATH)

    solve_instance_cp(instance, time_limit=100)

    return


if __name__ == "__main__":
    main()
