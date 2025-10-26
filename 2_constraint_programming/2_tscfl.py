"""
COS888

TSCFL com CPLEX

Gabriel Braun, 2025
"""

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
from docplex.mp.model import Model


@dataclass(frozen=True)
class TSCFLInstance:
    """
    Instância do TSCFL
    """

    nI: int  # |I| plantas
    nJ: int  # |J| depósitos
    nK: int  # |K| clientes

    f: np.ndarray  # f_i = custo fixo da planta i
    g: np.ndarray  # g_j = custo fixo do depósito j
    c: np.ndarray  # c_ij = custo unitário planta i -> depósito j
    d: np.ndarray  # d_jk = custo unitário depósito j -> cliente k
    p: np.ndarray  # p_i = capacidade da planta i
    q: np.ndarray  # q_j = capacidade do depósito j
    r: np.ndarray  # r_k = demanda do cliente k

    @property
    def I(self) -> list[int]:
        return list(range(self.nI))

    @property
    def J(self) -> list[int]:
        return list(range(self.nJ))

    @property
    def K(self) -> list[int]:
        return list(range(self.nK))

    @property
    def IJ(self) -> list[tuple[int, int]]:
        return list(product(self.I, self.J))

    @property
    def JK(self) -> list[tuple[int, int]]:
        return list(product(self.J, self.K))

    @classmethod
    def from_txt(cls, path: str) -> "TSCFLInstance":
        """
        Retorna: Instância a partir de um arquivo de instância .txt
        """
        arr = np.fromstring(Path(path).read_text(), sep=" ", dtype=float)

        nI, nJ, nK = arr[:3].astype(int)
        data = arr[3:]

        s1 = nK
        s2 = s1 + 2 * nJ
        s3 = s2 + nI * nJ
        s4 = s3 + 2 * nI
        s5 = s4 + nJ * nK

        r = data[:s1]

        qg = data[s1:s2].reshape(nJ, 2)
        q, g = qg[:, 0], qg[:, 1]

        c = data[s2:s3].reshape(nI, nJ)

        pf = data[s3:s4].reshape(nI, 2)
        p, f = pf[:, 0], pf[:, 1]

        d = data[s4:s5].reshape(nJ, nK)

        return cls(nI=nI, nJ=nJ, nK=nK, f=f, g=g, c=c, d=d, p=p, q=q, r=r)


def solve_instance(inst: TSCFLInstance, log_output: bool = True):
    """
    Resolve a instância TSCFL usando o CPLEX.
    """
    mdl = Model(name="TSCFL", log_output=log_output)

    # variáveis
    a = mdl.binary_var_dict(inst.I, name="a")
    b = mdl.binary_var_dict(inst.J, name="b")
    x = mdl.continuous_var_dict(inst.IJ, lb=0.0, name="x")
    y = mdl.continuous_var_dict(inst.JK, lb=0.0, name="y")

    # capacidades
    mdl.add_constraints_(
        (mdl.sum(x[i, j] for j in inst.J) <= inst.p[i] * a[i]) for i in inst.I
    )
    mdl.add_constraints_(
        (mdl.sum(y[j, k] for k in inst.K) <= inst.q[j] * b[j]) for j in inst.J
    )
    # balanço nos depósitos
    mdl.add_constraints_(
        (mdl.sum(x[i, j] for i in inst.I) == mdl.sum(y[j, k] for k in inst.K))
        for j in inst.J
    )
    # demanda dos clientes
    mdl.add_constraints_(
        (mdl.sum(y[j, k] for j in inst.J) == inst.r[k]) for k in inst.K
    )
    # VUBs
    mdl.add_constraints_((x[i, j] <= inst.q[j] * b[j]) for i, j in inst.IJ)
    mdl.add_constraints_((y[j, k] <= inst.r[k] * b[j]) for j, k in inst.JK)

    # objetivo
    cost_stage1 = mdl.sum(inst.c[i, j] * x[i, j] for i, j in inst.IJ)
    cost_stage2 = mdl.sum(inst.d[j, k] * y[j, k] for j, k in inst.JK)

    cost_fixed = mdl.sum(inst.f[i] * a[i] for i in inst.I) + mdl.sum(
        inst.g[j] * b[j] for j in inst.J
    )

    mdl.minimize(cost_fixed + cost_stage1 + cost_stage2)

    sol = mdl.solve()

    return sol.objective_value


def solve_instance_cp(
    inst: TSCFLInstance,
    log_output: bool = True,
    time_limit: float | None = None,
    granularity: int = 30,
):
    """
    CP Optimizer (docplex.cp) for TSCFL with integer flows X,Y and global constraints.

    granularity = G (>=1): 1 unit of X,Y represents G original units.
      - p,q,r are snapped to nearest multiples of G (in 'block' units).
      - If after snapping sum capacities < sum demand (in blocks), we minimally
        increase capacities (plants/warehouses) by +1 block to restore feasibility.
      - Variable costs are multiplied by G so the objective stays in original units.
    """
    import sys

    from docplex.cp.model import CpoModel, CpoParameters

    assert granularity >= 1
    G = int(granularity)

    # --- helper: nearest rounding in blocks, with minimal bump to satisfy a sum lower bound
    def _nearest_blocks_with_min_bump(vals: np.ndarray, target_sum_blocks: int | None):
        """Return z = round(vals/G) as ints; if target_sum_blocks is given and sum(z) < target,
        bump the most 'rounded-down' entries by +1 until sum(z) >= target."""
        x = vals / G
        z = np.rint(x).astype(int)  # nearest (banker's). Fine for our purpose.
        if target_sum_blocks is not None:
            deficit = int(target_sum_blocks - z.sum())
            if deficit > 0:
                # errors: z - x (most negative means we rounded down the most)
                err = z.astype(float) - x
                idx = np.argsort(err)  # ascending: most negative first
                take = min(deficit, len(idx))
                z[idx[:take]] += 1
        # ensure nonnegative
        z = np.maximum(z, 0)
        return z

    # --- round demands first (nearest in blocks)
    R_blk = _nearest_blocks_with_min_bump(inst.r, target_sum_blocks=None)
    sumR_blk = int(R_blk.sum())

    # --- round capacities nearest, then bump minimally to cover total demand
    P_blk = _nearest_blocks_with_min_bump(inst.p, target_sum_blocks=sumR_blk)
    Q_blk = _nearest_blocks_with_min_bump(inst.q, target_sum_blocks=sumR_blk)

    mdl = CpoModel(name=f"TSCFL_CP_g{G}")

    params = CpoParameters()
    if time_limit and time_limit > 0:
        params.TimeLimit = float(time_limit)

    # ---------- variables ----------
    # openings
    a = {i: mdl.binary_var(name=f"a_{i}") for i in inst.I}
    b = {j: mdl.binary_var(name=f"b_{j}") for j in inst.J}

    # integer flows in blocks
    X = {
        (i, j): mdl.integer_var(0, int(P_blk[i]), name=f"X_{i}_{j}") for i, j in inst.IJ
    }
    Y = {
        (j, k): mdl.integer_var(0, int(R_blk[k]), name=f"Y_{j}_{k}") for j, k in inst.JK
    }

    # presence (used arcs)
    UposX = {(i, j): mdl.binary_var(name=f"UposX_{i}_{j}") for i, j in inst.IJ}
    VposY = {(j, k): mdl.binary_var(name=f"VposY_{j}_{k}") for j, k in inst.JK}

    # counts
    nzX = {i: mdl.integer_var(0, inst.nJ, name=f"nzX_{i}") for i in inst.I}
    nzY = {j: mdl.integer_var(0, inst.nK, name=f"nzY_{j}") for j in inst.J}

    # ---------- constraints ----------
    # demand (in blocks)
    for k in inst.K:
        mdl.add(mdl.sum(Y[j, k] for j in inst.J) == int(R_blk[k]))

    # warehouse balance
    for j in inst.J:
        mdl.add(mdl.sum(X[i, j] for i in inst.I) == mdl.sum(Y[j, k] for k in inst.K))

    # plant capacity + closing link (in blocks)
    for i in inst.I:
        mdl.add(mdl.sum(X[i, j] for j in inst.J) <= int(P_blk[i]))
        mdl.add(mdl.if_then(a[i] == 0, mdl.sum(X[i, j] for j in inst.J) == 0))

    # warehouse capacity + closing link (in blocks)
    for j in inst.J:
        mdl.add(mdl.sum(X[i, j] for i in inst.I) <= int(Q_blk[j]))
        mdl.add(mdl.if_then(b[j] == 0, mdl.sum(X[i, j] for i in inst.I) == 0))
        mdl.add(mdl.if_then(b[j] == 0, mdl.sum(Y[j, k] for k in inst.K) == 0))

    # strong global feasibility in blocks
    mdl.add(mdl.sum(float(P_blk[i]) * a[i] for i in inst.I) >= float(sumR_blk))
    mdl.add(mdl.sum(float(Q_blk[j]) * b[j] for j in inst.J) >= float(sumR_blk))

    # reification: presence ↔ flow (min block = 1)
    for i, j in inst.IJ:
        mdl.add(mdl.if_then(UposX[i, j] == 0, X[i, j] == 0))
        if int(P_blk[i]) >= 1:
            mdl.add(mdl.if_then(UposX[i, j] == 1, X[i, j] >= 1))
        mdl.add(mdl.if_then(a[i] == 0, UposX[i, j] == 0))
    for j, k in inst.JK:
        mdl.add(mdl.if_then(VposY[j, k] == 0, Y[j, k] == 0))
        if int(R_blk[k]) >= 1:
            mdl.add(mdl.if_then(VposY[j, k] == 1, Y[j, k] >= 1))
        mdl.add(mdl.if_then(b[j] == 0, VposY[j, k] == 0))

    # GCC via COUNT: each positive leg consumes ≥1 block
    for i in inst.I:
        row = [UposX[i, j] for j in inst.J]
        mdl.add(nzX[i] == mdl.count(row, 1))
        mdl.add(nzX[i] <= int(P_blk[i]))
        mdl.add(mdl.if_then(a[i] == 0, nzX[i] == 0))
    for j in inst.J:
        row = [VposY[j, k] for k in inst.K]
        mdl.add(nzY[j] == mdl.count(row, 1))
        mdl.add(nzY[j] <= int(Q_blk[j]))
        mdl.add(mdl.if_then(b[j] == 0, nzY[j] == 0))

    # ---------- objective in ORIGINAL units ----------
    # each block is G units, so variable cost = (G*c) X + (G*d) Y
    cost_fixed = mdl.sum(float(inst.f[i]) * a[i] for i in inst.I) + mdl.sum(
        float(inst.g[j]) * b[j] for j in inst.J
    )
    cost_stage1 = mdl.sum(float(G) * float(inst.c[i, j]) * X[i, j] for i, j in inst.IJ)
    cost_stage2 = mdl.sum(float(G) * float(inst.d[j, k]) * Y[j, k] for j, k in inst.JK)
    mdl.minimize(cost_fixed + cost_stage1 + cost_stage2)

    msol = mdl.solve(params=params, log_output=(sys.stdout if log_output else None))
    if not msol:
        raise RuntimeError("CP Optimizer não encontrou solução.")

    return float(msol.get_objective_value())


def main():
    """
    Rotina principal
    """
    PATH = "instances/tscfl/tscfl_11_50.txt"

    instance = TSCFLInstance.from_txt(PATH)

    print(f"nI={instance.nI}, nJ={instance.nJ}, nK={instance.nK}")
    print(
        f"Σp={instance.p.sum():.0f}, Σq={instance.q.sum():.0f}, Σr={instance.r.sum():.0f}"
    )

    obj = solve_instance_cp(instance)

    print(f"objective: {obj}")

    return


if __name__ == "__main__":
    main()
