"""
COS888

CFL com CPLEX

Gabriel Braun, 2025
"""

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
from docplex.mp.model import Model


@dataclass(frozen=True)
class CFLInstance:
    """
    Instância do CFL
    """

    m: int
    n: int
    f_cap: np.ndarray  # (m,)
    f_cst: np.ndarray  # (m,)
    c_dem: np.ndarray  # (n,)
    c_cst: np.ndarray  # (m, n)

    @classmethod
    def from_orlib(cls, path: str) -> "CFLInstance":
        """
        Retorna: Instância a partir de um arquivo de instância .txt
        """
        arr = np.fromstring(Path(path).read_text(), sep=" ", dtype=float)

        m = int(arr[0])
        n = int(arr[1])

        data = arr[2:]

        facility_data = data[: 2 * m].reshape(m, 2)
        f_cap = facility_data[:, 0]
        f_cst = facility_data[:, 1]

        customer_data = data[2 * m :].reshape(n, 1 + m)
        c_dem = customer_data[:, 0]
        c_cst = customer_data[:, 1:].T / c_dem[None, :]

        return cls(m=m, n=n, f_cap=f_cap, f_cst=f_cst, c_dem=c_dem, c_cst=c_cst)


def solve_instance(inst: CFLInstance, log_output: bool = True):
    """
    Resolve a instância usando o CPLEX.
    """
    F = range(inst.m)
    C = range(inst.n)

    mdl = Model(name="CFL", log_output=log_output)

    # variáveis
    x = mdl.binary_var_list(inst.m, name="x")
    y = mdl.continuous_var_dict(((i, j) for i in C for j in F), lb=0.0, name="y")

    # restrições
    mdl.add_constraints_(
        (mdl.sum(y[i, j] for i in C) <= inst.f_cap[j] * x[j]) for j in F
    )
    mdl.add_constraints_((mdl.sum(y[i, j] for j in F) == inst.c_dem[i]) for i in C)

    mdl.add_constraints_(
        (y[i, j] <= min(inst.c_dem[i], inst.f_cap[j]) * x[j]) for i, j in product(C, F)
    )

    # objetivo
    total_c_cst = mdl.sum(inst.c_cst[j, i] * y[i, j] for i in C for j in F)
    total_f_cst = mdl.sum(inst.f_cst[j] * x[j] for j in F)

    mdl.minimize(total_c_cst + total_f_cst)

    # configurações
    mdl.parameters.threads = 1

    sol = mdl.solve()

    return sol.objective_value


def solve_instance_relaxcut(
    inst,
    *,
    max_iter: int = 300,
    threads: int = 1,
    gamma: float = 1.0,  # Polyak factor
    mu0: float = 10.0,  # fallback step before UB exists
    tol_stop: float = 1e-6,
    ndrc_keep: int = 5,
    seed: int = 0,
    log: bool = False,
) -> float:
    """
    Relax-and-Cut for CFL (amounts model). Returns only the best UB (objective value).
    If no feasible UB is obtained, returns float('inf').
    """
    rng = np.random.default_rng(seed)

    m, n = inst.m, inst.n
    d = inst.f_cap  # (m,)
    f = inst.f_cst  # (m,)
    a = inst.c_dem  # (n,)
    c = inst.c_cst  # (m,n) per-unit

    # multipliers for demand equalities (free)
    u = np.zeros(n, dtype=float)

    # active VUB cuts: for each j, store customers i with y[i,j] <= a[i] x[j]
    active_vub = [dict() for _ in range(m)]

    best_lb = -np.inf
    best_ub = np.inf
    best_x_open = np.zeros(m, dtype=int)

    def lagrangian_subproblem(u_vec: np.ndarray):
        r = c - u_vec[None, :]  # reduced costs (m,n)
        lb = float(np.dot(u_vec, a))  # constant term
        x_open = np.zeros(m, dtype=int)
        y = np.zeros((n, m), dtype=float)

        for j in range(m):
            order = np.argsort(r[j, :])
            if r[j, order[0]] >= 0.0:
                continue
            cap_left = d[j]
            accum = 0.0
            for i in order:
                if cap_left <= 1e-12:
                    break
                rc = r[j, i]
                if rc >= 0.0:
                    break
                y_cap = a[i] if (i in active_vub[j]) else np.inf
                take = min(cap_left, y_cap)
                if take > 0.0:
                    y[i, j] = take
                    cap_left -= take
                    accum += rc * take
            if f[j] + accum < 0.0:
                x_open[j] = 1
                lb += f[j] + accum
        return lb, x_open, y

    def separate_vub_violations(x_open, y) -> int:
        newcuts = 0
        for j in range(m):
            if x_open[j] == 0:
                continue
            over = np.where(y[:, j] > a + 1e-9)[0]
            for i in over:
                if i not in active_vub[j]:
                    active_vub[j][i] = 0
                    newcuts += 1
        return newcuts

    def age_and_prune_cuts():
        for j in range(m):
            dead = []
            for i, age in active_vub[j].items():
                age += 1
                if age > ndrc_keep:
                    dead.append(i)
                else:
                    active_vub[j][i] = age
            for i in dead:
                del active_vub[j][i]

    def transportation_repair(x_open):
        if float(np.dot(d, x_open)) + 1e-9 < float(np.sum(a)):
            return None
        mdl = Model(name="CFL_repair", log_output=False)
        mdl.parameters.threads = threads
        I = range(n)
        J = [j for j in range(m) if x_open[j] == 1]
        yvar = {(i, j): mdl.continuous_var(lb=0.0) for j in J for i in I}
        mdl.add_constraints_(mdl.sum(yvar[i, j] for j in J) == a[i] for i in I)
        mdl.add_constraints_(mdl.sum(yvar[i, j] for i in I) <= d[j] for j in J)
        mdl.minimize(
            mdl.sum(c[j, i] * yvar[i, j] for j in J for i in I)
            + mdl.sum(f[j] for j in J)
        )
        sol = mdl.solve()
        if not sol:
            return None
        return float(sol.objective_value)

    for k in range(1, max_iter + 1):
        lb, x_open, y_lag = lagrangian_subproblem(u)
        if lb > best_lb:
            best_lb = lb
        newcuts = separate_vub_violations(x_open, y_lag)
        g = a - np.sum(y_lag, axis=1)  # subgradient
        ng2 = float(np.dot(g, g))

        # try to build UB occasionally / when improving
        if (k == 1) or (k % 5 == 0):
            rep = transportation_repair(x_open)
            if rep is not None and rep + 1e-8 < best_ub:
                best_ub = rep
                best_x_open = x_open.copy()

        # Polyak step (if UB known), else fallback
        if ng2 > 0:
            if np.isfinite(best_ub):
                step = gamma * max(best_ub - lb, 0.0) / ng2
                if step < 1e-12:
                    step = 1e-12
            else:
                step = mu0 / np.sqrt(k)
            u = u + step * g

        if newcuts == 0:
            age_and_prune_cuts()

        if np.isfinite(best_ub) and (
            best_ub - best_lb <= tol_stop * max(1.0, abs(best_ub))
        ):
            break

    if not np.isfinite(best_ub):
        # last-chance repair with the last x_open (or zeros if never improved)
        rep = transportation_repair(best_x_open if best_x_open.sum() else x_open)
        if rep is not None:
            best_ub = rep

    return float(best_ub) if np.isfinite(best_ub) else float("inf")


def main():
    """
    Rotina principal
    """
    PATH = "/home/braun/Documents/Developer/cos888/instances/cfl/cap41.txt"

    instance = CFLInstance.from_orlib(PATH)
    print(f"m={instance.m}, n={instance.n}")

    obj_true = solve_instance(instance)

    obj_rac = solve_instance_relaxcut(instance)

    print(f"true objective: {obj_true}")
    print(f"relax and cut: {obj_rac}")

    print(f"{100*abs(obj_true - obj_rac) / obj_true}%")

    return


if __name__ == "__main__":
    main()
