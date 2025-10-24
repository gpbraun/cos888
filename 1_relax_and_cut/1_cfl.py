"""
COS888

CFL com CPLEX

Gabriel Braun, 2025
"""

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Optional

import numpy as np
from docplex.mp.model import Model


@dataclass(frozen=True)
class CFLInstance:
    """
    Instância do CFL
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
    def from_txt(cls, path: str) -> "CFLInstance":
        """
        Retorna: Instância a partir de um arquivo de instância .txt
        """
        arr = np.fromstring(Path(path).read_text(), sep=" ", dtype=float)

        nI, nJ = arr[:2].astype(int)
        data = arr[2:]

        facility_data = data[: 2 * nI].reshape(nI, 2)
        p = facility_data[:, 0]
        f = facility_data[:, 1]

        customer_data = data[2 * nI :].reshape(nJ, 1 + nI)
        r = customer_data[:, 0]
        c = customer_data[:, 1:].T / r[None, :]

        return cls(nI=nI, nJ=nJ, f=f, p=p, r=r, c=c)


def solve_instance(inst: CFLInstance, log_output: bool = True):
    """
    Resolve a instância usando o CPLEX.
    """
    mdl = Model(name="CFL", log_output=log_output)

    # variáveis
    a = mdl.binary_var_dict(inst.I, name="a")
    x = mdl.continuous_var_dict(inst.IJ, lb=0.0, name="x")

    # restrições
    mdl.add_constraints_(
        (mdl.sum(x[i, j] for j in inst.J) <= inst.p[i] * a[i]) for i in inst.I
    )
    mdl.add_constraints_(
        (mdl.sum(x[i, j] for i in inst.I) == inst.r[j]) for j in inst.J
    )
    mdl.add_constraints_(
        (x[i, j] <= min(inst.p[i], inst.r[j]) * a[i]) for i, j in inst.IJ
    )

    # objetivo
    cost_fixed = mdl.sum(inst.f[i] * a[i] for i in inst.I)
    cost_transport = mdl.sum(inst.c[i, j] * x[i, j] for i, j in inst.IJ)

    mdl.minimize(cost_fixed + cost_transport)

    sol = mdl.solve()

    return sol.objective_value


# ============================================================================
# Relax-and-Cut (Non-Delayed, slide-faithful ε-schedule)
# ============================================================================
def solve_instance_rc(
    inst: "CFLInstance",
    *,
    max_iter: int = 10000,
    threads: int = 1,
    # Polyak (used when UB is finite)
    gamma: float = 1.0,
    # Fallback ε-schedule (used only while UB is not known)
    eps0: float = 1.0,
    stall_halve: int = 200,
    eps_min: float = 1e-12,
    # CA/PA/CI & cuts
    viol_tol: float = 1e-8,
    dual_keep: int = 10,
    ndrc_keep: int = 5,
    # stopping / logs
    tol_stop: float = 1e-6,
    log: bool = False,
) -> float:
    """
    Non-Delayed Relax-and-Cut (NDRC) close to the slides:
      - Dualize demand equalities with free multipliers u[j]
      - Explicit CA/PA/CI; set g_j=0 for j in CI (non-delayed)
      - EXTRA aging on PA\CA (drop multipliers after 'dual_keep')
      - Step: Polyak with UB if available; else ε-schedule fallback
      - VUB cuts separated on demand (aged non-delayed)
      - Repair only to produce UB (not in step length when ε-fallback is active)
    Returns: best feasible UB (float) or +inf if none found.
    """
    import numpy as np
    from docplex.mp.model import Model

    TOL = 1e-12

    # ---------- dual state ----------
    u = np.zeros(inst.nJ, dtype=float)
    dual_age = np.zeros(inst.nJ, dtype=int)

    # ---------- cut state (VUB) ----------
    active_vub: list[dict[int, int]] = [dict() for _ in range(inst.nI)]

    # ---------- bounds ----------
    best_lb = -np.inf
    best_ub = np.inf
    best_a_open = np.zeros(inst.nI, dtype=int)

    # ε-schedule bookkeeping (used only while UB is inf)
    eps = float(eps0)
    since_improve = 0

    # ---------------- helpers ----------------
    def lrp_solve(u_vec: np.ndarray):
        """
        Greedy LRP at u: reduced costs c̃[i,j] = c[i,j] - u[j].
        """
        ctilde = inst.c - u_vec[None, :]
        lb_k = float(np.dot(u_vec, inst.r))
        a_rel = np.zeros(inst.nI, dtype=int)
        x_rel = np.zeros((inst.nI, inst.nJ), dtype=float)

        for i in inst.I:
            neg = np.where(ctilde[i, :] < -TOL)[0]
            if neg.size == 0:
                continue

            order = neg[np.argsort(ctilde[i, neg])]
            cap_left = float(inst.p[i])
            var_part = 0.0

            for j in order:
                if cap_left <= TOL:
                    break
                rc = ctilde[i, j]
                if rc >= 0.0:
                    break
                # VUB only when activated (non-delayed strengthening)
                per_pair_cap = inst.r[j] if (j in active_vub[i]) else np.inf
                take = min(cap_left, per_pair_cap)
                if take > TOL:
                    x_rel[i, j] = take
                    cap_left -= take
                    var_part += rc * take

            if inst.f[i] + var_part < 0.0:
                a_rel[i] = 1
                lb_k += inst.f[i] + var_part
            else:
                if np.any(x_rel[i, :] > 0):
                    x_rel[i, :] = 0.0

        g = inst.r - np.sum(x_rel, axis=0)
        return lb_k, a_rel, x_rel, g

    def separate_vub(a_rel: np.ndarray, x_rel: np.ndarray) -> int:
        newcuts = 0
        for i in inst.I:
            if a_rel[i] == 0:
                continue
            viol = np.where(x_rel[i, :] > inst.r + 1e-9)[0]
            for j in viol:
                if j not in active_vub[i]:
                    active_vub[i][j] = 0
                    newcuts += 1
        return newcuts

    def age_and_prune_vub():
        for i in inst.I:
            if not active_vub[i]:
                continue
            drop = []
            for j, age in active_vub[i].items():
                age += 1
                if age > ndrc_keep:
                    drop.append(j)
                else:
                    active_vub[i][j] = age
            for j in drop:
                del active_vub[i][j]

    def repair_ub(a_open: np.ndarray) -> Optional[float]:
        """
        Feasibility-completion + transportation LP on open set → UB.
        """
        total_r = float(np.sum(inst.r))
        a_fix = a_open.astype(int).copy()
        cap = float(np.dot(inst.p, a_fix))

        if cap + 1e-9 < total_r:
            closed = [i for i in inst.I if a_fix[i] == 0 and inst.p[i] > 0]
            closed.sort(key=lambda i: inst.f[i] / max(inst.p[i], 1e-12))
            for i in closed:
                a_fix[i] = 1
                cap += float(inst.p[i])
                if cap + 1e-9 >= total_r:
                    break
            if cap + 1e-9 < total_r:
                a_fix[:] = 1  # last resort

        mdl = Model(name="CFL_repair", log_output=False)
        mdl.parameters.threads = threads

        I_open = [i for i in inst.I if a_fix[i] == 1]
        if not I_open:
            return None

        xvar = {
            (i, j): mdl.continuous_var(lb=0.0, name=f"x_{i}_{j}")
            for i in I_open
            for j in inst.J
        }

        mdl.add_constraints_(
            mdl.sum(xvar[i, j] for i in I_open) == inst.r[j] for j in inst.J
        )
        mdl.add_constraints_(
            mdl.sum(xvar[i, j] for j in inst.J) <= inst.p[i] for i in I_open
        )

        mdl.minimize(
            mdl.sum(inst.c[i, j] * xvar[i, j] for i in I_open for j in inst.J)
            + mdl.sum(inst.f[i] for i in I_open)
        )
        sol = mdl.solve()
        if not sol:
            return None
        return float(sol.objective_value)

    # ---------------- main loop ----------------
    for k in range(1, max_iter + 1):
        lb_k, a_rel_k, x_rel_k, g_k = lrp_solve(u)

        # Best LB / stall logic
        improved = lb_k > best_lb + 1e-12
        if improved:
            best_lb = lb_k
            since_improve = 0
            ub_try = repair_ub(a_rel_k)
            if ub_try is not None and ub_try + 1e-8 < best_ub:
                best_ub = ub_try
                best_a_open = a_rel_k.copy()
        else:
            since_improve += 1

        # VUB separation + aging
        if separate_vub(a_rel_k, x_rel_k) == 0:
            age_and_prune_vub()

        # CA/PA/CI on dualized demands
        ca = np.where(np.abs(g_k) > viol_tol)[0]
        pa = np.where(u != 0.0)[0]
        ci = np.setdiff1d(np.arange(inst.nJ), np.union1d(ca, pa), assume_unique=False)

        g_nd = g_k.copy()
        if ci.size:
            g_nd[ci] = 0.0

        dual_age[ca] = 0
        pa_not_ca = np.setdiff1d(pa, ca, assume_unique=False)
        if pa_not_ca.size:
            dual_age[pa_not_ca] += 1
            kill = pa_not_ca[dual_age[pa_not_ca] > dual_keep]
            if kill.size:
                u[kill] = 0.0
                dual_age[kill] = 0

        # Step: Polyak if UB is finite, else ε-schedule fallback
        denom = float(np.dot(g_nd, g_nd))
        if denom > 0.0:
            if np.isfinite(best_ub):
                step = gamma * max(best_ub - lb_k, 0.0) / denom
            else:
                step = eps / denom
            u = u + step * g_nd

        # ε-schedule decay only active while using ε (i.e., UB is inf)
        if not np.isfinite(best_ub) and since_improve >= stall_halve and eps > eps_min:
            eps = max(eps * 0.5, eps_min)
            since_improve = 0

        # Stop if we have a UB and gap is small
        if np.isfinite(best_ub):
            gap = best_ub - best_lb
            if gap <= tol_stop * max(1.0, abs(best_ub)):
                if log:
                    print(
                        f"[NDRC] it={k} LBk={lb_k:.6f} LB={best_lb:.6f} "
                        f"UB={best_ub:.6f} ||g_nd||={np.linalg.norm(g_nd):.3e} "
                        f"|CA|={ca.size} |PA|={pa.size} |CI|={ci.size}"
                    )
                break

        if log and (k % 50 == 0 or k == 1):
            print(
                f"[NDRC] it={k:4d} LBk={lb_k:.6f} LB={best_lb:.6f} "
                f"UB={(best_ub if np.isfinite(best_ub) else float('inf')):.6f} "
                f"||g_nd||={np.linalg.norm(g_nd):.3e} |CA|={ca.size} |PA|={pa.size} |CI|={ci.size}"
            )

    # last-chance UB
    if not np.isfinite(best_ub):
        fallback = best_a_open if best_a_open.sum() else a_rel_k
        ub_try = repair_ub(fallback)
        if ub_try is not None:
            best_ub = ub_try

    return float(best_ub) if np.isfinite(best_ub) else np.inf


def main():
    """
    Rotina principal
    """
    PATH = "instances/cfl/cfl_a2.txt"

    instance = CFLInstance.from_txt(PATH)

    obj = solve_instance_rc(instance, log=True)

    print(f"objective: {obj}")

    return


if __name__ == "__main__":
    main()
