"""
COS888

CFL com Non-Delayed Relax-and-Cut (NDRC)

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

        s1 = 2 * nI

        pf = data[:s1].reshape(nI, 2)
        p = pf[:, 0]
        f = pf[:, 1]

        rc = data[s1:].reshape(nJ, 1 + nI)
        r = rc[:, 0]
        c = rc[:, 1:].T / r[None, :]

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


def solve_instance_rc(
    inst: "CFLInstance",
    *,
    max_iter: int = 10_000,
    threads: int = 1,
    # Polyak (used when UB is finite; slides denote target w or z(·))
    gamma: float = 1,
    # Fallback ε-schedule (only while UB is unknown)
    eps0: float = 1.0,
    stall_halve: int = 200,
    eps_min: float = 1e-12,
    # CA/PA/CI & aging
    viol_tol: float = 1e-4,  # threshold to mark a row "currently active" (in CA(k))
    dual_keep: int = 10,  # EXTRA life for PA(k)\CA(k) before dropping to CI(k)
    ndrc_keep: int = 10,  # life for strengthening cuts (non-delayed)
    # stopping / logs
    tol_stop: float = 1e-6,
    log: bool = False,
) -> float:
    """
    Non-Delayed Relax-and-Cut (NDRC)
    """
    TOL = 1e-12

    # ---------- Dual state: λ over the dualized equalities (free multipliers) ----------
    lam = np.zeros(inst.nJ, dtype=float)  # slides: λ
    age_lam = np.zeros(inst.nJ, dtype=int)  # EXTRA aging for PA(k)\CA(k)

    # ---------- Strengthening pool S (kept non-dualized; NDRC “non-delayed”) ----------
    # S[i] is a dict {j: age} for active VUB(i,j): x[i,j] ≤ r[j] a[i]
    S = [dict() for _ in range(inst.nI)]

    # ---------- Bounds ----------
    L_best = -np.inf  # best dual lower bound (max LRP(λ))
    z_best = np.inf  # best primal feasible cost (upper bound)
    a_inc = np.zeros(
        inst.nI, dtype=int
    )  # incumbent open pattern for last-chance repair

    # ε-schedule (only while z_best is inf)
    eps = float(eps0)
    since_improve = 0

    # ---------------------------------------------------------------------------
    # STEP 1 — LRP(λ): solve relaxed problem at current multipliers
    # ---------------------------------------------------------------------------
    def LRP(lam_vec: np.ndarray):
        """
        Returns:
            L_val : LRP(λ^k)  (dual value at λ)
            a_rel : relaxed open/close per facility (0/1 in this greedy)
            x_rel : relaxed flows
            g     : subgradient of dualized equalities (demands): g = r − ∑_i x[i,·]
        """
        ctil = inst.c - lam_vec[None, :]  # reduced costs: c̃ = c − λ
        L_val = float(np.dot(lam_vec, inst.r))
        a_rel = np.zeros(inst.nI, dtype=int)
        x_rel = np.zeros((inst.nI, inst.nJ), dtype=float)

        for i in inst.I:
            # consider arcs with negative reduced cost only
            cols = np.where(ctil[i, :] < -TOL)[0]
            if cols.size == 0:
                continue
            order = cols[np.argsort(ctil[i, cols])]
            cap_left = float(inst.p[i])
            var_part = 0.0

            for j in order:
                if cap_left <= TOL:
                    break
                rc = ctil[i, j]
                if rc >= 0.0:
                    break

                # Non-delayed strengthening: enforce pair-cap only if VUB(i,j) ∈ S
                cap_pair = inst.r[j] if (j in S[i]) else np.inf
                take = min(cap_left, cap_pair)
                if take > TOL:
                    x_rel[i, j] = take
                    cap_left -= take
                    var_part += rc * take

            # open i in the relaxation iff fixed+variable part is negative
            if inst.f[i] + var_part < 0.0:
                a_rel[i] = 1
                L_val += inst.f[i] + var_part
            else:
                if np.any(x_rel[i, :] > 0.0):
                    x_rel[i, :] = 0.0  # rollback shipments if not opening

        g = inst.r - np.sum(x_rel, axis=0)
        return L_val, a_rel, x_rel, g

    # ---------------------------------------------------------------------------
    # STEP 2 — Separation of valid inequalities (Strengthening) & aging of S
    # ---------------------------------------------------------------------------
    def separate_strengthening(a_rel, x_rel) -> int:
        """
        Separate violated VUB(i,j): x[i,j] ≤ r[j] a[i].
        Activate into S with age=0. Return number of new activations.
        """
        newS = 0
        for i in inst.I:
            if a_rel[i] == 0:
                continue
            viol = np.where(x_rel[i, :] > inst.r + 1e-9)[0]
            for j in viol:
                if j not in S[i]:
                    S[i][j] = 0
                    newS += 1
        return newS

    def age_and_prune_S():
        """Age active strengthening; prune when age > ndrc_keep."""
        for i in inst.I:
            if not S[i]:
                continue
            drop = []
            for j, age in S[i].items():
                age += 1
                if age > ndrc_keep:
                    drop.append(j)
                else:
                    S[i][j] = age
            for j in drop:
                del S[i][j]

    # ---------------------------------------------------------------------------
    # STEP 5 — Lagrangian heuristic to produce a primal feasible UB (ẑ)
    # ---------------------------------------------------------------------------
    def lagrangian_heuristic(a_rel):
        """
        Complete capacity if needed, then solve transportation LP over open set.
        Return feasible cost (UB) or None.
        """
        total_r = float(np.sum(inst.r))
        a_fix = a_rel.astype(int).copy()
        cap = float(np.dot(inst.p, a_fix))

        if cap + 1e-9 < total_r:
            closed = [i for i in inst.I if (a_fix[i] == 0 and inst.p[i] > 0)]
            # open cheapest per unit of capacity
            closed.sort(key=lambda i: inst.f[i] / max(inst.p[i], 1e-12))
            for i in closed:
                a_fix[i] = 1
                cap += float(inst.p[i])
                if cap + 1e-9 >= total_r:
                    break
            if cap + 1e-9 < total_r:
                a_fix[:] = 1  # last resort: open all

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

        # meet demands & respect capacities
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

    # ================================
    # MAIN NDRC LOOP (k = 1..max_iter)
    # ================================
    for k in range(1, max_iter + 1):
        # (1) LRP(λ^k)
        L_k, a_k, x_k, g_k = LRP(lam)

        # track best LB and try to build a UB when LRP improves
        improved = L_k > L_best + 1e-12
        if improved:
            L_best = L_k
            since_improve = 0
            z_try = lagrangian_heuristic(a_k)
            if (z_try is not None) and (z_try + 1e-8 < z_best):
                z_best = z_try
                a_inc = a_k.copy()
        else:
            since_improve += 1

        # (2) Separation & aging for strengthening pool S
        if separate_strengthening(a_k, x_k) == 0:
            age_and_prune_S()

        # (3) CA/PA/CI on dualized family (demands)
        CA_k = np.where(np.abs(g_k) > viol_tol)[0]  # “currently active”
        PA_k = np.where(lam != 0.0)[0]  # “previously active”
        CI_k = np.setdiff1d(
            np.arange(inst.nJ), np.union1d(CA_k, PA_k), assume_unique=False
        )

        # Non-Delayed: zero subgradient on CI(k)
        g_nd_k = g_k.copy()
        if CI_k.size:
            g_nd_k[CI_k] = 0.0

        # EXTRA aging for PA(k)\CA(k)
        age_lam[CA_k] = 0
        PA_not_CA = np.setdiff1d(PA_k, CA_k, assume_unique=False)
        if PA_not_CA.size:
            age_lam[PA_not_CA] += 1
            to_drop = PA_not_CA[age_lam[PA_not_CA] > dual_keep]
            if to_drop.size:
                lam[to_drop] = 0.0
                age_lam[to_drop] = 0

        # (4) Step length μ_k and update λ^{k+1} = λ^k + μ_k * g_nd_k
        denom = float(np.dot(g_nd_k, g_nd_k))
        if denom > 0.0:
            if np.isfinite(z_best):
                mu_k = gamma * max(z_best - L_k, 0.0) / denom  # Polyak with target ẑ
            else:
                mu_k = eps / denom  # ε-schedule fallback
            lam = lam + mu_k * g_nd_k

        # ε-halving while no UB exists
        if (
            (not np.isfinite(z_best))
            and (since_improve >= stall_halve)
            and (eps > eps_min)
        ):
            eps = max(eps * 0.5, eps_min)
            since_improve = 0

        # stop by gap once UB exists
        if np.isfinite(z_best):
            gap = z_best - L_best
            if gap <= tol_stop * max(1.0, abs(z_best)):
                if log:
                    print(
                        f"[NDRC] it={k}  LRP(λ^k)={L_k:.6f}  LB={L_best:.6f}  "
                        f"UB={z_best:.6f}  ||g_nd||={np.linalg.norm(g_nd_k):.3e}  "
                        f"|CA(k)|={CA_k.size} |PA(k)|={PA_k.size} |CI(k)|={CI_k.size}"
                    )
                break

        if log and (k % 50 == 0 or k == 1):
            print(
                f"[NDRC] it={k:4d}  LRP(λ^k)={L_k:.6f}  LB={L_best:.6f}  "
                f"UB={(z_best if np.isfinite(z_best) else float('inf')):.6f}  "
                f"||g_nd||={np.linalg.norm(g_nd_k):.3e}  "
                f"|CA(k)|={CA_k.size} |PA(k)|={PA_k.size} |CI(k)|={CI_k.size}"
            )

    # last-chance UB if none was produced (use best/last relaxed pattern)
    if not np.isfinite(z_best):
        fallback = a_inc if a_inc.sum() else a_k
        z_try = lagrangian_heuristic(fallback)
        if z_try is not None:
            z_best = z_try

    return float(z_best) if np.isfinite(z_best) else np.inf


def main():
    """
    Rotina principal
    """
    PATH = "instances/cfl/cfl_41.txt"

    instance = CFLInstance.from_txt(PATH)

    obj = solve_instance_rc(instance, log=True)

    print(f"objective: {obj}")

    return


if __name__ == "__main__":
    main()
