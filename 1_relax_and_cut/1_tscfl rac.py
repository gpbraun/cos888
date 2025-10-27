"""
COS888

TSCFL com CPLEX

Gabriel Braun, 2025
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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


# =====================================
# Relax-and-Cut class (simple NDRC)
# =====================================
class RelaxAndCutTSCFL:
    def __init__(
        self,
        inst: TSCFLInstance,
        *,
        max_iter: int = 1000,
        threads: int = 8,
        # Stepsizes: Polyak (when UB exists) or epsilon fallback
        gamma: float = 1.0,
        eps0: float = 1.0,
        eps_min: float = 1e-12,
        stall_halve: int = 200,
        # NDRC bookkeeping (for equalities only; gamma handled simply)
        viol_tol: float = 1e-1,
        dual_keep: int = 5,
        # UB heuristic frequency
        ub_every: int = 5,
        # Decouple LR by dualizing x-VUB as well (recommended)
        dualize_x_vub: bool = True,
        # Logging
        log: bool = True,
    ) -> None:
        self.inst = inst
        self.max_iter = max_iter
        self.threads = threads
        self.gamma = gamma
        self.eps = float(eps0)
        self.eps_min = eps_min
        self.stall_halve = stall_halve
        self.viol_tol = viol_tol
        self.dual_keep = dual_keep
        self.ub_every = ub_every
        self.dualize_x_vub = dualize_x_vub
        self.log = log

        # Multipliers
        self.alpha = np.zeros(inst.nJ, dtype=float)  # depot balance
        self.beta = np.zeros(inst.nK, dtype=float)  # client demand
        self.gamma_x = (
            np.zeros((inst.nI, inst.nJ), dtype=float) if dualize_x_vub else None
        )

        # Aging for NDRC (alpha/beta)
        self.age_alpha = np.zeros(inst.nJ, dtype=int)
        self.age_beta = np.zeros(inst.nK, dtype=int)

        # Bounds & incumbents
        self.best_lb = -np.inf
        self.best_ub = np.inf
        self.a_inc = np.zeros(inst.nI, dtype=int)
        self.b_inc = np.zeros(inst.nJ, dtype=int)

        # VA-like running averages (cheap stabilization for UB rounding)
        self.a_bar = np.zeros(inst.nI, dtype=float)
        self.b_bar = np.zeros(inst.nJ, dtype=float)
        self.smooth_w = 0.0  # weight accumulator

    # ------------- public API -------------
    def solve(self) -> Dict[str, Any]:
        it_since_lb = 0
        for it in range(1, self.max_iter + 1):
            L_k, a_k, b_k, x_k, y_k, gA_k, gB_k, gG_k = self._solve_lrp()

            # LB update
            if L_k > self.best_lb + 1e-12:
                self.best_lb = L_k
                it_since_lb = 0
            else:
                it_since_lb += 1

            # Keep VA-like averages (for UB rounding)
            self._update_averages(a_k, b_k)

            # UB heuristic (periodic & on LB improvement)
            if (it % self.ub_every == 0) or (it == 1) or (it_since_lb == 0):
                # Try a fast LH first (score-based), then repair MCF
                z_try = self._lh_open_by_score_and_repair()
                if (z_try is not None) and (z_try + 1e-8 < self.best_ub):
                    self.best_ub = z_try
                    self.a_inc = (
                        self.a_last_open if hasattr(self, "a_last_open") else a_k
                    ).copy()
                    self.b_inc = (
                        self.b_last_open if hasattr(self, "b_last_open") else b_k
                    ).copy()

            # NDRC: CA/PA/CI for equalities only (kept simple)
            gA_nd, gB_nd = self._ndrc_zeroing(gA_k, gB_k)

            # Stepsize and dual update (Polyak or epsilon)
            denom = float(np.dot(gA_nd, gA_nd) + np.dot(gB_nd, gB_nd))
            if self.dualize_x_vub and gG_k is not None:
                denom += float(np.sum(gG_k * gG_k))
            if denom > 0.0:
                if np.isfinite(self.best_ub):
                    mu = self.gamma * max(self.best_ub - L_k, 0.0) / denom
                else:
                    mu = self.eps / denom

                self.alpha = self.alpha + mu * gA_nd
                self.beta = self.beta + mu * gB_nd
                if self.dualize_x_vub and gG_k is not None:
                    self.gamma_x = np.maximum(
                        0.0, self.gamma_x + mu * gG_k
                    )  # projection

            # Epsilon schedule while no UB
            if (
                (not np.isfinite(self.best_ub))
                and (it % self.stall_halve == 0)
                and (self.eps > self.eps_min)
            ):
                self.eps = max(self.eps * 0.5, self.eps_min)

            # Log
            if self.log and (it % 5 == 0 or it == 1):
                denom_view = np.sqrt(denom)
                ub_view = self.best_ub if np.isfinite(self.best_ub) else float("inf")
                print(
                    f"[RaC] it={it:4d}  LRP={L_k:.6f}  LB={self.best_lb:.6f}  UB={ub_view:.6f}  ||g||={denom_view:.3e}"
                )

            # Early stop if small gap
            if np.isfinite(self.best_ub):
                gap = self.best_ub - self.best_lb
                if gap <= 1e-6 * max(1.0, abs(self.best_ub)):
                    break

        # Final attempt if no UB
        if not np.isfinite(self.best_ub):
            z_try = self._repair_given_open(
                self.a_inc if self.a_inc.sum() else a_k,
                self.b_inc if self.b_inc.sum() else b_k,
            )
            if z_try is not None:
                self.best_ub = z_try

        return {
            "LB": float(self.best_lb),
            "UB": float(self.best_ub),
            "a_inc": self.a_inc.copy(),
            "b_inc": self.b_inc.copy(),
            "alpha": self.alpha.copy(),
            "beta": self.beta.copy(),
            "gamma_x": None if self.gamma_x is None else self.gamma_x.copy(),
        }

    # ------------- LR subproblem -------------
    def _solve_lrp(
        self,
    ) -> Tuple[
        float,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        Optional[np.ndarray],
    ]:
        """
        Greedy/separable evaluation of LRP(α,β[,γ]) with:
          Plants: fixed-charge continuous knapsack (no per-arc caps if we dualize x-VUB).
          Depots: bounded continuous knapsack with per-arc bound y[j,k] <= r[k].
        """
        inst = self.inst
        TOL = 1e-12

        # Reduced costs
        if self.dualize_x_vub:
            ctil = inst.c + self.alpha[None, :] + self.gamma_x  # (nI, nJ)
        else:
            ctil = inst.c + self.alpha[None, :]
        dtil = inst.d - self.alpha[:, None] + self.beta[None, :]  # (nJ, nK)

        a_rel = np.zeros(inst.nI, dtype=np.int8)
        b_rel = np.zeros(inst.nJ, dtype=np.int8)
        x_rel = np.zeros((inst.nI, inst.nJ), dtype=float)
        y_rel = np.zeros((inst.nJ, inst.nK), dtype=float)

        # Constant term from demand equalities
        L_val = -float(np.dot(self.beta, inst.r))

        # ---------- Plants (i) ----------
        for i in range(inst.nI):
            row = ctil[i]
            # All capacity p_i can go to the single most negative reduced cost (no per-arc cap if dualize_x_vub)
            j_star = int(np.argmin(row))
            rc = float(row[j_star])
            if rc < -TOL:
                take = float(inst.p[i])
                x_rel[i, j_star] = take
                var_part = rc * take
                if inst.f[i] + var_part < 0.0:
                    a_rel[i] = 1
                    L_val += inst.f[i] + var_part
                else:
                    x_rel[i, j_star] = 0.0  # not worth opening

        # ---------- Depots (j) ----------
        for j in range(inst.nJ):
            row = dtil[j]
            neg_ks = np.nonzero(row < -TOL)[0]
            if neg_ks.size == 0:
                continue

            order = neg_ks[np.argsort(row[neg_ks])]  # best (most negative) first
            cap_left = float(inst.q[j])
            var_part = 0.0
            yj = y_rel[j]

            for k in order:
                if cap_left <= TOL:
                    break
                take = min(
                    cap_left, float(inst.r[k])
                )  # y[j,k] <= r[k] (b_j is binary later in repair)
                if take <= TOL:
                    continue
                rc = float(row[k])  # negative
                yj[k] = take
                cap_left -= take
                var_part += rc * take

            if yj.any() and (inst.g[j] + var_part < 0.0):
                b_rel[j] = 1
                L_val += inst.g[j] + var_part
            else:
                yj.fill(0.0)

        # Subgradients
        sum_x_j = x_rel.sum(axis=0)  # (nJ,)
        sum_y_j = y_rel.sum(axis=1)  # (nJ,)
        sum_y_k = y_rel.sum(axis=0)  # (nK,)

        gA = sum_x_j - sum_y_j  # depot balance residuals
        gB = sum_y_k - inst.r  # demand residuals
        gG = None
        if self.dualize_x_vub:
            # g_gamma = x - q*b  (here b is 0/1 only if we open; in LR, use current b_rel as the natural primal choice)
            gG = x_rel - inst.q[None, :] * b_rel[None, :]

        return L_val, a_rel, b_rel, x_rel, y_rel, gA, gB, gG

    # ------------- Simple NDRC zeroing for equalities -------------
    def _ndrc_zeroing(
        self, gA: np.ndarray, gB: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        inst = self.inst

        CA_a = np.where(np.abs(gA) > self.viol_tol)[0]
        PA_a = np.where(self.alpha != 0.0)[0]
        CI_a = np.setdiff1d(
            np.arange(inst.nJ), np.union1d(CA_a, PA_a), assume_unique=False
        )

        CA_b = np.where(np.abs(gB) > self.viol_tol)[0]
        PA_b = np.where(self.beta != 0.0)[0]
        CI_b = np.setdiff1d(
            np.arange(inst.nK), np.union1d(CA_b, PA_b), assume_unique=False
        )

        gA_nd = gA.copy()
        gB_nd = gB.copy()
        if CI_a.size:
            gA_nd[CI_a] = 0.0
        if CI_b.size:
            gB_nd[CI_b] = 0.0

        # Aging
        self.age_alpha[CA_a] = 0
        dropA = np.setdiff1d(PA_a, CA_a, assume_unique=False)
        if dropA.size:
            self.age_alpha[dropA] += 1
            to_zero = dropA[self.age_alpha[dropA] > self.dual_keep]
            if to_zero.size:
                self.alpha[to_zero] = 0.0
                self.age_alpha[to_zero] = 0

        self.age_beta[CA_b] = 0
        dropB = np.setdiff1d(PA_b, CA_b, assume_unique=False)
        if dropB.size:
            self.age_beta[dropB] += 1
            to_zero = dropB[self.age_beta[dropB] > self.dual_keep]
            if to_zero.size:
                self.beta[to_zero] = 0.0
                self.age_beta[to_zero] = 0

        return gA_nd, gB_nd

    # ------------- VA-like averages for UB rounding -------------
    def _update_averages(
        self, a_rel: np.ndarray, b_rel: np.ndarray, w: float = 1.0
    ) -> None:
        self.a_bar = (self.smooth_w * self.a_bar + w * a_rel) / (self.smooth_w + w)
        self.b_bar = (self.smooth_w * self.b_bar + w * b_rel) / (self.smooth_w + w)
        self.smooth_w += w

    # ------------- Lagrangian heuristic: score-based open + repair -------------
    def _lh_open_by_score_and_repair(self) -> Optional[float]:
        """
        LH-1: open-by-score (reduced-cost driven) then min-cost flow repair.
        Also try VA-based rounding as fallback if capacity is insufficient.
        """
        inst = self.inst

        # Reduced costs at current multipliers
        if self.dualize_x_vub:
            ctil = inst.c + self.alpha[None, :] + self.gamma_x
        else:
            ctil = inst.c + self.alpha[None, :]
        dtil = inst.d - self.alpha[:, None] + self.beta[None, :]

        # --- Depot scores: open cheapest-by-reduced-cost mass up to q_j
        depot_scores = np.full(inst.nJ, np.inf, dtype=float)
        for j in range(inst.nJ):
            row = dtil[j]
            order = np.argsort(row)  # cheapest first (can be positive too)
            cap = float(inst.q[j])
            acc = 0.0
            for k in order:
                if cap <= 0.0:
                    break
                take = min(cap, float(inst.r[k]))
                if take > 0.0:
                    acc += row[k] * take
                    cap -= take
            depot_scores[j] = acc + inst.g[j]

        # --- Plant scores: fill p_i on best j (or spread if not dualizing x-VUB)
        plant_scores = np.full(inst.nI, np.inf, dtype=float)
        if self.dualize_x_vub:
            j_best = np.argmin(ctil, axis=1)
            rc = ctil[np.arange(inst.nI), j_best]
            plant_scores = inst.f + rc * inst.p
        else:
            # spread greedily over js (slower, but still OK)
            for i in range(inst.nI):
                row = ctil[i]
                order = np.argsort(row)
                cap = float(inst.p[i])
                acc = 0.0
                for j in order:
                    if cap <= 0.0:
                        break
                    take = cap
                    acc += row[j] * take
                    cap -= take
                plant_scores[i] = inst.f[i] + acc

        # Greedy open until global capacity covers total demand
        R = float(np.sum(inst.r))
        a_open = np.zeros(inst.nI, dtype=int)
        b_open = np.zeros(inst.nJ, dtype=int)

        # First depots by best score per unit capacity
        depot_ratio = depot_scores / np.maximum(inst.q, 1e-12)
        for j in np.argsort(depot_ratio):
            if b_open.sum() == 0 or np.dot(inst.q, b_open) < R - 1e-9:
                b_open[j] = 1

        # Then plants by best score per unit capacity
        plant_ratio = plant_scores / np.maximum(inst.p, 1e-12)
        for i in np.argsort(plant_ratio):
            if a_open.sum() == 0 or np.dot(inst.p, a_open) < R - 1e-9:
                a_open[i] = 1

        # If still insufficient, use VA-rounded openings to top-up
        if np.dot(inst.q, b_open) < R - 1e-9:
            for j in np.argsort(-self.b_bar):
                if b_open[j] == 0:
                    b_open[j] = 1
                if np.dot(inst.q, b_open) >= R - 1e-9:
                    break
        if np.dot(inst.p, a_open) < R - 1e-9:
            for i in np.argsort(-self.a_bar):
                if a_open[i] == 0:
                    a_open[i] = 1
                if np.dot(inst.p, a_open) >= R - 1e-9:
                    break

        # Save last-opened sets (for reporting incumbents)
        self.a_last_open = a_open.copy()
        self.b_last_open = b_open.copy()

        # Final repair by min-cost flow on open sets
        return self._repair_given_open(a_open, b_open)

    # ------------- Repair: min-cost flow (fixed openings) -------------
    def _repair_given_open(
        self, a_open: np.ndarray, b_open: np.ndarray
    ) -> Optional[float]:
        inst = self.inst
        R = float(np.sum(inst.r))

        # Ensure total capacity (open cheapest per-unit capacity)
        cap_p = float(np.dot(inst.p, a_open))
        cap_q = float(np.dot(inst.q, b_open))

        if cap_p + 1e-9 < R:
            closed = [i for i in range(inst.nI) if a_open[i] == 0 and inst.p[i] > 0]
            closed.sort(key=lambda i: inst.f[i] / max(inst.p[i], 1e-12))
            for i in closed:
                a_open[i] = 1
                cap_p += float(inst.p[i])
                if cap_p >= R - 1e-9:
                    break

        if cap_q + 1e-9 < R:
            closed = [j for j in range(inst.nJ) if b_open[j] == 0 and inst.q[j] > 0]
            closed.sort(key=lambda j: inst.g[j] / max(inst.q[j], 1e-12))
            for j in closed:
                b_open[j] = 1
                cap_q += float(inst.q[j])
                if cap_q >= R - 1e-9:
                    break

        I_open = [i for i in range(inst.nI) if a_open[i] == 1]
        J_open = [j for j in range(inst.nJ) if b_open[j] == 1]
        if not I_open or not J_open:
            return None

        # Build min-cost flow on open sets
        mdl = Model(name="TSCFL_repair", log_output=False)
        mdl.parameters.threads = self.threads

        xR = {
            (i, j): mdl.continuous_var(lb=0.0, name=f"x_{i}_{j}")
            for i in I_open
            for j in J_open
        }
        yR = {
            (j, k): mdl.continuous_var(lb=0.0, name=f"y_{j}_{k}")
            for j in J_open
            for k in inst.K
        }

        # Capacities
        mdl.add_constraints_(
            mdl.sum(xR[i, j] for j in J_open) <= inst.p[i] for i in I_open
        )
        mdl.add_constraints_(
            mdl.sum(yR[j, k] for k in inst.K) <= inst.q[j] for j in J_open
        )

        # Depot balance (open depots only)
        mdl.add_constraints_(
            mdl.sum(xR[i, j] for i in I_open) == mdl.sum(yR[j, k] for k in inst.K)
            for j in J_open
        )

        # Client demand
        mdl.add_constraints_(
            mdl.sum(yR[j, k] for j in J_open) == inst.r[k] for k in inst.K
        )

        # Linking (optional but safe): y[j,k] <= r[k], x[j] unconstrained per-arc here
        mdl.add_constraints_(yR[j, k] <= inst.r[k] for (j, k) in yR)

        flow_cost = mdl.sum(inst.c[i, j] * xR[i, j] for (i, j) in xR) + mdl.sum(
            inst.d[j, k] * yR[j, k] for (j, k) in yR
        )

        fixed_cost = float(
            np.dot(inst.f[I_open], np.ones(len(I_open)))
            + np.dot(inst.g[J_open], np.ones(len(J_open)))
        )
        mdl.minimize(flow_cost + fixed_cost)

        sol = mdl.solve()
        return None if not sol else float(sol.objective_value)


def main():
    """
    Rotina principal
    """
    PATH = "instances/tscfl/tscfl_11_50.txt"

    instance = TSCFLInstance.from_txt(PATH)

    solver = RelaxAndCutTSCFL(
        instance,
        max_iter=1000,
        threads=16,
        gamma=1.0,
        eps0=1.0,
        stall_halve=200,
        eps_min=1e-12,
        viol_tol=1e-1,
        dual_keep=5,
        ub_every=5,
        dualize_x_vub=True,  # keep LR fully separable and stronger
        log=True,
    )
    result = solver.solve()

    print("\n=== RESULT ===")
    print(f"LB = {result['LB']:.6f}")
    print(f"UB = {result['UB']:.6f}")
    print(f"gap = {result['UB'] - result['LB']:.6f}")

    return


if __name__ == "__main__":
    main()
