"""
COS888

TSCFL com CPLEX

Gabriel Braun, 2025
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

    @property
    def R(self) -> float:
        return float(self.r.sum())

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


# === Relax-and-Cut (NDRC) for TSCFL =================================================
class RelaxAndCutTSCFL:
    """
    NDRC Relax-and-Cut for TSCFL (lecture notation):

    - LRP(λ): inner Lagrangian subproblem at multipliers (alpha, beta, gamma, phi, psi)
    - LDP:    dual problem max LRP(λ)
    - Subgradient g^k from constraint residuals at current primal minimizer of LRP
    - CA/PA/CI buckets; only one "maximal" cut per node; PA aging with EXTRA

    Dualized constraints:
      1) Depot balance:     sum_i x_ij - sum_k y_jk = 0        (multipliers alpha_j ∈ R)
      2) Client demand:     sum_j y_jk - r_k = 0               (multipliers beta_k ∈ R)
      3) x-VUB:             x_ij - q_j b_j ≤ 0                 (multipliers gamma_ij ≥ 0)

    Dualized flow-cover families (relax-and-cut):
      4) Plant cover (i,S): sum_{j∈S} x_ij ≤ min(p_i, sum_{j∈S} q_j) a_i  (phi_{i,S} ≥ 0)
      5) Depot cover (j,T): sum_{k∈T} y_jk ≤ min(q_j, sum_{k∈T} r_k) b_j  (psi_{j,T} ≥ 0)

    Hard constraints kept (always enforced): global coverage
        sum_i p_i a_i ≥ R,  sum_j q_j b_j ≥ R
    """

    def __init__(self, inst: TSCFLInstance, seed: int = 0):
        self.inst = inst
        self.rng = np.random.default_rng(seed)

        # Multipliers (λ) — initialize at zero
        self.alpha = np.zeros(inst.nJ)  # R
        self.beta = np.zeros(inst.nK)  # R
        self.gamma = np.zeros((inst.nI, inst.nJ))  # >=0

        # Dualized cut multipliers for relax-and-cut (>=0)
        # store as dicts keyed by (i, frozenset(S)) and (j, frozenset(T))
        self.phi: Dict[Tuple[int, frozenset], float] = {}  # plant covers
        self.psi: Dict[Tuple[int, frozenset], float] = {}  # depot covers

        # Aging (NDRC): PA → CI after EXTRA iterations without violation
        self.cut_age: Dict[Tuple[str, Tuple], int] = (
            {}
        )  # key: ("phi",(i,S)) or ("psi",(j,T))
        self.EXTRA = 5

        # Best known UB and incumbent
        self.best_ub = math.inf
        self.best_sol = None  # (a, b, x, y)

        # Hard polishing model (full MIP, used occasionally to repair UB)
        self._build_hard_model()

    # ---------- Hard model (for UB polishing) ---------------------------------------
    def _build_hard_model(self):
        inst = self.inst
        mdl = Model("TSCFL_polish")

        # Variables (same names as requested)
        self.a = mdl.binary_var_dict(inst.I, name="a")
        self.b = mdl.binary_var_dict(inst.J, name="b")
        self.x = mdl.continuous_var_dict(inst.IJ, lb=0.0, name="x")
        self.y = mdl.continuous_var_dict(inst.JK, lb=0.0, name="y")

        # Capacities
        mdl.add_constraints_(
            (mdl.sum(self.x[i, j] for j in inst.J) <= inst.p[i] * self.a[i])
            for i in inst.I
        )
        mdl.add_constraints_(
            (mdl.sum(self.y[j, k] for k in inst.K) <= inst.q[j] * self.b[j])
            for j in inst.J
        )
        # Balance and demand (original constraints)
        mdl.add_constraints_(
            (
                mdl.sum(self.x[i, j] for i in inst.I)
                == mdl.sum(self.y[j, k] for k in inst.K)
            )
            for j in inst.J
        )
        mdl.add_constraints_(
            (mdl.sum(self.y[j, k] for j in inst.J) == inst.r[k]) for k in inst.K
        )
        # Linking VUBs
        mdl.add_constraints_(
            (self.x[i, j] <= inst.q[j] * self.b[j]) for i, j in inst.IJ
        )
        mdl.add_constraints_(
            (self.y[j, k] <= inst.r[k] * self.b[j]) for j, k in inst.JK
        )

        # Global coverage (hard stabilizers)
        mdl.add_constraint(
            mdl.sum(inst.p[i] * self.a[i] for i in inst.I) >= inst.R,
            ctname="cov_plants",
        )
        mdl.add_constraint(
            mdl.sum(inst.q[j] * self.b[j] for j in inst.J) >= inst.R,
            ctname="cov_depots",
        )

        # Objective
        cost_fixed1 = mdl.sum(inst.f[i] * self.a[i] for i in inst.I)
        cost_fixed2 = mdl.sum(inst.g[j] * self.b[j] for j in inst.J)
        cost_stage1 = mdl.sum(inst.c[i, j] * self.x[i, j] for i, j in inst.IJ)
        cost_stage2 = mdl.sum(inst.d[j, k] * self.y[j, k] for j, k in inst.JK)
        mdl.minimize(cost_fixed1 + cost_fixed2 + cost_stage1 + cost_stage2)

        self.mdl_polish = mdl

    def _polish(
        self,
        core_I: Optional[List[int]] = None,
        core_J: Optional[List[int]] = None,
        timelimit: float = 2.0,
    ):
        """
        Solve the original MIP on a small 'core' (optional), to update the UB.
        """
        mdl = self.mdl_polish
        inst = self.inst

        # Optional core restriction (fix some opens to zero if requested)
        fixed0 = []
        if core_I is not None:
            for i in inst.I:
                if i not in core_I:
                    fixed0.append(self.a[i].equals(0))
        if core_J is not None:
            for j in inst.J:
                if j not in core_J:
                    fixed0.append(self.b[j].equals(0))
        for ct in fixed0:
            mdl.add(ct)

        mdl.parameters.timelimit = timelimit
        sol = mdl.solve(log_output=False)

        # remove temporary fixings
        for ct in fixed0:
            mdl.remove(ct)

        if sol:
            UB = sol.objective_value
            if UB < self.best_ub:
                self.best_ub = UB
                # extract solution
                a = np.array([sol.get_value(self.a[i]) for i in inst.I])
                b = np.array([sol.get_value(self.b[j]) for j in inst.J])
                x = np.zeros((inst.nI, inst.nJ))
                y = np.zeros((inst.nJ, inst.nK))
                for i, j in inst.IJ:
                    x[i, j] = sol.get_value(self.x[i, j])
                for j, k in inst.JK:
                    y[j, k] = sol.get_value(self.y[j, k])
                self.best_sol = (a, b, x, y)
        return self.best_ub

    # ---------- LRP(λ) evaluation (greedy separable subproblems) ---------------------
    def _evaluate_LR(self):
        """
        Solve LRP(λ) for current multipliers (alpha,beta,gamma,phi,psi).

        Returns:
          value (LR bound), (a,b,x,y), residuals for subgradient updates.
        """
        inst = self.inst
        I, J, K = inst.I, inst.J, inst.K

        # Adjusted fixed charges from dualized covers and x-VUBs
        fhat = inst.f.copy().astype(float)
        ghat = inst.g.copy().astype(float)
        # - contribution from x-VUB multipliers on g_j ( - q_j * sum_i gamma_ij )
        ghat -= inst.q * self.gamma.sum(axis=0)

        # - plant-cover cuts: f_i - sum_S phi_{i,S} * min(p_i, q(S))
        # - depot-cover cuts: g_j - sum_T psi_{j,T} * min(q_j, r(T))
        for (i, S), lam in self.phi.items():
            if lam > 0:
                capS = min(inst.p[i], float(inst.q[list(S)].sum()))
                fhat[i] -= lam * capS
        for (j, T), lam in self.psi.items():
            if lam > 0:
                capT = min(inst.q[j], float(inst.r[list(T)].sum()))
                ghat[j] -= lam * capT

        # Reduced costs for flows (include alpha,beta,gamma and cover-lambdas)
        tilde_c = inst.c + self.alpha[None, :] + self.gamma  # shape (nI, nJ)
        tilde_d = inst.d - self.alpha[:, None] + self.beta[None, :]  # (nJ, nK)

        # Add +phi on x_ij for j in S; +psi on y_jk for k in T
        if self.phi:
            for (i, S), lam in self.phi.items():
                if lam > 0:
                    idx = np.array(list(S), dtype=int)
                    tilde_c[i, idx] += lam
        if self.psi:
            for (j, T), lam in self.psi.items():
                if lam > 0:
                    idx = np.array(list(T), dtype=int)
                    tilde_d[j, idx] += lam

        # Solve plant subproblems (fixed-charge continuous knapsacks)
        a = np.zeros(inst.nI, dtype=float)
        x = np.zeros((inst.nI, inst.nJ), dtype=float)
        for i in I:
            j_best = int(np.argmin(tilde_c[i, :]))
            if tilde_c[i, j_best] < 0:
                a[i] = 1.0
                x[i, j_best] = inst.p[i]  # push all to the most negative cost
            # else: leave closed, x=0

        # Solve depot subproblems (bounded continuous knapsacks)
        b = np.zeros(inst.nJ, dtype=float)
        y = np.zeros((inst.nJ, inst.nK), dtype=float)
        for j in J:
            # select negative reduced-cost clients
            order = np.argsort(tilde_d[j, :])  # increasing
            cap = inst.q[j]
            used = 0.0
            any_neg = False
            for k in order:
                if tilde_d[j, k] < 0 and used < cap:
                    any_neg = True
                    take = min(inst.r[k], cap - used)
                    y[j, k] = take
                    used += take
                else:
                    break
            if any_neg and used > 0:
                b[j] = 1.0  # open depot if any flow assigned

        # LRP value:
        val = float(
            fhat @ a
            + ghat @ b
            + (tilde_c * x).sum()
            + (tilde_d * y).sum()
            - self.beta @ inst.r
        )

        # Residuals (subgradients of relaxed constraints)
        #  g^alpha_j = sum_i x_ij - sum_k y_jk
        g_alpha = x.sum(axis=0) - y.sum(axis=1)
        #  g^beta_k  = sum_j y_jk - r_k
        g_beta = y.sum(axis=0) - inst.r
        #  g^gamma_ij = x_ij - q_j b_j
        g_gamma = x - inst.q[None, :] * b[None, :]

        # Residuals for dualized covers (flow-covers):
        g_phi: Dict[Tuple[int, frozenset], float] = {}
        for (i, S), _ in self.phi.items():
            v = (
                x[i, list(S)].sum()
                - min(inst.p[i], float(inst.q[list(S)].sum())) * a[i]
            )
            g_phi[(i, S)] = v
        g_psi: Dict[Tuple[int, frozenset], float] = {}
        for (j, T), _ in self.psi.items():
            v = (
                y[j, list(T)].sum()
                - min(inst.q[j], float(inst.r[list(T)].sum())) * b[j]
            )
            g_psi[(j, T)] = v

        return val, (a, b, x, y), (g_alpha, g_beta, g_gamma, g_phi, g_psi)

    # ---------- Separation (heuristic, “maximal” per node) ---------------------------
    def _separate_depot_flow_cover(self, y: np.ndarray, b: np.ndarray, tol=1e-9):
        """
        For each depot j, build one 'maximal' T ⊆ K with large flow y_jk to create:
          sum_{k∈T} y_jk ≤ min(q_j, Σ_{k∈T} r_k) b_j
        Returns a dict { (j, frozenset(T)) : violation } for violated only.
        """
        inst = self.inst
        cand: Dict[Tuple[int, frozenset], float] = {}
        for j in inst.J:
            if b[j] <= 0.5 and y[j, :].sum() <= tol:
                continue
            # Sort clients by y_jk descending
            order = np.argsort(-y[j, :])
            T = []
            lhs = 0.0
            rhs_sum_r = 0.0
            best_viol = 0.0
            best_T = None
            for k in order:
                if y[j, k] <= tol:
                    break
                T.append(k)
                lhs += y[j, k]
                rhs_sum_r += inst.r[k]
                rhs = min(inst.q[j], rhs_sum_r) * b[j]
                viol = lhs - rhs
                if viol > best_viol + 1e-12:
                    best_viol = viol
                    best_T = list(T)
            if best_T and best_viol > tol:
                key = (j, frozenset(best_T))
                cand[key] = best_viol
        return cand

    def _separate_plant_flow_cover(self, x: np.ndarray, a: np.ndarray, tol=1e-9):
        """
        For each plant i, build one 'maximal' S ⊆ J (big x_ij) to create:
          sum_{j∈S} x_ij ≤ min(p_i, Σ_{j∈S} q_j) a_i
        Returns a dict { (i, frozenset(S)) : violation } for violated only.
        """
        inst = self.inst
        cand: Dict[Tuple[int, frozenset], float] = {}
        for i in inst.I:
            if a[i] <= 0.5 and x[i, :].sum() <= tol:
                continue
            order = np.argsort(-x[i, :])
            S = []
            lhs = 0.0
            rhs_sum_q = 0.0
            best_viol = 0.0
            best_S = None
            for j in order:
                if x[i, j] <= tol:
                    break
                S.append(j)
                lhs += x[i, j]
                rhs_sum_q += inst.q[j]
                rhs = min(inst.p[i], rhs_sum_q) * a[i]
                viol = lhs - rhs
                if viol > best_viol + 1e-12:
                    best_viol = viol
                    best_S = list(S)
            if best_S and best_viol > tol:
                key = (i, frozenset(best_S))
                cand[key] = best_viol
        return cand

    # ---------- Subgradient update (Polyak-like, NDRC rules) ------------------------
    def _update_multipliers(self, residuals, LR_val, eps, ub_for_stepsize):
        """
        Polyak-like step with NDRC masking (CI entries -> 0).
        """
        g_alpha, g_beta, g_gamma, g_phi, g_psi = residuals

        # Build a single vector 2-norm^2 (only over *active* entries)
        # NDRC masking: only entries tied to currently dualized inequalities (PA)
        # plus *newly violated* (CA) that we just decided to dualize.
        def sqnorm(arr):
            return float(np.square(arr).sum())

        norm2 = sqnorm(g_alpha) + sqnorm(g_beta) + sqnorm(g_gamma)
        for key, v in g_phi.items():
            norm2 += float(v * v)
        for key, v in g_psi.items():
            norm2 += float(v * v)

        if norm2 <= 1e-16:
            return  # nothing to do

        # Polyak stepsize μ = ε (UB - LRP) / ||g||^2  (projection for γ,φ,ψ)
        # If no UB yet, use a diminishing stepsize based on |LRP|
        gap = (
            (ub_for_stepsize - LR_val)
            if math.isfinite(ub_for_stepsize)
            else abs(LR_val) + 1.0
        )
        mu = max(0.0, eps * gap / (norm2 + 1e-16))

        # Update α, β (free) and γ, φ, ψ (projected to ≥0)
        self.alpha -= mu * g_alpha
        self.beta -= mu * g_beta
        self.gamma = np.maximum(0.0, self.gamma - mu * g_gamma)

        # Covers (projected)
        for key, v in g_phi.items():
            self.phi[key] = max(0.0, self.phi.get(key, 0.0) - mu * v)
        for key, v in g_psi.items():
            self.psi[key] = max(0.0, self.psi.get(key, 0.0) - mu * v)

    # ---------- NDRC cut pool mgmt (CA/PA/CI, maximal per node, aging) --------------
    def _ndrc_manage_cuts(
        self,
        new_phi: Dict[Tuple[int, frozenset], float],
        new_psi: Dict[Tuple[int, frozenset], float],
    ):
        """
        - Keep at most one 'maximal' per i (plant) and per j (depot) in CA.
        - Age PA cuts; drop when age > EXTRA.
        """
        # Choose one CA per node (max violation)
        # Plants:
        best_per_i: Dict[int, Tuple[Tuple[int, frozenset], float]] = {}
        for key, viol in new_phi.items():
            i, S = key
            if (i not in best_per_i) or (viol > best_per_i[i][1]):
                best_per_i[i] = (key, viol)
        # Depots:
        best_per_j: Dict[int, Tuple[Tuple[int, frozenset], float]] = {}
        for key, viol in new_psi.items():
            j, T = key
            if (j not in best_per_j) or (viol > best_per_j[j][1]):
                best_per_j[j] = (key, viol)

        # Add/update chosen CA into pool (becomes PA if keeps positive multiplier)
        for i, (key, viol) in best_per_i.items():
            if key not in self.phi:
                self.phi[key] = 0.0
            self.cut_age[("phi", key)] = 0  # reset age

        for j, (key, viol) in best_per_j.items():
            if key not in self.psi:
                self.psi[key] = 0.0
            self.cut_age[("psi", key)] = 0

        # Age the rest (PA\CA)
        to_delete = []
        for key in list(self.phi.keys()):
            tag = ("phi", key)
            if tag not in self.cut_age:
                self.cut_age[tag] = 0
            if key not in [bp[0] for bp in best_per_i.values()]:
                self.cut_age[tag] += 1
                if self.cut_age[tag] > self.EXTRA:
                    to_delete.append(("phi", key))
        for key in list(self.psi.keys()):
            tag = ("psi", key)
            if tag not in self.cut_age:
                self.cut_age[tag] = 0
            if key not in [bp[0] for bp in best_per_j.values()]:
                self.cut_age[tag] += 1
                if self.cut_age[tag] > self.EXTRA:
                    to_delete.append(("psi", key))

        # Remove expired
        for tag, key in to_delete:
            if tag == "phi":
                self.phi.pop(key, None)
            else:
                self.psi.pop(key, None)
            self.cut_age.pop((tag, key), None)

    # ---------- Main driver ----------------------------------------------------------
    def run(
        self,
        max_iter: int = 500,
        eps: float = 2.0,
        polish_every: int = 10,
        polish_time: float = 2.0,
        seed: int = 0,
        verbose: bool = True,
    ):
        """
        NDRC Relax-and-Cut loop.
        - eps: Polyak epsilon (can be reduced on stalls, e.g., eps *= 0.5)
        """
        self.rng = np.random.default_rng(seed)
        best_lb = -math.inf
        stall = 0

        for k in range(1, max_iter + 1):
            # 1) Solve LRP(λ)
            LR_val, (a, b, x, y), residuals = self._evaluate_LR()
            if LR_val > best_lb + 1e-9:
                best_lb = LR_val
                stall = 0
            else:
                stall += 1

            # 2) Separation (CA): one 'maximal' per node
            cand_psi = self._separate_depot_flow_cover(y, b)
            cand_phi = self._separate_plant_flow_cover(x, a)
            self._ndrc_manage_cuts(cand_phi, cand_psi)

            # 3) Subgradient update (NDRC masking handled by our pool)
            # Build residuals including only dualized covers in pool
            g_alpha, g_beta, g_gamma, g_phi_all, g_psi_all = residuals
            g_phi = {key: g_phi_all.get(key, 0.0) for key in self.phi.keys()}
            g_psi = {key: g_psi_all.get(key, 0.0) for key in self.psi.keys()}
            self._update_multipliers(
                (g_alpha, g_beta, g_gamma, g_phi, g_psi), LR_val, eps, self.best_ub
            )

            # 4) Heuristic polish (UB) and stepsize stabilization
            if (k % polish_every) == 0:
                # Build cores from current usage
                Icore = [i for i in self.inst.I if a[i] > 0.5 or x[i, :].sum() > 1e-9]
                Jcore = [j for j in self.inst.J if b[j] > 0.5 or y[j, :].sum() > 1e-9]
                self._polish(core_I=Icore, core_J=Jcore, timelimit=polish_time)

            # 5) Optional epsilon schedule on stall
            if stall > 50:
                eps = max(0.25, 0.5 * eps)
                stall = 0

            if verbose and (k % 10 == 0):
                gap = (
                    (self.best_ub - best_lb) / max(1.0, abs(self.best_ub))
                    if math.isfinite(self.best_ub)
                    else math.inf
                )
                print(
                    f"it={k:4d}  LRP={LR_val:,.3f}  LB*={best_lb:,.3f}  UB*={self.best_ub:,.3f}  gap={gap:.4%}  |phi|={len(self.phi)}  |psi|={len(self.psi)}"
                )

        return dict(
            LB=best_lb,
            UB=self.best_ub,
            best=self.best_sol,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            phi=self.phi,
            psi=self.psi,
        )


def main():
    """
    Rotina principal
    """
    PATH = "instances/tscfl/tscfl_11_50.txt"

    instance = TSCFLInstance.from_txt(PATH)

    solver = RelaxAndCutTSCFL(instance, seed=0)

    solver.run(max_iter=500, eps=2.0, polish_every=10, polish_time=2.0, verbose=True)

    return


if __name__ == "__main__":
    main()
