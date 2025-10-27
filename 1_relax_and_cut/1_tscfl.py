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


def solve_instance_rc(
    inst: "TSCFLInstance",
    *,
    max_iter: int = 1000,
    threads: int = 20,
    # Polyak (when UB exists) / epsilon fallback
    gamma: float = 1.0,
    eps0: float = 1.0,
    stall_halve: int = 200,
    eps_min: float = 1e-12,
    # CA/PA/CI bookkeeping
    viol_tol: float = 1e-1,
    dual_keep: int = 5,
    # stopping / logs
    tol_stop: float = 1e-6,
    log: bool = False,
) -> None:
    """
    NDRC
    """
    alpha = np.zeros(inst.nJ, dtype=float)  # for depot balances
    beta = np.zeros(inst.nK, dtype=float)  # for customer demands
    age_alpha = np.zeros(inst.nJ, dtype=int)
    age_beta = np.zeros(inst.nK, dtype=int)

    # ---------- bounds ----------
    L_best = -np.inf
    z_best = np.inf
    a_inc = np.zeros(inst.nI, dtype=int)
    b_inc = np.zeros(inst.nJ, dtype=int)

    # epsilon schedule (used only while UB unknown)
    eps = float(eps0)
    since_improve = 0

    # ------------------------------------------------------------------
    # Greedy LRP at (alpha, beta) — fully decoupled, no CPLEX here
    # ------------------------------------------------------------------
    def solve_lrp_greedy(alpha_vec: np.ndarray, beta_vec: np.ndarray):
        TOL = 1e-12

        # Reduced costs
        ctil = inst.c + alpha_vec[None, :]  # (nI, nJ)
        dtil = inst.d - alpha_vec[:, None] + beta_vec[None, :]  # (nJ, nK)

        # Allocations and open decisions
        a_rel = np.zeros(inst.nI, dtype=np.int8)
        b_rel = np.zeros(inst.nJ, dtype=np.int8)
        x_rel = np.zeros((inst.nI, inst.nJ), dtype=float)
        y_rel = np.zeros((inst.nJ, inst.nK), dtype=float)

        # Constant from dualizing customer demands
        L_val = -float(np.dot(beta_vec, inst.r))

        # Local views (tiny speedup + readability)
        p, q, r = inst.p, inst.q, inst.r
        f, g = inst.f, inst.g

        # -------- Plant-side subproblems (independent over i) --------
        for i in range(inst.nI):
            row = ctil[i]  # view
            neg_js = np.nonzero(row < -TOL)[0]
            if neg_js.size == 0:
                continue

            order = neg_js[np.argsort(row[neg_js])]
            cap_left = float(p[i])
            var_part = 0.0
            x_row = x_rel[i]  # view

            for j in order:
                if np.isclose(cap_left, 0.0, atol=TOL):
                    break

                # per-arc bound: x[i,j] <= q[j]
                take = q[j] if cap_left >= q[j] else cap_left
                if np.isclose(take, 0.0, atol=TOL):
                    continue

                rc = row[j]  # negative by construction
                x_row[j] = take
                cap_left -= take
                var_part += rc * take

            # open i iff variable part + f[i] is negative; otherwise wipe tentative flow
            if x_row.any() and (f[i] + var_part < 0.0):
                a_rel[i] = 1
                L_val += float(f[i] + var_part)
            elif x_row.any():
                x_row.fill(0.0)

        # -------- Depot-side subproblems (independent over j) --------
        for j in range(inst.nJ):
            row = dtil[j]  # view
            neg_ks = np.nonzero(row < -TOL)[0]
            if neg_ks.size == 0:
                continue

            order = neg_ks[np.argsort(row[neg_ks])]
            cap_left = float(q[j])
            var_part = 0.0
            y_row = y_rel[j]  # view

            for k in order:
                if np.isclose(cap_left, 0.0, atol=TOL):
                    break

                # per-arc bound: y[j,k] <= r[k]
                take = r[k] if cap_left >= r[k] else cap_left
                if np.isclose(take, 0.0, atol=TOL):
                    continue

                rc = row[k]  # negative by construction
                y_row[k] = take
                cap_left -= take
                var_part += rc * take

            # open j iff variable part + g[j] is negative; otherwise wipe tentative flow
            if y_row.any() and (g[j] + var_part < 0.0):
                b_rel[j] = 1
                L_val += float(g[j] + var_part)
            elif y_row.any():
                y_row.fill(0.0)

        # -------- Subgradients --------
        sum_x_j = x_rel.sum(axis=0)  # (nJ,)
        sum_y_j = y_rel.sum(axis=1)  # (nJ,)
        sum_y_k = y_rel.sum(axis=0)  # (nK,)

        gA = sum_x_j - sum_y_j
        gB = sum_y_k - r

        return L_val, a_rel, b_rel, x_rel, y_rel, gA, gB

    # ------------------------------------------------------------------
    # Repair (CPLEX) to build a feasible primal UB given (a,b)
    # *Creates variables only on the open sets to speed up.*
    # ------------------------------------------------------------------
    def repair_ub(a_open: np.ndarray, b_open: np.ndarray) -> float | None:
        total_r = float(np.sum(inst.r))
        a_fix = a_open.astype(int).copy()
        b_fix = b_open.astype(int).copy()

        cap_p = float(np.dot(inst.p, a_fix))
        cap_q = float(np.dot(inst.q, b_fix))

        # ensure enough total capacity (open cheapest per unit capacity)
        if cap_p + 1e-9 < total_r:
            closed_plants = [
                i for i in range(inst.nI) if a_fix[i] == 0 and inst.p[i] > 0
            ]
            closed_plants.sort(key=lambda i: inst.f[i] / max(inst.p[i], 1e-12))
            for i in closed_plants:
                a_fix[i] = 1
                cap_p += float(inst.p[i])
                if cap_p + 1e-9 >= total_r:
                    break

        if cap_q + 1e-9 < total_r:
            closed_depots = [
                j for j in range(inst.nJ) if b_fix[j] == 0 and inst.q[j] > 0
            ]
            closed_depots.sort(key=lambda j: inst.g[j] / max(inst.q[j], 1e-12))
            for j in closed_depots:
                b_fix[j] = 1
                cap_q += float(inst.q[j])
                if cap_q + 1e-9 >= total_r:
                    break

        if cap_p + 1e-9 < total_r or cap_q + 1e-9 < total_r:
            a_fix[:] = 1
            b_fix[:] = 1

        I_open = [i for i in range(inst.nI) if a_fix[i] == 1]
        J_open = [j for j in range(inst.nJ) if b_fix[j] == 1]
        if not I_open or not J_open:
            return None

        mdl = Model(name="TSCFL_repair", log_output=False)
        mdl.parameters.threads = threads

        xR = {
            (i, j): mdl.continuous_var(lb=0.0, name=f"x_{i}_{j}")
            for i in I_open
            for j in J_open
        }
        yR = {
            (j, k): mdl.continuous_var(lb=0.0, name=f"y_{j}_{k}")
            for j in J_open
            for k in range(inst.nK)
        }

        # capacities (only open sets)
        mdl.add_constraints_(
            mdl.sum(xR[i, j] for j in J_open) <= inst.p[i] for i in I_open
        )
        mdl.add_constraints_(
            mdl.sum(yR[j, k] for k in range(inst.nK)) <= inst.q[j] for j in J_open
        )
        # balance at depots (only open depots)
        mdl.add_constraints_(
            mdl.sum(xR[i, j] for i in I_open)
            == mdl.sum(yR[j, k] for k in range(inst.nK))
            for j in J_open
        )
        # customer demands (sum over open depots)
        mdl.add_constraints_(
            mdl.sum(yR[j, k] for j in J_open) == inst.r[k] for k in range(inst.nK)
        )

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

    # ==============================
    # Main NDRC loop
    # ==============================
    for it in range(1, max_iter + 1):
        # (1) solve LRP(α,β) greedily
        L_k, a_k, b_k, x_k, y_k, gA_k, gB_k = solve_lrp_greedy(alpha, beta)

        # (2) LB/UB maintenance
        improved = L_k > L_best + 1e-12
        if improved:
            L_best = L_k
            since_improve = 0
        else:
            since_improve += 1

        if (it == 1) or improved or (it % 25 == 0):
            z_try = repair_ub(a_k, b_k)
            if (z_try is not None) and (z_try + 1e-8 < z_best):
                z_best = z_try
                a_inc = a_k.copy()
                b_inc = b_k.copy()

        # (3) CA/PA/CI & non-delayed zeroing for both families
        CA_a = np.where(np.abs(gA_k) > viol_tol)[0]
        PA_a = np.where(alpha != 0.0)[0]
        CI_a = np.setdiff1d(
            np.arange(inst.nJ), np.union1d(CA_a, PA_a), assume_unique=False
        )

        CA_b = np.where(np.abs(gB_k) > viol_tol)[0]
        PA_b = np.where(beta != 0.0)[0]
        CI_b = np.setdiff1d(
            np.arange(inst.nK), np.union1d(CA_b, PA_b), assume_unique=False
        )

        gA_nd = gA_k.copy()
        gB_nd = gB_k.copy()
        if CI_a.size:
            gA_nd[CI_a] = 0.0
        if CI_b.size:
            gB_nd[CI_b] = 0.0

        age_alpha[CA_a] = 0
        dropA = np.setdiff1d(PA_a, CA_a, assume_unique=False)
        if dropA.size:
            age_alpha[dropA] += 1
            to_zero = dropA[age_alpha[dropA] > dual_keep]
            if to_zero.size:
                alpha[to_zero] = 0.0
                age_alpha[to_zero] = 0

        age_beta[CA_b] = 0
        dropB = np.setdiff1d(PA_b, CA_b, assume_unique=False)
        if dropB.size:
            age_beta[dropB] += 1
            to_zero = dropB[age_beta[dropB] > dual_keep]
            if to_zero.size:
                beta[to_zero] = 0.0
                age_beta[to_zero] = 0

        # (4) stepsize and dual update
        denom = float(np.dot(gA_nd, gA_nd) + np.dot(gB_nd, gB_nd))
        if denom > 0.0:
            if np.isfinite(z_best):
                mu = gamma * max(z_best - L_k, 0.0) / denom
            else:
                mu = eps / denom
            alpha = alpha + mu * gA_nd
            beta = beta + mu * gB_nd

        # (5) epsilon schedule while UB unknown
        if (
            (not np.isfinite(z_best))
            and (since_improve >= stall_halve)
            and (eps > eps_min)
        ):
            eps = max(eps * 0.5, eps_min)
            since_improve = 0

        # (6) stopping by gap when UB exists
        if np.isfinite(z_best):
            gap = z_best - L_best
            if gap <= tol_stop * max(1.0, abs(z_best)):
                if log:
                    print(
                        f"[NDRC] it={it}  LRP(α,β)={L_k:.6f}  LB={L_best:.6f}  UB={z_best:.6f}  "
                        f"||g_nd||={np.sqrt(denom):.3e}  "
                        f"|CAα|={CA_a.size} |PAα|={PA_a.size} |CIα|={CI_a.size}  "
                        f"|CAβ|={CA_b.size} |PAβ|={PA_b.size} |CIβ|={CI_b.size}"
                    )
                break

        if log and (it % 5 == 0 or it == 1):
            print(
                f"[NDRC] it={it:4d}  LRP(α,β)={L_k:.6f}  LB={L_best:.6f}  "
                f"UB={(z_best if np.isfinite(z_best) else float('inf')):.6f}  "
                f"||g_nd||={np.sqrt(denom):.3e}  "
                f"|CAα|={CA_a.size} |PAα|={PA_a.size} |CIα|={CI_a.size}  "
                f"|CAβ|={CA_b.size} |PAβ|={PA_b.size} |CIβ|={CI_b.size}"
            )

    # last-chance UB if none found
    if not np.isfinite(z_best):
        fallback_a = (
            a_inc
            if a_inc.sum()
            else (a_k if "a_k" in locals() else np.zeros(inst.nI, dtype=int))
        )
        fallback_b = (
            b_inc
            if b_inc.sum()
            else (b_k if "b_k" in locals() else np.zeros(inst.nJ, dtype=int))
        )
        z_try = repair_ub(fallback_a, fallback_b)
        if z_try is not None:
            z_best = z_try


def main():
    """
    Rotina principal
    """
    PATH = "instances/tscfl/tscfl_11_50.txt"

    instance = TSCFLInstance.from_txt(PATH)

    solve_instance_rc(instance, log=True)

    return


if __name__ == "__main__":
    main()
