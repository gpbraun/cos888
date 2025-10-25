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
    max_iter: int = 10_000,
    threads: int = 20,
    # Polyak (when UB exists) / epsilon fallback
    gamma: float = 1.0,
    eps0: float = 1.0,
    stall_halve: int = 200,
    eps_min: float = 1e-12,
    # CA/PA/CI
    viol_tol: float = 1e-4,
    dual_keep: int = 10,
    # LRP robustness / debug
    per_iter_timelimit: float | None = None,  # e.g., 1.0 for debugging; None for exact
    mip_node_limit: int | None = None,  # e.g., 5_000 for debugging; None for exact
    use_x_vub: bool = True,  # try False if you see stalls
    use_y_vub: bool = True,  # usually cheap to keep True
    log_lrp: bool = False,  # show CPLEX log for the LRP
    # stopping / logs
    tol_stop: float = 1e-6,
    log: bool = False,
) -> float:
    """
    NDRC for TSCFL dualizing BOTH families:
      - Satellite balances:   sum_i x[i,j] = sum_k y[j,k]  (α_j free)
      - Customer demands:     sum_j y[j,k] = r[k]          (β_k free)
    LRP solved by CPLEX each iteration. Strengthening VUBs kept inside LRP.
    """
    from typing import Optional

    import numpy as np
    from docplex.mp.model import Model

    TOL = 1e-12

    # --- dual state
    alpha = np.zeros(inst.nJ, dtype=float)  # for satellite balances
    beta = np.zeros(inst.nK, dtype=float)  # for customer demands
    age_alpha = np.zeros(inst.nJ, dtype=int)
    age_beta = np.zeros(inst.nK, dtype=int)

    # --- bounds
    L_best = -np.inf
    z_best = np.inf
    a_inc = np.zeros(inst.nI, dtype=int)
    b_inc = np.zeros(inst.nJ, dtype=int)

    # epsilon schedule
    eps = float(eps0)
    since_improve = 0

    # --- Build LRP once
    LRP = Model(name="TSCFL_LRP", log_output=log_lrp)
    LRP.parameters.threads = threads
    if mip_node_limit is not None:
        LRP.parameters.mip.limits.nodes = int(mip_node_limit)

    a = LRP.binary_var_dict(inst.I, name="a")
    b = LRP.binary_var_dict(inst.J, name="b")
    x = LRP.continuous_var_dict(inst.IJ, lb=0.0, name="x")
    y = LRP.continuous_var_dict(inst.JK, lb=0.0, name="y")

    # capacities
    LRP.add_constraints_(
        LRP.sum(x[i, j] for j in inst.J) <= inst.p[i] * a[i] for i in inst.I
    )
    LRP.add_constraints_(
        LRP.sum(y[j, k] for k in inst.K) <= inst.q[j] * b[j] for j in inst.J
    )

    # strengthening (toggle-able)
    if use_x_vub:
        LRP.add_constraints_(x[i, j] <= inst.q[j] * b[j] for i, j in inst.IJ)
    if use_y_vub:
        LRP.add_constraints_(y[j, k] <= inst.r[k] * b[j] for j, k in inst.JK)

    def set_lrp_objective(alpha_vec: np.ndarray, beta_vec: np.ndarray):
        # reduced costs:
        #   ĉ_ij = c_ij + α_j
        #   đ_jk = d_jk - α_j + β_k
        stage1 = LRP.sum((inst.c[i, j] + alpha_vec[j]) * x[i, j] for i, j in inst.IJ)
        stage2 = LRP.sum(
            (inst.d[j, k] - alpha_vec[j] + beta_vec[k]) * y[j, k] for j, k in inst.JK
        )
        fixed = LRP.sum(inst.f[i] * a[i] for i in inst.I) + LRP.sum(
            inst.g[j] * b[j] for j in inst.J
        )
        LRP.minimize(fixed + stage1 + stage2)  # constant −∑β_k r_k added after solve

    # simple warm starts for a,b (optional)
    a_ws = {i: 0 for i in inst.I}
    b_ws = {j: 0 for j in inst.J}

    def repair_ub(a_open: np.ndarray, b_open: np.ndarray) -> Optional[float]:
        total_r = float(np.sum(inst.r))
        a_fix = a_open.astype(int).copy()
        b_fix = b_open.astype(int).copy()

        cap_p = float(np.dot(inst.p, a_fix))
        cap_q = float(np.dot(inst.q, b_fix))

        if cap_p + 1e-9 < total_r:
            closed_plants = [i for i in inst.I if a_fix[i] == 0 and inst.p[i] > 0]
            closed_plants.sort(key=lambda i: inst.f[i] / max(inst.p[i], 1e-12))
            for i in closed_plants:
                a_fix[i] = 1
                cap_p += float(inst.p[i])
                if cap_p + 1e-9 >= total_r:
                    break

        if cap_q + 1e-9 < total_r:
            closed_depots = [j for j in inst.J if b_fix[j] == 0 and inst.q[j] > 0]
            closed_depots.sort(key=lambda j: inst.g[j] / max(inst.q[j], 1e-12))
            for j in closed_depots:
                b_fix[j] = 1
                cap_q += float(inst.q[j])
                if cap_q + 1e-9 >= total_r:
                    break

        if cap_p + 1e-9 < total_r or cap_q + 1e-9 < total_r:
            a_fix[:] = 1
            b_fix[:] = 1

        mdl = Model(name="TSCFL_repair", log_output=False)
        mdl.parameters.threads = threads
        xR = mdl.continuous_var_dict(inst.IJ, lb=0.0, name="x")
        yR = mdl.continuous_var_dict(inst.JK, lb=0.0, name="y")

        mdl.add_constraints_(
            mdl.sum(xR[i, j] for j in inst.J) <= inst.p[i] * a_fix[i] for i in inst.I
        )
        mdl.add_constraints_(
            mdl.sum(yR[j, k] for k in inst.K) <= inst.q[j] * b_fix[j] for j in inst.J
        )
        mdl.add_constraints_(
            mdl.sum(xR[i, j] for i in inst.I) == mdl.sum(yR[j, k] for k in inst.K)
            for j in inst.J
        )
        mdl.add_constraints_(
            mdl.sum(yR[j, k] for j in inst.J) == inst.r[k] for k in inst.K
        )

        flow_cost = mdl.sum(inst.c[i, j] * xR[i, j] for i, j in inst.IJ) + mdl.sum(
            inst.d[j, k] * yR[j, k] for j, k in inst.JK
        )
        fixed_cost = float(np.dot(inst.f, a_fix) + np.dot(inst.g, b_fix))
        mdl.minimize(flow_cost + fixed_cost)
        sol = mdl.solve()
        if not sol:
            return None
        return float(sol.objective_value)

    # =============== main loop ===============
    for it in range(1, max_iter + 1):
        if log and (it % 50 == 1):  # early heartbeat so you see it’s alive
            print(f"[NDRC] it={it:4d}  (solving LRP...)")

        # (1) solve LRP(α,β)
        set_lrp_objective(alpha, beta)
        for i in inst.I:
            a[i].start = a_ws[i]
        for j in inst.J:
            b[j].start = b_ws[j]

        if per_iter_timelimit is None:
            sol = LRP.solve()
        else:
            sol = LRP.solve(time_limit=float(per_iter_timelimit))

        if not sol:
            # If you set a time limit, CPLEX may stop without a solution.
            # In that case, just shrink eps to try different multipliers.
            if (not np.isfinite(z_best)) and (eps > eps_min):
                eps = max(eps * 0.5, eps_min)
            if log:
                print(f"[NDRC] it={it:4d}  LRP returned no solution; eps={eps:g}")
            continue

        val_sub = float(sol.objective_value)
        L_k = val_sub - float(np.dot(beta, inst.r))  # subtract constant −∑ β_k r_k

        # extract pattern
        a_k = np.array(
            [int(round(a[i].solution_value or 0.0)) for i in inst.I], dtype=int
        )
        b_k = np.array(
            [int(round(b[j].solution_value or 0.0)) for j in inst.J], dtype=int
        )
        for i in inst.I:
            a_ws[i] = a_k[i]
        for j in inst.J:
            b_ws[j] = b_k[j]

        # subgradients
        sum_x_j = np.zeros(inst.nJ, dtype=float)
        sum_y_j = np.zeros(inst.nJ, dtype=float)
        sum_y_k = np.zeros(inst.nK, dtype=float)
        for i, j in inst.IJ:
            xv = x[i, j].solution_value or 0.0
            sum_x_j[j] += xv
        for j, k in inst.JK:
            yv = y[j, k].solution_value or 0.0
            sum_y_j[j] += yv
            sum_y_k[k] += yv

        g_alpha = sum_x_j - sum_y_j
        g_beta = sum_y_k - inst.r

        # LB / UB maintenance
        improved = L_k > L_best + 1e-12
        if improved:
            L_best = L_k
            since_improve = 0
        else:
            since_improve += 1

        if (k == 1) or improved or (k % 25 == 0):
            z_try = repair_ub(a_k, b_k)
            if (z_try is not None) and (z_try + 1e-8 < z_best):
                z_best = z_try
                a_inc = a_k.copy()
                b_inc = b_k.copy()

        # (2) CA/PA/CI for both families; non-delayed zeroing
        CA_a = np.where(np.abs(g_alpha) > viol_tol)[0]
        PA_a = np.where(alpha != 0.0)[0]
        CI_a = np.setdiff1d(
            np.arange(inst.nJ), np.union1d(CA_a, PA_a), assume_unique=False
        )

        CA_b = np.where(np.abs(g_beta) > viol_tol)[0]
        PA_b = np.where(beta != 0.0)[0]
        CI_b = np.setdiff1d(
            np.arange(inst.nK), np.union1d(CA_b, PA_b), assume_unique=False
        )

        gA_nd = g_alpha.copy()
        gB_nd = g_beta.copy()
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

        # (3) stepsize update
        denom = float(np.dot(gA_nd, gA_nd) + np.dot(gB_nd, gB_nd))
        if denom > 0.0:
            if np.isfinite(z_best):
                mu = gamma * max(z_best - L_k, 0.0) / denom
            else:
                mu = eps / denom
            alpha = alpha + mu * gA_nd
            beta = beta + mu * gB_nd

        # epsilon halving while no UB exists
        if (
            (not np.isfinite(z_best))
            and (since_improve >= stall_halve)
            and (eps > eps_min)
        ):
            eps = max(eps * 0.5, eps_min)
            since_improve = 0

        # stopping by gap
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

        if log and (it % 50 == 0 or k == 1):
            print(
                f"[NDRC] it={it:4d}  LRP(α,β)={L_k:.6f}  LB={L_best:.6f}  "
                f"UB={(z_best if np.isfinite(z_best) else float('inf')):.6f}  "
                f"||g_nd||={np.sqrt(denom):.3e}  "
                f"|CAα|={CA_a.size} |PAα|={PA_a.size} |CIα|={CI_a.size}  "
                f"|CAβ|={CA_b.size} |PAβ|={PA_b.size} |CIβ|={CI_b.size}"
            )

    # last-chance UB
    if not np.isfinite(z_best):
        # use best/last patterns we saw
        fallback_a = a_inc if a_inc.sum() else a_k
        fallback_b = b_inc if b_inc.sum() else b_k
        z_try = repair_ub(fallback_a, fallback_b)
        if z_try is not None:
            z_best = z_try

    return float(z_best) if np.isfinite(z_best) else np.inf


def main():
    """
    Rotina principal
    """
    PATH = "instances/tscfl/tscfl_11_50.txt"

    instance = TSCFLInstance.from_txt(PATH)

    print(f"nI={instance.nI}, nJ={instance.nJ}, nK={instance.nK}")

    # obj = solve_instance(instance)
    # print(f"objective: {obj}")

    obj_rac = solve_instance_rc(instance, log=True)
    print(f"objective (relax and cut): {obj_rac}")

    # print(f"GAP: {100.0 * (obj_rac - obj) / obj:.6f}%")

    return


if __name__ == "__main__":
    main()
