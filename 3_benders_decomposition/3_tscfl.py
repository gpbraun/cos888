"""
COS888

TSCFL com CPLEX — Benders (callbacks) + full MIP + flow recovery

Gabriel Braun, 2025
"""

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, Optional

import cplex.callbacks as cpx_cb
import numpy as np
from docplex.mp.callbacks.cb_mixin import ConstraintCallbackMixin
from docplex.mp.model import Model

# =============================== Instance ===============================


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


# =============================== Full MIP (baseline) ===============================


def solve_instance(inst: TSCFLInstance, log_output: bool = True):
    """
    Resolve a instância TSCFL usando o CPLEX (modelo completo).
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
    # VUBs (sem prejuízo)
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


# ========================== Worker LP (Dual of flow subproblem) ==========================


class _BendersWorkerDual:
    """
    Worker LP (dual correto do subproblema de fluxo).
    Primal (min): duas ≤, duas =, u,v ≥ 0
      => Dual (max):
        variáveis: alpha_i ≤ 0 , beta_j ≤ 0 , gamma_j livre , delta_k livre
        restrições:
            alpha_i + gamma_j            ≤ c_ij
            beta_j  - gamma_j + delta_k  ≤ d_jk
        objetivo:
            max  sum_i p_i x_i * alpha_i + sum_j q_j y_j * beta_j + sum_k r_k * delta_k

    O corte de Benders é:
        eta >= [sum_k r_k delta_k*] + sum_i (p_i alpha_i*) x_i + sum_j (q_j beta_j*) y_j
    (coeficientes nas inclinações são negativos, pois alpha*, beta* ≤ 0)
    """

    def __init__(self, inst: TSCFLInstance, log: bool = False):
        self.inst = inst
        self.log = log
        self.m = Model(name="TSCFL_worker_dual", log_output=log)

        I, J, K = inst.I, inst.J, inst.K
        inf = self.m.infinity

        # alpha, beta ≤ 0  ;  gamma, delta livres
        self.alpha = self.m.continuous_var_list(inst.nI, lb=-inf, ub=0.0, name="alpha")
        self.beta = self.m.continuous_var_list(inst.nJ, lb=-inf, ub=0.0, name="beta")
        self.delta = self.m.continuous_var_list(inst.nK, lb=-inf, name="delta")
        self.gamma = self.m.continuous_var_list(inst.nJ, lb=-inf, name="gamma")

        # alpha_i + gamma_j <= c_ij
        for i in I:
            for j in J:
                self.m.add(self.alpha[i] + self.gamma[j] <= float(inst.c[i, j]))

        # beta_j - gamma_j + delta_k <= d_jk
        for j in J:
            for k in K:
                self.m.add(
                    self.beta[j] - self.gamma[j] + self.delta[k] <= float(inst.d[j, k])
                )

        # dummy objective (será trocado a cada chamada)
        self.m.maximize(0)

        # Worker seguro em callback: 1 thread
        self.m.context.cplex_parameters.threads = 1

    def solve_with(self, x_val: np.ndarray, y_val: np.ndarray):
        inst = self.inst

        # objetivo para (x,y) atuais
        obj = (
            self.m.sum(inst.p[i] * float(x_val[i]) * self.alpha[i] for i in inst.I)
            + self.m.sum(inst.q[j] * float(y_val[j]) * self.beta[j] for j in inst.J)
            + self.m.sum(inst.r[k] * self.delta[k] for k in inst.K)
        )
        self.m.set_objective("max", obj)

        sol = self.m.solve(log_output=self.log)
        if sol is None:
            raise RuntimeError("Worker dual did not solve (unexpected).")

        theta = float(self.m.objective_value)

        # multiplicadores
        alpha_v = np.array([self.alpha[i].solution_value for i in inst.I], dtype=float)
        beta_v = np.array([self.beta[j].solution_value for j in inst.J], dtype=float)
        delta_v = np.array([self.delta[k].solution_value for k in inst.K], dtype=float)

        # Coeficientes do corte (notar sinais):
        coef_x = inst.p * alpha_v  # (≤ 0)
        coef_y = inst.q * beta_v  # (≤ 0)
        rhs = float(np.dot(inst.r, delta_v))  # constante

        return theta, coef_x, coef_y, rhs


# =============================== Callbacks ===============================


class _BendersLazyCallback(ConstraintCallbackMixin, cpx_cb.LazyConstraintCallback):
    def __init__(self, env):
        cpx_cb.LazyConstraintCallback.__init__(self, env)
        ConstraintCallbackMixin.__init__(self)
        self.inst = None
        self.worker = None
        self.x = None
        self.y = None
        self.eta = None
        self.eps = 1e-6

    def __call__(self):
        try:
            # incumbent candidate
            sol = self.make_solution_from_vars(list(self.x) + list(self.y) + [self.eta])
            x_val = np.array([sol.get_value(v) for v in self.x], dtype=float)
            y_val = np.array([sol.get_value(v) for v in self.y], dtype=float)
            eta_val = float(sol.get_value(self.eta))

            theta, coef_x, coef_y, rhs = self.worker.solve_with(x_val, y_val)

            if theta - eta_val > self.eps:
                cut_ct = self.eta >= rhs + sum(
                    float(coef_x[i]) * self.x[i] for i in self.inst.I
                ) + sum(float(coef_y[j]) * self.y[j] for j in self.inst.J)
                lhs, sense, rhs_num = self.linear_ct_to_cplex(cut_ct)
                self.add(lhs, sense, rhs_num)  # corta o incumbente
        except Exception:
            # Protege a execução do solver; se algo der errado, não aborta o MIP
            pass


class _BendersUserCutCallback(ConstraintCallbackMixin, cpx_cb.UserCutCallback):
    def __init__(self, env):
        cpx_cb.UserCutCallback.__init__(self, env)
        ConstraintCallbackMixin.__init__(self)
        self.inst = None
        self.worker = None
        self.x = None
        self.y = None
        self.eta = None
        self.eps = 1e-6

    def __call__(self):
        try:
            # valores fracionários do nó
            sol = self.make_complete_solution()
            x_val = np.array([sol.get_value(v) for v in self.x], dtype=float)
            y_val = np.array([sol.get_value(v) for v in self.y], dtype=float)
            eta_val = float(sol.get_value(self.eta))

            theta, coef_x, coef_y, rhs = self.worker.solve_with(x_val, y_val)

            if theta - eta_val > self.eps:
                cut_ct = self.eta >= rhs + sum(
                    float(coef_x[i]) * self.x[i] for i in self.inst.I
                ) + sum(float(coef_y[j]) * self.y[j] for j in self.inst.J)
                lhs, sense, rhs_num = self.linear_ct_to_cplex(cut_ct)
                self.add(lhs, sense, rhs_num)  # global user cut
        except Exception:
            pass


# =============================== Master (Benders) ===============================


def solve_instance_benders(
    inst: TSCFLInstance,
    time_limit: Optional[float] = None,
    separate_fractional: bool = True,
    threads: Optional[int] = None,
    eps: float = 1e-6,
    log: bool = True,
) -> Dict[str, Any]:
    """
    Benders via callbacks (estilo ilobendersatsp2.cpp):
      - Master: x (plantas), y (depósitos), eta
      - Worker (dual) construído uma vez, reutilizado nas callbacks
      - Lazy (incumbente inteiro) + opcionalmente User cuts (fracionário)
      - Busca tradicional (obrigatória para callbacks legados)
    """
    I, J, K = inst.I, inst.J, inst.K
    demand_total = float(inst.r.sum())

    mdl = Model(name="TSCFL_Benders_Master", log_output=log)

    # parâmetros: Traditional search + time/threads se informados
    mdl.parameters.mip.strategy.search = 1  # Traditional (necessário p/ callbacks)
    if time_limit is not None:
        mdl.parameters.timelimit = float(time_limit)
    if threads is not None:
        mdl.parameters.threads = int(threads)

    # variáveis do master
    x = mdl.binary_var_list(inst.nI, name="x")
    y = mdl.binary_var_list(inst.nJ, name="y")
    eta = mdl.continuous_var(lb=0, name="eta")

    # capacidade agregada (garante viabilidade do subproblema)
    mdl.add(mdl.sum(inst.p[i] * x[i] for i in I) >= demand_total, "cap_plants")
    mdl.add(mdl.sum(inst.q[j] * y[j] for j in J) >= demand_total, "cap_depots")

    # objetivo: fixo + eta
    mdl.minimize(
        mdl.sum(inst.f[i] * x[i] for i in I)
        + mdl.sum(inst.g[j] * y[j] for j in J)
        + eta
    )

    # worker dual (uma vez só)
    worker = _BendersWorkerDual(inst, log=False)

    # callbacks
    lazy_cb = mdl.register_callback(_BendersLazyCallback)
    lazy_cb.inst = inst
    lazy_cb.worker = worker
    lazy_cb.x = x
    lazy_cb.y = y
    lazy_cb.eta = eta
    lazy_cb.eps = eps

    if separate_fractional:
        user_cb = mdl.register_callback(_BendersUserCutCallback)
        user_cb.inst = inst
        user_cb.worker = worker
        user_cb.x = x
        user_cb.y = y
        user_cb.eta = eta
        user_cb.eps = eps

    # solve master
    msol = mdl.solve(log_output=log)
    if msol is None:
        raise RuntimeError(
            "No solution found by Benders master. Enable log to inspect callbacks."
        )

    x_val = np.array([v.solution_value for v in x], dtype=float)
    y_val = np.array([v.solution_value for v in y], dtype=float)

    # ================== Flow recovery (fixa aberturas) ==================
    flow = Model(name="TSCFL_FlowRecover", log_output=False)
    u = {(i, j): flow.continuous_var(lb=0, name=f"u_{i}_{j}") for i in I for j in J}
    v = {(j, k): flow.continuous_var(lb=0, name=f"v_{j}_{k}") for j in J for k in K}

    # capacidades
    for i in I:
        flow.add(flow.sum(u[i, j] for j in J) <= inst.p[i] * round(x_val[i]))
    for j in J:
        flow.add(flow.sum(v[j, k] for k in K) <= inst.q[j] * round(y_val[j]))

    # balanço no depósito
    for j in J:
        flow.add(flow.sum(u[i, j] for i in I) - flow.sum(v[j, k] for k in K) == 0)

    # demanda
    for k in K:
        flow.add(flow.sum(v[j, k] for j in J) == inst.r[k])

    flow.minimize(
        flow.sum(inst.c[i, j] * u[i, j] for i in I for j in J)
        + flow.sum(inst.d[j, k] * v[j, k] for j in J for k in K)
    )
    fsol = flow.solve(log_output=log)
    if fsol is None:
        raise RuntimeError("Flow recovery LP infeasible (unexpected).")

    u_mat = np.zeros((inst.nI, inst.nJ))
    v_mat = np.zeros((inst.nJ, inst.nK))
    for (i, j), var in u.items():
        u_mat[i, j] = var.solution_value
    for (j, k), var in v.items():
        v_mat[j, k] = var.solution_value

    # custos
    x_int = np.round(x_val).astype(int)
    y_int = np.round(y_val).astype(int)
    fixed_cost = float(np.dot(inst.f, x_int) + np.dot(inst.g, y_int))
    transport_cost = float(flow.objective_value)
    total_cost = fixed_cost + transport_cost

    return dict(
        x=x_int,
        y=y_int,
        u=u_mat,
        v=v_mat,
        fixed_cost=fixed_cost,
        transport_cost=transport_cost,
        obj=total_cost,
        benders_eta=float(msol.get_value(eta)),
        gap=mdl.solve_details.mip_relative_gap,
        nodes=mdl.solve_details.nb_nodes_processed,
        time=mdl.solve_details.time,
    )


# =============================== CLI quick test ===============================


def main():
    PATH = "instances/tscfl/tscfl_11_50.txt"
    inst = TSCFLInstance.from_txt(PATH)

    print(f"nI={inst.nI}, nJ={inst.nJ}, nK={inst.nK}")

    # Baseline full MIP (optional)
    # obj_full = solve_instance(inst, log_output=False)
    # print("Full MIP OBJ:", obj_full)

    benders = solve_instance_benders(
        inst,
        time_limit=None,
        separate_fractional=True,
        threads=None,
        eps=1e-6,
        log=True,
    )
    print("Benders OBJ:", benders["obj"])


if __name__ == "__main__":
    main()
