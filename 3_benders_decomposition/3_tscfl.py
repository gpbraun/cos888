"""
COS888

TSCFL com CPLEX — Benders (callbacks) + recuperação de fluxo

Gabriel Braun, 2025
"""

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Optional

import cplex.callbacks as cpx_cb
import numpy as np
from docplex.mp.callbacks.cb_mixin import ConstraintCallbackMixin
from docplex.mp.model import Model


@dataclass(frozen=True)
class TSCFLInstance:
    """
    Instância do TSCFL
    """

    nI: int  # |I| plantas
    nJ: int  # |J| depósitos
    nK: int  # |K| clientes

    f: np.ndarray  # f_i  = custo fixo da planta i
    g: np.ndarray  # g_j  = custo fixo do depósito j
    c: np.ndarray  # c_ij = custo unitário planta i -> depósito j
    d: np.ndarray  # d_jk = custo unitário depósito j -> cliente k
    p: np.ndarray  # p_i  = capacidade da planta i
    q: np.ndarray  # q_j  = capacidade do depósito j
    r: np.ndarray  # r_k  = demanda do cliente k

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
        Retorna: Instância a partir de um arquivo .txt
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


class _BendersWorkerDual:
    """
    Worker LP.
    """

    def __init__(self, inst: TSCFLInstance, *, log: bool = False) -> None:
        self.inst = inst
        self.log = log

        m = Model(name="TSCFL_worker_dual", log_output=log)
        self.m = m
        inf = m.infinity

        # variáveis: alpha, beta ≤ 0 ; gamma, delta livres
        self.alpha = m.continuous_var_list(inst.nI, lb=-inf, ub=0.0, name="alpha")
        self.beta = m.continuous_var_list(inst.nJ, lb=-inf, ub=0.0, name="beta")
        self.gamma = m.continuous_var_list(inst.nJ, lb=-inf, name="gamma")
        self.delta = m.continuous_var_list(inst.nK, lb=-inf, name="delta")

        m.add(self.alpha[i] + self.gamma[j] <= inst.c[i, j] for i, j in self.IJ)
        m.add(
            self.beta[j] - self.gamma[j] + self.delta[k] <= inst.d[j, k]
            for j, k in self.JK
        )

        # objeto será trocado a cada chamada
        m.maximize(0)

        # worker em callbacks: 1 thread
        m.context.cplex_parameters.threads = 1

    def solve_with(self, x_val: np.ndarray, y_val: np.ndarray):
        """
        Resolve o dual para (x,y) dados e retorna (θ, coef_x, coef_y, rhs).
        """
        inst = self.inst
        m = self.m

        # objetivo dependente de (x,y)
        obj = (
            m.sum(inst.p[i] * float(x_val[i]) * self.alpha[i] for i in inst.I)
            + m.sum(inst.q[j] * float(y_val[j]) * self.beta[j] for j in inst.J)
            + m.sum(inst.r[k] * self.delta[k] for k in inst.K)
        )
        m.set_objective("max", obj)

        sol = m.solve(log_output=self.log)
        if sol is None:
            raise RuntimeError("Worker dual não resolveu (inesperado).")

        theta = float(m.objective_value)

        # multiplicadores
        alpha_v = np.array([self.alpha[i].solution_value for i in inst.I], dtype=float)
        beta_v = np.array([self.beta[j].solution_value for j in inst.J], dtype=float)
        delta_v = np.array([self.delta[k].solution_value for k in inst.K], dtype=float)

        # coeficientes do corte
        coef_x = inst.p * alpha_v  # ≤ 0
        coef_y = inst.q * beta_v  # ≤ 0
        rhs = np.dot(inst.r, delta_v)

        return theta, coef_x, coef_y, rhs


class _BendersLazyCallback(ConstraintCallbackMixin, cpx_cb.LazyConstraintCallback):
    """
    Lazy: corta soluções incumbentes inteiras.
    """

    def __init__(self, env):
        cpx_cb.LazyConstraintCallback.__init__(self, env)
        ConstraintCallbackMixin.__init__(self)
        self.inst: Optional[TSCFLInstance] = None
        self.worker: Optional[_BendersWorkerDual] = None
        self.x = None
        self.y = None
        self.eta = None

    def __call__(self):
        try:
            sol = self.make_solution_from_vars(list(self.x) + list(self.y) + [self.eta])

            x_val = np.array([sol.get_value(v) for v in self.x], dtype=float)
            y_val = np.array([sol.get_value(v) for v in self.y], dtype=float)
            eta_val = sol.get_value(self.eta)

            theta, coef_x, coef_y, rhs = self.worker.solve_with(x_val, y_val)

            # comparação direta
            if theta > eta_val:
                cut_ct = self.eta >= rhs + sum(
                    coef_x[i] * self.x[i] for i in self.inst.I
                ) + sum(coef_y[j] * self.y[j] for j in self.inst.J)
                lhs, sense, rhs_num = self.linear_ct_to_cplex(cut_ct)
                self.add(lhs, sense, rhs_num)

        except Exception:
            pass


class _BendersUserCutCallback(ConstraintCallbackMixin, cpx_cb.UserCutCallback):
    """
    User cut: separa cortes em nós fracionários (global).
    """

    def __init__(self, env):
        cpx_cb.UserCutCallback.__init__(self, env)
        ConstraintCallbackMixin.__init__(self)
        self.inst: Optional[TSCFLInstance] = None
        self.worker: Optional[_BendersWorkerDual] = None
        self.x = None
        self.y = None
        self.eta = None

    def __call__(self):
        try:
            sol = self.make_complete_solution()

            x_val = np.array([sol.get_value(v) for v in self.x], dtype=float)
            y_val = np.array([sol.get_value(v) for v in self.y], dtype=float)
            eta_val = sol.get_value(self.eta)

            theta, coef_x, coef_y, rhs = self.worker.solve_with(x_val, y_val)

            # comparação direta
            if theta > eta_val:
                cut_ct = self.eta >= rhs + sum(
                    coef_x[i] * self.x[i] for i in self.inst.I
                ) + sum(coef_y[j] * self.y[j] for j in self.inst.J)
                lhs, sense, rhs_num = self.linear_ct_to_cplex(cut_ct)
                self.add(lhs, sense, rhs_num)

        except Exception:
            pass


class BendersTSCFL:
    """
    Resolve TSCFL via Benders (callbacks) com recuperação do fluxo.
    """

    def __init__(
        self,
        inst: TSCFLInstance,
        *,
        threads: int = 1,
        time_limit: int = 1000,
        log_output: bool = True,
    ) -> None:
        self.inst = inst
        self.time_limit = time_limit
        self.threads = threads
        self.log_output = log_output

    def solve(self) -> None:
        """
        Master + callbacks + recuperação de fluxo.
        """
        inst = self.inst
        I, J, K = inst.I, inst.J, inst.K
        demand_total = inst.r.sum()

        # master
        mdl = Model(
            name="TSCFL_Benders_Master",
            log_output=self.log_output,
            time_limit=self.time_limit,
        )

        # Traditional search (necessário p/ callbacks)
        mdl.parameters.mip.strategy.search = 1
        mdl.parameters.threads = self.threads

        # variáveis do master
        x = mdl.binary_var_list(inst.nI, name="x")
        y = mdl.binary_var_list(inst.nJ, name="y")
        eta = mdl.continuous_var(lb=0.0, name="eta")

        # capacidade agregada (viabilidade)
        mdl.add(mdl.sum(inst.p[i] * x[i] for i in I) >= demand_total)
        mdl.add(mdl.sum(inst.q[j] * y[j] for j in J) >= demand_total)

        # objetivo: custos fixos + eta
        mdl.minimize(
            mdl.sum(inst.f[i] * x[i] for i in I)
            + mdl.sum(inst.g[j] * y[j] for j in J)
            + eta
        )

        # worker dual
        worker = _BendersWorkerDual(inst, log=False)

        # callbacks
        lazy_cb = mdl.register_callback(_BendersLazyCallback)
        lazy_cb.inst = inst
        lazy_cb.worker = worker
        lazy_cb.x = x
        lazy_cb.y = y
        lazy_cb.eta = eta

        user_cb = mdl.register_callback(_BendersUserCutCallback)
        user_cb.inst = inst
        user_cb.worker = worker
        user_cb.x = x
        user_cb.y = y
        user_cb.eta = eta

        # solve master
        master_solution = mdl.solve(log_output=self.log_output)
        if master_solution is None:
            raise RuntimeError("Benders master não encontrou solução. Ative o log.")

        x_val = np.array([v.solution_value for v in x], dtype=float)
        y_val = np.array([v.solution_value for v in y], dtype=float)

        # log master
        print(
            f"[BENDERS] master: OBJ={master_solution.objective_value:.3f}  "
            f"eta={master_solution.get_value(eta):.3f}  "
            f"gap={mdl.solve_details.mip_relative_gap:.3%}  "
            f"nodes={mdl.solve_details.nb_nodes_processed}  "
            f"time={mdl.solve_details.time:.2f}s"
        )

        # recuperação de fluxo com aberturas fixadas
        flow = Model(name="TSCFL_FlowRecover", log_output=False)

        # variáveis
        u = flow.continuous_var_dict(inst.IJ, lb=0.0, name="u")
        v = flow.continuous_var_dict(inst.JK, lb=0.0, name="v")

        # capacidades
        flow.add(flow.sum(u[i, j] for j in J) <= inst.p[i] * round(x_val[i]) for i in I)
        flow.add(flow.sum(v[j, k] for k in K) <= inst.q[j] * round(y_val[j]) for j in J)
        # balanço no depósito
        flow.add(
            flow.sum(u[i, j] for i in I) == flow.sum(v[j, k] for k in K) for j in J
        )
        # demanda dos clientes
        flow.add(flow.sum(v[j, k] for j in J) == inst.r[k] for k in K)

        flow.minimize(
            flow.sum(inst.c[i, j] * u[i, j] for i in I for j in J)
            + flow.sum(inst.d[j, k] * v[j, k] for j in J for k in K)
        )
        flow_solution = flow.solve(log_output=self.log_output)
        if flow_solution is None:
            raise RuntimeError("LP de recuperação de fluxo inviável (inesperado).")

        # custos
        x_int = np.round(x_val).astype(int)
        y_int = np.round(y_val).astype(int)

        fixed_cost = np.dot(inst.f, x_int) + np.dot(inst.g, y_int)
        flow_cost = flow.objective_value
        total_cost = fixed_cost + flow_cost

        # log recuperação de fluxo
        print(f"[BENDERS] OBJ={total_cost:.3f}")

        return None


def main():
    """
    Rotina principal
    """
    PATH = "instances/tscfl/tscfl_11_50.txt"

    inst = TSCFLInstance.from_txt(PATH)

    solver = BendersTSCFL(inst, time_limit=None, threads=None, log_output=True)
    solver.solve()

    return


if __name__ == "__main__":
    main()
