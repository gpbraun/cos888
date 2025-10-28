"""
COS888

TSCFL por Decomposição de Benders

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
    Worker LP (dual do subproblema de fluxo).
    """

    def __init__(self, inst: TSCFLInstance, *, log: bool = False) -> None:
        self.inst = inst
        self.log = log

        mdl = Model(name="TSCFL_worker_dual", log_output=log)
        self.mdl = mdl
        inf = mdl.infinity

        # variáveis: alpha, beta ≤ 0 ; gamma, delta livres
        self.alpha = mdl.continuous_var_list(inst.nI, lb=-inf, ub=0.0, name="alpha")
        self.beta = mdl.continuous_var_list(inst.nJ, lb=-inf, ub=0.0, name="beta")
        self.gamma = mdl.continuous_var_list(inst.nJ, lb=-inf, name="gamma")
        self.delta = mdl.continuous_var_list(inst.nK, lb=-inf, name="delta")

        # alpha_i + gamma_j ≤ c_ij
        for i in inst.I:
            for j in inst.J:
                mdl.add(self.alpha[i] + self.gamma[j] <= float(inst.c[i, j]))

        # beta_j - gamma_j + delta_k ≤ d_jk
        for j in inst.J:
            for k in inst.K:
                mdl.add(
                    self.beta[j] - self.gamma[j] + self.delta[k] <= float(inst.d[j, k])
                )

        mdl.maximize(0)

        # worker em callback: 1 thread por segurança
        mdl.context.cplex_parameters.threads = 1

    def solve_with(self, a_val: np.ndarray, b_val: np.ndarray):
        """
        Resolve o dual para (a, b) dados. Retorna (theta, coef_a, coef_b, rhs).
        """
        inst = self.inst
        mdl = self.mdl

        # objetivo dependente de (a, b)
        obj = (
            mdl.sum(inst.p[i] * a_val[i] * self.alpha[i] for i in inst.I)
            + mdl.sum(inst.q[j] * b_val[j] * self.beta[j] for j in inst.J)
            + mdl.sum(inst.r[k] * self.delta[k] for k in inst.K)
        )
        mdl.set_objective("max", obj)

        sol = mdl.solve(log_output=self.log)
        if sol is None:
            raise RuntimeError("Worker dual não resolveu (inesperado).")

        theta = mdl.objective_value

        # multiplicadores ótimos
        alpha_v = np.array([self.alpha[i].solution_value for i in inst.I], dtype=float)
        beta_v = np.array([self.beta[j].solution_value for j in inst.J], dtype=float)
        delta_v = np.array([self.delta[k].solution_value for k in inst.K], dtype=float)

        # coeficientes do corte (notar sinais: alpha,beta ≤ 0)
        coef_a = inst.p * alpha_v  # coeficientes para a_i
        coef_b = inst.q * beta_v  # coeficientes para b_j
        rhs = np.dot(inst.r, delta_v)

        return theta, coef_a, coef_b, rhs


class _BendersLazyCallback(ConstraintCallbackMixin, cpx_cb.LazyConstraintCallback):
    """
    Lazy: corta incumbentes inteiros (a, b) com η subestimando φ(a, b).
    """

    def __init__(self, env):
        cpx_cb.LazyConstraintCallback.__init__(self, env)
        ConstraintCallbackMixin.__init__(self)
        self.inst: Optional[TSCFLInstance] = None
        self.worker: Optional[_BendersWorkerDual] = None
        self.a = None
        self.b = None
        self.eta = None

    def __call__(self):
        try:
            sol = self.make_solution_from_vars(list(self.a) + list(self.b) + [self.eta])

            a_val = np.array([sol.get_value(v) for v in self.a], dtype=float)
            b_val = np.array([sol.get_value(v) for v in self.b], dtype=float)
            eta_val = float(sol.get_value(self.eta))

            theta, coef_a, coef_b, rhs = self.worker.solve_with(a_val, b_val)

            # violação direta
            if theta > eta_val:
                cut_ct = self.eta >= rhs + sum(
                    float(coef_a[i]) * self.a[i] for i in self.inst.I
                ) + sum(float(coef_b[j]) * self.b[j] for j in self.inst.J)
                lhs, sense, rhs_num = self.linear_ct_to_cplex(cut_ct)
                self.add(lhs, sense, rhs_num)

        except Exception:
            pass


class _BendersUserCutCallback(ConstraintCallbackMixin, cpx_cb.UserCutCallback):
    """
    User cut: separa cortes globalmente em nós fracionários.
    """

    def __init__(self, env):
        cpx_cb.UserCutCallback.__init__(self, env)
        ConstraintCallbackMixin.__init__(self)
        self.inst: Optional[TSCFLInstance] = None
        self.worker: Optional[_BendersWorkerDual] = None
        self.a = None
        self.b = None
        self.eta = None

    def __call__(self):
        try:
            sol = self.make_complete_solution()

            a_val = np.array([sol.get_value(v) for v in self.a], dtype=float)
            b_val = np.array([sol.get_value(v) for v in self.b], dtype=float)
            eta_val = sol.get_value(self.eta)

            theta, coef_a, coef_b, rhs = self.worker.solve_with(a_val, b_val)

            if theta > eta_val:
                cut_ct = self.eta >= rhs + sum(
                    coef_a[i] * self.a[i] for i in self.inst.I
                ) + sum(coef_b[j] * self.b[j] for j in self.inst.J)
                lhs, sense, rhs_num = self.linear_ct_to_cplex(cut_ct)
                self.add(lhs, sense, rhs_num)

        except Exception:
            pass


class BendersTSCFL:
    """
    Resolve a instância usando o método: Benders Decomposition.
    """

    def __init__(
        self,
        inst: TSCFLInstance,
        *,
        time_limit: Optional[float] = None,
        threads: Optional[int] = None,
        log_output: bool = True,
    ) -> None:
        self.inst = inst
        self.time_limit = time_limit
        self.threads = threads
        self.log_output = log_output

    def solve(self) -> None:
        """
        Master + callbacks + recuperação de fluxo (x,y).
        Imprime informações relevantes; não retorna nada.
        """
        # MASTER BENDERS
        mdl = Model(name="TSCFL_Benders_Master", log_output=self.log_output)
        mdl.parameters.mip.strategy.search = 1  # Traditional (para callbacks)

        # binárias a,b e variável eta
        a = mdl.binary_var_list(self.inst.nI, name="a")
        b = mdl.binary_var_list(self.inst.nJ, name="b")
        eta = mdl.continuous_var(lb=0.0, name="eta")

        # capacidade
        mdl.add(
            mdl.sum(self.inst.p[i] * a[i] for i in self.inst.I) >= self.inst.r.sum()
        )
        mdl.add(
            mdl.sum(self.inst.q[j] * b[j] for j in self.inst.J) >= self.inst.r.sum()
        )

        # OBJETIVO
        cost_fixed1 = mdl.sum(self.inst.f[i] * a[i] for i in self.inst.I)
        cost_fixed2 = mdl.sum(self.inst.g[j] * b[j] for j in self.inst.J)

        mdl.minimize(cost_fixed1 + cost_fixed2 + eta)

        # worker dual
        worker = _BendersWorkerDual(self.inst, log=False)

        # callbacks (sempre: lazy + user)
        lazy_cb = mdl.register_callback(_BendersLazyCallback)
        lazy_cb.inst = self.inst
        lazy_cb.worker = worker
        lazy_cb.a = a
        lazy_cb.b = b
        lazy_cb.eta = eta

        user_cb = mdl.register_callback(_BendersUserCutCallback)
        user_cb.inst = self.inst
        user_cb.worker = worker
        user_cb.a = a
        user_cb.b = b
        user_cb.eta = eta

        # solve master
        msol = mdl.solve(log_output=self.log_output)

        print(
            f"[BENDERS] master: OBJ={msol.objective_value:.3f}  "
            f"eta={float(msol.get_value(eta)):.3f}  "
            f"gap={mdl.solve_details.mip_relative_gap:.3%}  "
            f"nodes={mdl.solve_details.nb_nodes_processed}  "
            f"time={mdl.solve_details.time:.2f}s"
        )

        # RECUPERAÇÃO DE FLUXO
        a_val = np.round([v.solution_value for v in a]).astype(int)
        b_val = np.round([v.solution_value for v in b]).astype(int)

        x_val, y_val = self.recover_flows(a_val, b_val)

    def recover_flows(
        self, a_val: np.ndarray, b_val: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Resolve um LP de transporte com aberturas fixadas para materializar os fluxos.
        """
        mdl = Model(name="TSCFL_FlowRecover", log_output=False)

        # VARIÁVEIS
        #   x_ij = fluxo: planta i -> satélite j
        #   y_jk = fluxo: satélite j -> cliente k
        x = mdl.continuous_var_dict(self.inst.IJ, lb=0.0, name="x")  # i→j
        y = mdl.continuous_var_dict(self.inst.JK, lb=0.0, name="y")  # j→k

        # Capacidades
        mdl.add_constraints_(
            mdl.sum(x[i, j] for j in self.inst.J) <= self.inst.p[i] * a_val[i]
            for i in self.inst.I
        )
        mdl.add_constraints_(
            mdl.sum(y[j, k] for k in self.inst.K) <= self.inst.q[j] * b_val[j]
            for j in self.inst.J
        )

        # Balanço nos depósitos
        mdl.add_constraints_(
            mdl.sum(x[i, j] for i in self.inst.I)
            == mdl.sum(y[j, k] for k in self.inst.K)
            for j in self.inst.J
        )
        # Demandas
        mdl.add_constraints_(
            mdl.sum(y[j, k] for j in self.inst.J) == self.inst.r[k] for k in self.inst.K
        )

        # OBJETIVO
        cost_flow1 = mdl.sum(self.inst.c[i, j] * x[i, j] for i, j in self.inst.IJ)
        cost_flow2 = mdl.sum(self.inst.d[j, k] * y[j, k] for j, k in self.inst.JK)

        mdl.minimize(cost_flow1 + cost_flow2)
        mdl.solve()

        # Monta matrizes x,y
        x_val = np.zeros((self.inst.nI, self.inst.nJ), dtype=float)
        y_val = np.zeros((self.inst.nJ, self.inst.nK), dtype=float)
        for i, j in self.inst.IJ:
            x_val[i, j] = x[i, j].solution_value
        for j, k in self.inst.JK:
            y_val[j, k] = y[j, k].solution_value

        return x_val, y_val


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
