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


def solve_instance(inst: TSCFLInstance) -> None:
    """
    Resolve a instância TSCFL (two-stage) usando o CPLEX.
    """
    mdl = Model(name="TSCFL", log_output=True)

    # VARIÁVEIS
    #   a_i  = decisão: abre a planta i
    #   b_j  = decisão: abre o satélite j
    #   x_ij = fluxo: planta i -> satélite j
    #   y_jk = fluxo: satélite i -> cliente k
    a = mdl.binary_var_dict(inst.I, name="a")
    b = mdl.binary_var_dict(inst.J, name="b")
    x = mdl.continuous_var_dict(inst.IJ, lb=0.0, name="x")
    y = mdl.continuous_var_dict(inst.JK, lb=0.0, name="y")

    # RESTRIÇÕES
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

    # capacidade agregada (viabilidade)
    mdl.add_constraints_((x[i, j] <= inst.q[j] * b[j]) for i, j in inst.IJ)
    mdl.add_constraints_((y[j, k] <= inst.r[k] * b[j]) for j, k in inst.JK)

    # OBJETIVO
    cost_fixed1 = mdl.sum(inst.f[i] * a[i] for i in inst.I)
    cost_fixed2 = mdl.sum(inst.g[j] * b[j] for j in inst.J)

    cost_flow1 = mdl.sum(inst.c[i, j] * x[i, j] for i, j in inst.IJ)
    cost_flow2 = mdl.sum(inst.d[j, k] * y[j, k] for j, k in inst.JK)

    mdl.minimize(cost_fixed1 + cost_fixed2 + cost_flow1 + cost_flow2)

    # SOLVE
    solution = mdl.solve()

    if solution:
        print(f"\nSolved.\n")
        print(f"objective  = {solution.objective_value:.2f}")
        print(f"best bound = {solution.solve_details.best_bound:.2f}")

        print(f"\n{solution.solve_details}")


def main():
    """
    Rotina principal
    """
    PATH = "instances/tscfl/tscfl_12_50.txt"

    instance = TSCFLInstance.from_txt(PATH)

    solve_instance(instance)

    return


if __name__ == "__main__":
    main()
