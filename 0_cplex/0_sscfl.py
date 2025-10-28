"""
COS888

CFL com CPLEX

Gabriel Braun, 2025
"""

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
from docplex.mp.model import Model


@dataclass(frozen=True)
class SSCFLInstance:
    """
    Instância do SSCFL
    """

    nI: int  # |I| plantas
    nJ: int  # |J| clientes

    f: np.ndarray  # f_i  = custo fixo da planta i
    c: np.ndarray  # c_ij = custo unitário planta i -> cliente j
    p: np.ndarray  # p_i  = capacidade da planta i
    r: np.ndarray  # r_j  = demanda do cliente j

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
    def from_txt(cls, path: str) -> "SSCFLInstance":
        """
        Retorna: Instância SSCFL a partir de um arquivo .txt.
        """
        arr = np.fromstring(Path(path).read_text(), sep=" ", dtype=float).astype(int)

        nI, nJ = arr[:2]
        data = arr[2:]

        s1 = 2 * nI
        s2 = s1 + nJ
        s3 = s2 + nI * nJ

        pf = data[:s1].reshape(nI, 2)
        p = pf[:, 0]
        f = pf[:, 1]

        r = data[s1:s2]
        c = data[s2:s3].reshape(nI, nJ)

        return cls(nI=nI, nJ=nJ, f=f, c=c, p=p, r=r)


def solve_instance(inst: SSCFLInstance) -> None:
    """
    Resolve a instância SSCFL (single-source) usando CPLEX.
    """
    mdl = Model(name="SSCFL", log_output=True)

    # VARIÁVEIS
    #   a_i  = decisão: abre instalação i
    #   x_ij = decisão: instalação i -> cliente j
    a = mdl.binary_var_dict(inst.I, name="a")
    x = mdl.binary_var_dict(inst.IJ, name="x")

    # RESTRIÇÕES
    # cada cliente é atendido por exatamente uma instalação
    mdl.add_constraints_(mdl.sum(x[i, j] for i in inst.I) == 1 for j in inst.J)

    # capacidade das instalações
    mdl.add_constraints_(
        mdl.sum(inst.r[j] * x[i, j] for j in inst.J) <= inst.p[i] * a[i] for i in inst.I
    )

    # vinculação: se instalação está fechada, ninguém pode ser atendido por ela
    mdl.add_constraints_((x[i, j] <= a[i]) for i, j in inst.IJ)

    # OBJETIVO
    cost_fixed = mdl.sum(inst.f[i] * a[i] for i in inst.I)
    cost_flow = mdl.sum(inst.c[i, j] * x[i, j] for i, j in inst.IJ)

    mdl.minimize(cost_fixed + cost_flow)

    # SOLVE
    solution = mdl.solve()

    if solution:
        print(f"\nSolved.\n")
        print(f"objective  = {solution.objective_value:.2f}")
        print(f"best bound = {solution.solve_details.best_bound:.2f}")

        print(f"\n{solution.solve_details}")

    return solution


def main():
    """
    Rotina principal
    """
    PATH = "instances/sscfl/holmberg/sscfl_h_40.txt"

    instance = SSCFLInstance.from_txt(PATH)

    solve_instance(instance)

    return


if __name__ == "__main__":
    main()
