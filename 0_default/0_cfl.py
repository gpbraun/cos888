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

        facility_data = data[: 2 * nI].reshape(nI, 2)
        p = facility_data[:, 0]
        f = facility_data[:, 1]

        customer_data = data[2 * nI :].reshape(nJ, 1 + nI)
        r = customer_data[:, 0]
        c = customer_data[:, 1:].T / r[None, :]

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


def main():
    """
    Rotina principal
    """
    PATH = "instances/cfl/cfl_41.txt"

    instance = CFLInstance.from_txt(PATH)

    obj = solve_instance(instance)

    print(f"objective: {obj}")

    return


if __name__ == "__main__":
    main()
