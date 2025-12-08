"""
COS888

sscfl_solver_mp.py

Gabriel Braun, 2025
"""

import numpy as np
from docplex.mp.model import Model

from ..sscfl_instance import SSCFLInstance


class SSCFLSolverMP:
    """
    Solver CPLEX para o SSCFL.
    """

    def __init__(self, inst: SSCFLInstance):
        self.inst = inst
        self.mdl = Model(name="SSCFL_MP", log_output=False)
        # Parâmetros do solver
        self.mdl.parameters.mip.tolerances.mipgap = 1e-7

        # VARIÁVEIS
        # a_i  = decisão: abre instalação i
        # x_ij = decisão: instalação i -> cliente j
        self.a = self.mdl.binary_var_dict(inst.I, name="a")
        self.x = self.mdl.binary_var_dict(inst.IJ, name="x")

        # Resultado e estatísticas
        self.lb = 0.0
        self.ub = np.inf
        self.gap = np.inf
        self.time = 0.0
        self.nodes = 0
        self.status = None

        # Monta o modelo
        self._build_model()

    def _build_model(self) -> None:
        """
        Construção do modelo MIP no CPLEX.
        """
        inst = self.inst
        mdl = self.mdl

        # RESTRIÇÕES
        # cada cliente é atendido por exatamente uma instalação
        mdl.add_constraints_(mdl.sum(self.x[i, j] for i in inst.I) == 1 for j in inst.J)

        # capacidade das instalações
        mdl.add_constraints_(
            mdl.sum(inst.r[j] * self.x[i, j] for j in inst.J) <= inst.p[i] * self.a[i]
            for i in inst.I
        )

        # vinculação: se instalação está fechada, ninguém pode ser atendido por ela
        mdl.add_constraints_(self.x[i, j] <= self.a[i] for i, j in inst.IJ)

        # OBJETIVO
        cost_fixed = mdl.sum(inst.f[i] * self.a[i] for i in inst.I)
        cost_flow = mdl.sum(inst.c[i, j] * self.x[i, j] for i, j in inst.IJ)

        mdl.minimize(cost_fixed + cost_flow)

    def solve(
        self,
        log_output: bool = True,
        time_limit: float | None = None,
    ):
        """
        Resolve o modelo MIP com CPLEX.
        """
        self.mdl.log_output = log_output

        solve_kwargs: dict = {}
        if time_limit is not None:
            solve_kwargs["time_limit"] = time_limit

        sol = self.mdl.solve(**solve_kwargs)

        # Resultado e statísticas
        details = self.mdl.solve_details

        self.status = details.status
        self.lb = float(details.best_bound)
        self.ub = float(sol.objective_value)
        self.gap = float(details.mip_relative_gap)
        self.nodes = int(details.nb_nodes_processed)
        self.time = float(details.time)

        # Log final
        print("\n\n[MP] Solver finalizado\n")
        print(f"LB    = {self.lb:.0f}")
        print(f"UB    = {self.ub:.0f}")
        print(f"gap   = {self.gap:.2e}")
        print(f"time  = {self.time:.1f}")
        print(f"nodes = {self.nodes}")

        return sol
