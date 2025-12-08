"""
COS888

sscfl_solver_cp.py

Gabriel Braun, 2025
"""

import numpy as np
from docplex.cp.model import CpoModel

from ..sscfl_instance import SSCFLInstance


class SSCFLSolverCP:
    """
    Solver CP (CP Optimizer) para o SSCFL.
    """

    def __init__(self, inst: SSCFLInstance):
        self.inst = inst
        self.mdl = CpoModel(name="SSCFL_CP")
        # Parâmetros do solver
        self.mdl.set_parameters(RelativeOptimalityTolerance=1e-7)

        # VARIÁVEIS
        # a_i = decisão: abre planta i
        # Y_j = associação: cliente j -> planta i
        self.a = self.mdl.binary_var_list(inst.nI, name="a")
        self.Y = self.mdl.integer_var_list(inst.nJ, 0, inst.nI - 1, name="Y")

        # VARIÁVEIS AUXILIARES
        # L_i = carga total atendida pela planta i
        # N_i = número de clientes atendidos pela planta i
        self.L = self.mdl.integer_var_list(inst.nI, 0, inst.r.sum(), name="L")
        self.N = self.mdl.integer_var_list(inst.nI, 0, inst.nJ, name="N")

        # Resultado e statísticas
        self.status = None
        self.lb = 0.0
        self.ub = np.inf
        self.gap = np.inf
        self.time = 0.0
        self.branches = 0

        # Monta o modelo
        self._build_model()

    def _build_model(self) -> None:
        """
        Construção do modelo CP.
        """
        inst = self.inst
        mdl = self.mdl

        # FILTRO: só permite plantas que suportam a demanda do cliente
        for j in inst.J:
            feas_i = [i for i in inst.I if inst.p[i] >= inst.r[j]]
            if len(feas_i) < inst.nI:  # só adiciona se realmente podar algo
                mdl.add(
                    mdl.allowed_assignments(
                        [self.Y[j]],
                        mdl.tuple_set((i,) for i in feas_i),
                    )
                )

        # GLOBAL CONSTRAINTS
        # Bin packing das demandas (r) nas plantas (pela variável Y)
        mdl.add(mdl.pack(self.L, self.Y, inst.r))
        # Conta quantos clientes cada planta recebe
        mdl.add(mdl.distribute(self.N, self.Y, values=inst.I))

        # CONSTRAINTS
        # capacidade
        mdl.add(self.L[i] <= inst.p[i] for i in inst.I)
        # Open <-> used: closed ⇒ L=0 & N=0 ; used ⇒ open
        mdl.add(mdl.if_then(self.a[i] == 0, self.L[i] == 0) for i in inst.I)
        mdl.add(mdl.if_then(self.a[i] == 0, self.N[i] == 0) for i in inst.I)
        mdl.add(mdl.if_then(self.N[i] >= 1, self.a[i] == 1) for i in inst.I)

        # QUEBRA DE SIMETRIA
        # Cria uma chave para ordenar os grupos idênticos
        keys = np.column_stack((inst.p, inst.f, inst.c)).astype(int, copy=False)
        _, grp = np.unique(keys, axis=0, return_inverse=True)

        # Para cada grupo idêntico, forçar ordem lexicográfica em [a, L, N]
        ng = grp.max() + 1
        for g in range(ng):
            idxs = np.where(grp == g)[0]
            if idxs.size > 1:
                idxs.sort()
                for i1, i2 in zip(idxs[:-1], idxs[1:]):
                    mdl.add(
                        mdl.lexicographic(
                            [self.a[i2], self.L[i2], self.N[i2]],
                            [self.a[i1], self.L[i1], self.N[i1]],
                        )
                    )

        # OBJETIVO
        cost_fixed = mdl.sum(inst.f[i] * self.a[i] for i in inst.I)
        cost_flow = mdl.sum(mdl.element(inst.c[:, j], self.Y[j]) for j in inst.J)

        mdl.minimize(cost_fixed + cost_flow)

    def solve(
        self,
        log_output: bool = True,
        time_limit: float | None = None,
    ):
        """
        Resolve o modelo CP.
        """
        sol = self.mdl.solve(
            LogVerbosity=("Terse" if log_output else "Quiet"),
            TimeLimit=time_limit,
        )

        self.solution = sol

        # Resultado e statísticas
        infos = sol.get_solver_infos()

        self.status = sol.get_solve_status()
        self.ub = float(sol.get_objective_value())
        self.lb = float(sol.get_objective_bound())
        self.gap = float(sol.get_objective_gap())
        self.time = float(sol.get_solve_time())
        self.branches = int(infos.get_number_of_branches())

        # Log final
        print("\n\n[CP] Solver finalizado\n")
        print(f"LB    = {self.lb:.0f}")
        print(f"UB    = {self.ub:.0f}")
        print(f"gap   = {self.gap:.2e}")
        print(f"time  = {self.time:.1f}")
        print(f"brncs = {self.branches}")

        return sol
