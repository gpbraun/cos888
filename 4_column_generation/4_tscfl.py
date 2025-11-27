"""
COS888 — TSCFL resolvido com Branch-and-Price + Geração de Colunas
Implementação apenas para fins didáticos, baseada na relaxação Lagrangeana
por blocos (plantas e satélites) e decomposição de Dantzig–Wolfe.

Gabriel Braun, 2025
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import docplex
import numpy as np
from docplex.mp.model import Model

# ============================================================================
# Instância do TSCFL
# ============================================================================


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
        text = Path(path).read_text()
        arr = np.fromstring(text, sep=" ", dtype=float)

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


# ============================================================================
# Estruturas de dados do Branch-and-Price
# ============================================================================


class ColumnType(Enum):
    PLANT = 1
    SAT = 2


@dataclass
class Column:
    """
    Uma coluna do RMP (padrão de planta ou de satélite).

    cost    = custo original (não reduzido) da coluna.
    balance = contribuição na restrição de balanço nos depósitos (tamanho nJ).
    demand  = contribuição nas restrições de demanda dos clientes (tamanho nK).
    """

    col_id: int
    col_type: ColumnType
    block_id: int  # i se planta, j se satélite
    cost: float
    balance: np.ndarray  # shape (nJ,)
    demand: np.ndarray  # shape (nK,)


@dataclass
class Duals:
    """
    Duais do RMP:
      pi[j]    = dual das restrições de balanço nos depósitos
      sigma[k] = dual das restrições de demanda dos clientes
      alpha[i] = dual das restrições de convexidade das plantas
      nu[j]    = dual das restrições de convexidade dos satélites
    """

    pi: np.ndarray
    sigma: np.ndarray
    alpha: np.ndarray
    nu: np.ndarray


@dataclass
class NodeContext:
    """
    Contexto de um nó da árvore de Branch-and-Price:
      fixed_a[i] ∈ { -1 (livre), 0 (forçado a 0), 1 (forçado a 1) }
      fixed_b[j] ∈ { -1 (livre), 0, 1 }
    """

    fixed_a: np.ndarray  # shape (nI,), int
    fixed_b: np.ndarray  # shape (nJ,), int

    def copy(self) -> "NodeContext":
        return NodeContext(self.fixed_a.copy(), self.fixed_b.copy())


@dataclass
class NodeLPResult:
    """
    Resultado da resolução do RMP (LP) em um nó.
    """

    z_lp: float
    a_hat: np.ndarray  # solução aproximada para a_i (via colunas de plantas)
    b_hat: np.ndarray  # solução aproximada para b_j (via colunas de satélites)
    duals: Duals
    max_balance_slack: float
    max_demand_slack: float


@dataclass
class SearchNode:
    node_id: int
    depth: int
    ctx: NodeContext


# ============================================================================
# RMP (Restricted Master Problem) com DOcplex
# ============================================================================


class MasterProblem:
    """
    Modelo mestre restrito (RMP) em formulação de Dantzig–Wolfe:

      - Variáveis: z_col >= 0 (colunas/padrões).
      - Restrições:
          • Balanço nos depósitos j (com folgas s_pos[j], s_neg[j]).
          • Demanda dos clientes k (com folga s_dem[k]).
          • Convexidade por planta i: sum_{cols de i} z_col <= 1.
          • Convexidade por satélite j: sum_{cols de j} z_col <= 1.

      As colunas são adicionadas dinamicamente.
    """

    def __init__(self, inst: TSCFLInstance, big_M: float = 1e7) -> None:
        self.inst = inst
        self.big_M = big_M

        self.mdl = Model(name="TSCFL_RMP", log_output=False)

        # z para cada coluna ativa no RMP
        self.z_vars: Dict[int, "docplex.mp.dvar.Var"] = {}
        self.columns_in_model: Dict[int, Column] = {}

        # restrições
        self.balance_ct: Dict[int, "docplex.mp.constr.LinearConstraint"] = {}
        self.demand_ct: Dict[int, "docplex.mp.constr.LinearConstraint"] = {}
        self.conv_plant_ct: Dict[int, "docplex.mp.constr.LinearConstraint"] = {}
        self.conv_sat_ct: Dict[int, "docplex.mp.constr.LinearConstraint"] = {}

        # variáveis de folga
        self.s_pos: Dict[int, "docplex.mp.dvar.Var"] = {}
        self.s_neg: Dict[int, "docplex.mp.dvar.Var"] = {}
        self.s_dem: Dict[int, "docplex.mp.dvar.Var"] = {}

        # objetivo acumulado
        self.obj_expr = self.mdl.linear_expr()

        self._build_base_constraints()

    # ------------------------------------------------------------------ #
    # Construção das restrições base (sem colunas)
    # ------------------------------------------------------------------ #

    def _build_base_constraints(self) -> None:
        inst = self.inst
        M = self.big_M

        # Balanço nos depósitos j: sum_cols balance[j]*z + s_pos[j] - s_neg[j] = 0
        for j in inst.J:
            s_pos = self.mdl.continuous_var(lb=0.0, name=f"s_pos_{j}")
            s_neg = self.mdl.continuous_var(lb=0.0, name=f"s_neg_{j}")
            self.s_pos[j] = s_pos
            self.s_neg[j] = s_neg

            ct = self.mdl.add_constraint(s_pos - s_neg == 0.0, ctname=f"bal_{j}")
            self.balance_ct[j] = ct

            self.obj_expr += M * (s_pos + s_neg)

        # Demanda dos clientes k:
        #   sum_cols demand[k]*z + s_dem[k] = r_k
        for k in inst.K:
            s_dem = self.mdl.continuous_var(lb=0.0, name=f"s_dem_{k}")
            self.s_dem[k] = s_dem

            ct = self.mdl.add_constraint(s_dem == float(inst.r[k]), ctname=f"dem_{k}")
            self.demand_ct[k] = ct

            self.obj_expr += M * s_dem

        # Convexidade por planta i: sum_{cols de i} z <= 1
        for i in inst.I:
            lhs = self.mdl.linear_expr()
            ct = self.mdl.add_constraint(lhs <= 1.0, ctname=f"convI_{i}")
            self.conv_plant_ct[i] = ct

        # Convexidade por satélite j: sum_{cols de j} z <= 1
        for j in inst.J:
            lhs = self.mdl.linear_expr()
            ct = self.mdl.add_constraint(lhs <= 1.0, ctname=f"convJ_{j}")
            self.conv_sat_ct[j] = ct

        self.mdl.minimize(self.obj_expr)

    # ------------------------------------------------------------------ #
    # Colunas
    # ------------------------------------------------------------------ #

    def add_columns(self, new_columns: List[Column]) -> None:
        """
        Adiciona novas colunas (padrões) ao RMP.
        """
        if not new_columns:
            return

        inst = self.inst

        for col in new_columns:
            if col.col_id in self.z_vars:
                continue

            z = self.mdl.continuous_var(lb=0.0, name=f"z_{col.col_id}")
            self.z_vars[col.col_id] = z
            self.columns_in_model[col.col_id] = col

            # objetivo: custo original da coluna
            self.obj_expr += col.cost * z

            # convexidade
            if col.col_type is ColumnType.PLANT:
                ct = self.conv_plant_ct[col.block_id]
                ct.left_expr += z
            else:
                ct = self.conv_sat_ct[col.block_id]
                ct.left_expr += z

            # balanço nos depósitos
            for j in inst.J:
                coef = float(col.balance[j])
                if abs(coef) > 1e-12:
                    self.balance_ct[j].left_expr += coef * z

            # demanda dos clientes
            for k in inst.K:
                coef = float(col.demand[k])
                if abs(coef) > 1e-12:
                    self.demand_ct[k].left_expr += coef * z

    # ------------------------------------------------------------------ #
    # Resolução do LP
    # ------------------------------------------------------------------ #

    def solve_lp(self) -> NodeLPResult:
        sol = self.mdl.solve(log_output=False)
        if sol is None:
            raise RuntimeError("RMP infeasible ou sem solução LP.")

        z_lp = float(sol.objective_value)

        # Reconstrói a_hat, b_hat a partir das colunas
        a_hat = np.zeros(self.inst.nI, dtype=float)
        b_hat = np.zeros(self.inst.nJ, dtype=float)

        for col_id, z_var in self.z_vars.items():
            z_val = float(sol.get_value(z_var))
            if abs(z_val) < 1e-9:
                continue
            col = self.columns_in_model[col_id]
            if col.col_type is ColumnType.PLANT:
                a_hat[col.block_id] += z_val
            else:
                b_hat[col.block_id] += z_val

        # Slacks máximos
        max_bal_slack = 0.0
        for j in self.inst.J:
            sp = float(sol.get_value(self.s_pos[j]))
            sn = float(sol.get_value(self.s_neg[j]))
            max_bal_slack = max(max_bal_slack, sp, sn)

        max_dem_slack = 0.0
        for k in self.inst.K:
            sd = float(sol.get_value(self.s_dem[k]))
            max_dem_slack = max(max_dem_slack, abs(sd))

        # Duais
        pi = np.zeros(self.inst.nJ, dtype=float)
        sigma = np.zeros(self.inst.nK, dtype=float)
        alpha = np.zeros(self.inst.nI, dtype=float)
        nu = np.zeros(self.inst.nJ, dtype=float)

        for j in self.inst.J:
            pi[j] = float(self.balance_ct[j].dual_value)
        for k in self.inst.K:
            sigma[k] = float(self.demand_ct[k].dual_value)
        for i in self.inst.I:
            alpha[i] = float(self.conv_plant_ct[i].dual_value)
        for j in self.inst.J:
            nu[j] = float(self.conv_sat_ct[j].dual_value)

        duals = Duals(pi=pi, sigma=sigma, alpha=alpha, nu=nu)

        return NodeLPResult(
            z_lp=z_lp,
            a_hat=a_hat,
            b_hat=b_hat,
            duals=duals,
            max_balance_slack=max_bal_slack,
            max_demand_slack=max_dem_slack,
        )


# ============================================================================
# Branch-and-Price
# ============================================================================


class BranchAndPriceSolver:
    """
    Branch-and-Price para o TSCFL, com:

      - RMP em Dantzig–Wolfe (MasterProblem).
      - Pricing por blocos:
          • subproblemas de planta i: decisão x_ij agregada.
          • subproblemas de satélite j: decisão y_jk agregada.
      - Branching em a_i ou b_j (via a_hat, b_hat).
    """

    def __init__(
        self,
        inst: TSCFLInstance,
        *,
        cg_tol: float = 1e-6,
        cg_max_iter: int = 200,
        log_output: bool = True,
    ) -> None:
        self.inst = inst
        self.cg_tol = cg_tol
        self.cg_max_iter = cg_max_iter
        self.log_output = log_output

        self._next_col_id: int = 0

    # ------------------------------------------------------------------ #
    # Colunas iniciais (dummy: abre planta/satélite sem fluxo)
    # ------------------------------------------------------------------ #

    def _build_initial_columns(self, ctx: NodeContext) -> List[Column]:
        inst = self.inst
        cols: List[Column] = []

        # Colunas de plantas (sem fluxo)
        for i in inst.I:
            if ctx.fixed_a[i] == 0:
                continue
            balance = np.zeros(inst.nJ, dtype=float)
            demand = np.zeros(inst.nK, dtype=float)
            cost = float(inst.f[i])  # abre a planta, sem fluxo
            col = Column(
                col_id=self._next_col_id,
                col_type=ColumnType.PLANT,
                block_id=i,
                cost=cost,
                balance=balance,
                demand=demand,
            )
            cols.append(col)
            self._next_col_id += 1

        # Colunas de satélites (sem fluxo)
        for j in inst.J:
            if ctx.fixed_b[j] == 0:
                continue
            balance = np.zeros(inst.nJ, dtype=float)
            demand = np.zeros(inst.nK, dtype=float)
            cost = float(inst.g[j])  # abre o satélite, sem fluxo
            col = Column(
                col_id=self._next_col_id,
                col_type=ColumnType.SAT,
                block_id=j,
                cost=cost,
                balance=balance,
                demand=demand,
            )
            cols.append(col)
            self._next_col_id += 1

        return cols

    # ------------------------------------------------------------------ #
    # Pricing: plantas
    # ------------------------------------------------------------------ #

    def _price_plants(self, duals: Duals, ctx: NodeContext) -> List[Column]:
        """
        Subproblema de planta i:

          min  f_i + sum_j c_ij x_ij - alpha_i - sum_j pi_j x_ij
          s.a. sum_j x_ij <= p_i, x_ij >= 0

        que equivale a:

          min  (f_i - alpha_i) + sum_j (c_ij - pi_j) x_ij
        """
        inst = self.inst
        new_cols: List[Column] = []

        for i in inst.I:
            if ctx.fixed_a[i] == 0:
                continue

            p_i = float(inst.p[i])
            if p_i <= 0:
                continue

            # coeficiente reduzido por unidade de fluxo para cada j
            phi = inst.c[i, :] - duals.pi  # shape (nJ,)
            j_best = int(np.argmin(phi))
            phi_min = float(phi[j_best])

            base_cost = float(inst.f[i])
            # x = 0
            cost0 = base_cost
            rc0 = cost0 - duals.alpha[i]  # sem contribuição em balanço/demanda

            if phi_min >= 0.0:
                # melhor é x = 0
                rc = rc0
                if rc < -self.cg_tol:
                    balance = np.zeros(inst.nJ, dtype=float)
                    demand = np.zeros(inst.nK, dtype=float)
                    col = Column(
                        col_id=self._next_col_id,
                        col_type=ColumnType.PLANT,
                        block_id=i,
                        cost=cost0,
                        balance=balance,
                        demand=demand,
                    )
                    new_cols.append(col)
                    self._next_col_id += 1
                continue

            # padrão com fluxo máximo na melhor aresta
            x_val = p_i
            balance = np.zeros(inst.nJ, dtype=float)
            balance[j_best] = x_val
            demand = np.zeros(inst.nK, dtype=float)

            cost = base_cost + float(inst.c[i, j_best]) * x_val
            rc = cost - duals.alpha[i] - duals.pi[j_best] * x_val

            if rc < -self.cg_tol:
                col = Column(
                    col_id=self._next_col_id,
                    col_type=ColumnType.PLANT,
                    block_id=i,
                    cost=cost,
                    balance=balance,
                    demand=demand,
                )
                new_cols.append(col)
                self._next_col_id += 1

        return new_cols

    # ------------------------------------------------------------------ #
    # Pricing: satélites
    # ------------------------------------------------------------------ #

    def _price_sats(self, duals: Duals, ctx: NodeContext) -> List[Column]:
        """
        Subproblema de satélite j:

          min  g_j + sum_k d_jk y_jk - nu_j
               - pi_j * (-sum_k y_jk) - sum_k sigma_k y_jk
          s.a. sum_k y_jk <= q_j
               0 <= y_jk <= r_k

        isto é:

          min  (g_j - nu_j) + sum_k (d_jk + pi_j - sigma_k) y_jk
        """
        inst = self.inst
        new_cols: List[Column] = []

        for j in inst.J:
            if ctx.fixed_b[j] == 0:
                continue

            q_j = float(inst.q[j])
            if q_j <= 0:
                continue

            base_cost = float(inst.g[j])  # custo fixo do satélite

            psi = inst.d[j, :] + duals.pi[j] - duals.sigma  # shape (nK,)

            cap = q_j
            y = np.zeros(inst.nK, dtype=float)

            # preenche clientes com menor psi_k primeiro (os "mais negativos")
            order = np.argsort(psi)
            for k in order:
                if psi[k] >= 0.0:
                    break
                if cap <= 1e-9:
                    break
                assign = float(min(inst.r[k], cap))
                if assign <= 1e-9:
                    continue
                y[k] = assign
                cap -= assign

            total_y = float(y.sum())

            if total_y <= 1e-9:
                # padrão sem fluxo
                cost = base_cost
                balance = np.zeros(inst.nJ, dtype=float)
                demand = np.zeros(inst.nK, dtype=float)
                rc = cost - duals.nu[j]
            else:
                cost = base_cost + float(np.dot(inst.d[j, :], y))

                balance = np.zeros(inst.nJ, dtype=float)
                balance[j] = -total_y  # sai fluxo do satélite j

                demand = y.copy()

                rc = (
                    cost
                    - duals.nu[j]
                    - duals.pi[j] * balance[j]
                    - float(np.dot(duals.sigma, demand))
                )

            if rc < -self.cg_tol:
                col = Column(
                    col_id=self._next_col_id,
                    col_type=ColumnType.SAT,
                    block_id=j,
                    cost=cost,
                    balance=balance,
                    demand=demand,
                )
                new_cols.append(col)
                self._next_col_id += 1

        return new_cols

    # ------------------------------------------------------------------ #
    # Utilitários: integrality, branching
    # ------------------------------------------------------------------ #

    @staticmethod
    def _is_integral(lp_res: NodeLPResult, int_tol: float = 1e-5) -> bool:
        a_hat = lp_res.a_hat
        b_hat = lp_res.b_hat

        frac_a = (a_hat > int_tol) & (a_hat < 1.0 - int_tol)
        frac_b = (b_hat > int_tol) & (b_hat < 1.0 - int_tol)

        if np.any(frac_a) or np.any(frac_b):
            return False
        return True

    def _choose_branch_var(
        self, lp_res: NodeLPResult, ctx: NodeContext
    ) -> Optional[Tuple[str, int, float]]:
        """
        Escolhe variável de branching mais fracionária dentre a_hat e b_hat.
        Retorna tupla (tipo, índice, valor) ou None se não houver candidato.
        """
        int_tol = 1e-5
        best = None
        best_dist = 1.0

        # a_i
        for i in self.inst.I:
            if ctx.fixed_a[i] != -1:
                continue
            val = float(lp_res.a_hat[i])
            if val <= int_tol or val >= 1.0 - int_tol:
                continue
            dist = abs(val - 0.5)
            if dist < best_dist:
                best_dist = dist
                best = ("a", i, val)

        # b_j
        for j in self.inst.J:
            if ctx.fixed_b[j] != -1:
                continue
            val = float(lp_res.b_hat[j])
            if val <= int_tol or val >= 1.0 - int_tol:
                continue
            dist = abs(val - 0.5)
            if dist < best_dist:
                best_dist = dist
                best = ("b", j, val)

        return best

    # ------------------------------------------------------------------ #
    # Resolve o LP (RMP + CG) em um nó da árvore
    # ------------------------------------------------------------------ #

    def _solve_node_lp(
        self, node: SearchNode, current_UB: float
    ) -> Optional[NodeLPResult]:
        ctx = node.ctx
        inst = self.inst

        master = MasterProblem(inst)

        initial_columns = self._build_initial_columns(ctx)

        if self.log_output:
            fa = int(np.count_nonzero(ctx.fixed_a != -1))
            fb = int(np.count_nonzero(ctx.fixed_b != -1))
            print(
                f"  [CG] start |fixed_a|={fa} |fixed_b|={fb} init_cols={len(initial_columns)}"
            )

        master.add_columns(initial_columns)

        last_lp_res: Optional[NodeLPResult] = None

        for it in range(1, self.cg_max_iter + 1):
            lp_res = master.solve_lp()
            last_lp_res = lp_res
            z_lp = lp_res.z_lp

            if self.log_output:
                print(
                    f"  [CG] it={it:3d}  z_lp={z_lp:,.3f}  |cols|={len(master.z_vars):4d}",
                    end="",
                )

            # Parada antecipada por bound (LB não pode diminuir)
            if current_UB < float("inf") and z_lp >= current_UB - 1e-6:
                if self.log_output:
                    print(f"  (early stop: z_lp={z_lp:,.3f} >= UB={current_UB:,.3f})")
                return lp_res

            # Pricing
            plant_cols = self._price_plants(lp_res.duals, ctx)
            sat_cols = self._price_sats(lp_res.duals, ctx)
            new_cols = plant_cols + sat_cols

            if self.log_output:
                print(
                    f"  new={len(new_cols):3d} (plants={len(plant_cols)}, "
                    f"sats={len(sat_cols)})"
                )

            if not new_cols:
                if self.log_output:
                    print("  [CG] no column with negative reduced cost, stopping.")
                break

            master.add_columns(new_cols)

        return last_lp_res

    # ------------------------------------------------------------------ #
    # Loop principal de Branch-and-Price
    # ------------------------------------------------------------------ #

    def solve(self):
        inst = self.inst

        # Contexto raiz: tudo livre (-1)
        root_ctx = NodeContext(
            fixed_a=np.full(inst.nI, -1, dtype=int),
            fixed_b=np.full(inst.nJ, -1, dtype=int),
        )

        best_obj = float("inf")
        best_a: Optional[np.ndarray] = None
        best_b: Optional[np.ndarray] = None

        stack: List[SearchNode] = []
        stack.append(SearchNode(node_id=1, depth=0, ctx=root_ctx))

        if self.log_output:
            print("[BnP] starting search\n")

        node_counter = 1

        while stack:
            node = stack.pop()
            ctx = node.ctx

            if self.log_output:
                fa = int(np.count_nonzero(ctx.fixed_a != -1))
                fb = int(np.count_nonzero(ctx.fixed_b != -1))
                ub_str = f"{best_obj:,.3f}" if best_obj < float("inf") else "inf"
                print(
                    f"[BnP] Node #{node.node_id}  depth={node.depth}  "
                    f"|fixed_a|={fa}  |fixed_b|={fb}  current_UB={ub_str}"
                )

            lp_res = self._solve_node_lp(node, best_obj)
            if lp_res is None:
                if self.log_output:
                    print("[BnP]   LP infeasible, pruning.")
                continue

            z_lp = lp_res.z_lp

            # poda por bound
            if z_lp >= best_obj - 1e-6:
                if self.log_output:
                    print(
                        f"[BnP]   prune by bound: z_lp={z_lp:,.3f} >= best_obj={best_obj:,.3f}"
                    )
                continue

            # verifica integralidade em a_hat, b_hat
            if self._is_integral(lp_res):
                if self.log_output:
                    print("[BnP]   LP solution is integral.")

                # checa slacks (garante viabilidade nas restrições originais)
                if lp_res.max_balance_slack > 1e-6 or lp_res.max_demand_slack > 1e-6:
                    if self.log_output:
                        print(
                            f"[BnP]   discarding integral LP: slacks "
                            f"(bal={lp_res.max_balance_slack:.3e}, "
                            f"dem={lp_res.max_demand_slack:.3e})"
                        )
                    continue

                if z_lp < best_obj - 1e-6:
                    best_obj = z_lp
                    best_a = np.round(lp_res.a_hat).astype(int)
                    best_b = np.round(lp_res.b_hat).astype(int)
                    if self.log_output:
                        print(
                            f"[BnP]   new incumbent: UB={best_obj:,.3f}  "
                            f"(bal_slack={lp_res.max_balance_slack:.3e}, "
                            f"dem_slack={lp_res.max_demand_slack:.3e})"
                        )
                continue

            # caso contrário: precisa ramificar
            var_info = self._choose_branch_var(lp_res, ctx)
            if var_info is None:
                # nada claramente fracionário (provavelmente numérico) -> considera incumbente
                if self.log_output:
                    print("[BnP]   no branching candidate, treating as incumbent.")
                if z_lp < best_obj - 1e-6:
                    best_obj = z_lp
                    best_a = np.round(lp_res.a_hat).astype(int)
                    best_b = np.round(lp_res.b_hat).astype(int)
                continue

            vname, idx, val = var_info
            if self.log_output:
                print(f"[BnP]   branching on {vname}_{idx} = {val:.3f}")

            # filho esquerdo: fixa a/b = 0
            ctx_left = ctx.copy()
            if vname == "a":
                ctx_left.fixed_a[idx] = 0
            else:
                ctx_left.fixed_b[idx] = 0
            node_counter += 1
            child_left = SearchNode(
                node_id=node_counter, depth=node.depth + 1, ctx=ctx_left
            )

            # filho direito: fixa a/b = 1
            ctx_right = ctx.copy()
            if vname == "a":
                ctx_right.fixed_a[idx] = 1
            else:
                ctx_right.fixed_b[idx] = 1
            node_counter += 1
            child_right = SearchNode(
                node_id=node_counter, depth=node.depth + 1, ctx=ctx_right
            )

            # DFS: processa primeiro o filho "0", depois o "1" (ordem na pilha)
            stack.append(child_right)
            stack.append(child_left)

        if self.log_output:
            print("\n[BnP] search finished.")
            if best_obj < float("inf"):
                print(f"[BnP] best objective = {best_obj:,.3f}")
                print(f"[BnP] best a = {best_a}")
                print(f"[BnP] best b = {best_b}")
            else:
                print("[BnP] no feasible integer solution found.")

        return {
            "best_obj": best_obj,
            "best_a": best_a,
            "best_b": best_b,
        }


# ============================================================================
# main()
# ============================================================================


def main():
    """
    Rotina principal
    """
    PATH = "instances/tscfl/tscfl_11_50.txt"

    inst = TSCFLInstance.from_txt(PATH)

    solver = BranchAndPriceSolver(inst, cg_max_iter=200, log_output=True)
    solver.solve()

    return


if __name__ == "__main__":
    main()
