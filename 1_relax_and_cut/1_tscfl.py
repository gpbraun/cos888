"""
COS888

TSCFL por Relax-and-Cut

Gabriel Braun, 2025
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from time import perf_counter

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


class RelaxAndCutTSCFL:
    """
    Resolve a instância usando o método: Non-Delayed Relax-and-Cut.
    """

    def __init__(
        self,
        inst: "TSCFLInstance",
        *,
        gamma: float = 1.0,
        dual_keep: int = 5,
        tol_stop: float = 1e-6,
        max_iter: int = 10_000,
        time_limit: int = 1000,
        log_output: bool = False,
    ) -> None:
        self.inst = inst
        self.gamma = gamma
        self.dual_keep = dual_keep
        self.tol_stop = tol_stop

        self.max_iter = max_iter
        self.time_limit = time_limit
        self.log_output = log_output

        self.lamb = np.zeros(self.inst.nJ + self.inst.nK, dtype=float)
        self.lamb_age = np.zeros(self.inst.nJ + self.inst.nK, dtype=int)

        self.L_best = -np.inf
        self.z_best = np.inf
        self.gap = np.inf

        self.a_best = np.zeros(self.inst.nI, dtype=int)
        self.b_best = np.zeros(self.inst.nJ, dtype=int)

        self.iter = 0
        self.time = 0.0

    def _solve_lrp(self, lamb: np.ndarray):
        """
        Resolve a relaxação Lagrangeana para obter um limite inferior.
        """
        alph = lamb[: self.inst.nJ]
        beta = lamb[self.inst.nJ :]

        # custos reduzidos
        ctil = self.inst.c + alph[None, :]  # (nI, nJ)
        dtil = self.inst.d - alph[:, None] + beta[None, :]  # (nJ, nK)

        a_rel = np.zeros(self.inst.nI, dtype=np.int8)
        b_rel = np.zeros(self.inst.nJ, dtype=np.int8)
        L_val = -np.dot(beta, self.inst.r)

        # agregadores para subgradientes
        sum_x_j = np.zeros(self.inst.nJ, dtype=float)  # soma_i x[i,j]
        sum_y_j = np.zeros(self.inst.nJ, dtype=float)  # soma_k y[j,k]
        sum_y_k = np.zeros(self.inst.nK, dtype=float)  # soma_j y[j,k]

        def _greedy_take(sorted_caps: np.ndarray, cap_total: float) -> np.ndarray:
            """
            Dado: vetor de capacidades dos itens (já ordenados) e uma capacidade total.
            Retorna: `take` com quanto é pego de cada item, na mesma ordem.
            """
            if np.isclose(cap_total, 0.0) or sorted_caps.size == 0:
                return np.zeros_like(sorted_caps, dtype=float)
            cum = np.cumsum(sorted_caps)
            full_cnt = np.searchsorted(cum, cap_total, side="right")
            take = np.zeros_like(sorted_caps, dtype=float)
            if full_cnt > 0:
                take[:full_cnt] = sorted_caps[:full_cnt]
            rem = cap_total - (cum[full_cnt - 1] if full_cnt > 0 else 0.0)
            if (
                (full_cnt < sorted_caps.size)
                and (rem > 0.0)
                and (not np.isclose(rem, 0.0))
            ):
                take[full_cnt] = rem
            return take

        # resolve uma planta i
        def _solve_plant(i: int):
            row = ctil[i]
            mask = (row < 0.0) & (~np.isclose(row, 0.0))
            if not np.any(mask):
                return (i, 0, 0.0, None, None)  # fechado

            js = np.flatnonzero(mask)
            # ordenar por custo reduzido crescente
            order = np.argsort(row[js])
            js = js[order]
            rc = row[js]
            qj = self.inst.q[js]

            take_sorted = _greedy_take(qj, self.inst.p[i])
            var_part = np.dot(rc, take_sorted)
            open_i = take_sorted.any() and (self.inst.f[i] + var_part < 0.0)

            if open_i:
                return i, 1, (self.inst.f[i] + var_part), js, take_sorted
            else:
                return i, 0, 0.0, None, None

        # resolve um depósito j
        def _solve_depot(j: int):
            row = dtil[j]
            mask = (row < 0.0) & (~np.isclose(row, 0.0))
            if not np.any(mask):
                return j, 0, 0.0, None, None

            ks = np.flatnonzero(mask)
            order = np.argsort(row[ks])
            ks = ks[order]
            rc = row[ks]
            rk = self.inst.r[ks]

            take_sorted = _greedy_take(rk, self.inst.q[j])
            var_part = np.dot(rc, take_sorted)
            open_j = take_sorted.any() and (self.inst.g[j] + var_part < 0.0)

            if open_j:
                return j, 1, (self.inst.g[j] + var_part), ks, take_sorted
            else:
                return j, 0, 0.0, None, None

        # Resolução das plantas e depósitos em paralelo
        with ThreadPoolExecutor() as ex:
            for i, open_i, contrib, js, take in ex.map(_solve_plant, self.inst.I):
                if open_i:
                    a_rel[i] = 1
                    L_val += contrib
                    # soma_i x[i,j] (apenas nas colunas usadas)
                    sum_x_j[js] += take

        with ThreadPoolExecutor() as ex:
            for j, open_j, contrib, ks, take in ex.map(_solve_depot, self.inst.J):
                if open_j:
                    b_rel[j] = 1
                    L_val += contrib
                    # soma_k y[j,k] e soma_j y[j,k]
                    tj = np.sum(take)
                    if not np.isclose(tj, 0.0):
                        sum_y_j[j] = tj
                        sum_y_k[ks] += take

        # Cálculo dos subgradientes
        subgrad = np.empty_like(lamb)
        subgrad[: self.inst.nJ] = sum_x_j - sum_y_j
        subgrad[self.inst.nJ :] = sum_y_k - self.inst.r

        return L_val, a_rel, b_rel, subgrad

    def _solve_heuristic(self, a_open: np.ndarray, b_open: np.ndarray):
        """
        Resolve a heurística lagrangeana para obter um limite superior.
        """
        total_r = np.sum(self.inst.r)

        a_fix = a_open.astype(int).copy()
        b_fix = b_open.astype(int).copy()

        cap_p = np.dot(self.inst.p, a_fix)
        cap_q = np.dot(self.inst.q, b_fix)

        # garante a capacidade total (abre o mais barato por unidade)
        if (cap_p < total_r) and (not np.isclose(cap_p, total_r)):
            closed_plants = [
                i for i in range(self.inst.nI) if a_fix[i] == 0 and self.inst.p[i] > 0
            ]
            closed_plants.sort(key=lambda i: self.inst.f[i] / self.inst.p[i])
            for i in closed_plants:
                a_fix[i] = 1
                cap_p += self.inst.p[i]
                if (cap_p > total_r) or np.isclose(cap_p, total_r):
                    break

        if (cap_q < total_r) and (not np.isclose(cap_q, total_r)):
            closed_depots = [
                j for j in range(self.inst.nJ) if b_fix[j] == 0 and self.inst.q[j] > 0
            ]
            closed_depots.sort(key=lambda j: self.inst.g[j] / self.inst.q[j])
            for j in closed_depots:
                b_fix[j] = 1
                cap_q += self.inst.q[j]
                if (cap_q > total_r) or np.isclose(cap_q, total_r):
                    break

        if (cap_p < total_r and not np.isclose(cap_p, total_r)) or (
            cap_q < total_r and not np.isclose(cap_q, total_r)
        ):
            a_fix[:] = 1
            b_fix[:] = 1

        I_open = [i for i in range(self.inst.nI) if a_fix[i] == 1]
        J_open = [j for j in range(self.inst.nJ) if b_fix[j] == 1]
        if not I_open or not J_open:
            return None

        # Modelo CPLEX (contínuo)
        mdl = Model(name="TSCFL_heuristic", log_output=False)

        xR = mdl.continuous_var_dict(
            [(i, j) for i in I_open for j in J_open], lb=0.0, name="x"
        )
        yR = mdl.continuous_var_dict(
            [(j, k) for j in J_open for k in self.inst.K], lb=0.0, name="y"
        )

        # capacidades
        mdl.add_constraints_(
            mdl.sum(xR[i, j] for j in J_open) <= self.inst.p[i] for i in I_open
        )
        mdl.add_constraints_(
            mdl.sum(yR[j, k] for k in range(self.inst.nK)) <= self.inst.q[j]
            for j in J_open
        )
        # balanço dos depósitos
        mdl.add_constraints_(
            mdl.sum(xR[i, j] for i in I_open)
            == mdl.sum(yR[j, k] for k in range(self.inst.nK))
            for j in J_open
        )
        # demandas dos consumidores
        mdl.add_constraints_(
            mdl.sum(yR[j, k] for j in J_open) == self.inst.r[k]
            for k in range(self.inst.nK)
        )

        # objetivo
        cost_fixed1 = np.dot(self.inst.f[I_open], np.ones(len(I_open)))
        cost_fixed2 = np.dot(self.inst.g[J_open], np.ones(len(J_open)))

        cost_flow1 = mdl.sum(self.inst.c[i, j] * xR[i, j] for (i, j) in xR)
        cost_flow2 = mdl.sum(self.inst.d[j, k] * yR[j, k] for (j, k) in yR)

        mdl.minimize(cost_fixed1 + cost_fixed2 + cost_flow1 + cost_flow2)

        solution = mdl.solve()

        return None if not solution else float(solution.objective_value)

    def solve(self) -> None:
        """
        Loop principal do NDRC.
        """
        time_start = perf_counter()

        while self.iter < self.max_iter and self.time <= self.time_limit:
            self.iter += 1

            # (1) resolve o subproblema lagrangeano
            L_k, a_k, b_k, subgrad_k = self._solve_lrp(self.lamb)

            # (2) manutenção de LB/UB
            lb_improved = L_k > self.L_best
            if lb_improved:
                self.L_best = L_k

            if (self.iter == 1) or lb_improved or (self.iter % 25 == 0):
                z_try = self._solve_heuristic(a_k, b_k)
                if (z_try is not None) and (z_try < self.z_best):
                    self.z_best = z_try
                    self.a_best = a_k.copy()
                    self.b_best = b_k.copy()

            # (3) gerenciamento de CA/PA/CI
            CA_idx = np.where(~np.isclose(subgrad_k, 0.0))[0]
            PA_idx = np.where(~np.isclose(self.lamb, 0.0))[0]
            CI_idx = np.setdiff1d(
                np.arange(self.lamb.size),
                np.union1d(CA_idx, PA_idx),
                assume_unique=False,
            )

            subgrad_k[CI_idx] = 0.0
            self.lamb_age[CA_idx] = 0

            drop_idx = np.setdiff1d(PA_idx, CA_idx, assume_unique=False)
            if drop_idx.size:
                self.lamb_age[drop_idx] += 1
                to_zero = drop_idx[self.lamb_age[drop_idx] > self.dual_keep]
                if to_zero.size:
                    self.lamb[to_zero] = 0.0
                    self.lamb_age[to_zero] = 0

            # (4) tamanho de passo e atualização do dual
            denom = np.dot(subgrad_k, subgrad_k)
            if denom > 0.0:
                if np.isfinite(self.z_best):
                    mu = self.gamma * max(self.z_best - L_k, 0.0) / denom
                else:
                    mu = self.gamma / denom

                self.lamb = self.lamb + mu * subgrad_k

            # (5) condição de encerramento
            if np.isfinite(self.z_best):
                self.gap = (self.z_best - self.L_best) / max(1.0, abs(self.z_best))
                if self.gap <= self.tol_stop:
                    print("Solução ótima encontrada!")
                    break

            # log do loop
            self.time = perf_counter() - time_start

            if self.log_output and (self.iter % 20 == 0):
                norm_g = float(np.sqrt(denom))
                print(
                    f"[NDRC] it={self.iter:4d}  time={round(self.time):4d}s    "
                    f"LRP={L_k:10.3f}  LB={self.L_best:10.3f}  UB={self.z_best:10.3f}  "
                    f"||subgrad||^2={norm_g:.3e}  |CA|={CA_idx.size:3d}  |PA|={PA_idx.size:3d}  |CI|={CI_idx.size:3d}"
                )

        return None


def main():
    """
    Rotina principal
    """
    PATH = "instances/tscfl/tscfl_11_50.txt"

    instance = TSCFLInstance.from_txt(PATH)

    solver = RelaxAndCutTSCFL(instance, time_limit=20, log_output=True)
    solver.solve()

    return


if __name__ == "__main__":
    main()
