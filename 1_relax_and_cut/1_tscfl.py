"""
COS888

TSCFL por Non-Delayed Relax-and-Cut

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
        cut_tol: float = 1e-9,
        solve_master_every: int = 25,
    ) -> None:
        self.inst = inst
        self.gamma = gamma
        self.dual_keep = dual_keep
        self.tol_stop = tol_stop

        self.max_iter = max_iter
        self.time_limit = time_limit
        self.log_output = log_output
        self.cut_tol = cut_tol
        self.solve_master_every = solve_master_every

        # Multiplicadores e aging
        self.lamb = np.zeros(self.inst.nJ + self.inst.nK, dtype=float)
        self.lamb_age = np.zeros(self.inst.nJ + self.inst.nK, dtype=int)

        # LB/UB e melhor (a,b)
        self.L_best = -np.inf
        self.z_best = np.inf
        self.gap = np.inf
        self.a_best = np.zeros(self.inst.nI, dtype=int)
        self.b_best = np.zeros(self.inst.nJ, dtype=int)

        # Loop state
        self.iter = 0
        self.time = 0.0

        # Modelo mestre (construído sob demanda)
        self.master = None
        self._a = self._b = self._x = self._y = None

    def _build_master(self):
        mdl = Model(name="RAC_master", log_output=False)
        I, J, K = self.inst.I, self.inst.J, self.inst.K

        a = mdl.binary_var_dict(I, name="a")
        b = mdl.binary_var_dict(J, name="b")
        x = mdl.continuous_var_dict([(i, j) for i in I for j in J], lb=0.0, name="x")
        y = mdl.continuous_var_dict([(j, k) for j in J for k in K], lb=0.0, name="y")

        # Capacidades agregadas (planta e depósito)
        mdl.add_constraints_(
            mdl.sum(x[i, j] for j in J) <= self.inst.p[i] * a[i] for i in I
        )
        mdl.add_constraints_(
            mdl.sum(y[j, k] for k in K) <= self.inst.q[j] * b[j] for j in J
        )

        # Balanço nos depósitos e atendimento da demanda
        mdl.add_constraints_(
            mdl.sum(x[i, j] for i in I) == mdl.sum(y[j, k] for k in K) for j in J
        )
        mdl.add_constraints_(mdl.sum(y[j, k] for j in J) == self.inst.r[k] for k in K)

        # Objetivo
        mdl.minimize(
            mdl.sum(self.inst.f[i] * a[i] for i in I)
            + mdl.sum(self.inst.g[j] * b[j] for j in J)
            + mdl.sum(self.inst.c[i, j] * x[i, j] for i in I for j in J)
            + mdl.sum(self.inst.d[j, k] * y[j, k] for j in J for k in K)
        )

        self.master = mdl
        self._a, self._b, self._x, self._y = a, b, x, y

    def _separate_flow_covers_depot(self, y_rel: np.ndarray) -> list[tuple]:
        """
        'Flow-cover' no depósito j.
        """
        cuts = []
        r = self.inst.r
        for j in self.inst.J:
            yj = y_rel[j, :]
            if np.allclose(yj, 0.0):
                continue
            order = np.argsort(-yj)  # desc
            S = []
            rS = 0.0
            for k in order:
                if yj[k] <= 0.0:
                    break
                S.append(k)
                rS += r[k]
                if rS > self.inst.q[j] + self.cut_tol:
                    # cover encontrado
                    overflow = rS - self.inst.q[j]
                    # RHS com b_j=1 (checagem conservadora)
                    rhs_const = np.sum(np.minimum(r, overflow))  # sum over all k
                    rhs = self.inst.q[j] * 1.0 + (
                        rhs_const - np.minimum(r[S], overflow).sum()
                    )
                    # LHS atual
                    lhs = yj[S].sum()
                    if lhs > rhs + self.cut_tol:
                        cuts.append(("depot_cover", j, tuple(S), overflow))
                    break
        return cuts

    def _separate_flow_covers_plant(self, x_rel: np.ndarray) -> list[tuple]:
        """
        'Flow-cover' na planta i.
        """
        cuts = []
        q = self.inst.q
        for i in self.inst.I:
            xi = x_rel[i, :]
            if np.allclose(xi, 0.0):
                continue
            order = np.argsort(-xi)  # desc
            T = []
            qT = 0.0
            for j in order:
                if xi[j] <= 0.0:
                    break
                T.append(j)
                qT += q[j]
                if qT > self.inst.p[i] + self.cut_tol:
                    overflow = qT - self.inst.p[i]
                    rhs_const = np.minimum(q, overflow).sum()
                    rhs = self.inst.p[i] * 1.0 + (
                        rhs_const - np.minimum(q[T], overflow).sum()
                    )
                    lhs = xi[T].sum()
                    if lhs > rhs + self.cut_tol:
                        cuts.append(("plant_cover", i, tuple(T), overflow))
                    break
        return cuts

    def _add_cuts_to_master(self, cuts: list[tuple]) -> int:
        """
        Adiciona cortes ao mestre.
        """
        if not cuts:
            return 0
        if self.master is None:
            self._build_master()

        m = self.master
        added = 0
        allK = set(self.inst.K)
        allJ = set(self.inst.J)

        for typ, idx, subset, overflow in cuts:
            if typ == "depot_cover":
                j = idx
                S = set(subset)
                compl = allK - S
                rhs_const = float(np.minimum(self.inst.r[list(compl)], overflow).sum())
                m.add(
                    m.sum(self._y[j, k] for k in S)
                    <= self.inst.q[j] * self._b[j] + rhs_const
                )
                added += 1
            else:  # "plant_cover"
                i = idx
                T = set(subset)
                compl = allJ - T
                rhs_const = float(np.minimum(self.inst.q[list(compl)], overflow).sum())
                m.add(
                    m.sum(self._x[i, j] for j in T)
                    <= self.inst.p[i] * self._a[i] + rhs_const
                )
                added += 1

        return added

    def _solve_lrp(self, lamb: np.ndarray):
        """
        Resolve a relaxação Lagrangeana para obter um limite inferior
        """
        nI, nJ, nK = self.inst.nI, self.inst.nJ, self.inst.nK

        alph = lamb[:nJ]
        beta = lamb[nJ:]

        # custos reduzidos
        ctil = self.inst.c + alph[None, :]  # (nI, nJ)
        dtil = self.inst.d - alph[:, None] + beta[None, :]  # (nJ, nK)

        a_rel = np.zeros(nI, dtype=np.int8)
        b_rel = np.zeros(nJ, dtype=np.int8)
        L_val = -np.dot(beta, self.inst.r)

        # agregadores para subgradientes
        sum_x_j = np.zeros(nJ, dtype=float)  # soma_i x[i,j]
        sum_y_j = np.zeros(nJ, dtype=float)  # soma_k y[j,k]
        sum_y_k = np.zeros(nK, dtype=float)  # soma_j y[j,k]

        # fluxos detalhados para cortes
        x_rel = np.zeros((nI, nJ), dtype=float)
        y_rel = np.zeros((nJ, nK), dtype=float)

        def _greedy_take(sorted_caps: np.ndarray, cap_total: float) -> np.ndarray:
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
                return (i, 0, 0.0, None, None)
            js = np.flatnonzero(mask)
            order = np.argsort(row[js])  # crescente
            js = js[order]
            rc = row[js]
            qj = self.inst.q[js]
            take_sorted = _greedy_take(qj, self.inst.p[i])
            var_part = float(np.dot(rc, take_sorted))
            open_i = (take_sorted.any()) and (self.inst.f[i] + var_part < 0.0)
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
            var_part = float(np.dot(rc, take_sorted))
            open_j = (take_sorted.any()) and (self.inst.g[j] + var_part < 0.0)
            if open_j:
                return j, 1, (self.inst.g[j] + var_part), ks, take_sorted
            else:
                return j, 0, 0.0, None, None

        # plantas (fora das threads aplicamos efeitos)
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor() as ex:
            for i, open_i, contrib, js, take in ex.map(_solve_plant, self.inst.I):
                if open_i:
                    a_rel[i] = 1
                    L_val += contrib
                    sum_x_j[js] += take
                    x_rel[i, js] = take

        # depósitos
        with ThreadPoolExecutor() as ex:
            for j, open_j, contrib, ks, take in ex.map(_solve_depot, self.inst.J):
                if open_j:
                    b_rel[j] = 1
                    L_val += contrib
                    tj = float(np.sum(take))
                    if not np.isclose(tj, 0.0):
                        sum_y_j[j] = tj
                        sum_y_k[ks] += take
                        y_rel[j, ks] = take

        # subgradientes (igualdades)
        subgrad = np.empty_like(lamb)
        subgrad[: self.inst.nJ] = sum_x_j - sum_y_j
        subgrad[self.inst.nJ :] = sum_y_k - self.inst.r

        return L_val, a_rel, b_rel, subgrad, x_rel, y_rel

    def solve(self) -> None:
        """
        Loop principal do NDRC.
        """
        time_start = perf_counter()

        while self.iter < self.max_iter and self.time <= self.time_limit:
            self.iter += 1

            # (1) LRP
            L_k, a_k, b_k, subgrad_k, x_rel, y_rel = self._solve_lrp(self.lamb)

            # (1.1) separa e adiciona cortes (non-delayed)
            cuts = []
            cuts += self._separate_flow_covers_depot(y_rel)
            cuts += self._separate_flow_covers_plant(x_rel)
            ncuts = self._add_cuts_to_master(cuts) if cuts else 0

            # (2) LB
            if L_k > self.L_best:
                self.L_best = L_k

            # (3) UB: resolver o master periodicamente (e na primeira iteração)
            if (
                (self.iter == 1)
                or (self.iter % self.solve_master_every == 0)
                or (ncuts > 0)
            ):
                if self.master is None:
                    self._build_master()
                # limite de tempo restante para o master
                remaining = max(0.0, self.time_limit - (perf_counter() - time_start))
                if remaining > 0.0:
                    self.master.parameters.timelimit = remaining
                sol = self.master.solve()
                if sol:
                    z_m = float(sol.objective_value)
                    if z_m < self.z_best:
                        self.z_best = z_m

            # (4) CA/PA/CI (igualdades → |g| > eps)
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

            # (5) passo do subgradiente
            denom = float(np.dot(subgrad_k, subgrad_k))
            if denom > 0.0:
                if np.isfinite(self.z_best):
                    mu = self.gamma * max(self.z_best - L_k, 0.0) / denom
                else:
                    mu = self.gamma / denom
                self.lamb = self.lamb + mu * subgrad_k

            # (6) parada
            if np.isfinite(self.z_best):
                self.gap = (self.z_best - self.L_best) / max(1.0, abs(self.z_best))
                if self.gap <= self.tol_stop:
                    if self.log_output:
                        print("Convergência: gap ≤ tol.")
                    break

            # (7) log
            self.time = perf_counter() - time_start
            if self.log_output and (self.iter % 20 == 0):
                norm_g = float(np.sqrt(denom))
                print(
                    f"[NDRC] it={self.iter:4d} time={round(self.time):4d}s  "
                    f"LRP={L_k:10.3f}  LB={self.L_best:10.3f}  UB={self.z_best:10.3f}  "
                    f"||g||={norm_g:.3e}  +cuts={ncuts:3d}  pool={len(self._cut_pool):4d}"
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
