"""
COS888

sscfl_instance.py

Gabriel Braun, 2025
"""

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np

INSTANCES_DIR = Path(__file__).resolve().parent.parent.joinpath("data")


@dataclass(frozen=True)
class SSCFLInstance:
    """
    Instância do SSCFL.
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
    def read(cls, path: Path | str) -> "SSCFLInstance":
        """
        Retorna: Instância SSCFL a partir de um arquivo `.txt`.
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

    @classmethod
    def load(cls, name: str) -> "SSCFLInstance":
        """
        Retorna: Instância SSCFL a partir de um arquivo `.txt` no diretório padrão.
        """
        return cls.read(INSTANCES_DIR.joinpath(name).with_suffix(".txt"))
