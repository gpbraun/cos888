"""
COS888

sscfl_experiments.py

Gabriel Braun, 2025
"""

from typing import Iterable

from sscfl import SSCFLInstance, SSCFLSolverCP, SSCFLSolverMP

# Instâncias dos testes
INSTANCES = [
    "holmberg/sscfl_h_40",
]

# Separação no log
SEP = "    "


def run_experiments(
    instances: Iterable[str],
    output_path: str = "out/sscfl_out.txt",
    time_limit: float | None = 100.0,
):
    """
    Roda MP e CP para cada instância e imprime resumo no formato desejado.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 52 + "\n")

        for idx, path in enumerate(instances, start=1):
            f.write(f"#{idx}. {path}\n\n")

            inst = SSCFLInstance.load(path)

            # CPLEX
            solver_mp = SSCFLSolverMP(inst)
            solver_mp.solve(time_limit=time_limit, log_output=False)

            f.write(
                f"[MP]{SEP}"
                f"lb={solver_mp.lb:.0f}{SEP}"
                f"ub={solver_mp.ub:.0f}{SEP}"
                f"nodes={solver_mp.nodes:d}{SEP}"
                f"time={solver_mp.time:.1f}\n"
            )
            f.flush()

            # CONSTRAINT PROGRAMMING
            solver_cp = SSCFLSolverCP(inst)
            solver_cp.solve(time_limit=time_limit, log_output=False)

            f.write(
                f"[CP]{SEP}"
                f"lb={solver_cp.lb:.0f}{SEP}"
                f"ub={solver_cp.ub:.0f}{SEP}"
                f"branches={solver_cp.branches:d}{SEP}"
                f"time={solver_cp.time:.1f}\n\n"
            )
            f.flush()


def main():
    run_experiments(INSTANCES)


if __name__ == "__main__":
    main()
