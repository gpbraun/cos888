"""
COS888

sscfl_experiments.py

Gabriel Braun, 2025
"""

from pathlib import Path
from typing import Iterable

from sscfl import SSCFLInstance, SSCFLSolverCP, SSCFLSolverMP

# Instâncias dos testes
INSTANCES = [
    "holmberg/sscfl_h_01",
    "holmberg/sscfl_h_40",
]


def log_num(label, value, fmt, width=15):
    return f"{label}={value:{fmt}}".ljust(width)


def run_experiments(
    instances: Iterable[str],
    output_path: str = Path("out/sscfl_out.txt"),
    time_limit: float | None = 100.0,
):
    """
    Roda MP e CP para cada instância e imprime resumo no formato desejado.
    """
    Path(output_path).parent.mkdir(exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for idx, path in enumerate(instances, start=1):
            f.write("=" * 75 + "\n")
            f.write(f"#{idx}. {path}\n\n")

            inst = SSCFLInstance.load(path)

            # CPLEX
            solver_mp = SSCFLSolverMP(inst)
            solver_mp.solve(time_limit=time_limit, log_output=False)

            line = "".join(
                [
                    "[ MP ]    ",
                    log_num("lb", solver_mp.lb, ".0f"),
                    log_num("ub", solver_mp.ub, ".0f"),
                    log_num("nodes", solver_mp.nodes, "d", width=18),
                    log_num("time", solver_mp.time, ".1f"),
                    "\n",
                ]
            )
            f.write(line)
            f.flush()

            # CONSTRAINT PROGRAMMING
            solver_cp = SSCFLSolverCP(inst)
            solver_cp.solve(time_limit=time_limit, log_output=False)

            line = "".join(
                [
                    f"[ CP ]    ",
                    log_num("lb", solver_cp.lb, ".0f"),
                    log_num("ub", solver_cp.ub, ".0f"),
                    log_num("brchs", solver_cp.branches, "d", width=18),
                    log_num("time", solver_cp.time, ".1f"),
                    "\n",
                ]
            )
            f.write(line)
            f.write("\n")
            f.flush()


def main():
    run_experiments(INSTANCES)


if __name__ == "__main__":
    main()
