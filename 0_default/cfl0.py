# cap_solve_simple.py
# pip install docplex
import sys

from docplex.mp.model import Model


def read_cap(path):
    """OR-Library format:
    m n
    (m lines): capacity_i fixed_cost_i
    (n lines): demand_j  c_1j ... c_mj   (cost to assign all of j to i)
    """
    toks = open(path).read().split()
    it = iter(toks)
    m, n = int(next(it)), int(next(it))
    cap, f = [], []
    for _ in range(m):
        cap.append(float(next(it)))
        f.append(float(next(it)))
    d = [0.0] * n
    # c[i][j] matrix
    c = [[0.0] * n for _ in range(m)]
    for j in range(n):
        d[j] = float(next(it))
        for i in range(m):
            c[i][j] = float(next(it))
    return m, n, cap, f, d, c


def solve_cap(path):
    m, n, cap, f, d, c = read_cap(path)
    mdl = Model(name=path)

    # variables
    y = mdl.binary_var_list(m, name="y")  # open facility i?
    x = mdl.binary_var_matrix(m, n, name="x")  # assign customer j to facility i?

    # constraints
    for j in range(n):
        mdl.add(mdl.sum(x[i, j] for i in range(m)) == 1)  # one facility per customer
    for i in range(m):
        mdl.add(mdl.sum(d[j] * x[i, j] for j in range(n)) <= cap[i] * y[i])  # capacity
    mdl.add_constraints(x[i, j] <= y[i] for i in range(m) for j in range(n))  # link

    # objective
    mdl.minimize(
        mdl.sum(f[i] * y[i] for i in range(m))
        + mdl.sum(c[i][j] * x[i, j] for i in range(m) for j in range(n))
    )

    sol = mdl.solve(log_output=True)
    if not sol:
        print("No solution.")
        return

    obj = sol.objective_value
    opened = [i for i in range(m) if sol.get_value(y[i]) > 0.5]

    print(f"Instance: {path}")
    print(f"Objective: {obj:.3f}")
    print(f"Opened facilities ({len(opened)}/{m}): {opened}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python cap_solve_simple.py /path/to/cap41.txt")
    else:
        solve_cap(sys.argv[1])
