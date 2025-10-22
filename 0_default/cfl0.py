# pip install docplex
from docplex.mp.model import Model


def read_cap_orlib(path: str):
    """
    OR-Library CAP format (cap41, ..., cap134; capa/b/c):
      line 1: m n
      next m lines (or tokens): for i=1..m -> (capacity_i, fixed_i)
      next n blocks: for j=1..n -> demand_j, then m costs c_{1j}..c_{mj}
    Costs c_{ij} are for serving ALL of demand j from facility i.
    """
    with open(path, "r") as f:
        toks = f.read().split()
    it = iter(toks)

    m = int(next(it))
    n = int(next(it))

    cap = [0.0] * m
    fixed = [0.0] * m
    for i in range(m):
        cap[i] = float(next(it))
        fixed[i] = float(next(it))

    dem = [0.0] * n
    # cost[i][j] with i in 0..m-1, j in 0..n-1
    cost = [[0.0] * n for _ in range(m)]
    for j in range(n):
        dem[j] = float(next(it))
        for i in range(m):
            cost[i][j] = float(next(it))

    return m, n, cap, fixed, dem, cost


def build_and_solve(path: str):
    m, n, cap, fixed, dem, cost = read_cap_orlib(path)
    I = range(m)
    J = range(n)

    mdl = Model(name="CAP_Beasley_DOCplex", log_output=True)

    # Decision vars
    y = mdl.binary_var_dict(I, name="y")  # open facility?
    x = mdl.continuous_var_dict(
        ((i, j) for i in I for j in J), lb=0.0, ub=1.0, name="x"
    )  # fraction of demand j served by i

    # Each customer fully served
    for j in J:
        mdl.add_constraint(mdl.sum(x[i, j] for i in I) == 1, f"cover_{j}")

    # Capacities
    for i in I:
        mdl.add_constraint(
            mdl.sum(dem[j] * x[i, j] for j in J) <= cap[i] * y[i], f"cap_{i}"
        )

    # Standard linking (tightened version with min{1, cap_i/dem_j} also OK)
    for i in I:
        for j in J:
            mdl.add_constraint(x[i, j] <= y[i], f"link_{i}_{j}")

    # Objective: c_ij already equals the cost of serving ALL of j from i
    # so use c_ij * x_ij (DO NOT multiply by demand again)
    assign_cost = mdl.sum(cost[i][j] * x[i, j] for i in I for j in J)
    fixed_cost = mdl.sum(fixed[i] * y[i] for i in I)
    mdl.minimize(assign_cost + fixed_cost)

    # Quick sanity check
    tot_dem = sum(dem)
    tot_cap = sum(cap)
    print(f"m={m}, n={n}, total demand={tot_dem:.3f}, total capacity={tot_cap:.3f}")

    sol = mdl.solve()
    if not sol:
        print("No solution found.")
        return None

    print(f"Objective value: {sol.objective_value:.2f}")
    opened = [i for i in I if y[i].solution_value > 0.5]
    print(f"Opened facilities ({len(opened)}): {opened}")

    print("\nShipments (i -> j : qty) with qty > 1e-3:")
    for i in I:
        for j in J:
            frac = x[i, j].solution_value
            qty = frac * dem[j]
            if qty > 1e-3:
                print(f"{i:2d} -> {j:2d} : {qty:.3f}")

    return sol


if __name__ == "__main__":
    PATH = "/home/braun/Documents/Developer/cos888/instances/cfl/capc1.txt"
    build_and_solve(PATH)
