# pip install docplex
from docplex.cp.model import CpoModel

n = 50
H = [2, 4, 5, 6, 11, 12, 17, 20, 21, 25, 27, 28, 30, 31, 33, 37, 44, 46]
isH = {i: (i in H) for i in range(1, n + 1)}

mdl = CpoModel()

# Folds y_k for k = 1..n-1
y = {k: mdl.binary_var(name=f"y_{k}") for k in range(1, n)}

# --- BEST FIX: force-extract all y_k (even if unconstrained) ---
for k in range(1, n):
    mdl.add(y[k])  # ensures y_k is extracted and appears in the solution

# Eligible hydrophobic pairs
x = {}  # (i,j) -> var
for i in range(1, n + 1):
    if not isH[i]:
        continue
    for j in range(i + 1, n + 1):
        if not isH[j]:
            continue
        if j == i + 1:
            continue
        if (i + j - 1) % 2 != 0:
            continue

        x[(i, j)] = mdl.binary_var(name=f"x_{i}_{j}")
        m = (i + j - 1) // 2
        mdl.add(x[(i, j)] == y[m])  # exactly one fold at midpoint

        # forbid any other fold in [i, j)
        for k in range(i, j):
            if k != m:
                mdl.add(y[k] + x[(i, j)] <= 1)

# Objective
mdl.maximize(mdl.sum(x.values()))

# Solve
res = mdl.solve(TimeLimit=60, LogVerbosity="Quiet")

if res:
    print("Objective:", res.get_objective_values()[0])

    folds = [k for k in range(1, n) if res.get_value(y[k]) > 0.5]
    print("Folds (between k and k+1):", folds)

    pairs = [(i, j) for (i, j), var in x.items() if res.get_value(var) > 0.5]
    print("Matched hydrophobic pairs:", pairs)
else:
    print("No solution.")
