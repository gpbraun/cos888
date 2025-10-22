# test_mip.py
from docplex.mp.model import Model

m = Model()
x = m.binary_var()
y = m.binary_var()
m.maximize(3 * x + 2 * y)
m.add(x + 2 * y <= 2)
print(m.solve(log_output=True).objective_value)
