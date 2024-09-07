from cbx.dynamics import CBO

f = lambda x: x[0] ** 2 + x[1] ** 2
dyn = CBO(f, d=2)
x = dyn.optimize()
