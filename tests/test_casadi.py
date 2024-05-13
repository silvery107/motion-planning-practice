import casadi as ca
import numpy as np


opti = ca.Opti()

x = opti.variable()
y = opti.variable()

opti.minimize((y-x**2)**2)
opti.subject_to(x**2+y**2==1)
opti.subject_to(x+y>=1)

opti.solver('ipopt')
sol = opti.solve()

sol.value(x)
sol.value(y)