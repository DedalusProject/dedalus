

import numpy as np
from dedalus.core import coords, distributor, basis, field, operators, problems, solvers
from dedalus.core import future
from dedalus.tools.array import apply_matrix

N = 8
ang_res = 6

c = coords.SphericalCoordinates('phi', 'theta', 'r')
d = distributor.Distributor((c,))
b = basis.BallBasis(c, (ang_res, ang_res, N), radius=1)
phi, theta, r = b.local_grids((1, 1, 1))
x = r * np.sin(theta) * np.cos(phi)
y = r * np.sin(theta) * np.sin(phi)
z = r * np.cos(theta)

f = field.Field(dist=d, bases=(b.radial_basis,), dtype=np.complex128)
g = field.Field(dist=d, bases=(b,), dtype=np.complex128)
v = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)

f['g'] = r**6
g['g'] = 3*x**2 + 2*y*z
u = operators.Gradient(g, c).evaluate()

v['g'] = u['g']*g['g']
w_op = f * u

print(f.domain.bases[0].k)
print(u.domain.bases[0].k)
print(w_op.domain.bases[0].k)
w_op = w_op.reinitialize(ncc=True, ncc_vars=[u])
print(w_op.domain.bases[0].k)

