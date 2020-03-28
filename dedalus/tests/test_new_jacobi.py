

import numpy as np
from dedalus.core import coords, distributor, basis, field, operators
from mpi4py import MPI

comm = MPI.COMM_WORLD

results = []


# Jacobi derivatives
c = coords.CartesianCoordinates('x')
d = distributor.Distributor(c.coords)
xb = basis.ChebyshevT(c.coords[0], size=16, bounds=(0, 1))
x = xb.local_grid(1)
f = field.Field(dist=d, bases=[xb], dtype=np.complex128)
f['g'] = x**5
fx= operators.Differentiate(f, c.coords[0]).evaluate()
fxg = 5 * x**4
result = np.allclose(fx['g'], fxg)
results.append(result)
print(len(results), ':', result, '(Jacobi 1st derivative)')
fxx = operators.Differentiate(fx, c.coords[0]).evaluate()
fxxg = 20 * x**3
result = np.allclose(fxx['g'], fxxg)
results.append(result)
print(len(results), ':', result, '(Jacobi 2nd derivative)')

# Jacobi conversion
f['c']
fx['c']
q = (f + fx).evaluate()
qg = f['g'] + fx['g']
result = np.allclose(q['g'], qg)
results.append(result)
print(len(results), ':', result, '(Jacobi conversion 1)')
f['c']
fxx['c']
q = (f + fxx).evaluate()
qg = f['g'] + fxx['g']
result = np.allclose(q['g'], qg)
results.append(result)
print(len(results), ':', result, '(Jacobi conversion 2)')
