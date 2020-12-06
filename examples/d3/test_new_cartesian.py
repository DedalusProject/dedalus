
import numpy as np
from dedalus.core import coords, distributor, basis, field, operators
from mpi4py import MPI

comm = MPI.COMM_WORLD

results = []


## 2D Fourier * Chebyshev
c = coords.CartesianCoordinates('x', 'y')
d = distributor.Distributor((c,))
xb = basis.ComplexFourier(c.coords[0], size=16, bounds=(0, 2*np.pi))
yb = basis.ChebyshevT(c.coords[1], size=16, bounds=(0, 1))
x = xb.local_grid(1)
y = yb.local_grid(1)

# Gradient of a scalar
f = field.Field(dist=d, bases=(xb, yb), dtype=np.complex128)
f.name = 'f'
f['g'] = np.sin(x) * y**5
u = operators.Gradient(f, c)
u = u.evaluate()
u.name = 'u'
ug = np.array([np.cos(x) * y**5, np.sin(x) * 5 * y**4])
result = np.allclose(u['g'], ug)
results.append(result)
print(len(results), ':', result, '(gradient of a scalar)')

# Gradient of a vector
T = operators.Gradient(u, c)
T = T.evaluate()
T.name = 'T'
Tg = np.array([[-np.sin(x) * y**5, np.cos(x) * 5 * y**4],
               [np.cos(x) * 5 * y**4, np.sin(x) * 20 * y**3]])
result = np.allclose(T['g'], Tg)
results.append(result)
print(len(results), ':', result, '(gradient of a vector)')

# Divergence of a vector
h = operators.Divergence(u)
h = h.evaluate()
h.name = 'h'
hg = - np.sin(x) * y**5 + np.sin(x) * 20 * y**3
result = np.allclose(h['g'], hg)
results.append(result)
print(len(results), ':', result, '(divergence of a vector)')

# Laplacian of a scalar
k = operators.Laplacian(f, c)
k = k.evaluate()
k.name = 'k'
result = np.allclose(k['g'], hg)
results.append(result)
print(len(results), ':', result, '(laplacian of a scalar)')

