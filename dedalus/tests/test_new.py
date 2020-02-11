
import numpy as np
from dedalus.core import coords, distributor, basis, field

results = []

## 1D complex Fourier
c = coords.Coordinate('x')
d = distributor.Distributor([c])
xb = basis.ComplexFourier(c, size=16, bounds=(0, 1))
x = xb.local_grid(1)
# Scalar transforms
u = field.Field(dist=d, bases=[xb], dtype=np.complex128)
ug = np.exp(2*np.pi*1j*x)
u['g'] = ug
u['c']
result = np.allclose(u['g'], ug)
results.append(result)
print(len(results), ':', result)

## 1D Chebyshev
c = coords.Coordinate('x')
d = distributor.Distributor([c])
xb = basis.ChebyshevT(c, size=16, bounds=(0, 1))
x = xb.local_grid(1)
# Scalar transforms
u = field.Field(dist=d, bases=[xb], dtype=np.complex128)
u['g'] = ug = 2*x**2 - 1
u['c']
result = np.allclose(u['g'], ug)
results.append(result)
print(len(results), ':', result)

## 2D Fourier * Chebyshev
c = coords.CartesianCoordinates('x', 'y')
d = distributor.Distributor(c.coords)
xb = basis.ComplexFourier(c.coords[0], size=8, bounds=(0, 2*np.pi))
yb = basis.ChebyshevT(c.coords[1], size=16, bounds=(0, 1))
x = xb.local_grid(1)
y = yb.local_grid(1)
# Scalar transforms
u = field.Field(dist=d, bases=[xb,yb], dtype=np.complex128)
u['g'] = ug = 2*y**3 + np.cos(x)
u['c']
result = np.allclose(u['g'], ug)
results.append(result)
print(len(results), ':', result)
# Vector transforms
u = field.Field(dist=d, bases=[xb,yb], tensorsig=[c], dtype=np.complex128)
u['g'] = ug = [np.cos(x) * 2 * y**2, np.sin(x) * y + y]
u['c']
result = np.allclose(u['g'], ug)
results.append(result)
print(len(results), ':', result)
# Vector transforms 1D
u = field.Field(dist=d, bases=[xb], tensorsig=[c], dtype=np.complex128)
u['g'] = ug = [np.cos(x) * 2, np.sin(x) + 1]
u['c']
result = np.allclose(u['g'], ug)
results.append(result)
print(len(results), ':', result)

## S2
c = coords.S2Coordinates('phi','theta')
d = distributor.Distributor(c.coords)
sb = basis.SpinWeightedSphericalHarmonics(c, (32,16), radius=1)
phi, theta = sb.local_grids((1, 1))
# Scalar transforms
u = field.Field(dist=d, bases=(sb,), dtype=np.complex128)
u['c'][-2, 2] = 1
ug = np.sqrt(15) / 4 * np.sin(theta)**2 * np.exp(-2j*phi)
result = np.allclose(u['g'], ug)
results.append(result)
print(len(results), ':', result)
# Vector transforms
u = field.Field(dist=d, bases=(sb,), tensorsig=(c,), dtype=np.complex128)
u['c'][0,2,3] = 1
ug0 = - 1j * np.sqrt(35/512) / 2 * (np.sin(theta) - 4*np.sin(2*theta) - 3*np.sin(3*theta)) * np.exp(2j*phi)
result = np.allclose(u['g'][0], ug0)
results.append(result)
print(len(results), ':', result)
# Tensor transforms
T = field.Field(dist=d, bases=(sb,), tensorsig=(c,c), dtype=np.complex128)
T['c'][0,0,2,3] = 1
Tg00 = - 0.5 * np.sqrt(7/2) * (np.cos(theta/2)**4 * (-2 + 3*np.cos(theta))) * np.exp(2j*phi)
result = np.allclose(T['g'][0,0], Tg00)
results.append(result)
print(len(results), ':', result)

raise

c = coords.SphericalCoordinates('phi', 'theta', 'r')
d = distributor.Distributor(c.coords)
b = basis.BallBasis(c, 7, 7, 1, fourier_library='matrix')

f = field.Field(dist=d, bases=(b,), dtype=np.complex128)

f['c'][2,4,5] = 1.
fc0 = f['c'].copy()
f['g']
print('Ball scalar transform check:')
print(np.allclose(f['c'],fc0))

