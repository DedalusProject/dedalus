
import numpy as np
from dedalus.core import coords, distributor, basis, field, operators

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
f = field.Field(dist=d, bases=[xb,yb], dtype=np.complex128)
f['g'] = fg = np.sin(x) * y**5
f['c']
result = np.allclose(f['g'], fg)
results.append(result)
print(len(results), ':', result)
# Vector transforms
u = field.Field(dist=d, bases=[xb,yb], tensorsig=[c], dtype=np.complex128)
u['g'] = ug = np.array([np.cos(x) * 2 * y**2, np.sin(x) * y + y])
u['c']
result = np.allclose(u['g'], ug)
results.append(result)
print(len(results), ':', result)
# Vector transforms 1D
v = field.Field(dist=d, bases=[xb], tensorsig=[c], dtype=np.complex128)
v['g'] = vg = np.array([np.cos(x) * 2, np.sin(x) + 1])
v['c']
result = np.allclose(v['g'], vg)
results.append(result)
print(len(results), ':', result)
# Gradient operator
w = operators.Gradient(f, c).evaluate()
wg = np.array([np.cos(x) * y**5, np.sin(x) * 5 * y**4])
result = np.allclose(w['g'], wg)
results.append(result)
print(len(results), ':', result)
# Vector addition 2D+2D
uw = u + w
uw = uw.evaluate()
result = np.allclose(uw['g'], ug+wg)
results.append(result)
print(len(results), ':', result)
# Vector addition 1D+2D
# u.domain + v.domain
# uv = u + v
# print(uv.bases, uv.tensorsig)
# uv = uv.evaluate()
# result = np.allclose(uv['g'], ug+vg)
# results.append(result)
# print(len(results), ':', result)
# raise

## S2
c = coords.S2Coordinates('phi', 'theta')
d = distributor.Distributor(c.coords)
sb = basis.SpinWeightedSphericalHarmonics(c, (32,16), radius=1)
phi, theta = sb.local_grids((1, 1))
# Scalar transforms
f = field.Field(dist=d, bases=(sb,), dtype=np.complex128)
f['c'][-2,2] = 1
fg = np.sqrt(15) / 4 * np.sin(theta)**2 * np.exp(-2j*phi)
result = np.allclose(f['g'], fg)
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
# Gradient of a scalar
f['g'] = 1/2*np.sin(theta)**2*np.exp(-2j*phi)
u = operators.Gradient(f, c).evaluate()
ug = [-1j * np.sin(theta)*np.exp(-2j*phi), np.cos(theta)*np.sin(theta)*np.exp(-2j*phi)]
result = np.allclose(u['g'], ug)
results.append(result)
print(len(results), ':', result)
# Gradient of a vector
T = operators.Gradient(u, c).evaluate()
Tg = np.array([[(-2+np.cos(theta)**2)*np.exp(-2j*phi),-1j*np.cos(theta)*np.exp(-2j*phi)],
               [-1j*np.cos(theta)    *np.exp(-2j*phi),np.cos(2*theta)  *np.exp(-2j*phi)]])
result = np.allclose(T['g'], Tg)
results.append(result)
print(len(results), ':', result)

## Ball
c = coords.SphericalCoordinates('phi', 'theta', 'r')
d = distributor.Distributor(c.coords)
b = basis.BallBasis(c, (16,16,16), radius=1)
phi, theta, r = b.local_grids((1, 1, 1))
x = r * np.sin(theta) * np.cos(phi)
y = r * np.sin(theta) * np.sin(phi)
z = r * np.cos(theta)
# Scalar transforms
f = field.Field(dist=d, bases=(b,), dtype=np.complex128)
f['g'] = fg = 3*x**2 + 2*y*z
f['c']
result = np.allclose(f['g'], fg)
results.append(result)
print(len(results), ':', result)
# Vector transforms
u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
ug = np.copy(u['g'])
# component order is phi, theta, r
ug[2] =  4*r**3*np.cos(theta)**2
ug[1] = -2*r**3*np.cos(theta)*np.sin(theta)
u['g'] = ug
u['c']
result = np.allclose(u['g'], ug)
results.append(result)
print(len(results), ':', result)
# phi-dependent function
# note: both these test functions are the gradient of a scalar
ug[2] =  4*r**3*np.sin(theta)*np.cos(theta)*np.exp(1j*phi)
ug[1] =    r**3*np.cos(2*theta)*np.exp(1j*phi)
ug[0] = 1j*r**3*np.cos(theta)*np.exp(1j*phi)
u['g'] = ug
u['c']
result = np.allclose(u['g'], ug)
results.append(result)
print(len(results), ':', result)
# gradient of a scalar
u = operators.Gradient(f, c).evaluate()
ug[2] = (6*x**2+4*y*z)/r
ug[1] = -2*(y**3+x**2*(y-3*z)-y*z**2)/(r**2*np.sin(theta))
ug[0] = 2*x*(-3*y+z)/(r*np.sin(theta))
result = np.allclose(u['g'], ug)
results.append(result)
print(len(results), ':', result)


