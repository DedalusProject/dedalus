
import numpy as np
from dedalus.core import coords, distributor, basis, field, operators
from mpi4py import MPI

comm = MPI.COMM_WORLD

results = []

## 1D complex Fourier
if comm.size == 1:
    c = coords.Coordinate('x')
    d = distributor.Distributor([c])
    xb = basis.ComplexFourier(c, size=16, bounds=(0, 1))
    x = xb.local_grid(1)
    # Scalar transforms
    u = field.Field(dist=d, bases=(xb,), dtype=np.complex128)
    ug = np.exp(2*np.pi*1j*x)
    u['g'] = ug
    u['c']
    result = np.allclose(u['g'], ug)
    results.append(result)
    print(len(results), ':', result, '(1D complex Fourier)')

## 1D Chebyshev
if comm.size == 1:
    c = coords.Coordinate('x')
    d = distributor.Distributor([c])
    xb = basis.ChebyshevT(c, size=16, bounds=(0, 1))
    x = xb.local_grid(1)
    # Scalar transforms
    u = field.Field(dist=d, bases=(xb,), dtype=np.complex128)
    u['g'] = ug = 2*x**2 - 1
    u['c']
    result = np.allclose(u['g'], ug)
    results.append(result)
    print(len(results), ':', result, '(1D complex Chebyshev)')

## 2D Fourier * Chebyshev
c = coords.CartesianCoordinates('x', 'y')
d = distributor.Distributor(c.coords)
xb = basis.ComplexFourier(c.coords[0], size=8, bounds=(0, 2*np.pi))
yb = basis.ChebyshevT(c.coords[1], size=16, bounds=(0, 1))
x = xb.local_grid(1)
y = yb.local_grid(1)

# Scalar transforms
f = field.Field(dist=d, bases=(xb,yb,), dtype=np.complex128)
f['g'] = fg = np.sin(x) * y**5
f['c']
result = np.allclose(f['g'], fg)
results.append(result)
print(len(results), ':', result, '(Fourier x Chebyshev scalar)')

# Vector transforms
u = field.Field(dist=d, bases=(xb,yb,), tensorsig=(c,), dtype=np.complex128)
u['g'] = ug = np.array([np.cos(x) * 2 * y**2, np.sin(x) * y + y])
u['c']
result = np.allclose(u['g'], ug)
results.append(result)
print(len(results), ':', result, '(Fourier x Chebyshev vector)')

# Vector transforms 1D
v = field.Field(dist=d, bases=(xb,), tensorsig=(c,), dtype=np.complex128)
v['g'] = vg = np.array([np.cos(x) * 2, np.sin(x) + 1])
print(v['g'].shape)
v['c']
result = np.allclose(v['g'], vg)
results.append(result)
print(len(results), ':', result, '(Fourier x Chebyshev x-dependent vector)')

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
print(len(results), ':', result, '(S2 scalar)')

# Vector transforms
u = field.Field(dist=d, bases=(sb,), tensorsig=(c,), dtype=np.complex128)
u['c'][0,2,3] = 1
ug0 = - 1j * np.sqrt(35/512) / 2 * (np.sin(theta) - 4*np.sin(2*theta) - 3*np.sin(3*theta)) * np.exp(2j*phi)
result = np.allclose(u['g'][0], ug0)
results.append(result)
print(len(results), ':', result, '(S2 vector)')

# Tensor transforms
T = field.Field(dist=d, bases=(sb,), tensorsig=(c,c), dtype=np.complex128)
T['c'][0,0,2,3] = 1
Tg00 = - 0.5 * np.sqrt(7/2) * (np.cos(theta/2)**4 * (-2 + 3*np.cos(theta))) * np.exp(2j*phi)
result = np.allclose(T['g'][0,0], Tg00)
results.append(result)
print(len(results), ':', result, '(S2 tensor)')

## S2, 3D vectors
c = coords.SphericalCoordinates('phi', 'theta', 'r')
d = distributor.Distributor( (c,) )
c_S2 = c.S2coordsys
sb = basis.SpinWeightedSphericalHarmonics(c_S2, (32,16), radius=1)

u = field.Field(dist=d, bases=(sb,), tensorsig=(c,), dtype=np.complex128)
u['c'][0,2,3] = 1
u['c'][2,-2,2] = 1
ug0 = - 1j * np.sqrt(35/512) / 2 * (np.sin(theta) - 4*np.sin(2*theta) - 3*np.sin(3*theta)) * np.exp(2j*phi)
ug2 =   np.sqrt(15) / 4 * np.sin(theta)**2 * np.exp(-2j*phi)
result = np.allclose(u['g'][0,:,:,0],ug0) and np.allclose(u['g'][2,:,:,0],ug2)
results.append(result)
print(len(results), ':', result, '(S2, 3D vector)')

## Spherical Shell
c = coords.SphericalCoordinates('phi', 'theta', 'r')
d = distributor.Distributor(c.coords)
b = basis.SphericalShellBasis(c, (16,16,16), radii=(1,3))
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
print(len(results), ':', result, '(Spherical Shell scalar)')

# Vector transforms
u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
ug = np.copy(u['g'])
# phi-dependent function
# note: this function is the gradient of a scalar
ug[2] =  4*r**3*np.sin(theta)*np.cos(theta)*np.exp(1j*phi)
ug[1] =    r**3*np.cos(2*theta)*np.exp(1j*phi)
ug[0] = 1j*r**3*np.cos(theta)*np.exp(1j*phi)
u['g'] = ug
u['c']
result = np.allclose(u['g'], ug)
results.append(result)
print(len(results), ':', result, '(Spherical Shell vector)')

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
print(len(results), ':', result, '(Ball scalar)')

# Vector transforms
u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
ug = np.copy(u['g'])
# phi-dependent function
# note: this function is the gradient of a scalar
ug[2] =  4*r**3*np.sin(theta)*np.cos(theta)*np.exp(1j*phi)
ug[1] =    r**3*np.cos(2*theta)*np.exp(1j*phi)
ug[0] = 1j*r**3*np.cos(theta)*np.exp(1j*phi)
u['g'] = ug
u['c']
result = np.allclose(u['g'], ug)
results.append(result)
print(len(results), ':', result, '(Ball vector)')

