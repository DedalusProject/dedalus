
import numpy as np
from dedalus.core import coords, distributor, basis, field, operators
from dedalus.core import future
from mpi4py import MPI

comm = MPI.COMM_WORLD

results = []

## Cartesian
c = coords.CartesianCoordinates('x', 'y')
d = distributor.Distributor((c,))
xb = basis.ComplexFourier(c.coords[0], size=8, bounds=(0, 2*np.pi))
yb = basis.ChebyshevT(c.coords[1], size=16, bounds=(0, 1))
x = xb.local_grid(1)
y = yb.local_grid(1)

# Gradient operator
f = field.Field(dist=d, bases=(xb,yb,), dtype=np.complex128)
f['g'] = fg = np.sin(x) * y**5
w = operators.Gradient(f, c).evaluate()
wg = np.array([np.cos(x) * y**5, np.sin(x) * 5 * y**4])
result = np.allclose(w['g'], wg)
results.append(result)
print(len(results), ':', result)

# Vector addition 2D+2D
u = field.Field(dist=d, bases=(xb,yb,), tensorsig=(c,), dtype=np.complex128)
u['g'] = ug = np.array([np.cos(x) * 2 * y**2, np.sin(x) * y + y])
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
d = distributor.Distributor((c,))
sb = basis.SpinWeightedSphericalHarmonics(c, (32,16), radius=1)
phi, theta = sb.local_grids((1, 1))

# Gradient of a scalar
f = field.Field(dist=d, bases=(sb,), dtype=np.complex128)
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
d = distributor.Distributor((c,))
b = basis.BallBasis(c, (16,16,16), radius=1)
phi, theta, r = b.local_grids((1, 1, 1))
x = r * np.sin(theta) * np.cos(phi)
y = r * np.sin(theta) * np.sin(phi)
z = r * np.cos(theta)

# gradient of a scalar
f = field.Field(dist=d, bases=(b,), dtype=np.complex128)
f['g'] = fg = 3*x**2 + 2*y*z
u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
ug = np.copy(u['g'])
u = operators.Gradient(f, c).evaluate()
ug[2] = (6*x**2+4*y*z)/r
ug[1] = -2*(y**3+x**2*(y-3*z)-y*z**2)/(r**2*np.sin(theta))
ug[0] = 2*x*(-3*y+z)/(r*np.sin(theta))
result = np.allclose(u['g'], ug)
results.append(result)
print(len(results), ':', result)

# gradient of a vector
T = operators.Gradient(u, c).evaluate()
Tg0 = np.copy(T['g'])
Tg0[2,2] = (6*x**2+4*y*z)/r**2
Tg0[2,1] = Tg0[1,2] = -2*(y**3+x**2*(y-3*z)-y*z**2)/(r**3*np.sin(theta))
Tg0[2,0] = Tg0[0,2] = 2*x*(z-3*y)/(r**2*np.sin(theta))
Tg0[1,1] = 6*x**2/(r**2*np.sin(theta)**2) - (6*x**2+4*y*z)/r**2
Tg0[1,0] = Tg0[0,1] = -2*x*(x**2+y**2+3*y*z)/(r**3*np.sin(theta)**2)
Tg0[0,0] = 6*y**2/(x**2+y**2)
result = np.allclose(T['g'], Tg0)
results.append(result)
print(len(results), ':', result)

# Cross product
f['g'] = z
ez = operators.Gradient(f, c).evaluate()
h = operators.CrossProduct(ez,u).evaluate()
hg = np.zeros(h['g'].shape, dtype=h['g'].dtype)
hg[0] = - ez['g'][1]*u['g'][2] + ez['g'][2]*u['g'][1]
hg[1] = - ez['g'][2]*u['g'][0] + ez['g'][0]*u['g'][2]
hg[2] = - ez['g'][0]*u['g'][1] + ez['g'][1]*u['g'][0]
result = np.allclose(h['g'],hg)
results.append(result)
print(len(results), ':', result)

# Dot product, vector-vector
h = operators.DotProduct(ez,u).evaluate()
hg = np.sum(ez['g']*u['g'],axis=0)
result = np.allclose(h['g'],hg)
results.append(result)
print(len(results), ':', result)

# Dot product, tensor-vector
h = operators.DotProduct(T,u).evaluate()
hg = np.sum(T['g']*u['g'][None,:,:,:,:],axis=1)
result = np.allclose(h['g'],hg)
results.append(result)
print(len(results), ':', result)

# Dot product, tensor-vector, using indices
op = u + operators.DotProduct(T,u,indices=(0,0))
h = op.evaluate()
hg = u['g'] + np.sum(T['g']*u['g'][:,None,:,:,:],axis=0)
result = np.allclose(h['g'],hg)
results.append(result)
print(len(results), ':', result)

# Divergence
f = field.Field(dist=d, bases=(b,), dtype=np.complex128)
f['g'] = x**3 + 2*y**3 + 3*z**3
u = operators.Gradient(f, c)
h = operators.Divergence(u).evaluate()
hg = 6*x + 12*y + 18*z
result = np.allclose(h['g'],hg)
results.append(result)
print(len(results), ':', result)

# Curl
u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
ct = np.cos(theta)
st = np.sin(theta)
cp = np.cos(phi)
sp = np.sin(phi)
u['g'][2] = r**2*st*(2*ct**2*cp-r*ct**3*sp+r**3*cp**3*st**5*sp**3+r*ct*st**2*(cp**3+sp**3))
u['g'][1] = r**2*(2*ct**3*cp-r*cp**3*st**4+r**3*ct*cp**3*st**5*sp**3-1/16*r*np.sin(2*theta)**2*(-7*sp+np.sin(3*phi)))
u['g'][0] = r**2*sp*(-2*ct**2+r*ct*cp*st**2*sp-r**3*cp**2*st**5*sp**3)
v = operators.Curl(u).evaluate()
v0 = 0*u['g']
v0[2] = -r*st*(r*ct**2*cp+r*cp*st**2*sp*(3*cp+sp)+ct*sp*(-4+3*r**3*cp**2*st**3*sp))
v0[1] = r*(-r*ct**3*cp+4*ct**2*sp+3*r**3*cp**2*st**5*sp**2-r*ct*cp*st**2*sp*(3*cp+sp))
v0[0] = r*(4*ct*cp+r*ct**2*sp+r*st**2*(-3*cp**3+sp**3))
result = np.allclose(v['g'],v0)
results.append(result)
print(len(results), ':', result)

# Laplacian scalar
f = field.Field(dist=d, bases=(b,), dtype=np.complex128)
f['g'] = x**4 + 2*y**4 + 3*z**4
h = operators.Laplacian(f, c).evaluate()
result = np.allclose(h['g'],12*x**2+24*y**2+36*z**2)
results.append(result)
print(len(results), ':', result)

# Laplacian vector
v = operators.Laplacian(u, c).evaluate()
v0 = 0*u['g']
v0[2] = 2*(2+3*r*ct)*cp*st+1/2*r**3*st**4*(4*np.sin(2*phi)+np.sin(4*phi))
v0[1] = 2*r*(-3*cp*st**2+sp)+1/2*ct*(8*cp+r**3*st**3*(4*np.sin(2*phi)+np.sin(4*phi)))
v0[0] = 2*r*ct*cp+2*sp*(-2-r**3*(2+np.cos(2*phi))*st**3*sp)
result = np.allclose(v['g'],v0)
results.append(result)
print(len(results), ':', result)

# 2nd order Convert test
u['g']
v['g']
op = u + v
w = op.evaluate()
result = np.allclose(w['g'],u['g']+v['g'])
results.append(result)
print(len(results), ':', result)

# Transpose
T = operators.Gradient(u, c).evaluate()
T['g']
Tg0 = np.transpose(np.copy(T['g']),(1,0,2,3,4))

# grid-space transpose
T = operators.TransposeComponents(T).evaluate()
result = np.allclose(T['g'],Tg0)
results.append(result)
print(len(results), ':', result)

# coeff-space transpose
T = operators.Gradient(u, c).evaluate()
T = operators.TransposeComponents(T).evaluate()
result = np.allclose(T['g'],Tg0)
results.append(result)
print(len(results), ':', result)

# Scalar Interpolation
f = field.Field(dist=d, bases=(b,), dtype=np.complex128)
f['g'] = x**4 + 2*y**4 + 3*z**4
h = operators.interpolate(f,r=1).evaluate()
h0 = 3*np.cos(theta)**4 + np.cos(phi)**4*np.sin(theta)**4 + 2*np.sin(theta)**4*np.sin(phi)**4
result = np.allclose(h['g'],h0)
results.append(result)
print(len(results), ':', result)

# Vector Interpolation
u['g'][2] = r**2*st*(2*ct**2*cp-r*ct**3*sp+r**3*cp**3*st**5*sp**3+r*ct*st**2*(cp**3+sp**3))
u['g'][1] = r**2*(2*ct**3*cp-r*cp**3*st**4+r**3*ct*cp**3*st**5*sp**3-1/16*r*np.sin(2*theta)**2*(-7*sp+np.sin(3*phi)))
u['g'][0] = r**2*sp*(-2*ct**2+r*ct*cp*st**2*sp-r**3*cp**2*st**5*sp**3)
v = operators.interpolate(u,r=1).evaluate()
v0[0] = sp*(-2*ct**2+ct*cp*st**2*sp-cp**2*st**5*sp**3)
v0[1] = (2*ct**3*cp-cp**3*st**4+ct*cp**3*st**5*sp**3-1/16*np.sin(2*theta)**2*(-7*sp+np.sin(3*phi)))
v0[2] = st*(2*ct**2*cp-ct**3*sp+cp**3*st**5*sp**3+ct*st**2*(cp**3+sp**3))
result = np.allclose(v['g'],v0)
results.append(result)
print(len(results), ':', result)

b_S2 = b.S2_basis()
v_S2 = field.Field(dist=d, bases=(b_S2,), tensorsig=(c,), dtype=np.complex128)
v_S2['g'][:,:,:,0] = v0[:,:,:,0]
result = np.allclose(v['c'],v_S2['c'])
results.append(result)
print(len(results), ':', result)

# Radial Component
w = operators.RadialComponent(v).evaluate()
result = np.allclose(w['g'],v['g'][2])
results.append(result)
print(len(results), ':', result)

# Angular Component
w = operators.AngularComponent(v).evaluate()
result = np.allclose(w['g'],v['g'][:2])
results.append(result)
print(len(results), ':', result)

# Tensor Interpolation
T['g'][2,2] = (6*x**2+4*y*z)/r**2
T['g'][2,1] = T['g'][1,2] = -2*(y**3+x**2*(y-3*z)-y*z**2)/(r**3*np.sin(theta))
T['g'][2,0] = T['g'][0,2] = 2*x*(z-3*y)/(r**2*np.sin(theta))
T['g'][1,1] = 6*x**2/(r**2*np.sin(theta)**2) - (6*x**2+4*y*z)/r**2
T['g'][1,0] = T['g'][0,1] = -2*x*(x**2+y**2+3*y*z)/(r**3*np.sin(theta)**2)
T['g'][0,0] = 6*y**2/(x**2+y**2)
A = operators.interpolate(T,r=1).evaluate()
Ag0 = 0*A['g']
Ag0[2,2] = 2*np.sin(theta)*(3*np.cos(phi)**2*np.sin(theta)+2*np.cos(theta)*np.sin(phi))
Ag0[2,1] = Ag0[1,2] = 6*np.cos(theta)*np.cos(phi)**2*np.sin(theta) + 2*np.cos(2*theta)*np.sin(phi)
Ag0[2,0] = Ag0[0,2] = 2*np.cos(phi)*(np.cos(theta) - 3*np.sin(theta)*np.sin(phi))
Ag0[1,1] = 2*np.cos(theta)*(3*np.cos(theta)*np.cos(phi)**2 - 2*np.sin(theta)*np.sin(phi))
Ag0[1,0] = Ag0[0,1] = -2*np.cos(phi)*(np.sin(theta) + 3*np.cos(theta)*np.sin(phi))
Ag0[0,0] = 6*np.sin(phi)**2
result = np.allclose(A['g'],Ag0)
results.append(result)
print(len(results), ':', result)

# Radial Component
Ar = operators.RadialComponent(A).evaluate()
result = np.allclose(Ar['g'],A['g'][2,:])
results.append(result)
print(len(results), ':', result)

# Angular Component
A_ang = operators.AngularComponent(A, index=1).evaluate()
result = np.allclose(A_ang['g'],A['g'][:,:2])
results.append(result)
print(len(results), ':', result)

# ScalarField - ScalarField multiplication
f = field.Field(dist=d, bases=(b,), dtype=np.complex128)
f['g'] = x**3 + 2*y**3 + 3*z**3
g = (f * f).evaluate()
result = np.allclose(g['g'], f['g']**2)
results.append(result)
print(len(results), ':', result)

# ScalarField - VectorField multiplication
u = operators.Gradient(f, c).evaluate()
v = (f * u).evaluate()
result = np.allclose(v['g'], f['g'][None,...]*u['g'])
results.append(result)
print(len(results), ':', result)
v = (u * f).evaluate()
result = np.allclose(v['g'], u['g']*f['g'][None,...])
results.append(result)
print(len(results), ':', result)

# VectorField - VectorField multiplication
v = (u * u).evaluate()
result = np.allclose(v['g'], u['g'][:,None,...]*u['g'][None,:,...])
results.append(result)
print(len(results), ':', result)

# VectorField - TensorField multiplication
T = operators.Gradient(u, c).evaluate()
Q = (u * T).evaluate()
result = np.allclose(Q['g'], u['g'][:,None,None,...]*T['g'][None,:,:,...])
results.append(result)
print(len(results), ':', result)
Q = (T * u).evaluate()
result = np.allclose(Q['g'], T['g'][:,:,None,...]*u['g'][None,None,:,...])
results.append(result)
print(len(results), ':', result)

# TensorField - TensorField multiplication
Q = (T * T).evaluate()
result = np.allclose(Q['g'], T['g'][:,:,None,None,...]*T['g'][None,None,:,:,...])
results.append(result)
print(len(results), ':', result)

# Number - ScalarField multiplication
g = (2 * f).evaluate()
result = np.allclose(g['g'], 2*f['g'])
results.append(result)
print(len(results), ':', result)
g = (f * 2).evaluate()
result = np.allclose(g['g'], f['g']*2)
results.append(result)
print(len(results), ':', result)
