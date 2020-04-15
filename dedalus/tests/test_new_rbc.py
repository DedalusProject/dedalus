

import numpy as np
from dedalus.core import coords, distributor, basis, field, operators, problems

# Parameters
Lx, Ly, Lz = (4, 4, 1)
Nx, Ny, Nz = 16, 16, 16
Prandtl = 1
Rayleigh = 3000

# Bases
c = coords.CartesianCoordinates('x', 'y', 'z')
d = distributor.Distributor((c,))
xb = basis.ComplexFourier(c.coords[0], size=Nx, bounds=(0, Lx))
yb = basis.ComplexFourier(c.coords[1], size=Ny, bounds=(0, Ly))
zb = basis.ChebyshevT(c.coords[2], size=Nz, bounds=(0, Lz))

# Fields
p = field.Field(name='p', dist=d, bases=(xb,yb,zb), dtype=np.complex128)
b = field.Field(name='b', dist=d, bases=(xb,yb,zb), dtype=np.complex128)
u = field.Field(name='u',dist=d, bases=(xb,yb,zb), dtype=np.complex128, tensorsig=(c,))

# Equations [M, L, F]
P = (Rayleigh * Prandtl)**(-1/2)
R = (Rayleigh / Prandtl)**(-1/2)
ez = field.Field(name='ez', dist=d, bases=(xb,yb,zb), dtype=np.complex128, tensorsig=(c,))
ez['g'][2] = 1
ghat = - ez
div = operators.Divergence
lap = operators.Laplacian
grad = operators.Gradient
dot = operators.DotProduct
dt = operators.TimeDerivative

problem = problems.IVP([p, b, u])
problem.add_equation([div(u,0), 0])
problem.add_equation([dt(b) - P*lap(b,c), -dot(u,grad(b,c))])
problem.add_equation([dt(u) - R*lap(u,c) + grad(p,c), -dot(u,grad(u,c)) - b*ghat])
problem.add_equation([u(z=0), 0])
problem.add_equation([u(z=Lz), 0])
problem.add_equation([b(z=0), Lz])
problem.add_equation([b(z=Lz), 0])
# Pressure gauge?

print("Problem built")
