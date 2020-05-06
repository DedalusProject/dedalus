
import numpy as np
from dedalus.core import coords, distributor, basis, field, operators, problems, solvers, timesteppers
from dedalus.core import future
from dedalus.tools.parsing import split_equation
from dedalus.tools.array import apply_matrix
from mpi4py import MPI

comm = MPI.COMM_WORLD

results = []

## Shell
c = coords.SphericalCoordinates('phi', 'theta', 'r')
c_S2 = c.S2coordsys
d = distributor.Distributor((c,))
b = basis.SphericalShellBasis(c, (16,16,16), radii=(0.5,3))
b_out = b.S2_basis(3)
phi, theta, r = b.local_grids((1, 1, 1))
x = r * np.sin(theta) * np.cos(phi)
y = r * np.sin(theta) * np.sin(phi)
z = r * np.cos(theta)

# make subproblems
p = field.Field(dist=d, bases=(b,), dtype=np.complex128)
u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
def eq_eval(eq_str):
    return [eval(expr) for expr in split_equation(eq_str)]
problem = problems.IVP([u, p])
problem.add_equation(eq_eval("operators.Gradient(p,c) - u = 0"))
problem.add_equation(eq_eval("operators.Divergence(u) = 0"))
solver = solvers.InitialValueSolver(problem, timesteppers.CNAB2)

u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
ct = np.cos(theta)
st = np.sin(theta)
cp = np.cos(phi)
sp = np.sin(phi)
u['g'][2] = r**2*st*(2*ct**2*cp-r*ct**3*sp+r**3*cp**3*st**5*sp**3+r*ct*st**2*(cp**3+sp**3))
u['g'][1] = r**2*(2*ct**3*cp-r*cp**3*st**4+r**3*ct*cp**3*st**5*sp**3-1/16*r*np.sin(2*theta)**2*(-7*sp+np.sin(3*phi)))
u['g'][0] = r**2*sp*(-2*ct**2+r*ct*cp*st**2*sp-r**3*cp**2*st**5*sp**3)
u_BC = field.Field(dist=d, bases=(b_out,), tensorsig=(c,), dtype=np.complex128)
u_BC['g'][0] = 3**2*sp*(-2*ct**2+3*ct*cp*st**2*sp-3**3*cp**2*st**5*sp**3)
u_BC['g'][1] = 3**2*(2*ct**3*cp-3*cp**3*st**4+3**3*ct*cp**3*st**5*sp**3-1/16*3*np.sin(2*theta)**2*(-7*sp+np.sin(3*phi)))
u_BC['g'][2] = 3**2*st*(2*ct**2*cp-3*ct**3*sp+3**3*cp**3*st**5*sp**3+3*ct*st**2*(cp**3+sp**3))

## Vector Interpolation
#u_BC2 = field.Field(dist=d, bases=(b_out,), tensorsig=(c,), dtype=np.complex128)
#
#for ell in b.local_l:
#    sp = solver.subproblems[ell]
#
#    for m in b.local_m:
#
#        if abs(m) <= ell:
#            matrix = operators.interpolate(u,r=3).subproblem_matrix(sp)
#
#            vector = u['c'][:,m,ell,:]
#            shape = vector.shape
#            view = vector.reshape((shape[0]*shape[1]))
#            u_BC2['c'][:,m,ell,0] = matrix @ view
#
#print(np.allclose(u_BC['g'],u_BC2['g']), '(interpolation of k=0 vector)')
#
## Radial component
#u_BC2 = field.Field(dist=d, bases=(b_out,), dtype=np.complex128)
#
#for ell in b.local_l:
#    sp = solver.subproblems[ell]
#
#    for m in b.local_m:
#
#        if abs(m) <= ell:
#            op = operators.RadialComponent(operators.interpolate(u,r=3))
#            matrix = op.expression_matrices(sp, (u,))[u]
#
#            vector = u['c'][:,m,ell,:]
#            shape = vector.shape
#            view = vector.reshape((shape[0]*shape[1]))
#            u_BC2['c'][m,ell,0] = np.inner(matrix, view)
#
#print(np.allclose(u_BC['g'][2],u_BC2['g']), '(radial component of k=0 vector)')

# Angular component

u_BC2 = field.Field(dist=d, bases=(b_out,), tensorsig=(c_S2,), dtype=np.complex128)

for ell in b.local_l:
    sp = solver.subproblems[ell]

    for m in b.local_m:

        if abs(m) <= ell:
            op = operators.AngularComponent(operators.interpolate(u,r=3))
            matrix = op.expression_matrices(sp, (u,))[u]

            vector = u['c'][:,m,ell,:]
            shape = vector.shape
            view = vector.reshape((shape[0]*shape[1]))
            u_BC2['c'][:,m,ell,0] = matrix @ view

print(np.allclose(u_BC['g'][:2],u_BC2['g']), '(angular component of k=0 vector)')

