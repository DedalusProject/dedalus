
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
b = basis.SphericalShellBasis(c, (32,16,16), radii=(1/3,1))
b_out = b.S2_basis(3)
b_in = b.S2_basis(0.5)
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

#u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
#ct = np.cos(theta)
#st = np.sin(theta)
#cp = np.cos(phi)
#sp = np.sin(phi)
#u['g'][2] = r**2*st*(2*ct**2*cp-r*ct**3*sp+r**3*cp**3*st**5*sp**3+r*ct*st**2*(cp**3+sp**3))
#u['g'][1] = r**2*(2*ct**3*cp-r*cp**3*st**4+r**3*ct*cp**3*st**5*sp**3-1/16*r*np.sin(2*theta)**2*(-7*sp+np.sin(3*phi)))
#u['g'][0] = r**2*sp*(-2*ct**2+r*ct*cp*st**2*sp-r**3*cp**2*st**5*sp**3)
#u_BC = field.Field(dist=d, bases=(b_out,), tensorsig=(c,), dtype=np.complex128)
#u_BC['g'][0] = 3**2*sp*(-2*ct**2+3*ct*cp*st**2*sp-3**3*cp**2*st**5*sp**3)
#u_BC['g'][1] = 3**2*(2*ct**3*cp-3*cp**3*st**4+3**3*ct*cp**3*st**5*sp**3-1/16*3*np.sin(2*theta)**2*(-7*sp+np.sin(3*phi)))
#u_BC['g'][2] = 3**2*st*(2*ct**2*cp-3*ct**3*sp+3**3*cp**3*st**5*sp**3+3*ct*st**2*(cp**3+sp**3))

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

## Angular component
#
#u_BC2 = field.Field(dist=d, bases=(b_out,), tensorsig=(c_S2,), dtype=np.complex128)
#
#for ell in b.local_l:
#    sp = solver.subproblems[ell]
#
#    for m in b.local_m:
#
#        if abs(m) <= ell:
#            op = operators.AngularComponent(operators.interpolate(u,r=3))
#            matrix = op.expression_matrices(sp, (u,))[u]
#
#            vector = u['c'][:,m,ell,:]
#            shape = vector.shape
#            view = vector.reshape((shape[0]*shape[1]))
#            u_BC2['c'][:,m,ell,0] = matrix @ view
#
#print(np.allclose(u_BC['g'][:2],u_BC2['g']), '(angular component of k=0 vector)')

#f = field.Field(dist=d, bases=(b,), dtype=np.complex128)
#f['g'] = r**4*np.sin(theta)*np.cos(theta)*np.exp(1j*phi)
#u = operators.Gradient(f, c).evaluate()
#ug = u['g']*0
#ug[2] = 4*r**3*np.sin(theta)*np.cos(theta)*np.exp(1j*phi)
#ug[1] =   r**3*np.cos(2*theta)*np.exp(1j*phi)
#ug[0] =1j*r**3*np.cos(theta)*np.exp(1j*phi)
#
##print(np.allclose(ug,u['g']))
#
#u_BC = field.Field(dist=d, bases=(b_out,), tensorsig=(c,), dtype=np.complex128)
#u_BC['g'][2] = 4*3**3*np.sin(theta)*np.cos(theta)*np.exp(1j*phi)
#u_BC['g'][1] =   3**3*np.cos(2*theta)*np.exp(1j*phi)
#u_BC['g'][0] =1j*3**3*np.cos(theta)*np.exp(1j*phi)
#
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
#print(np.allclose(u_BC['g'],u_BC2['g']), '(interpolation of k=1 vector)')
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
#print(np.allclose(u_BC['g'][2],u_BC2['g']), '(radial component of k=1 vector)')

f = field.Field(dist=d, bases=(b,), dtype=np.complex128)
f['g'] = r**4*np.sin(theta)*np.cos(theta)*np.exp(1j*phi)
u = operators.Gradient(f, c).evaluate()
T = operators.Gradient(u, c).evaluate()
Tg = T['g']*0
Tg[2,2] = 6*r**2*np.sin(2*theta)*np.exp(1j*phi)
Tg[2,1] = Tg[1,2] = 3*r**2*np.cos(2*theta)*np.exp(1j*phi)
Tg[2,0] = Tg[0,2] = 3j*r**2*np.cos(theta)*np.exp(1j*phi)
Tg[1,0] = Tg[0,1] = -1j*r**2*np.sin(theta)*np.exp(1j*phi)
Tg[0,0] = r**2*np.sin(2*theta)*np.exp(1j*phi)

#print(np.allclose(Tg,T['g']))

T_BC = field.Field(dist=d, bases=(b_out,), tensorsig=(c,c), dtype=np.complex128)
T_BC['g'][2,2] = 6*3**2*np.sin(2*theta)*np.exp(1j*phi)
T_BC['g'][2,1] = T_BC['g'][1,2] = 3*3**2*np.cos(2*theta)*np.exp(1j*phi)
T_BC['g'][2,0] = T_BC['g'][0,2] = 3j*3**2*np.cos(theta)*np.exp(1j*phi)
T_BC['g'][1,0] = T_BC['g'][0,1] = -1j*3**2*np.sin(theta)*np.exp(1j*phi)
T_BC['g'][0,0] = 3**2*np.sin(2*theta)*np.exp(1j*phi)

## Tensor Interpolation
#T_BC2 = field.Field(dist=d, bases=(b_out,), tensorsig=(c,c), dtype=np.complex128)
#
#for ell in b.local_l:
#    sp = solver.subproblems[ell]
#
#    for m in b.local_m:
#
#        if abs(m) <= ell:
#            matrix = operators.interpolate(T,r=3).subproblem_matrix(sp)
#
#            vector = T['c'][:,:,m,ell,:]
#            shape = vector.shape
#            view = vector.reshape((shape[0]*shape[1]*shape[2]))
#            T_BC2['c'][:,:,m,ell,0] = (matrix @ view).reshape((shape[0],shape[1]))
#
#print(np.allclose(T_BC['g'],T_BC2['g']), '(interpolation of k=2 tensor)')

## Radial component
#u_BC2 = field.Field(dist=d, bases=(b_out,), tensorsig=(c,), dtype=np.complex128)
#
#for ell in b.local_l:
#    sp = solver.subproblems[ell]
#
#    for m in b.local_m:
#
#        if abs(m) <= ell:
#            op = operators.RadialComponent(operators.interpolate(T,r=3))
#            matrix = op.expression_matrices(sp, (T,))[T]
#
#            vector = T['c'][:,:,m,ell,:]
#            shape = vector.shape
#            view = vector.reshape((shape[0]*shape[1]*shape[2]))
#            u_BC2['c'][:,m,ell,0] = (matrix @ view)
#
#print(np.allclose(T_BC['g'][2],u_BC2['g']), '(radial component of k=2 tensor)')

## Radial derivative
#u_BC2 = field.Field(dist=d, bases=(b_in,), tensorsig=(c,), dtype=np.complex128)
#
#T_BC = field.Field(dist=d, bases=(b_in,), tensorsig=(c,c), dtype=np.complex128)
#T_BC['g'][2,2] = 6*0.5**2*np.sin(2*theta)*np.exp(1j*phi)
#T_BC['g'][2,1] = T_BC['g'][1,2] = 3*0.5**2*np.cos(2*theta)*np.exp(1j*phi)
#T_BC['g'][2,0] = T_BC['g'][0,2] = 3j*0.5**2*np.cos(theta)*np.exp(1j*phi)
#T_BC['g'][1,0] = T_BC['g'][0,1] = -1j*0.5**2*np.sin(theta)*np.exp(1j*phi)
#T_BC['g'][0,0] = 0.5**2*np.sin(2*theta)*np.exp(1j*phi)
#
#for ell in b.local_l:
#    sp = solver.subproblems[ell]
#
#    for m in b.local_m:
#
#        if abs(m) <= ell:
#            op = operators.RadialComponent(operators.interpolate(operators.Gradient(u, c),r=0.5))
#            matrix = op.expression_matrices(sp, (u,))[u]
#
#            vector = u['c'][:,m,ell,:]
#            shape = vector.shape
#            view = vector.reshape((shape[0]*shape[1]))
#            u_BC2['c'][:,m,ell,0] = (matrix @ view)
#
#print(np.allclose(T_BC['g'][2],u_BC2['g']), '(radial derivative of k=1 vector)')


## Ball
bB = basis.BallBasis(c, (32,16,16), radius=1/3)
bB_out = bB.S2_basis(0.5)
phi, theta, r = bB.local_grids((1, 1, 1))
x = r * np.sin(theta) * np.cos(phi)
y = r * np.sin(theta) * np.sin(phi)
z = r * np.cos(theta)

# make subproblems
pB = field.Field(dist=d, bases=(bB,), dtype=np.complex128)
uB = field.Field(dist=d, bases=(bB,), tensorsig=(c,), dtype=np.complex128)
def eq_eval(eq_str):
    return [eval(expr) for expr in split_equation(eq_str)]
problemB = problems.IVP([uB, pB])
problemB.add_equation(eq_eval("operators.Gradient(pB,c) - uB = 0"))
problemB.add_equation(eq_eval("operators.Divergence(uB) = 0"))
solverB = solvers.InitialValueSolver(problemB, timesteppers.CNAB2)

fB = field.Field(dist=d, bases=(bB,), dtype=np.complex128)
fB['g'] = r**4*np.sin(theta)*np.cos(theta)*np.exp(1j*phi)
uB = operators.Gradient(fB, c).evaluate()

## Interpolate
#uB_BC2 = field.Field(dist=d, bases=(bB_out,), tensorsig=(c,), dtype=np.complex128)
#
#uB_BC = field.Field(dist=d, bases=(bB_out,), tensorsig=(c,), dtype=np.complex128)
#uB_BC['g'][2] = 4*0.5**3*np.sin(theta)*np.cos(theta)*np.exp(1j*phi)
#uB_BC['g'][1] =   0.5**3*np.cos(2*theta)*np.exp(1j*phi)
#uB_BC['g'][0] =1j*0.5**3*np.cos(theta)*np.exp(1j*phi)
#
#for ell in bB.local_l:
#    sp = solverB.subproblems[ell]
#
#    for m in bB.local_m:
#
#        if abs(m) <= ell:
#            op = operators.interpolate(uB,r=0.5)
#            matrix = op.expression_matrices(sp, (uB,))[uB]
#
#            slice = bB.n_slice((),ell)
#            vector = uB['c'][:,m,ell,slice]
#            shape = vector.shape
#            view = vector.reshape((shape[0]*shape[1]))
#            uB_BC2['c'][:,m,ell,0] = (matrix @ view)
#
#print(np.allclose(uB_BC['g'],uB_BC2['g']), '(radial interpolation of k=1 vector, Ball)')

## Radial derivative
#uB_BC2 = field.Field(dist=d, bases=(bB_out,), tensorsig=(c,), dtype=np.complex128)
#
#TB_BC = field.Field(dist=d, bases=(bB_out,), tensorsig=(c,c), dtype=np.complex128)
#TB_BC['g'][2,2] = 6*0.5**2*np.sin(2*theta)*np.exp(1j*phi)
#TB_BC['g'][2,1] = TB_BC['g'][1,2] = 3*0.5**2*np.cos(2*theta)*np.exp(1j*phi)
#TB_BC['g'][2,0] = TB_BC['g'][0,2] = 3j*0.5**2*np.cos(theta)*np.exp(1j*phi)
#TB_BC['g'][1,0] = TB_BC['g'][0,1] = -1j*0.5**2*np.sin(theta)*np.exp(1j*phi)
#TB_BC['g'][0,0] = 0.5**2*np.sin(2*theta)*np.exp(1j*phi)
#
#for ell in bB.local_l:
#    sp = solverB.subproblems[ell]
#
#    for m in bB.local_m:
#
#        if abs(m) <= ell:
#            op = operators.RadialComponent(operators.interpolate(operators.Gradient(uB, c),r=0.5))
#            matrix = op.expression_matrices(sp, (uB,))[uB]
#
#            slice = bB.n_slice((),ell)
#            vector = uB['c'][:,m,ell,slice]
#            shape = vector.shape
#            view = vector.reshape((shape[0]*shape[1]))
#            uB_BC2['c'][:,m,ell,0] = (matrix @ view)
#
#print(np.allclose(TB_BC['g'][2],uB_BC2['g']), '(radial derivative of k=1 vector, Ball)')

# BCs
import matplotlib.pyplot as plt
plt.figure()

u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
uB = field.Field(dist=d, bases=(bB,), tensorsig=(c,), dtype=np.complex128)

for ell in [1]:
    sp = solverB.subproblems[ell]

    # Shell interpolate
    L00 = -operators.interpolate(u,r=1/3).subproblem_matrix(sp)

    # Ball interpolate
    op = operators.interpolate(uB,r=1/3)
    L01 = op.expression_matrices(sp, (uB,))[uB]

    # Shell derivative
    op = -operators.RadialComponent(operators.interpolate(operators.Gradient(u, c),r=1/3))
    L10 = op.expression_matrices(sp, (u,))[u]

    # Ball derivative
    op = operators.RadialComponent(operators.interpolate(operators.Gradient(uB, c),r=1/3))
    L11 = op.expression_matrices(sp, (uB,))[uB]

    # Shell interpolate
    L20 = operators.interpolate(u,r=1).subproblem_matrix(sp)

    # Ball zero
    L21 = L11*0

    matrix = np.bmat([[L00, L01],
                      [L10, L11],
                      [L20, L21]])

    print(np.linalg.matrix_rank(matrix))

    plt.imshow(np.log10(np.abs(matrix)))
    plt.colorbar()
    plt.savefig("matrices/BCs.png", dpi=300)
    plt.clf()
    print(L11[0,:])
    print(L10[0,:])


