

import numpy as np
from dedalus.core import coords, distributor, basis, field, operators, problems, solvers, timesteppers, arithmetic
from dedalus.tools import logging
from dedalus.tools.parsing import split_equation
import matplotlib.pyplot as plt
from mpi4py import MPI
rank = MPI.COMM_WORLD.rank

import logging
logger = logging.getLogger(__name__)

from dedalus.tools.config import config
config['linear algebra']['MATRIX_FACTORIZER'] = 'SuperLUNaturalFactorizedTranspose'


# Parameters
Lx, Lz = (4, 1)
Nx, Nz = 256, 64
Prandtl = 1
Rayleigh = 1e6
timestep = 0.05
stop_iteration = 150

# Bases
c = coords.CartesianCoordinates('x', 'z')
d = distributor.Distributor((c,))
xb = basis.RealFourier(c.coords[0], size=Nx, bounds=(0, Lx))
zb = basis.ChebyshevT(c.coords[1], size=Nz, bounds=(0, Lz))
x = xb.local_grid(1)
z = zb.local_grid(1)

# Fields
p = field.Field(name='p', dist=d, bases=(xb,zb), dtype=np.float64)
b = field.Field(name='b', dist=d, bases=(xb,zb), dtype=np.float64)
u = field.Field(name='u', dist=d, bases=(xb,zb), dtype=np.float64, tensorsig=(c,))

# Taus
zb1 = basis.ChebyshevU(c.coords[1], size=Nz, bounds=(0, Lz), alpha0=0)
t1 = field.Field(name='t1', dist=d, bases=(xb,), dtype=np.float64)
t2 = field.Field(name='t2', dist=d, bases=(xb,), dtype=np.float64)
t3 = field.Field(name='t3', dist=d, bases=(xb,), dtype=np.float64, tensorsig=(c,))
t4 = field.Field(name='t4', dist=d, bases=(xb,), dtype=np.float64, tensorsig=(c,))
P1 = field.Field(name='P1', dist=d, bases=(zb1,), dtype=np.float64)
if rank == 0:
    P1['c'][0,-1] = 1

# Parameters and operators
P = (Rayleigh * Prandtl)**(-1/2)
R = (Rayleigh / Prandtl)**(-1/2)

ex = field.Field(name='ex', dist=d, bases=(zb,), dtype=np.float64, tensorsig=(c,))
ez = field.Field(name='ez', dist=d, bases=(zb,), dtype=np.float64, tensorsig=(c,))
ex['g'][0] = 1
ez['g'][1] = 1

ghat = - ez
grid_ghat = operators.Grid(ghat).evaluate()

B = field.Field(name='B', dist=d, bases=(zb,), dtype=np.float64)
B['g'] = Lz - z
grid_B = operators.Grid(B).evaluate()

div = lambda A: operators.Divergence(A, index=0)
lap = lambda A: operators.Laplacian(A, c)
grad = lambda A: operators.Gradient(A, c)
dot = lambda A, B: arithmetic.DotProduct(A, B)
dt = lambda A: operators.TimeDerivative(A)

dx = lambda A: operators.Differentiate(A, c.coords[0])
dz = lambda A: operators.Differentiate(A, c.coords[1])
P2 = dz(P1).evaluate()

# Problem
def eq_eval(eq_str):
    return [eval(expr) for expr in split_equation(eq_str)]
problem = problems.IVP([p, b, u, t1, t2, t3, t4])
problem.add_equation(eq_eval("div(u) - P1*dot(ez,t4) = 0"))
problem.add_equation(eq_eval("dt(b) - P*lap(b) + P1*t1 + P2*t2 + dot(u,grad(B)) = - dot(u,grad(b)) + P*lap(B)"))
problem.add_equation(eq_eval("dt(u) - R*lap(u) + grad(p) + P1*t3 + P2*t4 + b*ghat = - dot(u,grad(u)) - grid_B*grid_ghat"))
problem.add_equation(eq_eval("b(z=0) = Lz - B(z=0)"))
problem.add_equation(eq_eval("u(z=0) = 0"))
problem.add_equation(eq_eval("b(z=Lz) = 0 - B(z=Lz)"))
problem.add_equation(eq_eval("u(z=Lz) = 0"), condition="nx != 0")
problem.add_equation(eq_eval("dot(ex,u)(z=Lz) = 0"), condition="nx == 0")
problem.add_equation(eq_eval("p(z=Lz) = 0"), condition="nx == 0")
print("Problem built")

# Solver
solver = solvers.InitialValueSolver(problem, timesteppers.RK222)
solver.stop_iteration = stop_iteration

if False:
    for i, subproblem in enumerate(solver.subproblems):
        M = subproblem.M_min
        L = subproblem.L_min
        print(i, subproblem.group, np.linalg.cond((M+L).A))

# Initial conditions
np.random.seed(42)
b['g'] = 1e-1 * np.random.randn(*b['g'].shape)

# Main loop
plt.figure()
while solver.ok:
    if solver.iteration % 10 == 0:
        print('Iteration: %i' %solver.iteration)
        plt.imshow(b['g'].real.T, origin='lower')
        plt.savefig('rbc_frames/frames_r%i/frame_%06i.png' %(rank,solver.iteration))
        plt.clf()
    solver.step(timestep)

# Plot matrices
import matplotlib
import matplotlib.pyplot as plt

# Plot options
fig = plt.figure(figsize=(9,3))
cmap = matplotlib.cm.get_cmap("winter_r")
clim = (-16, 0)
lim_margin = 0.05

def plot_sparse(A):
    I, J = A.shape
    A_mag = np.log10(np.abs(A.A))
    ax.pcolor(A_mag[::-1], cmap=cmap, vmin=clim[0], vmax=clim[1])
    ax.set_xlim(-lim_margin, I+lim_margin)
    ax.set_ylim(-lim_margin, J+lim_margin)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal', 'box')
    ax.text(0.95, 0.95, 'nnz: %i' %A.nnz, horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)
    ax.text(0.95, 0.95, '\ncon: %.1e' %np.linalg.cond(A.A), horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)

if False:
    for sp in solver.subproblems:
        m = sp.group[0]
        # Plot LHS
        ax = fig.add_subplot(1, 3, 1)
        LHS = (sp.M_min + sp.L_min) @ sp.pre_right
        plot_sparse(LHS)
        ax.set_title('LHS (m = %i)' %m)
        # Plot L
        ax = fig.add_subplot(1, 3, 2)
        L = sp.LHS_solvers[-1].LU.L
        plot_sparse(L)
        ax.set_title('L (m = %i)' %m)
        # Plot U
        ax = fig.add_subplot(1, 3, 3)
        U = sp.LHS_solvers[-1].LU.U
        plot_sparse(U)
        ax.set_title('U (m = %i)' %m)
        plt.tight_layout()
        plt.savefig("rbc_mats/m_%i.pdf" %m)
        fig.clear()
