import time
from scipy.special import jn_zeros
from scipy.linalg import eig
import numpy as np
from dedalus.core import coords, distributor, basis, field, operators, problems, solvers, timesteppers, arithmetic
from dedalus.tools import logging
from dedalus.tools.parsing import split_equation
import dedalus_sphere
import logging
logger = logging.getLogger(__name__)

dtype = np.float64

# Parameters
radius = 1
Mmax = 31
Nmax = 31
Ampl = 1e-6
Pr = 1
Ra = 1e6
nu = np.sqrt(Pr/Ra)
kappa = nu/Pr

# Bases
c = coords.PolarCoordinates('phi', 'r')
d = distributor.Distributor((c,))
b = basis.DiskBasis(c, (Mmax+1, Nmax+1), radius=radius, dtype=dtype)
cb = b.S1_basis(r=radius)

phi, r = b.local_grids((1, 1, 1))

# Fields
u = field.Field(dist=d, bases=(b,), tensorsig=(c,),dtype=dtype)
T = field.Field(dist=d, bases=(b,), dtype=dtype)
p = field.Field(dist=d, bases=(b,), dtype=dtype)
tau_u = field.Field(dist=d, bases=(cb,), tensorsig=(c,), dtype=dtype)
tau_T = field.Field(dist=d, bases=(cb,), dtype=dtype)

# parameters
S = field.Field(dist=d, bases=(b,), dtype=dtype)
g = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)

# constant heating rate
# radially dependent gravity
S['g'] = 2.
g['g'][1] = -r

# Parameters and operators
lap = lambda A: operators.Laplacian(A, c)
div = lambda A: operators.Divergence(A)
grad = lambda A: operators.Gradient(A, c)
curl = lambda A: operators.Curl(A)
dot = lambda A,B: arithmetic.DotProduct(A, B)
cross = lambda A,B: arithmetic.CrossProduct(A, B)
dt = lambda A: operators.TimeDerivative(A)

# Problem
def eq_eval(eq_str):
    return [eval(expr) for expr in split_equation(eq_str)]
problem = problems.IVP([u, T, p, tau_u, tau_T])
#problem.add_equation(eq_eval("dt(u) - nu*lap(u) - grad(p) = -cross(u, curl(u)) - g*T"), condition='nphi != 0')
problem.add_equation(eq_eval("dt(u) - nu*lap(u) - grad(p) = -dot(u, grad(u)) - g*T"), condition='nphi != 0')
problem.add_equation(eq_eval("u = 0"), condition='nphi == 0')
problem.add_equation(eq_eval("dt(T) - kappa*lap(T) = -dot(u, grad(T)) + S"))
problem.add_equation(eq_eval("div(u) = 0"), condition='nphi != 0')
problem.add_equation(eq_eval("p = 0"), condition='nphi == 0')

problem.add_equation(eq_eval("u(r=1) = 0"), condition='nphi != 0')
problem.add_equation(eq_eval("tau_u = 0"), condition='nphi == 0')
problem.add_equation(eq_eval("T(r=1) = 0"))

print("Problem built")

# Solver
solver = solvers.InitialValueSolver(problem, timesteppers.RK111)

# Add taus
alpha_BC = 0

def C(N, m, s):
    ab = (alpha_BC, m+s)
    cd = (2,        m+s)
    return dedalus_sphere.jacobi.coefficient_connection(N - m//2 + 1,ab,cd)

def BC_rows(N, m, num_comp):
    N_list = (np.arange(num_comp)+1)*(N - m//2 + 1)
    return N_list

for subproblem in solver.subproblems:
    m = subproblem.group[0]
    L = subproblem.L_min
    M = subproblem.M_min
    if dtype == np.complex128:
        raise NotImplementedError()
    elif dtype == np.float64:
        N0, N1, N2, N3 = BC_rows(Nmax, m, 4) * 2
        NM = Nmax - m//2 + 1
        if m == 0:
            tau_columns = np.zeros((L.shape[0], 2))
            tau_columns[N1:N1+NM,0] = (C(Nmax, m, 0))[:,-1]
            tau_columns[N1+NM:N2,1] = (C(Nmax, m, 0))[:,-1]
            L[:,-2:] = tau_columns
        else:
            tau_columns = np.zeros((L.shape[0], 6))
            tau_columns[0:NM,0] = (C(Nmax, m, -1))[:,-1]
            tau_columns[NM:N0,1] = (C(Nmax, m, -1))[:,-1]
            tau_columns[N0:N0+NM,2] = (C(Nmax, m, +1))[:,-1]
            tau_columns[N0+NM:N1,3] = (C(Nmax, m, +1))[:,-1]
            tau_columns[N1:N1+NM,4] = (C(Nmax, m, 0))[:,-1]
            tau_columns[N1+NM:N2,5] = (C(Nmax, m, 0))[:,-1]
            L[:,-6:] = tau_columns

    L.eliminate_zeros()
    subproblem.expand_matrices(['M','L'])

    #print("m = {}: Condition number {}".format(m, np.linalg.cond((L+M).A)))
    
# initial conditions
noise = np.random.randn(*T['g'].shape)*Ampl
T['g'] = 1-r**2 + noise
T.require_scales((0.25,0.25))
T['g']
T.require_scales((1,1))

# Main loop
solver.stop_sim_time = 1
dt = 1e-2
start_time = time.time()
while solver.ok:
    print("max(abs(u)) = {}".format(np.max(np.abs(u['g']))))
    solver.step(dt)
end_time = time.time()
print('Run time:', end_time-start_time)

print("SUCCESS!")
# Plot matrices
# import matplotlib.pyplot as plt
# plt.figure()
# I = 2
# J = 2
# for i, sp in enumerate(solver.subproblems[:I]):
#    for j, mat in enumerate(['M_min', 'L_min']):
#        axes = plt.subplot(I,J,i*J+j+1)
#        A = getattr(sp, mat)
#        im = axes.pcolor(np.log10(np.abs(A.A[::-1])))
#        axes.set_title('sp %i, %s' %(i, mat))
#        axes.set_aspect('equal')
#        axes.axis('off')
#        plt.colorbar(im)
# plt.tight_layout()
# plt.savefig("nmh_matrices.pdf")
