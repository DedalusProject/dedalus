

import numpy as np
from dedalus.core import coords, distributor, basis, field, operators, problems, solvers, timesteppers, arithmetic
from dedalus.tools import logging
from dedalus.tools.parsing import split_equation
from dedalus.extras.flow_tools import GlobalArrayReducer
from scipy import sparse
import dedalus_sphere
import time
from mpi4py import MPI

import logging
logger = logging.getLogger(__name__)

# Parameters
radius = 1
Lmax = 15
L_dealias = 1
Nmax = 15
N_dealias = 1
dt = 8e-5
t_end = 20
ts = timesteppers.SBDF2
dtype = np.float64

Ekman = 3e-4
Rayleigh = 95
Prandtl = 1

# Bases
c = coords.SphericalCoordinates('phi', 'theta', 'r')
d = distributor.Distributor((c,))
b = basis.BallBasis(c, (2*(Lmax+1), Lmax+1, Nmax+1), radius=radius, dtype=dtype)
bk2 = basis.BallBasis(c, (2*(Lmax+1), Lmax+1, Nmax+1), k=2, radius=radius, dtype=dtype)
b_S2 = b.S2_basis()
phi, theta, r = b.local_grids((1, 1, 1))

# Fields
u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
p = field.Field(dist=d, bases=(b,), dtype=dtype)
T = field.Field(dist=d, bases=(b,), dtype=dtype)
tau_u = field.Field(dist=d, bases=(b_S2,), tensorsig=(c,), dtype=dtype)
tau_T = field.Field(dist=d, bases=(b_S2,), dtype=dtype)

r_vec = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
r_vec['g'][2] = r

T['g'] = 0.5*(1-r**2) + 0.1/8*np.sqrt(35/np.pi)*r**3*(1-r**2)*(np.cos(3*phi)+np.sin(3*phi))*np.sin(theta)**3

T_source = field.Field(dist=d, bases=(b,), dtype=dtype)
T_source['g'] = 3

# Boundary conditions
u_r_bc = operators.RadialComponent(operators.interpolate(u,r=1))

stress = operators.Gradient(u, c) + operators.TransposeComponents(operators.Gradient(u, c))
u_perp_bc = operators.RadialComponent(operators.AngularComponent(operators.interpolate(stress,r=1), index=1))

# Parameters and operators
ez = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
ez['g'][1] = -np.sin(theta)
ez['g'][2] =  np.cos(theta)
div = lambda A: operators.Divergence(A, index=0)
lap = lambda A: operators.Laplacian(A, c)
grad = lambda A: operators.Gradient(A, c)
dot = lambda A, B: arithmetic.DotProduct(A, B)
cross = lambda A, B: arithmetic.CrossProduct(A, B)
ddt = lambda A: operators.TimeDerivative(A)

# Problem
def eq_eval(eq_str):
    return [eval(expr) for expr in split_equation(eq_str)]
problem = problems.IVP([p, u, T, tau_u, tau_T])

problem.add_equation(eq_eval("div(u) = 0"), condition="ntheta != 0")
problem.add_equation(eq_eval("p = 0"), condition="ntheta == 0")
problem.add_equation(eq_eval("Ekman*ddt(u) - Ekman*lap(u) + grad(p) = - Ekman*dot(u,grad(u)) + Rayleigh*r_vec*T - cross(ez, u)"), condition = "ntheta != 0")
problem.add_equation(eq_eval("u = 0"), condition="ntheta == 0")
problem.add_equation(eq_eval("Prandtl*ddt(T) - lap(T) = - Prandtl*dot(u,grad(T)) + T_source"))
problem.add_equation(eq_eval("u_r_bc = 0"), condition="ntheta != 0")
problem.add_equation(eq_eval("u_perp_bc = 0"), condition="ntheta != 0")
problem.add_equation(eq_eval("tau_u = 0"), condition="ntheta == 0")
problem.add_equation(eq_eval("T(r=1) = 0"))

print("Problem built")

# Solver
solver = solvers.InitialValueSolver(problem, ts)
solver.stop_sim_time = t_end

# Add taus
alpha_BC = 0

def C(N, ell, deg):
    ab = (alpha_BC,ell+deg+0.5)
    cd = (2,       ell+deg+0.5)
    return dedalus_sphere.jacobi.coefficient_connection(N - ell//2 + 1,ab,cd)

def BC_rows(N, ell, num_comp):
    N_list = (np.arange(num_comp)+1)*(N - ell//2 + 1)
    return N_list

for subproblem in solver.subproblems:
    ell = subproblem.group[1]
    L = subproblem.L_min
    shape = L.shape
    if dtype == np.complex128:
        N0, N1, N2, N3, N4 = BC_rows(Nmax, ell, 5)
        tau_columns = np.zeros((shape[0], 4))
        if ell != 0:
            tau_columns[N0:N1,0] = (C(Nmax, ell, -1))[:,-1]
            tau_columns[N1:N2,1] = (C(Nmax, ell, +1))[:,-1]
            tau_columns[N2:N3,2] = (C(Nmax, ell,  0))[:,-1]
            tau_columns[N3:N4,3] = (C(Nmax, ell,  0))[:,-1]
            subproblem.L_min[:,-4:] = tau_columns
        else: # ell = 0
            tau_columns[N3:N4, 3] = (C(Nmax, ell, 0))[:,-1]
            subproblem.L_min[:,-1:] = tau_columns[:,6:]
    elif dtype == np.float64:
        NL = Nmax - ell//2 + 1
        N0, N1, N2, N3, N4 = BC_rows(Nmax, ell, 5) * 2
        tau_columns = np.zeros((shape[0], 8))
        if ell != 0:
            tau_columns[N0:N0+NL,0] = (C(Nmax, ell, -1))[:,-1]
            tau_columns[N1:N1+NL,2] = (C(Nmax, ell, +1))[:,-1]
            tau_columns[N2:N2+NL,4] = (C(Nmax, ell,  0))[:,-1]
            tau_columns[N3:N3+NL,6] = (C(Nmax, ell,  0))[:,-1]
            tau_columns[N0+NL:N0+2*NL,1] = (C(Nmax, ell, -1))[:,-1]
            tau_columns[N1+NL:N1+2*NL,3] = (C(Nmax, ell, +1))[:,-1]
            tau_columns[N2+NL:N2+2*NL,5] = (C(Nmax, ell,  0))[:,-1]
            tau_columns[N3+NL:N3+2*NL,7] = (C(Nmax, ell,  0))[:,-1]
            subproblem.L_min[:,-8:] = tau_columns
        else: # ell = 0
            tau_columns[N3:N3+NL,6] = (C(Nmax, ell,  0))[:,-1]
            tau_columns[N3+NL:N3+2*NL,7] = (C(Nmax, ell,  0))[:,-1]
            subproblem.L_min[:,-2:] = tau_columns[:,6:]
    L.eliminate_zeros()
    subproblem.expand_matrices(['M','L'])

# Analysis
t_list = []
E_list = []
weight_theta = b.local_colatitude_weights(1)
weight_r = b.local_radial_weights(1)
reducer = GlobalArrayReducer(d.comm_cart)
vol_test = np.sum(weight_r*weight_theta+0*p['g'])*np.pi/(Lmax+1)/L_dealias
vol_test = reducer.reduce_scalar(vol_test, MPI.SUM)
vol_correction = 4*np.pi/3/vol_test

# Main loop
start_time = time.time()
while solver.ok:
    if solver.iteration % 10 == 0:
        E0 = np.sum(vol_correction*weight_r*weight_theta*u['g'].real**2)
        E0 = 0.5*E0*(np.pi)/(Lmax+1)/L_dealias
        E0 = reducer.reduce_scalar(E0, MPI.SUM)
        logger.info("t = %f, E = %e" %(solver.sim_time, E0))
        t_list.append(solver.sim_time)
        E_list.append(E0)
    solver.step(dt)
end_time = time.time()
print('Run time:', end_time-start_time)
