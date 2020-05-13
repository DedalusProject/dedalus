

import numpy as np
from dedalus.core import coords, distributor, basis, field, operators, problems, solvers, timesteppers, arithmetic, timesteppers_sphere
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
Lmax = 31
L_dealias = 1
Nmax = 31
N_dealias = 1
dt = 5e-6
t_end = 20
ts = timesteppers.SBDF2

Ekman = 5e-4
Roberts = 7
Rayleigh = 200
Rossby = 5/7*1e-4
Source = 3*Roberts

# Bases
c = coords.SphericalCoordinates('phi', 'theta', 'r')
d = distributor.Distributor((c,), mesh=[16,16])
b = basis.BallBasis(c, (2*(Lmax+1), Lmax+1, Nmax+1), radius=radius)
bk2 = basis.BallBasis(c, (2*(Lmax+1), Lmax+1, Nmax+1), k=2, radius=radius)
b_S2 = b.S2_basis()
phi, theta, r = b.local_grids((1, 1, 1))

# Fields
u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
B = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
p = field.Field(dist=d, bases=(b,), dtype=np.complex128)
T = field.Field(dist=d, bases=(b,), dtype=np.complex128)
tau_u = field.Field(dist=d, bases=(b_S2,), tensorsig=(c,), dtype=np.complex128)
tau_B = field.Field(dist=d, bases=(b_S2,), tensorsig=(c,), dtype=np.complex128)
tau_T = field.Field(dist=d, bases=(b_S2,), dtype=np.complex128)

r_vec = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
r_vec['g'][2] = r

T['g'] = 0.5*(1-r**2) + 0.1/8*np.sqrt(35/np.pi)*r**3*(1-r**2)*(np.cos(3*phi)+np.sin(3*phi))*np.sin(theta)**3

# initial toroidal velocity field
u['g'][1] = -10*r**2/7/np.sqrt(3)*np.cos(theta)*(  3*(-147+343*r**2-217*r**4+29*r**6)*np.cos(phi)
                                                 +14*(-9 - 125*r**2 +39*r**4+27*r**6)*np.sin(phi) )
u['g'][2] = -5*r/5544*( 7*(           (43700-58113*r**2-15345*r**4+1881*r**6+20790*r**8)*np.sin(theta)
                           +1485*r**2*(-9 + 115*r**2 - 167*r**4 + 70*r**6)*np.sin(3*theta) )
                       +528*np.sqrt(3)*r*np.cos(2*theta)*( 14*(-9-125*r**2+39*r**4+27*r**6)*np.cos(phi)
                                                           +3*(147-343*r**2+217*r**4-29*r**6)*np.sin(phi) ) )

# initial toroidal magnetic field
B['g'][1] = -3./2.*r*(-1+4*r**2-6*r**4+3*r**6)*(np.cos(phi)+np.sin(phi))
B['g'][2] = -3./4.*r*(-1+r**2)*np.cos(theta)* \
                 ( 3*r*(2-5*r**2+4*r**4)*np.sin(theta)
                  +2*(1-3*r**2+3*r**4)*(np.cos(phi)-np.sin(phi)))


T_source = field.Field(dist=d, bases=(b,), dtype=np.complex128)
T_source['g'] = Source

# Boundary conditions
u_r_bc = operators.RadialComponent(operators.interpolate(u,r=1))

stress = operators.Gradient(u, c) + operators.TransposeComponents(operators.Gradient(u, c))
u_perp_bc = operators.RadialComponent(operators.AngularComponent(operators.interpolate(stress,r=1), index=1))

# Potential BC on B
r_out = 1
ell_func = lambda ell: ell+1
B_potential_bc = operators.RadialComponent(operators.interpolate(operators.Gradient(u, c), r=1)) + operators.interpolate(operators.SphericalEllProduct(u, c, ell_func), r=1)/r_out

# Parameters and operators
ez = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
ez['g'][1] = -np.sin(theta)
ez['g'][2] =  np.cos(theta)
div = lambda A: operators.Divergence(A, index=0)
lap = lambda A: operators.Laplacian(A, c)
grad = lambda A: operators.Gradient(A, c)
dot = lambda A, B: arithmetic.DotProduct(A, B)
cross = lambda A, B: operators.CrossProduct(A, B)
ddt = lambda A: operators.TimeDerivative(A)
curl = lambda A: operators.Curl(A)

# Problem
def eq_eval(eq_str):
    return [eval(expr) for expr in split_equation(eq_str)]
problem = problems.IVP([p, u, B, T, tau_u, tau_B, tau_T])

problem.add_equation(eq_eval("div(u) = 0"), condition="ntheta != 0")
problem.add_equation(eq_eval("p = 0"), condition="ntheta == 0")
problem.add_equation(eq_eval("Rossby*ddt(u) - Ekman*lap(u) + grad(p) = - Rossby*dot(u,grad(u)) + Roberts*Rayleigh*r_vec*T - cross(ez, u) + dot(B,grad(B))"), condition = "ntheta != 0")
problem.add_equation(eq_eval("u = 0"), condition="ntheta == 0")
problem.add_equation(eq_eval("ddt(B) - lap(B) = curl(cross(u, B))"), condition="ntheta != 0")
problem.add_equation(eq_eval("B = 0"), condition="ntheta == 0")
problem.add_equation(eq_eval("ddt(T) - Roberts*lap(T) = - dot(u,grad(T)) + T_source"))
problem.add_equation(eq_eval("u_r_bc = 0"), condition="ntheta != 0")
problem.add_equation(eq_eval("u_perp_bc = 0"), condition="ntheta != 0")
problem.add_equation(eq_eval("tau_u = 0"), condition="ntheta == 0")
problem.add_equation(eq_eval("B_potential_bc = 0"), condition="ntheta != 0")
problem.add_equation(eq_eval("tau_B = 0"), condition="ntheta == 0")
problem.add_equation(eq_eval("T(r=1) = 0"))

# Solver
solver = solvers.InitialValueSolver(problem, ts)
solver.stop_sim_time = t_end

# Add taus
alpha_BC = 0

def C(N, ell, deg):
    ab = (alpha_BC,ell+deg+0.5)
    cd = (2,       ell+deg+0.5)
    return dedalus_sphere.jacobi128.coefficient_connection(N - ell//2,ab,cd)

def BC_rows(N, ell, num_comp):
    N_list = (np.arange(num_comp)+1)*(N - ell//2 + 1)
    return N_list

for subproblem in solver.subproblems:
    ell = subproblem.group[1]
    M = subproblem.M_min
    L = subproblem.L_min
    shape = M.shape
    subproblem.M_min[:,-7:] = 0
    subproblem.M_min.eliminate_zeros()
    N0, N1, N2, N3, N4, N5, N6, N7 = BC_rows(Nmax, ell, 8)
    tau_columns = np.zeros((shape[0], 7))
    if ell != 0:
        tau_columns[N0:N1,0] = (C(Nmax, ell, -1))[:,-1]
        tau_columns[N1:N2,1] = (C(Nmax, ell, +1))[:,-1]
        tau_columns[N2:N3,2] = (C(Nmax, ell,  0))[:,-1]
        tau_columns[N3:N4,3] = (C(Nmax, ell, -1))[:,-1]
        tau_columns[N4:N5,4] = (C(Nmax, ell, +1))[:,-1]
        tau_columns[N5:N6,5] = (C(Nmax, ell,  0))[:,-1]
        tau_columns[N6:N7,6] = (C(Nmax, ell,  0))[:,-1]
        subproblem.L_min[:,-7:] = tau_columns
    else: # ell = 0
        tau_columns[N6:N7, 6] = (C(Nmax, ell, 0))[:,-1]
        subproblem.L_min[:,-1:] = tau_columns[:,6:]
    subproblem.L_min.eliminate_zeros()
    subproblem.expand_matrices(['M','L'])

# Analysis
t_list = []
KE_list = []
ME_list = []
weight_theta = b.local_colatitude_weights(1)
weight_r = b.local_radius_weights(1)
reducer = GlobalArrayReducer(d.comm_cart)
vol_test = np.sum(weight_r*weight_theta+0*p['g'])*np.pi/(Lmax+1)/L_dealias
vol_test = reducer.reduce_scalar(vol_test, MPI.SUM)
vol_correction = 4*np.pi/3/vol_test

# Main loop
start_time = time.time()
while solver.ok:
    if solver.iteration % 10 == 0:
        KE = np.sum(vol_correction*weight_r*weight_theta*u['g'].real**2)
        KE = 0.5*KE*(np.pi)/(Lmax+1)/L_dealias
        KE = reducer.reduce_scalar(KE, MPI.SUM)
        ME = np.sum(vol_correction*weight_r*weight_theta*B['g'].real**2)
        ME = 0.5*ME*(np.pi)/(Lmax+1)/L_dealias
        ME = reducer.reduce_scalar(ME, MPI.SUM)
        logger.info("t = {:f}, KE = {:e},  ME = {:e}".format(solver.sim_time, KE, ME))
        t_list.append(solver.sim_time)
        KE_list.append(KE)
        ME_list.append(ME)
    solver.step(dt)
end_time = time.time()
print('Run time:', end_time-start_time)
