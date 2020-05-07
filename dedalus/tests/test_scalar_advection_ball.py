"""
Dedalus script for full sphere testing scalar advection.

An ell=m pattern is advected by solid body rotation (t=1 is one full rotation
of the pattern).  This advected pattern is compared against a static pattern
that diffuses at the same diffusion rate.  Each time the pattern overlaps the
static pattern, the relative L2 error is computed.

Significant apparent (but not true) errors come about from slight mismatch
of the advected and static pattern.  To eliminate this source of misunderstanding,
the actual timestep taken is dt = <dt>/<ell_benchmark>, and ell-many comparisons
are made during the test.

At default settings, on 256 cores, this test takes about 1 minute to run.
The run time grows for higher ell_benchmark, since dt decreases.

Usage:
    test_scalar_advection.py [options]

Options:
    --Pe=<Pe>                            Peclet number of flow [default: 20]
    --c_source=<c_source>                Source function for scalar [default: 0]
    --ell_benchmark=<ell_benchmark>      Integer value of benchmark perturbation m=+-ell [default: 3]

    --polar_flow                         Test flow over the pole by using solid body flow about the x-axis
                                         (default is solid body flow about z-axis)

    --L=<L>                              Max spherical harmonic [default: 31]
    --N=<N>                              Max radial polynomial  [default: 31]
    --t_end=<t_end>                      Stop time of problem; 1 = one revolution [default: 1]
    --dt=<dt>                            Timestep size [default: 1e-3]

    --mesh=<mesh>                        Processor mesh for 3-D runs

    --verbose                            Output frequently during test
"""

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

from docopt import docopt
args = docopt(__doc__)
mesh = args['--mesh']
if mesh is not None:
    mesh = mesh.split(',')
    mesh = (int(mesh[0]), int(mesh[1]))
else:
    log2 = np.log2(MPI.COMM_WORLD.size)
    if log2 == int(log2):
        mesh = (int(2**np.ceil(log2/2)),int(2**np.floor(log2/2)))
    logger.info("running on processor mesh={}".format(mesh))


# Parameters
radius = 1
Lmax = int(args['--L'])
L_dealias = 1
Nmax = int(args['--N'])
N_dealias = 1

𝓁 = int(args['--ell_benchmark'])
dt = float(args['--dt'])

t_end = float(args['--t_end'])
ts = timesteppers.SBDF4

Pe = float(args['--Pe'])



# Bases
c = coords.SphericalCoordinates('phi', 'theta', 'r')
d = distributor.Distributor((c,), mesh=mesh)
b = basis.BallBasis(c, (2*(Lmax+1), Lmax+1, Nmax+1), radius=radius)
bk2 = basis.BallBasis(c, (2*(Lmax+1), Lmax+1, Nmax+1), k=2, radius=radius)
b_S2 = b.S2_basis()
phi, theta, r = b.local_grids((1, 1, 1))

# Fields
u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
T = field.Field(dist=d, bases=(b,), dtype=np.complex128)
T_c = field.Field(dist=d, bases=(b,), dtype=np.complex128)
T_IC = field.Field(dist=d, bases=(b,), dtype=np.complex128)
tau_T = field.Field(dist=d, bases=(b_S2,), dtype=np.complex128)
tau_T_c = field.Field(dist=d, bases=(b_S2,), dtype=np.complex128)

# solid body rotation
Omega = 2*np.pi
if args['--polar_flow']:
    u['g'][0] =  Omega*r*np.sin(phi)*np.cos(theta)
    u['g'][1] = -Omega*r*np.cos(phi)
else:
    dt /= 𝓁
    u['g'][0] = Omega*r*np.sin(theta)

# multi-armed perturbation
A = 1
norm = 1/(2**𝓁*np.math.factorial(𝓁))*np.sqrt(np.math.factorial(2*𝓁+1)/(4*np.pi))
T_IC['g'] = A*norm*r**𝓁*(1-r**2)*(np.cos(𝓁*phi)+np.sin(𝓁*phi))*np.sin(theta)**𝓁
logger.info("benchmark run with perturbations at ell={} with norm={}".format(𝓁, norm))

T['g'] = T_IC['g']
T_c['g'] = T['g']

T_source = field.Field(dist=d, bases=(b,), dtype=np.complex128)
T_source['g'] = float(args['--c_source'])

# Parameters and operators
div = lambda A: operators.Divergence(A, index=0)
lap = lambda A: operators.Laplacian(A, c)
grad = lambda A: operators.Gradient(A, c)
dot = lambda A, B: arithmetic.DotProduct(A, B)
cross = lambda A, B: operators.CrossProduct(A, B)
ddt = lambda A: operators.TimeDerivative(A)

# Problem
def eq_eval(eq_str):
    return [eval(expr) for expr in split_equation(eq_str)]
problem = problems.IVP([T, T_c, tau_T, tau_T_c])

problem.add_equation(eq_eval("ddt(T) - 1/Pe*lap(T) = - dot(u,grad(T)) + T_source"))
problem.add_equation(eq_eval("ddt(T_c) - 1/Pe*lap(T_c) = T_source"))
problem.add_equation(eq_eval("T(r=1) = 0"))
problem.add_equation(eq_eval("T_c(r=1) = 0"))

logger.info("Problem built")

# Solver
solver = solvers.InitialValueSolver(problem, ts)
solver.stop_sim_time = t_end+dt

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
    subproblem.M_min[:,-4:] = 0
    subproblem.M_min.eliminate_zeros()
    N0, N1 = BC_rows(Nmax, ell, 2)
    tau_columns = np.zeros((shape[0], 2))
    tau_columns[:N0,   0] = (C(Nmax, ell, 0))[:,-1]
    tau_columns[N0:N1, 1] = (C(Nmax, ell, 0))[:,-1]
    subproblem.L_min[:,-2:] = tau_columns
    subproblem.L_min.eliminate_zeros()
    #if ell == 0 :  logger.debug("L_min for L={}:\n {}".format(ell, subproblem.L_min[:,-2:]))
    subproblem.expand_matrices(['M','L'])

# Analysis
t_list = []
E_list = []
T_list = []
T_err_list = []
weight_theta = b.local_colatitude_weights(1)
weight_r = b.local_radius_weights(1)
reducer = GlobalArrayReducer(d.comm_cart)
vol_test = np.sum(weight_r*weight_theta+0*T['g'])*np.pi/(Lmax+1)/L_dealias
vol_test = reducer.reduce_scalar(vol_test, MPI.SUM)
vol_correction = 4*np.pi/3/vol_test

report_cadence = 10

# Main loop
start_time = time.time()
while solver.ok:
    if args['--verbose'] and solver.iteration % report_cadence == 0 and solver.iteration > 0 :
        E0 = np.sum(vol_correction*weight_r*weight_theta*u['g'].real**2)
        E0 = 0.5*E0*(np.pi)/(Lmax+1)/L_dealias
        E0 = reducer.reduce_scalar(E0, MPI.SUM)
        T0 = np.sum(vol_correction*weight_r*weight_theta*T['g'].real**2)
        T0 = T0*(np.pi)/(Lmax+1)/L_dealias
        T0 = reducer.reduce_scalar(T0, MPI.SUM)
        T_err = np.sum(vol_correction*weight_r*weight_theta*(T['g'].real-T_c['g'].real)**2)
        T_err = T_err*(np.pi)/(Lmax+1)/L_dealias
        T_err = reducer.reduce_scalar(T_err, MPI.SUM)
        T_ref = np.sum(vol_correction*weight_r*weight_theta*(T_c['g'].real)**2)
        T_ref = T_ref*(np.pi)/(Lmax+1)/L_dealias
        T_ref = reducer.reduce_scalar(T_ref, MPI.SUM)
        logger.info("iter = {:d}, t = {:f}, E = {:g}, c = {:g}, err = {:g}".format(solver.iteration, solver.sim_time, E0, T0, T_err/T_ref))

    if args['--polar_flow']:
        overlap = 2*solver.sim_time
    else:
        overlap = 𝓁*solver.sim_time
    if np.isclose(overlap%1, 1, atol=0.1*dt) or np.isclose(overlap%1, 0, atol=0.1*dt):
        T_err = np.sum(vol_correction*weight_r*weight_theta*(T['g'].real-T_c['g'].real)**2)
        T_err = T_err*(np.pi)/(Lmax+1)/L_dealias
        T_err = reducer.reduce_scalar(T_err, MPI.SUM)
        T_ref = np.sum(vol_correction*weight_r*weight_theta*(T_c['g'].real)**2)
        T_ref = T_ref*(np.pi)/(Lmax+1)/L_dealias
        T_ref = reducer.reduce_scalar(T_ref, MPI.SUM)
        logger.info("at time {} ({}), <T_err**2>/<T_ref**2> =  {:g}".format(solver.sim_time, overlap, T_err/T_ref))
        T_err_list.append((solver.sim_time*𝓁, T_err/T_ref))

    solver.step(dt)
end_time = time.time()

logger.info('Run time: {}'.format(end_time-start_time))
logger.info("relative error comparison each time the pattern returns to original")
for n, err in T_err_list:
    logger.info("comparison {}: relative error = {:g}".format(n, err))
