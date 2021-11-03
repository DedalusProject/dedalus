

import numpy as np
import time
from scipy.special import sph_harm
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

# TODO: convert to float64 and remove imag cleaning once constants are working


# Parameters
Nphi = 128
Ntheta = 64
dealias = 3/2
Omega = 1
# kappa = 1e-2
# nu = 1e-3
# eta = 0.01
dtype = np.complex128
#timestep = 1e-1

LR = 1/4
R = 2

U = LR**2 / 2
Lep = LR / R
ep = (2 * Lep)**5
kappa = ep / U**2
nu = ep / (U * Ntheta)**2 / 10
print('ep:', ep)
print('kappa:', kappa)
print('nu:', nu)

timestep = LR / U
stop_iteration = 1000

# freq = 2 * m_ic * Omega / (l_ic * (l_ic + 1))
# period = 2 * np.pi / freq
# timestep = 1 / freq / 10

#[beta] = 1 / T / L

# Bases
coords = d3.S2Coordinates('phi', 'theta')
dist = d3.Distributor(coords, dtype=dtype)
basis = d3.SphereBasis(coords, (Nphi, Ntheta), radius=1, dealias=dealias, dtype=dtype)
phi, theta = basis.local_grids((1, 1))

# Fields
u = dist.VectorField(coords, name='u', bases=basis)
f = dist.VectorField(coords, name='f', bases=basis)
p = dist.Field(name='p', bases=basis)
g = dist.Field(name='g')

# Substitutions
zcross = lambda A: d3.MulCosine(d3.skew(A))
curl_vec = lambda A: - d3.div(d3.skew(A))

# Problem
problem = d3.IVP([u, p, g], namespace=locals())
problem.add_equation("dt(u) + grad(p) + kappa*u - nu*lap(u) + (2*Omega)*zcross(u) = - dot(u,grad(u)) + f")
problem.add_equation("div(u) + g = 0")
problem.add_equation("ave(p) = 0")

# Solver
solver = problem.build_solver(d3.RK222)
solver.stop_iteration = stop_iteration

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', iter=10, max_writes=10)
snapshots.add_task(curl_vec(u), name='vorticity')

# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.proceed:
        for field in problem.variables:
            field['g'].imag = 0
        f.fill_random(layout='g')
        f.high_pass_filter(scales=0.75)
        f['g'].imag = 0
        f['g'] *= (ep / timestep)**0.5
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            logger.info('Iteration=%i, Time=%e, dt=%e' %(solver.iteration, solver.sim_time, timestep))
            print(np.max(np.abs(u['g'])))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    run_time = end_time - start_time
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %run_time)
    logger.info('Run time: %f cpu-hr' %(run_time/60/60*dist.comm.size))
    DOF = Nphi * Ntheta * 3 + 1
    logger.info('Speed: %.2e DOF-iters/cpu-sec' %(DOF*solver.iteration/run_time/dist.comm.size))

