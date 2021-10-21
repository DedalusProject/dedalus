

import numpy as np
import time
from scipy.special import sph_harm
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

# TODO: convert to float64 and remove imag cleaning once constants are working
# TODO: clean up skew interface


# Parameters
Nphi = 64
Ntheta = 32
dealias = 3/2
amp_ic = 1e-2
l_ic = 10
m_ic = 3
Omega = 1
Re = 1e5
dtype = np.complex128

freq = 2 * m_ic * Omega / (l_ic * (l_ic + 1))
period = 2 * np.pi / freq
timestep = 1 / freq / 10

# Bases
coords = d3.S2Coordinates('phi', 'theta')
dist = d3.Distributor(coords, dtype=dtype)
basis = d3.SphereBasis(coords, (Nphi, Ntheta), radius=1, dealias=dealias, dtype=dtype)
phi, theta = basis.local_grids((1, 1))

# Fields
u = dist.VectorField(coords, name='u', bases=basis)
p = dist.Field(name='p', bases=basis)
g = dist.Field(name='g')

# Substitutions
from dedalus.core.basis import S2Skew
skew = lambda A: S2Skew(A)
zcross = lambda A: d3.MulCosine(skew(A))

# Problem
problem = d3.IVP([u, p, g], namespace=locals())
problem.add_equation("dt(u) + grad(p) - (1/Re)*lap(u) + (2*Omega)*zcross(u) = - dot(u,grad(u))")
problem.add_equation("div(u) + g = 0")
problem.add_equation("ave(p) = 0")

# Solver
solver = problem.build_solver(d3.RK443)
solver.stop_sim_time = 10 * period

# Initial conditions
psi = dist.Field(bases=basis)
psi['g'] = amp_ic * sph_harm(m_ic, l_ic, phi, theta).real
u['c'] = skew(d3.grad(psi)).evaluate()['c']

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', iter=10, max_writes=10)
snapshots.add_task(-d3.div(skew(u)), name='vorticity')

# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.proceed:
        for field in problem.variables:
            field['g'].imag = 0
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            logger.info('Iteration=%i, Time=%e, dt=%e' %(solver.iteration, solver.sim_time, timestep))
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

