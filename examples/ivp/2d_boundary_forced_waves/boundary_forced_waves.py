"""
Solve the 2D linear wave equation forced on the right y-boundary by an
x and t-dependent forcing function:

dt(f_t) - c**2 (dx(dx(f)) + dy(f_y)) = F(x,t)
dt(f) - f_t = 0
dy(f) - f_y = 0

We solve this on a 2pi * 2pi domain, with Chebyshev in y and Fourier
in x. On the left y boundary, f = 0, and on the right we force the
system. Initial conditions are zero everywhere.

The boundary forcing is periodic in time and localized using the
(spectrally) smooth cosine bump function:

F(x,t) = A*(((1-np.cos(2*np.pi/Lf * x))/2)**n)*np.cos(freq*t)

with n = 10, Lf = 1, and freq = 10.

"""

import time
import numpy as np
import dedalus.public as de
from dedalus.tools  import post

import logging
logger = logging.getLogger(__name__)


# Parameters
nx = 100
ny = 100
c = 1.0
dt = 0.01

# Bases and domain
x_basis = de.Fourier('x', nx, interval=(0, 2*np.pi))
y_basis = de.Chebyshev('y', ny, interval=(0, 2*np.pi))
domain = de.Domain([x_basis, y_basis], grid_dtype='float')

def BoundaryForcing(*args):
    """This function applies its arguments and returns the forcing"""
    t = args[0].value # this is a scalar; we use .value to get its value
    x = args[1].data # this is an array; we use .data to get its values
    ampl = args[2].value
    freq = args[3].value
    return ampl*cos_bump(x)*np.cos(t*freq)

def cos_bump(x, Lx=2*np.pi, n=10):
    """A simple, smooth bump function."""
    return (((1-np.cos(2*np.pi/Lx * x))/2)**n)

def Forcing(*args, domain=domain, F=BoundaryForcing):
    """
    This function takes arguments *args, a function F, and a domain and
    returns a Dedalus GeneralFunction that can be applied.
    """
    return de.operators.GeneralFunction(domain, layout='g', func=F, args=args)

# Now we make it parseable, so the symbol BF can be used in equations
# and the parser will know what it is.
de.operators.parseables['BF'] = Forcing

# Problem
waves = de.IVP(domain, variables=['f','f_t','f_y'])
waves.parameters['csq'] = c**2
waves.parameters['ampl'] = 1.
waves.parameters['freq'] = 10.
waves.add_equation("dt(f_t) - csq*(dx(dx(f)) + dy(f_y)) = 0")
waves.add_equation("f_t - dt(f) = 0")
waves.add_equation("f_y - dy(f) = 0")
# note that we apply the right() operator to the forcing
waves.add_bc("left(f) = 0")
waves.add_bc("right(f) = right(BF(t,x,ampl,freq))")

# Solver
solver = waves.build_solver(de.timesteppers.RK443)
solver.stop_sim_time = 10.

# Analysis
analysis_tasks = []
check = solver.evaluator.add_file_handler('checkpoints', iter=10, max_writes=200)
check.add_system(solver.state)
analysis_tasks.append(check)

# Main loop
start_time = time.time()
while solver.ok:
    solver.step(dt)
    if (solver.iteration-1) % 10 == 0:
        logger.info("Time step {}".format(solver.iteration))
end_time = time.time()

# Print statistics
logger.info('Total time: %f' %(end_time-start_time))
logger.info('Iterations: %i' %solver.iteration)
logger.info('Average timestep: %f' %(solver.sim_time/solver.iteration))

# Merge output
logger.info('beginning join operation')
for task in analysis_tasks:
    logger.info(task.base_path)
    post.merge_analysis(task.base_path)
