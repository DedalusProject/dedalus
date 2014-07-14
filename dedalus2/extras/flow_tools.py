"""
Extra tools that are useful in hydrodynamical problems.

"""

import logging
import numpy as np
from mpi4py import MPI

from ..data.operators import Operator

logger = logging.getLogger(__name__.split('.')[-1])


class GlobalArrayReducer:
    """Directs parallelized reduction of distributed array data."""

    def __init__(self, comm, dtype=np.float64):

        self.comm = comm
        self._scalar_buffer = np.zeros(1, dtype=dtype)

    def reduce_scalar(self, local_scalar, mpi_reduce_op):
        """Compute global reduction of a scalar from each process."""
        self._scalar_buffer[0] = local_scalar
        self.comm.Allreduce(MPI.IN_PLACE, self._scalar_buffer, op=mpi_reduce_op)
        return self._scalar_buffer[0]

    def global_min(self, data, empty=0):
        """Compute global min of all array data."""
        if data.size:
            local_min = np.min(data)
        else:
            local_min = empty
        return self.reduce_scalar(local_min, MPI.MIN)

    def global_max(self, data, empty=0):
        """Compute global max of all array data."""
        if data.size:
            local_max = np.max(data)
        else:
            local_max = empty
        return self.reduce_scalar(local_max, MPI.MAX)

    def global_mean(self, data):
        """Compute global mean of all array data."""
        local_sum = np.sum(data)
        local_size = data.size
        global_sum = self.reduce_scalar(local_sum, MPI.SUM)
        global_size = self.reduce_scalar(local_size, MPI.SUM)
        return global_sum / global_size


class GlobalFlowProperty:
    """
    Directs parallelized determination of a global flow property on the grid.

    Parameters
    ----------
    solver : solver object
        Problem solver
    cadence : int, optional
        Iteration cadence for property evaluation (default: 1)

    Examples
    --------
    >>> flow = GlobalFlowProperty(solver)
    >>> flow.add_task('sqrt(u*u + w*w) * Lz / nu', name='Re')
    ...
    >>> flow.max('Re')
    1024.5

    """

    def __init__(self, solver, cadence=1):

        self.solver = solver
        self.cadence = cadence
        self.reducer = GlobalArrayReducer(solver.domain.dist.comm_cart)
        self.dicthandler = solver.evaluator.add_dictionary_handler(iter=cadence)

    def add_property(self, property, name):
        """Add a property."""
        self.dicthandler.add_task(property, layout='g', name=name)

    def min(self, name):
        """Compute global min of a property on the grid."""
        gdata = self.dicthandler.fields[name]['g']
        return self.reducer.global_min(gdata)

    def max(self, name):
        """Compute global max of a property on the grid."""
        gdata = self.dicthandler.fields[name]['g']
        return self.reducer.global_max(gdata)

    def mean(self, name):
        """Compute global mean of a property on the grid."""
        gdata = self.dicthandler.fields[name]['g']
        return self.reducer.global_mean(gdata)


class CFL:
    """
    Compute CFL-limited timestep from a set of velocities/frequencies.

    Parameters
    ----------
    solver : solver object
        Problem solver
    first_dt : float
        Initial timestep
    cadence : int, optional
        Iteration cadence for computing new timestep (default: 1)
    safety : float, optional
        Safety factor for scaling computed timestep (default: 1.)
    max_dt : float, optional
        Maximum allowable timestep (default: inf)
    min_dt : float, optional
        Minimum allowable timestep (default: 0.)
    max_change : float, optional
        Maximum fractional change between timesteps (default: inf)
    min_change : float, optional
        Minimum fractional change between timesteps (default: 0.)

    Notes
    -----
    The new timestep is computed by summing across the provided frequencies
    for each grid point, and then reciprocating the maximum "total" frequency
    from the entire grid.

    """

    def __init__(self, solver, initial_dt, cadence=1, safety=1., max_dt=np.inf,
                 min_dt=0., max_change=np.inf, min_change=0.):

        self.solver = solver
        self.stored_dt = initial_dt
        self.cadence = cadence
        self.safety = safety
        self.max_dt = max_dt
        self.min_dt = min_dt
        self.max_change = max_change
        self.min_change = min_change

        domain = solver.domain
        self.grid_spacings = [domain.grid_spacing(axis, domain.dealias) for axis in range(domain.dim)]
        self.reducer = GlobalArrayReducer(solver.domain.dist.comm_cart)
        self.dicthandler = solver.evaluator.add_dictionary_handler(iter=cadence)

    def compute_dt(self):
        """Compute new timestep."""
        # Compute new timestep when cadence divides previous iteration
        # This is when the frequency dictionary handler is freshly updated
        if (self.solver.iteration-1) % self.cadence == 0:
            # Sum across frequencies for each local grid point
            local_freqs = np.sum(np.abs(field['g']) for field in self.dicthandler.fields.values())
            # Compute new timestep from max frequency across all grid points
            max_global_freq = self.reducer.global_max(local_freqs)
            if max_global_freq == 0:
                dt = np.inf
            else:
                dt = self.safety / max_global_freq
            dt = min(dt, self.max_dt, self.max_change*self.stored_dt)
            dt = max(dt, self.min_dt, self.min_change*self.stored_dt)
            self.stored_dt = dt

        return self.stored_dt

    def add_velocity(self, components):
        """Add grid-crossing frequencies from a set of velocity components."""
        if len(components) != self.solver.domain.dim:
            raise ValueError("Wrong number of components for domain.")
        for axis, component in enumerate(components):
            comp_op = Operator.from_string(component, self.solver.evaluator.vars, self.solver.domain)
            freq = comp_op / self.grid_spacings[axis]
            self.add_frequency(freq)

    def add_frequency(self, freq):
        """Add an on-grid frequency."""
        self.dicthandler.add_task(freq, layout='g')

