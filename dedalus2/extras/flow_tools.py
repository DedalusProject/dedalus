

import numpy as np
from mpi4py import MPI

from ..tools.logging import logger


class GlobalFlowProperty:

    def __init__(self, solver):

        self.solver = solver
        self.comm = solver.domain.dist.comm_cart
        self.comm_array = np.zeros(1, dtype=np.float64)

    def global_reduce(self, local_value, mpi_reduce_op):
        """Compute global reduction of a scalar from each process."""
        self.comm_array[0] = local_value
        self.comm.Allreduce(MPI.IN_PLACE, self.comm_array, op=mpi_reduce_op)
        return self.comm_array[0]

    def global_min(self, gdata, empty=0):
        """Compute global min of all local grid data."""
        if gdata.size:
            local_min = np.min(gdata)
        else:
            local_min = empty
        return self.global_reduce(local_min, MPI.MIN)

    def global_max(self, gdata, empty=0):
        """Compute global max of all local grid data."""
        if gdata.size:
            local_max = np.max(gdata)
        else:
            local_max = empty
        return self.global_reduce(local_max, MPI.MAX)

    def global_mean(self, gdata):
        """Compute global mean of all local grid data."""
        local_sum = np.sum(gdata)
        local_size = gdata.size
        global_sum = self.global_reduce(local_sum, MPI.SUM)
        global_size = self.global_reduce(local_size, MPI.SUM)
        return global_sum / global_size


class CFL(GlobalFlowProperty):
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

    """

    def __init__(self, solver, first_dt, cadence=1, safety=1., max_dt=np.inf,
                 min_dt=0., max_change=np.inf, min_change=0.):

        self.solver = solver
        self.stored_dt = first_dt
        self.cadence = cadence
        self.safety = safety
        self.max_dt = max_dt
        self.max_change = max_change
        self.min_change = min_change
        self.grid_spacings = [basis.grid_spacing(basis.dealias) for basis in solver.domain.bases]

    def compute_dt(self):
        """Compute new timestep."""
        # Compute new timestep when cadence divides previous iteration
        # This is when the frequency dictionary handler is freshly updated
        if (self.solver.iteration-1) % self.cadence == 0:
            # Sum across frequencies for each local grid point
            local_freqs = np.sum(np.abs(field['g']) for field in self.frequencies.values())
            # Take maximum frequency across all grid points
            max_global_freq = self.global_max(local_freqs, empty=0)
            # Compute new timestep
            if max_global_freq == 0:
                dt = np.inf
            else:
                dt = self.safety / max_global_freq
            dt = min(dt, self.max_dt, self.max_change*self.stored_dt)
            dt = max(dt, self.min_dt, self.min_change*self.stored_dt)
            # Store new timestep
            self.stored_dt = dt

        return self.stored_dt

    def add_velocity(self, components):
        """Add grid-crossing frequencies from a set of velocity components."""
        if len(components) != self.solver.domain.dim:
            raise ValueError("Wrong number of components for domain.")
        for axis, component in enumerate(components):
            freq = Operator(component) / self.grid_spacings[axis]
            self.add_frequency(freq)

    def add_frequency(self, freq):
        """Add an on-grid frequency."""
        self.frequencies.add_task(freq, layout='g')


class ReynoldsPeclet(GlobalFlowProperty):
    """
    Compute basic flow properties (Reynolds number, Peclet number) for a 2D flow.

    """
    def __init__(self, solver, flow_dict, report_cadence):

        GlobalFlowProperty.__init__(self, solver, flow_dict, report_cadence)

    def compute_Re_Pe(self):

        peak_Re = self.global_max(self.variables.fields['rms Re']['g'])
        rms_Re = self.global_mean(self.variables.fields['rms Re']['g'])
        peak_Pe = self.global_max(self.variables.fields['rms Re']['g'])
        rms_Pe = self.global_mean(self.variables.fields['rms Re']['g'])

        return peak_Re, peak_Pe, rms_Re, rms_Pe


class CFL_conv_2D(CFL):
    """
    Compute CFL limited timestep size for 2D flow.

    Assumes that x velocity is u and z velocity is w.
    """
    def __init__(self, solver, max_dt, cfl_cadence=1):

        solver.evaluator.vars['grid_delta_x'] = solver.domain.grid_spacing(0)
        solver.evaluator.vars['grid_delta_z'] = solver.domain.grid_spacing(1)

        freq_dict = {}
        freq_dict['f_u'] = 'u/grid_delta_x'
        freq_dict['f_w'] = 'w/grid_delta_z'

        CFL.__init__(self, solver, freq_dict, max_dt, cfl_cadence)

class RePe_conv_2D(ReynoldsPeclet):
    def __init__(self, solver, report_cadence=50):

        flow_dict = {}
        flow_dict['rms Re']='sqrt(u*u+w*w)*Lz/nu'
        flow_dict['rms Pe']='sqrt(u*u+w*w)*Lz/chi'

        ReynoldsPeclet.__init__(self, solver, flow_dict, report_cadence)

class CFL_KED_3D(CFL):
    def __init__(self, solver, max_dt, cfl_cadence=1):

        solver.evaluator.vars['grid_delta_x'] = solver.domain.grid_spacing(0)
        solver.evaluator.vars['grid_delta_y'] = solver.domain.grid_spacing(1)
        solver.evaluator.vars['grid_delta_z'] = solver.domain.grid_spacing(2)

        freq_dict = {}
        freq_dict['f_u'] = 'dy(p)/grid_delta_x'
        freq_dict['f_v'] = 'dx(p)/grid_delta_y'

        cfl.__init__(self, solver, freq_dict, max_dt, cfl_cadence)

class RePe_KED_3D(ReynoldsPeclet):
    def __init__(self, solver, report_cadence=50):

        flow_dict = {}
        flow_dict['rms Re']='sqrt(dy(p)*dy(p)+dx(p)*dx(p))*Lx/σ'
        flow_dict['rms Pe']='sqrt(dy(p)*dy(p)+dx(p)*dx(p))*Lx'

        ReynoldsPeclet.__init__(self, solver, flow_dict, report_cadence)
