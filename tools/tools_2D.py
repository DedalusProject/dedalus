import numpy as np
from dedalus2.tools.logging import logger

# for CFL and Re estimation
import mpi4py.MPI as MPI

class global_flow_property():
    def __init__(self, solver):
        self.comm = solver.domain.distributor.comm_world

        self.cfl_model = 'mpi allreduce inplace'

        # hacked solution for some processors coming in with zero z size.
        if solver.domain.grid(1).size > 0:
            self.participate = True
        else:
            self.participate = False

        self.comm_array = np.zeros(1, dtype=np.float64)

    def global_reduce(self, value, mpi_reduce_op, default_value = 0):
        if not self.participate:
            value = default_value

        if self.cfl_model == 'mpi allreduce inplace':
            self.comm_array[0] = value
            self.comm.Allreduce(MPI.IN_PLACE, self.comm_array, op=mpi_reduce_op)
            value = self.comm_array[0]
        else:
            logger.info('Invalid cfl_model chosen for CFL communication; terminating ({:s}).'.format(cfl_model))
            value = np.nan

        return value

    def global_min(self, value):
        value = self.global_reduce(np.min(value), MPI.MIN, default_value = np.inf)
        return value

    def global_max(self, value):
        value = self.global_reduce(np.max(value), MPI.MAX)
        return value

    def global_mean(self, value):
        value = self.global_reduce(np.mean(value), MPI.SUM)/self.comm.size
        # note, if there are non-participating cores, this will skew the mean slightly
        return value



class cfl(global_flow_property):
    """
    Compute CFL limited timestep size for 2D flow.

    Assumes that x velocity is u and z velocity is w.
    """
    def __init__(self, solver, max_dt, cfl_cadence=1):
        global_flow_property.__init__(self, solver)

        self.max_dt = max_dt
        self.cfl_cadence = cfl_cadence

        solver.evaluator.vars['grid_delta_x'] = solver.domain.grid_spacing(0)
        solver.evaluator.vars['grid_delta_z'] = solver.domain.grid_spacing(1)        

        self.cfl_variables = solver.evaluator.add_dictionary_handler(iter = cfl_cadence)
        self.cfl_variables.add_task('u/grid_delta_x', name='f_u')
        self.cfl_variables.add_task('w/grid_delta_z', name='f_w')
    
    def compute_dt(self, safety=1.):
        if self.participate:
            minut = 1./np.max(np.abs(self.cfl_variables.fields['f_u']['g']))
            minwt = 1./np.max(np.abs(self.cfl_variables.fields['f_w']['g']))
        else:
            # no data in these cores.
            minut = self.max_dt
            minwt = self.max_dt

        dt = safety * min(minut, minwt)

        if self.comm:
            dt = self.global_min(dt)

        dt = min(dt, self.max_dt)
        
        return dt

class basic_flow_properties(global_flow_property):
    """
    Compute basic flow properties (Reynolds number, Peclet number) for a 2D flow.

    """
    def __init__(self, solver, report_cadence=50):
        global_flow_property.__init__(self, solver)

        self.flow_properties = solver.evaluator.add_dictionary_handler(iter = report_cadence)
        self.flow_properties.add_task('sqrt(u*u+w*w)*Lz/nu',  name='rms Re')
        self.flow_properties.add_task('sqrt(u*u+w*w)*Lz/chi', name='rms Pe')

    def compute_Re_Pe(self):

        peak_Re = self.global_max(self.flow_properties.fields['rms Re']['g'])

        rms_Re = self.global_mean(self.flow_properties.fields['rms Re']['g'])

        peak_Pe = self.global_max(self.flow_properties.fields['rms Re']['g'])

        rms_Pe = self.global_mean(self.flow_properties.fields['rms Re']['g'])

        return peak_Re, peak_Pe, rms_Re, rms_Pe
