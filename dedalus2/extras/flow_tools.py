import numpy as np
import logging

# for CFL and Re estimation
import mpi4py.MPI as MPI

class GlobalFlowProperty():
    def __init__(self, solver, variable_dict, cadence):
        self.comm = solver.domain.distributor.comm_world

        self.cfl_model = 'mpi allreduce inplace'

        # hacked solution for some processors coming in with zero z size.
        if solver.domain.grid(solver.domain.dim-1).size > 0:
            self.participate = True
        else:
            self.participate = False

        self.comm_array = np.zeros(1, dtype=np.float64)

        self.variables = solver.evaluator.add_dictionary_handler(iter = cadence)
        for key in variable_dict.keys():
            self.variables.add_task(variable_dict[key], name=key)

        self.logger = logging.getLogger(__name__)


    def global_reduce(self, value, mpi_reduce_op, default_value = 0):
        if not self.participate:
            value = default_value

        if self.cfl_model == 'mpi allreduce inplace':
            self.comm_array[0] = value
            self.comm.Allreduce(MPI.IN_PLACE, self.comm_array, op=mpi_reduce_op)
            value = self.comm_array[0]
        else:
            self.logger.info('Invalid cfl_model chosen for CFL communication; terminating ({:s}).'.format(cfl_model))
            value = np.nan

        return value

    def global_min(self, value, default_value = np.inf):
        if not self.participate:
            value = default_value

        value = self.global_reduce(np.min(value), MPI.MIN, default_value = default_value)
        return value

    def global_max(self, value, default_value=0):
        if not self.participate:
            value = default_value

        value = self.global_reduce(np.max(value), MPI.MAX, default_value = default_value)
        return value

    def global_mean(self, value, default_value=0):
        if not self.participate:
            value = default_value

        value = self.global_reduce(np.mean(value), MPI.SUM, default_value=default_value)/self.comm.size
        # note, if there are non-participating cores, this will skew the mean slightly
        return value



class CFL(GlobalFlowProperty):
    """
    Compute CFL limited timestep size given a dictionary of frequencies.

    freq_dict contains a set of dictionary_handler values and keys.

    All variables used in freq_dict (e.g., delta_x_grid) must be set prior to calling cfl.
    """
    def __init__(self, solver, freq_dict, max_dt, cfl_cadence=1):

        GlobalFlowProperty.__init__(self, solver, freq_dict, cfl_cadence)

        self.max_dt = max_dt
        self.cfl_cadence = cfl_cadence

    def compute_dt(self, safety=1.):
        if self.participate:
            freq = 0
            for key in self.variables.fields.keys():
                freq += np.abs(self.variables.fields[key]['g'])

            mint = 1/np.max(freq)
        else:
            # no data in these cores.
            mint = self.max_dt

        dt = safety * mint

        if self.comm:
            dt = self.global_min(dt)

        dt = min(dt, self.max_dt)

        return dt


class ReynoldsPeclet(GlobalFlowProperty):
    """
    Compute basic flow properties (Reynolds number, Peclet number) for a 2D flow.

    """
    def __init__(self, solver, flow_dict, report_cadence):

        GlobalFlowProperty.__init__(self, solver, flow_dict, report_cadence)

    def compute_Re_Pe(self):

        peak_Re = self.global_max(self.variables.fields['rms Re']['g'])

        rms_Re = self.global_mean(self.variables.fields['rms Re']['g'])

        peak_Pe = self.global_max(self.variables.fields['rms Pe']['g'])

        rms_Pe = self.global_mean(self.variables.fields['rms Pe']['g'])

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
