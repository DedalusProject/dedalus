"""
Class for centralized evaluation of expression trees.

"""

import os
import h5py
import numpy as np
from .system import FieldSystem

from .operators import Operator
from ..tools.general import OrderedSet


class Evaluator:
    """
    Coordinated evaluation of operator trees through handlers.

    Parameters
    ----------
    domain : domain object
        Problem domain
    vars : dict
        Variables for parsing task expression strings

    """

    def __init__(self, domain, vars):

        self.domain = domain
        self.vars = vars
        self.handlers = []

    def add_system_handler(self, **kw):

        SH = SystemHandler(self.domain, self.vars, **kw)
        self.handlers.append(SH)

        return SH

    def add_file_handler(self, filename, **kw):

        FH = FileHandler(filename, self.domain, self.vars, **kw)
        self.handlers.append(FH)

        return FH

    # def add_snapshot_handler(self, filename):

    #     SH = SystemHandler(system, self.vars)
    #     self.handlers.append(SH)

    #     return SH

    def evaluate(self, wall_time, sim_time, iteration, force=False):
        """Evaluate scheduled handlers/tasks."""

        # Find scheduled tasks
        current_handlers = []
        tasks = []
        for handler in self.handlers:

            wall_div = wall_time // handler.wall_dt
            sim_div  = sim_time  // handler.sim_dt
            iter_div = iteration // handler.iter

            wall_up = (wall_div > handler.last_wall_div)
            sim_up  = (sim_div  > handler.last_sim_div)
            iter_up = (iter_div > handler.last_iter_div)

            if force or any((wall_up, sim_up, iter_up)):
                current_handlers.append(handler)
                tasks.extend(handler.tasks)
                handler.last_wall_div = wall_div
                handler.last_sim_div  = sim_div
                handler.last_iter_div = iter_div

        # Return if there are no scheduled handlers
        if not current_handlers:
            return None

        # Start from coefficient space
        fields = self.get_fields(tasks)
        for f in fields:
            f.require_coeff_space()
        L = 0
        tasks = self.attempt_tasks(tasks)

        # Oscillate through layouts until all tasks are evaluated
        Lmax = self.domain.distributor.grid_layout.index
        while tasks:
            # Change direction at first and last layouts
            if L == 0:
                dL = 1
            elif L == Lmax:
                dL = -1
            # Transform fields
            fields = self.get_fields(tasks)
            for f in fields:
                if dL > 0:
                    f.towards_grid_space()
                else:
                    f.towards_coeff_space()
            L += dL
            # Attempt evaluation
            tasks = self.attempt_tasks(tasks)

        # Transform all outputs to coefficient space to dealias
        for handler in current_handlers:
            for task in handler.tasks:
                task['out'].require_coeff_space()

        # Process
        for handler in current_handlers:
            handler.process(wall_time, sim_time, iteration)

    @staticmethod
    def get_fields(tasks):

        fields = OrderedSet()
        for task in tasks:
            fields.update(task['operator'].field_set())

        return fields

    @staticmethod
    def attempt_tasks(tasks):

        unfinished = []
        for task in tasks:
            output = task['operator'].attempt()
            if output is None:
                unfinished.append(task)
            else:
                task['out'] = output

        return unfinished


class Handler:
    """
    Group of tasks with associated scheduling data.

    Parameters
    ----------
    domain : domain object
        Problem domain
    vars : dict
        Variables for parsing task expression strings
    wall_dt : float, optional
        Wall time cadence for evaluating tasks (default: infinite)
    sim_dt : float, optional
        Simulation time cadence for evaluating tasks (default: infinite)
    iter : int, optional
        Iteration cadence for evaluating tasks (default: infinite)

    """

    def __init__(self, domain, vars, wall_dt=np.inf, sim_dt=np.inf, iter=np.inf):

        # Attributes
        self.domain = domain
        self.vars = vars
        self.wall_dt = wall_dt
        self.sim_dt = sim_dt
        self.iter = iter

        self.tasks = []
        self.last_wall_div = 0.
        self.last_sim_div = 0.
        self.last_iter_div = 0.

    def add_task(self, task_str, layout='g', name=None):
        """Add task in string form."""

        # Default name
        if name is None:
            name = task_str

        # Build task dictionary
        task = dict()
        task['operator'] = Operator.from_string(task_str, self.vars, self.domain)
        task['layout'] = layout
        task['name'] = name

        self.tasks.append(task)

    def add_tasks(self, tasks, **kw):
        """Add multiple tasks."""

        for task in tasks:
            self.add_task(task, **kw)


class SystemHandler(Handler):
    """Handler that sets fields in a FieldSystem. """

    def build_system(self):
        """Build FieldSystem and set task outputs."""

        nfields = len(self.tasks)
        self.system = FieldSystem(range(nfields), self.domain)

        for i, task in enumerate(self.tasks):
            task['operator'].out = self.system.fields[i]

        return self.system

    def process(self, wall_time, sim_time, iteration):
        """Gather fields into system."""

        self.system.gather()


class FileHandler(Handler):
    """
    Handler that writes tasks to an HDF5 file.

    Parameters
    ----------
    filename : str
        Base of filename, without an extension
    max_writes : int, optional
        Maximum number of writes to a single file (default: infinite)
    max_size : int, optional
        Maximum file size to write to, in bytes (default: 2**30 = 1 GB).
        (Note: files may be larger after final write.)

    """

    def __init__(self, filename, *args, max_writes=np.inf, max_size=2**30, **kw):

        Handler.__init__(self, *args, **kw)

        # Check filename
        if '.' in filename:
            raise ValueError("Provide filename without an extension.")

        # Attributes
        self.filename_base = filename
        self.max_writes = max_writes
        self.max_size = max_size

        self.file_num = 0
        self.current_file = '.'
        self.write_num = 0

    def get_file(self):
        """Return current HDF5 file, creating if necessary."""

        # Check file limits
        write_limit = ((self.write_num % self.max_writes) == 0)
        size_limit = (os.path.getsize(self.current_file) >= self.max_size)

        if (write_limit or size_limit):
            file = self.new_file()
        else:
            comm = self.domain.distributor.comm_cart
            file = h5py.File(self.current_file, 'a', driver='mpio', comm=comm)

        return file

    def new_file(self):
        """Generate new HDF5 file."""

        # Create next file
        self.file_num += 1
        self.current_file = '%s_%06i.hdf5' %(self.filename_base, self.file_num)
        comm = self.domain.distributor.comm_cart
        file = h5py.File(self.current_file, 'w', driver='mpio', comm=comm)

        # Metadeta
        file.create_group('tasks')
        #file.create_group('scales')

        return file

    def process(self, wall_time, sim_time, iteration):
        """Save task outputs to HDF5 file."""

        file = self.get_file()
        self.write_num += 1

        # Create task group and write timestep attributes
        task_group = file.create_group('tasks/write_%06i' %self.write_num)
        task_group.attrs['wall_time'] = wall_time
        task_group.attrs['sim_time'] = sim_time
        task_group.attrs['iteration'] = iteration

        # Create task datasets
        for task in self.tasks:
            out = task['out']
            name = task['name']
            out.require_layout(task['layout'])
            dtype = out.layout.dtype

            # Assemble nonconstant subspace
            subshape, subslices, subdata = self.get_subspace(out)
            dset = task_group.create_dataset(name=name, shape=subshape, dtype=dtype)
            dset[subslices] = subdata

        file.close()

    @staticmethod
    def get_subspace(field):
        """Return nonconstant subspace of a field, and the corresponding global parameters."""

        # References
        constant = field.constant
        gshape = field.layout.global_shape
        lshape = field.layout.shape
        start = field.layout.start
        slices = field.layout.slices

        # Build subshape from global shape
        subshape = gshape.copy()
        subshape[constant] = 1

        # Build global and local slices
        first = (start == 0)
        subslices = np.array(slices)
        datslices = np.array([slice(ls) for ls in lshape])
        subslices[constant & first] = slice(1)
        datslices[constant & first] = slice(1)
        subslices[constant & ~first] = slice(0)
        datslices[constant & ~first] = slice(0)

        subslices = tuple(subslices)
        subdata = field.data[tuple(datslices)]

        return subshape, subslices, subdata
