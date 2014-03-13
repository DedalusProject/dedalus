"""
Class for centralized evaluation of expression trees.

"""

import os
from collections import defaultdict
import h5py
import numpy as np

from .system import FieldSystem
from .operators import Operator, Cast
from ..tools.array import reshape_vector
from ..tools.general import OrderedSet


class Evaluator:
    """
    Coordinates evaluation of operator trees through various handlers.

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
        self.groups = defaultdict(list)

    def add_system_handler(self, **kw):
        """Create a system handler and add to evaluator."""

        SH = SystemHandler(self.domain, self.vars, **kw)
        return self.add_handler(SH)

    def add_file_handler(self, filename, **kw):
        """Create a file handler and add to evaluator."""

        FH = FileHandler(filename, self.domain, self.vars, **kw)
        return self.add_handler(FH)

    def add_handler(self, handler):
        """Add a handler to evaluator."""

        self.handlers.append(handler)
        # Register with group
        if handler.group is not None:
            self.groups[handler.group].append(handler)
        return handler

    def evaluate_group(self, group, wall_time, sim_time, iteration):
        """Evaluate all handlers in a group."""

        handlers = self.groups[group]
        self.evaluate_handlers(handlers, wall_time, sim_time, iteration)

    def evaluate_scheduled(self, wall_time, sim_time, iteration):
        """Evaluate all scheduled handlers."""

        scheduled_handlers = []
        for handler in self.handlers:
            # Get cadence devisors
            wall_div = wall_time // handler.wall_dt
            sim_div  = sim_time  // handler.sim_dt
            iter_div = iteration // handler.iter
            # Compare to divisor at last evaluation
            wall_up = (wall_div > handler.last_wall_div)
            sim_up  = (sim_div  > handler.last_sim_div)
            iter_up = (iter_div > handler.last_iter_div)

            if any((wall_up, sim_up, iter_up)):
                scheduled_handlers.append(handler)
                # Update all divisors
                handler.last_wall_div = wall_div
                handler.last_sim_div  = sim_div
                handler.last_iter_div = iter_div

        self.evaluate_handlers(scheduled_handlers, wall_time, sim_time, iteration)

    def evaluate_handlers(self, handlers, wall_time, sim_time, iteration):
        """Evaluate a collection of handlers."""

        # Attempt tasks in current layout
        tasks = [t for h in handlers for t in h.tasks]
        tasks = self.attempt_tasks(tasks)

        # Move all to coefficient layout
        fields = self.get_fields(tasks)
        for f in fields:
            f.require_coeff_space()
        tasks = self.attempt_tasks(tasks)

        # Oscillate through layouts until all tasks are evaluated
        L = 0
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

        # Transform all outputs to coefficient layout to dealias
        for handler in handlers:
            for task in handler.tasks:
                task['out'].require_coeff_space()

        # Process
        for handler in handlers:
            handler.process(wall_time, sim_time, iteration)

    @staticmethod
    def get_fields(tasks):
        """Get field set for a collection of tasks."""

        fields = OrderedSet()
        for task in tasks:
            fields.update(task['operator'].field_set())

        return fields

    @staticmethod
    def attempt_tasks(tasks):
        """Attempt tasks and return the unfinished ones."""

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
    group : str, optional
        Group name for forcing selected handelrs (default: None)
    wall_dt : float, optional
        Wall time cadence for evaluating tasks (default: infinite)
    sim_dt : float, optional
        Simulation time cadence for evaluating tasks (default: infinite)
    iter : int, optional
        Iteration cadence for evaluating tasks (default: infinite)

    """

    def __init__(self, domain, vars, group=None, wall_dt=np.inf, sim_dt=np.inf, iter=np.inf):

        # Attributes
        self.domain = domain
        self.vars = vars
        self.group = group
        self.wall_dt = wall_dt
        self.sim_dt = sim_dt
        self.iter = iter

        self.tasks = []
        # Set initial divisors to be scheduled for sim_time, iteration = 0
        self.last_wall_div = -1
        self.last_sim_div = -1
        self.last_iter_div = -1

    def add_task(self, task, layout='g', name=None):
        """Add task to handler."""

        # Default name
        if name is None:
            name = str(task)

        # Create operator
        if isinstance(task, Operator):
            op = task
        elif isinstance(task, str):
            op = Operator.from_string(task, self.vars, self.domain)
        else:
            op = Cast(task)

        # Build task dictionary
        task = dict()
        task['operator'] = op
        task['layout'] = layout
        task['name'] = name

        self.tasks.append(task)

    def add_tasks(self, tasks, **kw):
        """Add multiple tasks."""

        for task in tasks:
            self.add_task(task, **kw)

    def add_system(self, system, **kw):
        """Add fields from a FieldSystem."""

        self.add_tasks(system.fields, **kw)


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

        self._property_list = h5py.h5p.create(h5py.h5p.DATASET_XFER)
        self._property_list.set_dxpl_mpio(h5py.h5fd.MPIO_COLLECTIVE)

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

        # References
        domain = self.domain
        comm = domain.distributor.comm_cart

        # Create next file
        self.file_num += 1
        self.current_file = '%s_%06i.hdf5' %(self.filename_base, self.file_num)
        file = h5py.File(self.current_file, 'w', driver='mpio', comm=comm)

        # Metadeta
        file.attrs['file_number'] = self.file_num
        file.create_group('tasks')
        scale_group = file.create_group('scales')
        const = scale_group.create_dataset(name='constant', shape=[1]*domain.dim, dtype=np.float64)
        for axis, basis in enumerate(domain.bases):
            grid = reshape_vector(basis.grid, domain.dim, axis)
            elem = reshape_vector(basis.elements, domain.dim, axis)
            gdset = scale_group.create_dataset(name=basis.name, shape=grid.shape, dtype=grid.dtype)
            edset = scale_group.create_dataset(name=basis.element_label+basis.name, shape=elem.shape, dtype=elem.dtype)
            if domain.distributor.rank == 0:
                gdset[:] = grid
                edset[:] = elem

        return file

    def process(self, wall_time, sim_time, iteration):
        """Save task outputs to HDF5 file."""

        file = self.get_file()
        self.write_num += 1

        # Create task group and write timestep attributes
        task_group = file.create_group('tasks/write_%06i' %self.write_num)
        task_group.attrs['write_number'] = self.write_num
        task_group.attrs['wall_time'] = wall_time
        task_group.attrs['sim_time'] = sim_time
        task_group.attrs['iteration'] = iteration

        # Create task datasets
        for task_num, task in enumerate(self.tasks):
            out = task['out']
            name = task['name']
            out.require_layout(task['layout'])
            dtype = out.layout.dtype

            # Assemble nonconstant subspace
            subshape, memory_space, file_space = self.get_subspaces(out)
            dset = task_group.create_dataset(name=name, shape=subshape, dtype=dtype)
            dset.id.write(memory_space, file_space, out.data, dxpl=self._property_list)

            # Metadata and scales
            dset.attrs['task_number'] = task_num
            dset.attrs['constant'] = out.constant
            dset.attrs['grid_space'] = out.layout.grid_space
            for axis, basis in enumerate(self.domain.bases):
                if out.constant[axis]:
                    sn = 'constant'
                else:
                    if out.layout.grid_space[axis]:
                        sn = basis.name
                    else:
                        sn = basis.element_label + basis.name
                scale = file['scales'][sn]
                dset.dims.create_scale(scale, sn)
                dset.dims[axis].label = sn
                dset.dims[axis].attach_scale(scale)

        file.close()

    @staticmethod
    def get_subspaces(field):
        """Return HDF5 spaces for writing nonconstant subspace of a field."""

        # References
        constant = field.constant
        gshape = field.layout.global_shape
        lshape = field.layout.shape
        start = field.layout.start
        first = (start == 0)

        # Build subshape from global shape
        subshape = gshape.copy()
        subshape[constant] = 1

        # Build counts based on `constant` and `first`
        count = lshape.copy()
        count[constant & first] = 1
        count[constant & ~first] = 0

        # Build HDF5 spaces
        file_start = start.copy()
        file_start[constant & ~first] = 0
        file_space = h5py.h5s.create_simple(tuple(subshape))
        file_space.select_hyperslab(tuple(file_start), tuple(count))

        memory_start = 0 * start
        memory_space = h5py.h5s.create_simple(tuple(lshape))
        memory_space.select_hyperslab(tuple(memory_start), tuple(count))

        return subshape, memory_space, file_space

