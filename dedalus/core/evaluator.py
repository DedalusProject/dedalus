"""
Class for centralized evaluation of expression trees.

"""

import os
import re
from collections import defaultdict
import pathlib
import h5py
import shutil
import uuid
import numpy as np
from mpi4py import MPI
import hashlib

from .system import FieldSystem
#from .operators import Operator, Cast
#from .operators import FieldCopy
from .future import Future, FutureField
from .field import Field, LockedField
from ..tools.array import reshape_vector
from ..tools.general import OrderedSet
from ..tools.general import oscillate
from ..tools.parallel import Sync

from ..tools.config import config
FILEHANDLER_MODE_DEFAULT = config['analysis'].get('FILEHANDLER_MODE_DEFAULT')
FILEHANDLER_PARALLEL_DEFAULT = config['analysis'].getboolean('FILEHANDLER_PARALLEL_DEFAULT')
FILEHANDLER_TOUCH_TMPFILE = config['analysis'].getboolean('FILEHANDLER_TOUCH_TMPFILE')
FILEHANDLER_VIRTUAL_FILE = config['analysis'].getboolean('FILEHANDLER_VIRTUAL_FILE')

import logging
logger = logging.getLogger(__name__.split('.')[-1])


class Evaluator:
    """
    Coordinates evaluation of operator trees through various handlers.

    Parameters
    ----------
    dist : dist object
        Problem dist
    vars : dict
        Variables for parsing task expression strings

    """

    def __init__(self, dist, vars):
        self.dist = dist
        self.vars = vars
        self.handlers = []
        self.groups = defaultdict(list)

    def add_dictionary_handler(self, **kw):
        """Create a dictionary handler and add to evaluator."""
        DH = DictionaryHandler(self.dist, self.vars, **kw)
        return self.add_handler(DH)

    def add_system_handler(self, **kw):
        """Create a system handler and add to evaluator."""
        SH = SystemHandler(self.dist, self.vars, **kw)
        return self.add_handler(SH)

    def add_file_handler(self, filename, **kw):
        """Create a file handler and add to evaluator."""
        FH = FileHandler(filename, self.dist, self.vars, **kw)
        return self.add_handler(FH)

    def add_handler(self, handler):
        """Add a handler to evaluator."""
        self.handlers.append(handler)
        # Register with group
        if handler.group is not None:
            self.groups[handler.group].append(handler)
        return handler

    def evaluate_group(self, group, **kw):
        """Evaluate all handlers in a group."""
        handlers = self.groups[group]
        self.evaluate_handlers(handlers, **kw)

    def evaluate_scheduled(self, wall_time, sim_time, iteration, **kw):
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

        self.evaluate_handlers(scheduled_handlers, wall_time=wall_time, sim_time=sim_time, iteration=iteration, **kw)

    def evaluate_handlers(self, handlers, id=None, **kw):
        """Evaluate a collection of handlers."""

        # Default to uuid to cache within evaluation, but not across evaluations
        if id is None:
            id = uuid.uuid4()

        tasks = [t for h in handlers for t in h.tasks]
        for task in tasks:
            task['out'] = None

        # Attempt initial evaluation
        tasks = self.attempt_tasks(tasks, id=id)

        # Move all fields to coefficient layout
        fields = self.get_fields(tasks)
        self.require_coeff_space(fields)
        tasks = self.attempt_tasks(tasks, id=id)

        # Oscillate through layouts until all tasks are evaluated
        # Limit to 10 passes to break on potential infinite loops
        n_layouts = len(self.dist.layouts)
        oscillate_indices = oscillate(range(n_layouts), max_passes=10)
        current_index = next(oscillate_indices)
        while tasks:
            next_index = next(oscillate_indices)
            # Transform fields
            fields = self.get_fields(tasks)
            if current_index < next_index:
                self.dist.paths[current_index].increment(fields)
            else:
                self.dist.paths[next_index].decrement(fields)
            current_index = next_index
            # Attempt evaluation
            tasks = self.attempt_tasks(tasks, id=id)

        # # Transform all outputs to coefficient layout to dealias
        ## D3 note: need to worry about this for redundent tasks?
        # outputs = OrderedSet([t['out'] for h in handlers for t in h.tasks])
        # self.require_coeff_space(outputs)

        # # Copy redundant outputs so processing is independent
        # outputs = set()
        # for handler in handlers:
        #     for task in handler.tasks:
        #         if task['out'] in outputs:
        #             task['out'] = task['out'].copy()
        #         else:
        #             outputs.add(task['out'])
        if 'F' in self.groups.keys():
            for handler in self.groups['F']:
                for task in handler.tasks:
                    task['out'].require_coeff_space()

        # Process
        for handler in handlers:
            handler.process(**kw)

    def require_coeff_space(self, fields):
        """Move all fields to coefficient layout."""
        # Build dictionary of starting layout indices
        layouts = defaultdict(list, {0:[]})
        for f in fields:
            layouts[f.layout.index].append(f)
        # Decrement all fields down to layout 0
        max_index = max(layouts.keys())
        current_fields = []
        for index in range(max_index, 0, -1):
            current_fields.extend(layouts[index])
            self.dist.paths[index-1].decrement(current_fields)

    @staticmethod
    def get_fields(tasks):
        """Get field set for a collection of tasks."""
        fields = OrderedSet()
        for task in tasks:
            fields.update(task['operator'].atoms(Field))
        # Drop locked fields
        locked = [f for f in fields if isinstance(f, LockedField)]
        for field in locked:
            fields.pop(field)
        return fields

    @staticmethod
    def attempt_tasks(tasks, **kw):
        """Attempt tasks and return the unfinished ones."""
        unfinished = []
        for task in tasks:
            output = task['operator'].attempt(**kw)
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

    def __init__(self, dist, vars, group=None, wall_dt=np.inf, sim_dt=np.inf, iter=np.inf):

        # Attributes
        self.dist = dist
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

    def add_task(self, task, layout='g', name=None, scales=None):
        """Add task to handler."""

        # Default name
        if name is None:
            name = str(task)

        # Create operator
        if isinstance(task, str):
            op = FutureField.parse(task, self.vars, self.dist)
        else:
            # op = FutureField.cast(task, self.domain)
            # op = Cast(task)
            # TODO: figure out if we need to copying here
            op = task
        # Build task dictionary
        task = dict()
        task['operator'] = op
        task['layout'] = self.dist.get_layout_object(layout)
        task['name'] = name
        task['scales'] = self.dist.remedy_scales(scales)

        self.tasks.append(task)

    def add_tasks(self, tasks, **kw):
        """Add multiple tasks."""

        name = kw.pop('name', '')
        for task in tasks:
            tname = name + str(task)
            self.add_task(task, name=tname, **kw)

    def add_system(self, system, **kw):
        """Add fields from a FieldSystem."""

        self.add_tasks(system.fields, **kw)


class DictionaryHandler(Handler):
    """Handler that stores outputs in a dictionary."""

    def __init__(self, *args, **kw):
        Handler.__init__(self, *args, **kw)
        self.fields = dict()

    def __getitem__(self, item):
        return self.fields[item]

    def process(self, **kw):
        """Reference fields from dictionary."""
        for task in self.tasks:
            task['out'].require_scales(task['scales'])
            task['out'].require_layout(task['layout'])
            self.fields[task['name']] = task['out']


class SystemHandler(Handler):
    """Handler that sets fields in a FieldSystem."""

    def build_system(self):
        """Build FieldSystem and set task outputs."""

        # nfields = len(self.tasks)
        # names = ['sys'+str(i) for i in range(nfields)]
        # self.system = FieldSystem(names, self.domain)

        self.fields = []
        for i, task in enumerate(self.tasks):
            op = task['operator']
            if isinstance(op, FutureField):
                op.out = op.build_out()
                self.fields.append(op.out)
            else:
                self.fields.append(op)
            # field = Field(task['operator'].bases)
            # task['operator'].out = self.system.fields[i]

        #return self.system

    def process(self, **kw):
        """Gather fields into system."""
        #self.system.gather()
        pass



class FileHandler(Handler):
    """
    Handler that writes tasks to an HDF5 file.

    Parameters
    ----------
    base_path : str
        Base path for analyis output folder
    max_writes : int, optional
        Maximum number of writes per set (default: infinite)
    max_size : int, optional
        Maximum file size to write to, in bytes (default: 2**30 = 1 GB).
        (Note: files may be larger after final write.)
    parallel : bool, optional
        Perform parallel writes from each process to single file (True), or
        separately write to individual process files (False).
        Default behavior set by config option.
    mode : str, optional
        'overwrite' to delete any present analysis output with the same base path.
        'append' to begin with set number incremented past any present analysis output.
        Default behavior set by config option.

    """

    def __init__(self, base_path, *args, max_writes=np.inf, max_size=2**30, parallel=None, mode=None, virtual_file=None, **kw):

        Handler.__init__(self, *args, **kw)

        # Resolve defaults from config
        if parallel is None:
            parallel = FILEHANDLER_PARALLEL_DEFAULT
        if mode is None:
            mode = FILEHANDLER_MODE_DEFAULT
        if virtual_file is None:
            virtual_file = FILEHANDLER_VIRTUAL_FILE

        # Check base_path
        base_path = pathlib.Path(base_path).resolve()
        if any(base_path.suffixes):
            raise ValueError("base_path should indicate a folder for storing HDF5 files.")

        # Attributes
        self.base_path = base_path
        self.max_writes = max_writes
        self.max_size = max_size
        self.parallel = parallel
        self.virtual_file = virtual_file
        self._sl_array = np.zeros(1, dtype=int)

        # Resolve mode
        mode = mode.lower()
        if mode not in ['overwrite', 'append']:
            raise ValueError("Write mode {} not defined.".format(mode))

        comm = self.dist.comm_cart
        if comm.rank == 0:
            set_pattern = '%s_s*' % (self.base_path.stem)
            sets = list(self.base_path.glob(set_pattern))
            if mode == "overwrite":
                for set in sets:
                    if set.is_dir():
                        shutil.rmtree(str(set))
                    else:
                        set.unlink()
                set_num = 1
                total_write_num = 0
            elif mode == "append":
                set_nums = []
                if sets:
                    for set in sets:
                        m = re.match("{}_s(\d+)$".format(base_path.stem), set.stem)
                        if m:
                            set_nums.append(int(m.groups()[0]))
                    max_set = max(set_nums)
                    joined_file = base_path.joinpath("{}_s{}.h5".format(base_path.stem,max_set))
                    p0_file = base_path.joinpath("{0}_s{1}/{0}_s{1}_p0.h5".format(base_path.stem,max_set))
                    if os.path.exists(str(joined_file)):
                        with h5py.File(str(joined_file),'r') as testfile:
                            last_write_num = testfile['/scales/write_number'][-1]
                    elif os.path.exists(str(p0_file)):
                        with h5py.File(str(p0_file),'r') as testfile:
                            last_write_num = testfile['/scales/write_number'][-1]
                    else:
                        last_write_num = 0
                        logger.warn("Cannot determine write num from files. Restarting count.")
                else:
                    max_set = 0
                    last_write_num = 0
                set_num = max_set + 1
                total_write_num = last_write_num
        else:
            set_num = None
            total_write_num = None
        # Communicate set and write numbers
        self.set_num = comm.bcast(set_num, root=0)
        self.total_write_num = comm.bcast(total_write_num, root=0)

        # Create output folder
        with Sync(comm):
            if comm.rank == 0:
                base_path.mkdir(exist_ok=True)

        if parallel:
            # Set HDF5 property list for collective writing
            self._property_list = h5py.h5p.create(h5py.h5p.DATASET_XFER)
            self._property_list.set_dxpl_mpio(h5py.h5fd.MPIO_COLLECTIVE)

    def check_file_limits(self):
        """Check if write or size limits have been reached."""
        write_limit = (self.file_write_num >= self.max_writes)
        size_limit = (self.current_path.stat().st_size >= self.max_size)
        logger.debug("checking file limits. Write limit = {}; self.file_write_num = {}, self.max_writes = {}".format(write_limit, self.file_write_num, self.max_writes))

        if not self.parallel:
            # reduce(size_limit, or) across processes
            comm = self.dist.comm_cart
            self._sl_array[0] = size_limit
            comm.Allreduce(MPI.IN_PLACE, self._sl_array, op=MPI.LOR)
            size_limit = self._sl_array[0]
        return (write_limit or size_limit)

    def get_file(self):
        """Return current HDF5 file, creating if necessary."""
        # Create new file if necessary
        if os.path.exists(str(self.current_path)):
            if self.check_file_limits():
                self.set_num += 1
                self.create_current_file()
        else:
            self.create_current_file()
        # Open current file
        if self.parallel:
            comm = self.dist.comm_cart
            h5file = h5py.File(str(self.current_path), 'r+', driver='mpio', comm=comm)
        else:
            h5file = h5py.File(str(self.current_path), 'r+')
            self.file_write_num = h5file['/scales/write_number'].shape[0]
        return h5file

    @property
    def current_path(self):
        comm = self.dist.comm_cart
        set_num = self.set_num
        if self.parallel:
            # Save in base directory
            file_name = '%s_s%i.hdf5' %(self.base_path.stem, set_num)
            return self.base_path.joinpath(file_name)
        else:
            # Save in folders for each filenum in base directory
            folder_name = '%s_s%i' %(self.base_path.stem, set_num)
            folder_path = self.base_path.joinpath(folder_name)
            file_name = '%s_s%i_p%i.h5' %(self.base_path.stem, set_num, comm.rank)
            return folder_path.joinpath(file_name)

    @property
    def current_virtual_path(self):
        comm = self.dist.comm_cart
        set_num = self.set_num
        if comm.rank == 0 and self.virtual_file:
            file_name = '%s_s%i.h5' %(self.base_path.stem, set_num)
            return self.base_path.joinpath(file_name)
        else:
            return None

    def create_current_file(self):
        """Generate new HDF5 file in current_path."""
        self.file_write_num = 0
        comm = self.dist.comm_cart
        if self.parallel:
            file = h5py.File(str(self.current_path), 'w-', driver='mpio', comm=comm)
        else:
            # Create set folder
            with Sync(comm):
                if comm.rank == 0:
                    self.current_path.parent.mkdir()
            if FILEHANDLER_TOUCH_TMPFILE:
                tmpfile = self.base_path.joinpath('tmpfile_p%i' %(comm.rank))
                tmpfile.touch()
            file = h5py.File(str(self.current_path), 'w-')
            if FILEHANDLER_TOUCH_TMPFILE:
                tmpfile.unlink()
        self.setup_file(file)

        file.close()

        # def create_current_virtual_file(self):
        #     if comm.rank == 0:

    def construct_virtual_sources(self, task, file_shape):
        taskname = task['name']
        layout = task['layout']
        scales = task['scales']
        op = task['operator']
        virt_layout = h5py.VirtualLayout(shape=file_shape, dtype=op.dtype)

        for i in range(self.dist.comm_cart.size):
            file_name = '%s_s%i_p%i.h5' %(self.base_path.stem, self.set_num, i)
            folder_name = '%s_s%i' %(self.base_path.stem, self.set_num)
            folder_path = self.base_path.joinpath(folder_name)
            src_file_name = folder_path.joinpath(file_name).relative_to(self.base_path)
            gnc_shape, gnc_start, write_shape, write_start, write_count = self.get_write_stats(layout, scales, op.domain, op.tensorsig, index=0, virtual_file=True, rank=i)

            shape_stop = len(op.tensorsig) + 1
            src_shape = file_shape[slice(0,shape_stop)] + layout.local_shape(op.domain, scales, rank=i)
            start = gnc_start
            count = write_count

            spatial_slices = tuple(slice(s, s+c) for (s,c) in zip(start, count))

            slices = (slice(None),) + spatial_slices
            maxshape = (None,) + tuple(count)
            tname = 'tasks/{}'.format(taskname)

            vsource = h5py.VirtualSource(src_file_name, name=tname, shape=src_shape, maxshape=maxshape)
            virt_layout[slices] = vsource

        return virt_layout

    def setup_file(self, file, virtual_file=False):

        # Skip spatial scales for now
        skip_spatial_scales = True
        dist = self.dist
        comm = dist.comm_cart

        if virtual_file and comm.rank != 0:
            raise ValueError("Rank {} attemped to setup the virutal file. This should never happen.".format(comm.rank))
        # Metadeta
        file.attrs['set_number'] = self.set_num
        file.attrs['handler_name'] = self.base_path.stem
        file.attrs['writes'] = self.file_write_num
        if not self.parallel and not virtual_file:
            file.attrs['mpi_rank'] = dist.comm_cart.rank
            file.attrs['mpi_size'] = dist.comm_cart.size

        # Scales
        scale_group = file.create_group('scales')
        # Start time scales with shape=(0,) to chunk across writes
        scale_group.create_dataset(name='sim_time', shape=(0,), maxshape=(None,), dtype=np.float64)
        scale_group.create_dataset(name='timestep', shape=(0,), maxshape=(None,), dtype=np.float64)
        scale_group.create_dataset(name='world_time', shape=(0,), maxshape=(None,), dtype=np.float64)
        scale_group.create_dataset(name='wall_time', shape=(0,), maxshape=(None,), dtype=np.float64)
        scale_group.create_dataset(name='iteration', shape=(0,), maxshape=(None,), dtype=np.int)
        scale_group.create_dataset(name='write_number', shape=(0,), maxshape=(None,), dtype=np.int)
        scale_group.create_dataset(name='constant', data=np.array([0.], dtype=np.float64))
        scale_group['constant'].make_scale()

        # Tasks
        task_group =  file.create_group('tasks')
        for task_num, task in enumerate(self.tasks):
            op = task['operator']
            layout = task['layout']
            scales = task['scales']
            gnc_shape, gnc_start, write_shape, write_start, write_count = self.get_write_stats(layout, scales, op.domain, op.tensorsig, index=0, virtual_file=virtual_file)
            if np.prod(write_shape) <= 1:
                # Start with shape[0] = 0 to chunk across writes for scalars
                file_shape = (0,) + tuple(write_shape)
            else:
                # Start with shape[0] = 1 to chunk within writes
                file_shape = (1,) + tuple(write_shape)
            file_max = (None,) + tuple(write_shape)

            if virtual_file:
                # set up virtual layout
                virt_layout = self.construct_virtual_sources(task, file_shape)
                # create virtual dataset
                dset = task_group.create_virtual_dataset(task['name'], virt_layout, fillvalue=None)
            else:
                dset = task_group.create_dataset(name=task['name'], shape=file_shape, maxshape=file_max, dtype=op.dtype)

            self.dset_metadata(task, task_num, dset, scale_group, gnc_shape, gnc_start, write_count, virtual_file=virtual_file)

    def dset_metadata(self, task, task_num, dset, scale_group, gnc_shape, gnc_start, write_count, virtual_file=False):
        op = task['operator']
        layout = task['layout']
        scales = task['scales']

        if not self.parallel:
            dset.attrs['global_shape'] = gnc_shape
            dset.attrs['start'] = gnc_start
            dset.attrs['count'] = write_count

        # Metadata and scales
        dset.attrs['task_number'] = task_num
        dset.attrs['constant'] = op.domain.constant
        dset.attrs['grid_space'] = layout.grid_space
        dset.attrs['scales'] = scales

        # Time scales
        dset.dims[0].label = 't'
        for sn in ['sim_time', 'world_time', 'wall_time', 'timestep', 'iteration', 'write_number']:
            scale = scale_group[sn]
            dset.dims.create_scale(scale, sn)
            dset.dims[0].attach_scale(scale)

        # Spatial scales
        for axis in range(self.dist.dim):
            basis = op.domain.full_bases[axis]
            if basis is None:
                sn = lookup = 'constant'
            else:
                subaxis = axis - basis.axis
                if layout.grid_space[axis]:
                    sn = basis.coordsystem.coords[subaxis].name
                    if virtual_file:
                        data = basis.global_grids(scales)[subaxis].ravel()
                    else:
                        data = basis.local_grids(scales)[subaxis].ravel()
                else:
                    sn = 'k' + basis.coordsystem.coords[subaxis].name
                    if virtual_file:
                        data = basis.global_elements()[subaxis].ravel()
                    else:
                        data = basis.local_elements()[subaxis].ravel()


                if self.dist.comm_cart.rank == 0:
                    scale_hash = hashlib.sha1(data).hexdigest()
                else:
                    scale_hash = None
                if not virtual_file:
                    scale_hash = self.dist.comm_cart.bcast(scale_hash,  root=0)

                lookup = 'hash_' + scale_hash

                if lookup not in scale_group:
                    scale_group.create_dataset(name=lookup, data=data)
                    scale_group[lookup].make_scale()
            scale = scale_group[lookup]
            dset.dims[axis+1].label = sn
            dset.dims[axis+1].attach_scale(scale)


    def process(self, world_time=0, wall_time=0, sim_time=0, timestep=0, iteration=0, **kw):
        """Save task outputs to HDF5 file."""
        # HACK: fix world time and timestep inputs from solvers.py/timestepper.py
        file = self.get_file()
        self.total_write_num += 1
        self.file_write_num += 1
        file.attrs['writes'] = self.file_write_num
        index = self.file_write_num - 1

        # Update time scales
        scale_group = file['scales']
        sim_time_dset = file['scales/sim_time']
        world_time_dset = file['scales/world_time']
        wall_time_dset = file['scales/wall_time']
        timestep_dset = file['scales/timestep']
        iteration_dset = file['scales/iteration']
        write_num_dset = file['scales/write_number']

        sim_time_dset.resize(index+1, axis=0)
        sim_time_dset[index] = sim_time
        world_time_dset.resize(index+1, axis=0)
        world_time_dset[index] = world_time
        wall_time_dset.resize(index+1, axis=0)
        wall_time_dset[index] = wall_time
        timestep_dset.resize(index+1, axis=0)
        timestep_dset[index] = timestep
        iteration_dset.resize(index+1, axis=0)
        iteration_dset[index] = iteration
        write_num_dset.resize(index+1, axis=0)
        write_num_dset[index] = self.total_write_num

        # Create task datasets
        for task_num, task in enumerate(self.tasks):
            out = task['out']
            out.require_scales(task['scales'])
            out.require_layout(task['layout'])

            dset = file['tasks'][task['name']]
            dset.resize(index+1, axis=0)

            memory_space, file_space = self.get_hdf5_spaces(out.layout, task['scales'], out.domain, out.tensorsig, index)
            if self.parallel:
                dset.id.write(memory_space, file_space, out.data, dxpl=self._property_list)
            else:
                dset.id.write(memory_space, file_space, out.data)

        file.close()

        if self.check_file_limits() and self.virtual_file and self.dist.comm_cart.rank == 0:
            self.process_virtual_file()

    def process_virtual_file(self):

        if not self.dist.comm_cart.rank == 0:
            raise ValueError("Processing Virtual File not on root processor. This should never happen.")

        file = h5py.File(str(self.current_virtual_path), 'w-')
        self.setup_file(file, virtual_file=True)
        scale_group = file['scales']
        # get timescales from root processor
        file_name = '%s_s%i_p0.h5' %(self.base_path.stem, self.set_num)
        folder_name = '%s_s%i' %(self.base_path.stem, self.set_num)
        folder_path = self.base_path.joinpath(folder_name)
        src_file_name = folder_path.joinpath(file_name)

        file.attrs['writes'] = self.file_write_num

        with h5py.File(src_file_name,"r") as root_file:
            for time_scale in ['sim_time', 'world_time', 'wall_time', 'timestep', 'iteration', 'write_number']:
                dset = file['scales'][time_scale]
                dset.resize(self.file_write_num, axis=0)
                dset[:] = root_file['scales'][time_scale][:]

        for task_num, task in enumerate(self.tasks):
            # h5py does not support resizing virtual datasets
            # so we must rebuild at each write.
            op = task['operator']
            layout = task['layout']
            scales = task['scales']

            gnc_shape, gnc_start, write_shape, write_start, write_count = self.get_write_stats(layout, scales, op.domain, op.tensorsig, index=0, virtual_file=True)
            file_shape = (self.file_write_num,) + tuple(write_shape)

            virt_layout = self.construct_virtual_sources(task, file_shape)
            # create new virtual dataset
            del file['tasks'][task['name']]
            dset = file['tasks'].create_virtual_dataset(task['name'], virt_layout, fillvalue=None)
            # restore scales
            self.dset_metadata(task, task_num, dset, scale_group, gnc_shape, gnc_start, write_count, virtual_file=True)
        file.close()

    def get_write_stats(self, layout, scales, domain, tensorsig, index, virtual_file=False, rank=None):
        """Determine write parameters for nonconstant subspace of a field."""

        # References
        tensor_order = len(tensorsig)
        tensor_shape = tuple(cs.dim for cs in tensorsig)
        gshape = layout.global_shape(domain, scales)
        lshape = layout.local_shape(domain, scales)

        local_elements = layout.local_elements(domain, scales, rank=rank)
        start = []
        for axis, lei in enumerate(local_elements):
            if lei.size == 0:
                start.append(gshape[axis])
            else:
                start.append(lei[0])
        logger.debug("rank: {}, start = {}".format(rank, start))

        # Build counts, taking just the first entry along constant axes
        write_count = np.array(tensor_shape + lshape)

        # Collectively writing global data
        global_shape = np.array(tensor_shape + gshape)
        global_start = np.array([0 for i in range(tensor_order)] + start)

        if self.parallel or virtual_file:
            # Collectively writing global data
            write_shape = global_shape
            write_start = global_start
        else:
            # Independently writing local data
            write_shape = write_count
            write_start = 0 * global_start

        return global_shape, global_start, write_shape, write_start, write_count

    def get_hdf5_spaces(self, layout, scales, domain, tensorsig, index):
        """Create HDF5 space objects for writing nonconstant subspace of a field."""

        global_shape, global_start, write_shape, write_start, write_count = self.get_write_stats(layout, scales, domain, tensorsig, index)

        memory_shape = tuple(write_count)
        memory_start = tuple(0 * write_start)
        memory_count = tuple(write_count)
        memory_space = h5py.h5s.create_simple(memory_shape)
        memory_space.select_hyperslab(memory_start, memory_count)

        file_shape = (index+1,) + tuple(write_shape)
        file_start = (index,) + tuple(write_start)
        file_count = (1,) + tuple(write_count)
        file_space = h5py.h5s.create_simple(file_shape)
        file_space.select_hyperslab(file_start, file_count)

        return memory_space, file_space

