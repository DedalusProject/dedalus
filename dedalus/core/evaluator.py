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
from .future import Future, FutureField, FutureLockedField
from .field import Field, LockedField
from ..tools.array import reshape_vector
from ..tools.general import OrderedSet
from ..tools.general import oscillate
from ..tools.parallel import Sync

from ..tools.config import config
FILEHANDLER_MODE_DEFAULT = config['analysis'].get('FILEHANDLER_MODE_DEFAULT')
FILEHANDLER_PARALLEL_DEFAULT = config['analysis'].get('FILEHANDLER_PARALLEL_DEFAULT')
FILEHANDLER_TOUCH_TMPFILE = config['analysis'].getboolean('FILEHANDLER_TOUCH_TMPFILE')

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

    def add_file_handler(self, filename, parallel=None, **kw):
        """Create a file handler and add to evaluator."""
        if parallel is None:
            parallel = FILEHANDLER_PARALLEL_DEFAULT
        if parallel == 'gather':
            FileHandler = H5GatherFileHandler
        elif parallel == 'parallel':
            FileHandler = H5ParallelFileHandler
        else:
            raise ValueError(f"Parallel method '{parallel}' not recognized.")
        return self.add_handler(FileHandler(filename, self.dist, self.vars, **kw))

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
        outputs = OrderedSet([t['out'] for h in handlers for t in h.tasks if not isinstance(t['out'], LockedField)])
        self.require_coeff_space(outputs)

        # Process
        for handler in handlers:
            handler.process(**kw)

    def require_coeff_space(self, fields):
        """Move all fields to coefficient layout."""
        # Build dictionary of starting layout indices
        layouts = defaultdict(list, {0: []})
        for f in fields:
            layouts[f.layout.index].append(f)
        # Decrement all fields down to layout 0
        max_index = max(layouts.keys())
        current_fields = []
        for index in range(max_index, 0, -1):
            current_fields.extend(layouts[index])
            self.dist.paths[index-1].decrement(current_fields)

    def require_grid_space(self, fields):
        """Move all fields to grid layout."""
        # Build dictionary of starting layout indices
        layouts = defaultdict(list, {0: []})
        for f in fields:
            layouts[f.layout.index].append(f)
        # Increment all fields down to grid layout
        grid_index = len(self.dist.layouts) - 1
        min_index = min(layouts.keys())
        current_fields = []
        for index in range(min_index, grid_index):
            current_fields.extend(layouts[index])
            self.dist.paths[index].increment(current_fields)

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
        # Check scales
        if isinstance(op, (LockedField, FutureLockedField)):
            if scales is None:
                scales = op.domain.dealias
            else:
                scales = self.dist.remedy_scales(scales)
                if scales != op.domain.dealias:
                    scales = op.domain.dealias
                    logger.warning("Cannot specify non-delias scales for LockedFields")
        else:
            scales = self.dist.remedy_scales(scales)
        # Build task dictionary
        task = dict()
        task['operator'] = op
        task['layout'] = self.dist.get_layout_object(layout)
        task['name'] = name
        task['scales'] = scales
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
            task['out'].change_scales(task['scales'])
            task['out'].change_layout(task['layout'])
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


class H5FileHandlerBase(Handler):
    """
    Handler that writes tasks to an HDF5 file.

    Parameters
    ----------
    base_path : str
        Base path for analyis output folder
    max_writes : int, optional
        Maximum number of writes per set (default: infinite)
    mode : str, optional
        'overwrite' to delete any present analysis output with the same base path.
        'append' to begin with set number incremented past any present analysis output.
        Default behavior set by config option.

    """

    def __init__(self, base_path, *args, max_writes=np.inf, mode=None, **kw):
        Handler.__init__(self, *args, **kw)
        if mode is None:
            mode = FILEHANDLER_MODE_DEFAULT
        # Check base_path
        base_path = pathlib.Path(base_path).resolve()
        if any(base_path.suffixes):
            raise ValueError("base_path should indicate a folder for storing HDF5 files.")
        # Attributes
        self.base_path = base_path
        self.max_writes = max_writes

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
                        logger.warning("Cannot determine write num from files. Restarting count.")
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
        self.file_write_num = 0 # HACK: need to do this properly

        # Create output folder
        with Sync(comm):
            if comm.rank == 0:
                base_path.mkdir(exist_ok=True)

    @property
    def current_path(self):
        set_name = f"{self.base_path.stem}_s{self.set_num}"
        set_path = self.base_path.joinpath(set_name)
        return set_path

    @property
    def current_file(self):
        return self.current_path.with_suffix('.h5')

    def get_file(self, **kw):
        """Return current HDF5 file, creating if necessary."""
        # Create new file if necessary
        if not self.current_file.exists():
            self.create_current_file()
        # Open current file
        return self.open_file(**kw)

    def create_current_file(self):
        """Generate new HDF5 file in current_path from root node."""
        comm = self.dist.comm_cart
        with Sync(comm):
            if comm.rank == 0:
                file = h5py.File(f"{self.current_path}.h5", 'w-')
                self.setup_file(file)
                file.close()

    def setup_file(self, file):
        dist = self.dist
        comm = dist.comm_cart
        if comm.rank != 0:
            raise ValueError("Rank {} attemped to setup the set file. This should never happen.".format(comm.rank))
        # Metadeta
        file.attrs['set_number'] = self.set_num
        file.attrs['handler_name'] = self.base_path.stem
        file.attrs['writes'] = self.file_write_num
        # Scales
        file.create_group('scales')
        for name in ['sim_time', 'timestep', 'world_time', 'wall_time']:
            file['scales'].create_dataset(name=name, shape=(0,), maxshape=(None,), dtype=np.float64) # shape=(0,) to chunk across writes
            file['scales'][name].make_scale()
        for name in ['iteration', 'write_number']:
            file['scales'].create_dataset(name=name, shape=(0,), maxshape=(None,), dtype=int) # shape=(0,) to chunk across writes
            file['scales'][name].make_scale()
        file['scales'].create_dataset(name='constant', data=np.zeros(1), dtype=np.float64)
        file['scales']['constant'].make_scale()
        # Tasks
        file.create_group('tasks')
        for task in self.tasks:
            op = task['operator']
            layout = task['layout']
            scales = task['scales']
            global_shape = layout.global_shape(op.domain, scales)
            tensor_shape = tuple(cs.dim for cs in op.tensorsig)
            shape = (1,) + tensor_shape + global_shape # shape=(1,...) to chunk within writes
            maxshape = (None,) + tensor_shape + global_shape # shape=(None,...) to allow dynamic resizing
            dset = self.create_dataset(file, task['name'], shape, maxshape, op.dtype) # subclass dependent
            # Metadata
            dset.attrs['constant'] = op.domain.constant
            dset.attrs['grid_space'] = layout.grid_space
            dset.attrs['scales'] = scales
            # Time scales
            dset.dims[0].label = 't'
            for sn in ['sim_time', 'world_time', 'wall_time', 'timestep', 'iteration', 'write_number']:
                dset.dims[0].attach_scale(file['scales'][sn])
            # Spatial scales
            rank = 0
            for axis in range(self.dist.dim):
                basis = op.domain.full_bases[axis]
                if basis is None:
                    sn = lookup = 'constant'
                else:
                    subaxis = axis - basis.axis
                    if layout.grid_space[axis]:
                        sn = basis.coordsystem.coords[subaxis].name
                        data = basis.global_grids(scales)[subaxis].ravel()
                    else:
                        sn = 'k' + basis.coordsystem.coords[subaxis].name
                        data = layout.global_group_arrays(op.domain, scales)[subaxis]
                    scale_hash = hashlib.sha1(data).hexdigest()
                    lookup = f"{sn}_hash_{scale_hash}"
                    if lookup not in file['scales']:
                        file['scales'].create_dataset(name=lookup, data=data)
                        file['scales'][lookup].make_scale()
                scale = file['scales'][lookup]
                dset.dims[1 + rank + axis].label = sn
                dset.dims[1 + rank + axis].attach_scale(scale)

    def create_dataset(self, file, name, shape, maxshape, dtype):
        # Create standard h5py dataset
        return file['tasks'].create_dataset(name=name, shape=shape, maxshape=maxshape, dtype=dtype)

    def process(self, **kw):
        """Save task outputs to HDF5 file."""
        # HACK: fix world time and timestep inputs from solvers.py/timestepper.py
        kw['world_time'] = 0
        # Update write counts
        self.total_write_num += 1
        self.file_write_num += 1
        # Move to next set if necessary
        if self.file_write_num > self.max_writes:
            self.set_num += 1
            self.file_write_num = 1
        # Write file metadata
        file = self.get_file()
        self.write_file_metadata(file, write_number=self.total_write_num, **kw)
        # Write tasks
        for task in self.tasks:
            # Transform and process data
            out = task['out']
            out.change_scales(task['scales'])
            out.change_layout(task['layout'])
            self.write_task(file, task)
        # Finalize
        self.close_file(file)

    def write_file_metadata(self, file, **kw):
        """Write file metadata and time scales."""
        # Update file metadata
        file.attrs['writes'] = self.file_write_num
        # Update time scales
        for name in ['sim_time', 'world_time', 'wall_time', 'timestep', 'iteration', 'write_number']:
            dset = file['scales'][name]
            dset.resize(self.file_write_num, axis=0)
            dset[self.file_write_num-1] = kw[name]

    def write_task(self, file, task):
        """Write task data."""
        raise NotImplementedError("Subclasses must implement.")

    def open_file(self, mode='r+'):
        """Open current HDF5 file for processing."""
        raise NotImplementedError("Subclasses must implement.")

    def close_file(self, file):
        """Close current HDF5 file after processing."""
        raise NotImplementedError("Subclasses must implement.")


class H5GatherFileHandler(H5FileHandlerBase):
    """H5FileHandler that gathers global data to write from root process."""

    def open_file(self, mode='r+'):
        """Open current HDF5 file for processing."""
        # Only open on root process
        if self.dist.comm.rank == 0:
            return h5py.File(str(self.current_file), mode)

    def close_file(self, file):
        """Close current HDF5 file after processing."""
        # Close on root process
        if self.dist.comm.rank == 0:
            file.close()

    def write_file_metadata(self, file, **kw):
        """Write file metadata and time scales."""
        # Write from root process
        if self.dist.comm.rank == 0:
            super().write_file_metadata(file, **kw)

    def write_task(self, file, task):
        """Write task data."""
        # Gather data in parallel
        out = task['out']
        data = out.gather_data()
        # Write global data from root process
        if self.dist.comm.rank == 0:
            dset = file['tasks'][task['name']]
            dset.resize(self.file_write_num, axis=0)
            dset[self.file_write_num-1] = data


class H5ParallelFileHandler(H5FileHandlerBase):
    """H5FileHandler using parallel HDF5 writes."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        # Set HDF5 property list for collective writing
        self._property_list = h5py.h5p.create(h5py.h5p.DATASET_XFER)
        self._property_list.set_dxpl_mpio(h5py.h5fd.MPIO_COLLECTIVE)

    def open_file(self, mode='r+'):
        """Return current HDF5 file. Must already exist."""
        comm = self.dist.comm_cart
        return h5py.File(str(self.current_file), mode, driver='mpio', comm=comm)

    def close_file(self, file):
        """Close current HDF5 file after processing."""
        file.close()

    def write_task(self, file, task):
        """Write task data."""
        # Collectively resize
        dset = file['tasks'][task['name']]
        dset.resize(self.file_write_num, axis=0)
        # Collectively write local data
        out = task['out']
        index = self.file_write_num - 1
        memory_space, file_space = self.get_hdf5_spaces(out, index)
        # Fails when dxpl is included, but this seems to be required if some processes are empty.
        dset.id.write(memory_space, file_space, out.data[None])#, dxpl=self._property_list)

    def get_hdf5_spaces(self, field, index):
        """Create HDF5 space objects for writing local portion of a field."""
        global_shape, local_start, local_shape = self.get_write_stats(field, index)
        memory_shape = (1,) + local_shape
        memory_start = (0,) + (0,) * len(local_shape)
        memory_count = (1,) + local_shape
        memory_space = h5py.h5s.create_simple(memory_shape)
        memory_space.select_hyperslab(memory_start, memory_count)
        file_shape = (index+1,) + global_shape
        file_start = (index,) + local_start
        file_count = (1,) + local_shape
        file_space = h5py.h5s.create_simple(file_shape)
        file_space.select_hyperslab(file_start, file_count)
        return memory_space, file_space

    def get_write_stats(self, field, index):
        """Determine write parameters for nonconstant subspace of a field."""
        layout = field.layout
        scales = field.scales
        domain = field.domain
        # References
        gshape = layout.global_shape(domain, scales)
        lshape = layout.local_shape(domain, scales)
        local_elements = layout.local_elements(domain, scales)
        start = []
        for axis, lei in enumerate(local_elements):
            if lei.size == 0:
                start.append(gshape[axis])
            else:
                start.append(lei[0])
        start = tuple(start)
        # Include tensor axes
        tensor_shape = tuple(cs.dim for cs in field.tensorsig)
        global_shape = tensor_shape + gshape
        local_shape = tensor_shape + lshape
        local_start = (0,) * len(tensor_shape) + start
        return global_shape, local_start, local_shape


class H5VirtualFileHandler(H5FileHandlerBase):

    def create_current_file(self):
        """Generate new HDF5 file in current_path."""
        super().create_current_file()
        self.file_write_num = 0
        comm = self.dist.comm_cart
        # Save in folders for each filenum in base directory
        folder_path = self.current_path
        file_name = '%s_s%i_p%i.h5' %(self.base_path.stem, self.set_num, comm.rank)
        file_path = folder_path.joinpath(file_name)
        # Create set folder
        with Sync(comm):
            if comm.rank == 0:
                folder_path.mkdir()
        if FILEHANDLER_TOUCH_TMPFILE:
            tmpfile = self.base_path.joinpath('tmpfile_p%i' %(comm.rank))
            tmpfile.touch()
        file = h5py.File(str(file_path), 'w-')
        if FILEHANDLER_TOUCH_TMPFILE:
            tmpfile.unlink()
        self.setup_process_file(file)
        file.close()

    def setup_process_file(self, file):
        raise
        # if not self.parallel and not virtual_file:
        # file.attrs['mpi_rank'] = dist.comm_cart.rank
        # file.attrs['mpi_size'] = dist.comm_cart.size

    def create_dataset(self, group, name, shape, maxshape, dtype):
        # Create virtual h5py dataset
        virt_layout = self.construct_virtual_sources(task, maxshape)
        dset = group.create_virtual_dataset(name, virt_layout, fillvalue=None, dtype=dtype)

            # if not self.parallel:
            # dset.attrs['global_shape'] = gnc_shape
            # dset.attrs['start'] = gnc_start
            # dset.attrs['count'] = write_count

    @property
    def current_virtual_path(self):
        comm = self.dist.comm_cart
        set_num = self.set_num
        if comm.rank == 0 and self.virtual_file:
            file_name = '%s_s%i.h5' %(self.base_path.stem, set_num)
            return self.base_path.joinpath(file_name)
        else:
            return None

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
