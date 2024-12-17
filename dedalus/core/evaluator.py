"""Class for centralized evaluation of expression trees and handling the results."""

import os
import re
from collections import defaultdict
import pathlib
import h5py
import shutil
import uuid
import numpy as np
import hashlib
from math import prod

from .future import FutureField, FutureLockedField
from .field import Field, LockedField
from .operators import Copy
from ..tools.cache import CachedAttribute
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
            if self.dist.comm.size == 1:
                parallel = 'gather'
            else:
                parallel = FILEHANDLER_PARALLEL_DEFAULT
        if parallel == 'gather':
            FileHandler = H5GatherFileHandler
        elif parallel == 'virtual':
            FileHandler = H5VirtualFileHandler
        elif parallel == 'mpio':
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

    def evaluate_scheduled(self, **kw):
        """Evaluate all scheduled handlers."""
        handlers = [h for h in self.handlers if h.check_schedule(**kw)]
        self.evaluate_handlers(handlers, **kw)

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

        # Transform all outputs to coefficient layout to dealias
        outputs = OrderedSet([t['out'] for h in handlers for t in h.tasks if not isinstance(t['out'], LockedField)])
        self.require_coeff_space(outputs)

        # Copy redundant outputs so processing is independent
        outputs = set()
        for handler in handlers:
            for task in handler.tasks:
                if task['out'] in outputs:
                    task['out'] = task['out'].copy()
                else:
                    outputs.add(task['out'])

        # Process
        for handler in handlers:
            handler.process(**kw)

    def require_coeff_space(self, fields):
        """Move all fields to coefficient layout."""
        coeff_layout = self.dist.coeff_layout
        # Quickly return if all fields are already in coeff layout
        if all(f.layout is coeff_layout for f in fields):
            return
        # Build dictionary of starting layout indices
        layouts = defaultdict(list)
        for f in fields:
            if f.layout is not coeff_layout:
                layouts[f.layout.index].append(f)
        # Decrement all fields down to coeff layout
        current_fields = []
        for index in range(max(layouts.keys()), coeff_layout.index, -1):
            current_fields.extend(layouts[index])
            self.dist.paths[index-1].decrement(current_fields)

    def require_grid_space(self, fields):
        """Move all fields to grid layout."""
        grid_layout = self.dist.grid_layout
        # Quickly return if all fields are already in grid layout
        if all(f.layout is grid_layout for f in fields):
            return
        # Build dictionary of starting layout indices
        layouts = defaultdict(list)
        for f in fields:
            if f.layout is not grid_layout:
                layouts[f.layout.index].append(f)
        # Increment all fields up to grid layout
        current_fields = []
        for index in range(min(layouts.keys()), grid_layout.index):
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
    Group of tasks with associated evaluation schedule.

    Parameters
    ----------
    domain : domain object
        Problem domain
    vars : dict
        Variables for parsing task expression strings
    group : str, optional
        Group name for forcing selected handlers (default: None).
    wall_dt : float, optional
        Wall time cadence for evaluating tasks (default: None).
    sim_dt : float, optional
        Simulation time cadence for evaluating tasks (default: None).
    iter : int, optional
        Iteration cadence for evaluating tasks (default: None).
    custom_schedule : function, optional
        Custom scheduling function returning a boolean for triggering output (default: None).
        Signature for IVPs: custom_schedule(iteration, wall_time, sim_time, timestep)
        Signature for BVPs: custom_schedule(iteration)
    """

    def __init__(self, dist, vars, group=None, wall_dt=None, sim_dt=None, iter=None, custom_schedule=None):
        # Attributes
        self.dist = dist
        self.vars = vars
        self.group = group
        self.wall_dt = wall_dt
        self.sim_dt = sim_dt
        self.iter = iter
        self.custom_schedule = custom_schedule
        self.tasks = []
        # Set initial divisors to be -1 to trigger output on first iteration
        self.last_wall_div = -1
        self.last_sim_div = -1
        self.last_iter_div = -1

    def check_schedule(self, **kw):
        scheduled = False
        # Wall time
        if self.wall_dt:
            wall_div = kw['wall_time'] // self.wall_dt
            if wall_div > self.last_wall_div:
                scheduled = True
                self.last_wall_div = wall_div
        # Sim time
        if self.sim_dt:
            # Output if the output target closest to the current time hasn't triggered
            # an output yet, and the next timestep will not bring you closer.
            t = kw['sim_time']
            dt = kw['timestep']
            closest_sim_div = int(np.round(t / self.sim_dt))
            if closest_sim_div > self.last_sim_div:
                closest_sim_time = closest_sim_div * self.sim_dt
                if abs(t - closest_sim_time) < abs(t + dt - closest_sim_time):
                    scheduled = True
                    self.last_sim_div = closest_sim_div
        # Iteration
        if self.iter:
            iter_div = kw['iteration'] // self.iter
            if iter_div > self.last_iter_div:
                scheduled = True
                self.last_iter_div = iter_div
        # Custom call
        if self.custom_schedule:
            if self.custom_schedule(**kw):
                scheduled = True
        return scheduled

    def add_task(self, task, layout='g', name=None, scales=None):
        """Add task to handler."""
        # Default name
        if name is None:
            name = str(task)
        # Create operator
        if isinstance(task, str):
            op = FutureField.parse(task, self.vars, self.dist)
        elif isinstance(task, Field):
            op = Copy(task)
        else:
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
        task['dtype'] = op.dtype
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
        Maximum number of writes per set. Default: None (infinite).
    mode : str, optional
        'overwrite' to delete any present analysis output with the same base path.
        'append' to begin with set number incremented past any present analysis output.
        Default behavior set by config option.
    """

    def __init__(self, base_path, *args, max_writes=None, mode=None, **kw):
        Handler.__init__(self, *args, **kw)
        if mode is None:
            mode = FILEHANDLER_MODE_DEFAULT
        # Check base_path
        base_path = pathlib.Path(base_path).resolve()
        if base_path.is_file():
            raise ValueError("base_path should indicate a folder for storing HDF5 files.")
        # Attributes
        self.base_path = base_path
        self.name = base_path.stem
        self.max_writes = max_writes

        # Resolve mode
        mode = mode.lower()
        if mode not in ['overwrite', 'append']:
            raise ValueError("Write mode {} not defined.".format(mode))

        self.comm = comm = self.dist.comm_cart
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
                base_path.mkdir(exist_ok=True, parents=True)

    def add_task(self, *args, **kw):
        super().add_task(*args, **kw)
        # Add data distribution information to task dictionary
        task = self.tasks[-1]
        global_shape, local_start, local_shape = self.get_data_distribution(task)
        task['global_shape'] = global_shape
        task['local_start'] = local_start
        task['local_shape'] = local_shape
        task['local_size'] = prod(local_shape)
        task['local_slices'] = tuple(slice(start, start+size) for start, size in zip(local_start, local_shape))

    def get_data_distribution(self, task, rank=None):
        """Determine write parameters for a task."""
        if rank is None:
            rank = self.comm.rank
        layout = task['layout']
        scales = task['scales']
        domain = task['operator'].domain
        tensorsig = task['operator'].tensorsig
        # Domain shapes
        global_shape = layout.global_shape(domain, scales)
        local_shape = layout.local_shape(domain, scales, rank=rank)
        # Local start
        local_elements = layout.local_elements(domain, scales, rank=rank)
        local_start = []
        for axis, lei in enumerate(local_elements):
            if lei.size == 0:
                local_start.append(global_shape[axis])
            else:
                local_start.append(lei[0])
        local_start = tuple(local_start)
        # Field shapes with tensor axes
        tensor_shape = tuple(cs.dim for cs in tensorsig)
        global_shape = tensor_shape + global_shape
        local_shape = tensor_shape + local_shape
        local_start = (0,) * len(tensor_shape) + local_start
        return global_shape, local_start, local_shape

    @property
    def current_path(self):
        set_name = f"{self.name}_s{self.set_num}"
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
        """Generate and setup new HDF5 file from root process."""
        with Sync(self.comm):
            if self.comm.rank == 0:
                with h5py.File(str(self.current_file), 'w-') as file:
                    self.setup_file(file)

    def setup_file(self, file):
        """Prepare new HDF5 file for writing."""
        # Metadeta
        file.attrs['set_number'] = self.set_num
        file.attrs['handler_name'] = self.name
        file.attrs['writes'] = self.file_write_num
        # Scales
        file.create_group('scales')
        file['scales'].create_dataset(name='constant', data=np.zeros(1), dtype=np.float64)
        file['scales']['constant'].make_scale('constant')
        for name in ['sim_time', 'timestep', 'wall_time']:
            file['scales'].create_dataset(name=name, shape=(0,), maxshape=(self.max_writes,), dtype=np.float64) # shape[0] = 0 to chunk across writes
            file['scales'][name].make_scale(name)
        for name in ['iteration', 'write_number']:
            file['scales'].create_dataset(name=name, shape=(0,), maxshape=(self.max_writes,), dtype=int) # shape[0] = 0 to chunk across writes
            file['scales'][name].make_scale(name)
        # Tasks
        file.create_group('tasks')
        for task in self.tasks:
            op = task['operator']
            layout = task['layout']
            scales = task['scales']
            dset = self.create_task_dataset(file, task)
            # Metadata
            dset.attrs['constant'] = op.domain.constant
            dset.attrs['grid_space'] = layout.grid_space
            dset.attrs['scales'] = scales
            # Time scales
            dset.dims[0].label = 't'
            for sn in ['sim_time', 'wall_time', 'timestep', 'iteration', 'write_number']:
                dset.dims[0].attach_scale(file['scales'][sn])
            # Spatial scales
            rank = len(op.tensorsig)
            for axis in range(self.dist.dim):
                basis = op.domain.full_bases[axis]
                if basis is None:
                    sn = lookup = 'constant'
                else:
                    subaxis = axis - self.dist.get_basis_axis(basis)
                    if layout.grid_space[axis]:
                        sn = basis.coordsys.coords[subaxis].name
                        data = basis.global_grids(self.dist, scales)[subaxis].ravel()
                    else:
                        sn = 'k' + basis.coordsys.coords[subaxis].name
                        data = layout.global_group_arrays(op.domain, scales)[subaxis]
                    scale_hash = hashlib.sha1(data).hexdigest()
                    lookup = f"{sn}_hash_{scale_hash}"
                    if lookup not in file['scales']:
                        file['scales'].create_dataset(name=lookup, data=data)
                        file['scales'][lookup].make_scale(sn)
                scale = file['scales'][lookup]
                dset.dims[1 + rank + axis].label = sn
                dset.dims[1 + rank + axis].attach_scale(scale)

    def create_task_dataset(self, file, task):
        """Create dataset for a task."""
        # Create resizable dataset, automatically chunking within writes
        shape = (1,) + task['global_shape'] # shape[0] = 1 to automatically chunk within writes
        maxshape = (self.max_writes,) + task['global_shape']
        dset = file['tasks'].create_dataset(name=task['name'], shape=shape, maxshape=maxshape, dtype=task['dtype'])
        return dset

    def process(self, iteration, wall_time=0, sim_time=0, timestep=0):
        """Save task outputs to HDF5 file."""
        # Update write counts
        self.total_write_num += 1
        self.file_write_num += 1
        # Move to next set if necessary
        if self.max_writes is not None:
            if self.file_write_num > self.max_writes:
                self.set_num += 1
                self.file_write_num = 1
        # Write file metadata
        file = self.get_file()
        self.write_file_metadata(file, write_number=self.total_write_num, iteration=iteration, wall_time=wall_time, sim_time=sim_time, timestep=timestep)
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
        for name in ['sim_time', 'wall_time', 'timestep', 'iteration', 'write_number']:
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
        if self.comm.rank == 0:
            return h5py.File(str(self.current_file), mode)

    def close_file(self, file):
        """Close current HDF5 file after processing."""
        # Close on root process
        if self.comm.rank == 0:
            file.close()

    def write_file_metadata(self, file, **kw):
        """Write file metadata and time scales."""
        # Write from root process
        if self.comm.rank == 0:
            super().write_file_metadata(file, **kw)

    def write_task(self, file, task):
        """Write task data."""
        # Gather data in parallel
        out = task['out']
        data = out.gather_data()
        # Write global data from root process
        if self.comm.rank == 0:
            dset = file['tasks'][task['name']]
            dset.resize(self.file_write_num, axis=0)
            dset[self.file_write_num-1] = data


class H5ParallelFileHandler(H5FileHandlerBase):
    """H5FileHandler using parallel HDF5 writes."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        # Fail if not using MPI
        if not h5py.get_config().mpi:
            raise ValueError("H5ParallelFileHandler requires parallel build of h5py.")
        # Set HDF5 property list for collective writing
        self._property_list = h5py.h5p.create(h5py.h5p.DATASET_XFER)
        self._property_list.set_dxpl_mpio(h5py.h5fd.MPIO_COLLECTIVE)

    def open_file(self, mode='r+'):
        """Return current HDF5 file. Must already exist."""
        return h5py.File(str(self.current_file), mode, driver='mpio', comm=self.comm)

    def close_file(self, file):
        """Close current HDF5 file after processing."""
        file.close()

    def create_current_file(self):
        """Generate and setup new HDF5 file."""
        # Setup file with MPIO to avoid write errors and deadlocks during processing
        with h5py.File(str(self.current_file), 'w-', driver='mpio', comm=self.comm) as file:
            self.setup_file(file)

    def create_task_dataset(self, file, task):
        """Create dataset for a task."""
        # Determine chunk size from local data shape on root process
        local_chunks = None
        if self.comm.rank == 0:
            local_chunks = h5py.filters.guess_chunk(task['local_shape'], task['local_shape'], np.dtype(task['dtype']).itemsize)
        local_chunks = self.comm.bcast(local_chunks, root=0)
        # Create resizable dataset, chunking within local writes
        shape = (1,) + task['global_shape']
        maxshape = (self.max_writes,) + task['global_shape']
        chunks = (1,) + local_chunks
        dset = file['tasks'].create_dataset(name=task['name'], shape=shape, maxshape=maxshape, chunks=chunks, dtype=task['dtype'])
        return dset

    def write_task(self, file, task):
        """Write task data."""
        # Collectively resize
        dset = file['tasks'][task['name']]
        dset.resize(self.file_write_num, axis=0)
        index = self.file_write_num - 1
        # Collective low-level write to accomodate empty processes
        # Reshape local data to include leading time dimension to avoid dataspace conversion
        memory_space, file_space = self.get_hdf5_spaces(task, index)
        dset.id.write(memory_space, file_space, task['out'].data[None], dxpl=self._property_list)

        # Non-collective high-level write -- slow
        # local_slices = (index,) + task['local_slices']
        # dset[local_slices] = task['out'].data

        # Collective high-level write -- hangs with empty cores
        # local_slices = (index,) + task['local_slices']
        # with dset.collective:
        #     dset[local_slices] = task['out'].data

    def get_hdf5_spaces(self, task, index):
        """Create HDF5 space objects for writing local portion of a task."""
        memory_shape = (1,) + task['local_shape']
        memory_start = (0,) * len(memory_shape)
        memory_count = (1,) + task['local_shape']
        memory_space = h5py.h5s.create_simple(memory_shape)
        memory_space.select_hyperslab(memory_start, memory_count)
        file_shape = (index+1,) + task['global_shape']
        file_start = (index,) + task['local_start']
        file_count = (1,) + task['local_shape']
        file_space = h5py.h5s.create_simple(file_shape)
        file_space.select_hyperslab(file_start, file_count)
        return memory_space, file_space


class H5VirtualFileHandler(H5FileHandlerBase):
    """H5FileHandler using process files and virtual joint files."""

    @property
    def current_process_file(self):
        return self._current_process_file(self.comm.rank)

    def _current_process_file(self, rank):
        file_name = f"{self.name}_s{self.set_num}_p{rank}.h5"
        return self.current_path.joinpath(file_name)

    @CachedAttribute
    def empty(self):
        return not any(task['local_size'] for task in self.tasks)

    def open_file(self, mode='r+'):
        """Open current HDF5 file for processing."""
        # Only open joint file on root process
        if self.comm.rank == 0:
            joint_file = h5py.File(str(self.current_file), mode)
        else:
            joint_file = None
        # Open local process files on nonempty processes
        if not self.empty:
            proc_file = h5py.File(str(self.current_process_file), mode)
        else:
            proc_file = None
        return joint_file, proc_file

    def close_file(self, file):
        """Close current HDF5 file after processing."""
        joint_file, proc_file = file
        # Close joint file on root process
        if self.comm.rank == 0:
            joint_file.close()
        # Close process files on nonempty processes
        if not self.empty:
            proc_file.close()

    def create_current_file(self):
        """Generate and setup new HDF5 file."""
        # Create joint file
        super().create_current_file()
        # Create set folder
        with Sync(self.comm):
            if self.comm.rank == 0:
                self.current_path.mkdir(exist_ok=True)
        # Create process files on nonempty processes
        if not self.empty:
            # Touch temp files to update filesystem cache
            if FILEHANDLER_TOUCH_TMPFILE:
                tmpfile = self.current_path.joinpath(f"tmpfile_p{self.comm.rank}")
                tmpfile.touch()
            # Create and setup process files
            with h5py.File(str(self.current_process_file), 'w-') as file:
                self.setup_process_file(file)
            # Remove temp files
            if FILEHANDLER_TOUCH_TMPFILE:
                tmpfile.unlink()

    def create_task_dataset(self, file, task):
        """Create dataset for a task."""
        # Note: this is only called from root process during joint file setup
        # Setup joint virtual layout
        shape = (1,) + task['global_shape']
        maxshape = (self.max_writes,) + task['global_shape']
        virtual_layout = h5py.VirtualLayout(shape=shape, maxshape=maxshape, dtype=task['dtype'])
        dset_name = f"tasks/{task['name']}"
        # Add virtual sources from nonempty processes
        for rank in range(self.comm.size):
            global_shape, local_start, local_shape = self.get_data_distribution(task, rank=rank)
            if prod(local_shape):
                shape = (1,) + local_shape
                maxshape = (self.max_writes,) + local_shape
                filename = str(self._current_process_file(rank).relative_to(self.base_path))
                virtual_source = h5py.VirtualSource(filename, dset_name, shape=shape, maxshape=maxshape)
                layout_slices = [slice(h5py.h5s.UNLIMITED)] + [slice(start, start+size) for start, size in zip(local_start, local_shape)]
                source_slices = [slice(h5py.h5s.UNLIMITED)] + [slice(size) for size in local_shape]
                virtual_layout[tuple(layout_slices)] = virtual_source[tuple(source_slices)]
        # Create unlimited virtual dataset to automatically resize based on process files
        dset = file['tasks'].create_virtual_dataset(name=task['name'], layout=virtual_layout, fillvalue=None)
        return dset

    def setup_process_file(self, file):
        """Prepare new HDF5 file for writing."""
        # Tasks
        file.create_group('tasks')
        for task in self.tasks:
            # Create resizable datasets for nonempty processes, automatically chunking within writes
            if task['local_size']:
                shape = (1,) + task['local_shape'] # shape[0] = 1 to automatically chunk within writes
                maxshape = (self.max_writes,) + task['local_shape']
                dset = file['tasks'].create_dataset(name=task['name'], shape=shape, maxshape=maxshape, dtype=task['dtype'])
                # Save write attributes for merging
                dset.attrs['ext_mesh'] = tuple(task['layout'].ext_mesh)
                dset.attrs['ext_coords'] = tuple(task['layout'].ext_coords)
                dset.attrs['global_shape'] = task['global_shape']
                dset.attrs['local_start'] = task['local_start']
                dset.attrs['local_shape'] = task['local_shape']

    def write_file_metadata(self, file, **kw):
        """Write file metadata and time scales."""
        joint_file, proc_file = file
        # Write joint file metadata from root process
        if self.comm.rank == 0:
            super().write_file_metadata(joint_file, **kw)

    def write_task(self, file, task):
        """Write task data."""
        joint_file, proc_file = file
        out = task['out']
        # Write local data to process files from nonempty processes
        if task['local_size']:
            dset = proc_file['tasks'][task['name']]
            dset.resize(self.file_write_num, axis=0)
            dset[self.file_write_num-1] = out.data

    @staticmethod
    def merge_task(file, task_name, overwrite=False):
        """Merge virtual dataset into regular dataset."""
        # Create new dataset
        old_dset = file['tasks'][task_name]
        if not old_dset.is_virtual:
            raise ValueError("Specified dataset is not a virtual dataset.")
        new_name = f"{task_name}_merged"
        new_shape = (1,) + old_dset.shape[1:] # shape[0] = 1 to automatically chunk within writes
        new_dset = file['tasks'].create_dataset(name=new_name, shape=new_shape, maxshape=old_dset.maxshape, dtype=old_dset.dtype)
        new_dset.resize(old_dset.shape[0], axis=0)
        # Copy attributes and scales
        new_dset.attrs.update(old_dset.attrs)
        # Copy data chunk by chunk
        for chunk in new_dset.iter_chunks():
            new_dset[chunk] = old_dset[chunk]
        # Overwrite old dataset if requested
        if overwrite:
            del file['tasks'][task_name]
            file['tasks'].move(new_name, task_name)

