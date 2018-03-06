"""
Post-processing helpers.

"""

import pathlib
import h5py
import numpy as np
from mpi4py import MPI

from ..tools.general import natural_sort
from ..tools.parallel import sync_glob

import logging
logger = logging.getLogger(__name__.split('.')[-1])


def visit_writes(set_paths, function, comm=MPI.COMM_WORLD, **kw):
    """
    Apply function to writes from a list of analysis sets.

    Parameters
    ----------
    set_paths : list of str or pathlib.Path
        List of set paths
    function : function(set_path, start, count, **kw)
        A function on an HDF5 file, start index, and count.
    comm : mpi4py.MPI.Intracomm, optional
        MPI communicator (default: COMM_WORLD)

    Other keyword arguments are passed on to `function`

    Notes
    -----
    This function is parallelized over writes, and so can be effectively
    parallelized up to the number of writes from all specified sets.

    """
    set_paths = natural_sort(str(sp) for sp in set_paths)
    arg_list = zip(set_paths, *get_assigned_writes(set_paths, comm=comm))
    for set_path, start, count in arg_list:
        if count:
            logger.info("Visiting set {} (start: {}, end: {})".format(set_path, start, start+count))
            function(set_path, start, count, **kw)


def get_assigned_writes(set_paths, comm=MPI.COMM_WORLD):
    """
    Divide writes from a list of analysis sets between MPI processes.

    Parameters
    ----------
    set_paths : list of str or pathlib.Path
        List of set paths
    comm : mpi4py.MPI.Intracomm, optional
        MPI communicator (default: COMM_WORLD)

    """
    set_paths = natural_sort(str(sp) for sp in set_paths)
    # Distribute all writes in blocks
    writes = get_all_writes(set_paths)
    block = int(np.ceil(sum(writes) / comm.size))
    proc_start = comm.rank * block
    # Find set start/end indices
    writes = np.array(writes)
    set_ends = np.cumsum(writes)
    set_starts = set_ends - writes
    # Find proc start indices and counts for each set
    starts = np.clip(proc_start, a_min=set_starts, a_max=set_ends)
    counts = np.clip(proc_start+block, a_min=set_starts, a_max=set_ends) - starts
    return starts-set_starts, counts


def get_all_writes(set_paths):
    """
    Get write numbers from a list of analysis sets.

    Parameters
    ----------
    set_paths : list of str or pathlib.Path
        List of set paths

    """
    set_paths = natural_sort(str(sp) for sp in set_paths)
    writes = []
    for set_path in set_paths:
        with h5py.File(str(set_path), mode='r') as file:
            writes.append(file.attrs['writes'])
    return writes


def get_assigned_sets(base_path, distributed=False, comm=MPI.COMM_WORLD):
    """
    Divide analysis sets from a FileHandler between MPI processes.

    Parameters
    ----------
    base_path : str or pathlib.Path
        Base path of FileHandler output
    distributed : bool, optional
        Divide distributed sets instead of merged sets (default: False)
    comm : mpi4py.MPI.Intracomm, optional
        MPI communicator (default: COMM_WORLD)

    """
    base_path = pathlib.Path(base_path)
    base_stem = base_path.stem
    if distributed:
        pattern = "{}_*".format(base_stem)
        set_paths = sync_glob(base_path, pattern, comm)
        set_paths = filter(lambda path: path.is_dir(), set_paths)
    else:
        pattern = "{}_*.h5".format(base_stem)
        set_paths = sync_glob(base_path, pattern, comm)
    set_paths = natural_sort(set_paths)
    return set_paths[comm.rank::comm.size]


def merge_process_files(base_path, cleanup=False, comm=MPI.COMM_WORLD):
    """
    Merge process files from all distributed analysis sets in a folder.

    Parameters
    ----------
    base_path : str or pathlib.Path
        Base path of FileHandler output
    cleanup : bool, optional
        Delete distributed files after merging (default: False)
    comm : mpi4py.MPI.Intracomm, optional
        MPI communicator (default: COMM_WORLD)

    Notes
    -----
    This function is parallelized over sets, and so can be effectively
    parallelized up to the number of distributed sets.

    """
    set_path = pathlib.Path(base_path)
    logger.info("Merging files from {}".format(base_path))

    set_paths = get_assigned_sets(base_path, distributed=True, comm=comm)
    for set_path in set_paths:
        merge_process_files_single_set(set_path, cleanup=cleanup)


def merge_process_files_single_set(set_path, cleanup=False):
    """
    Merge process files from a single distributed analysis set.

    Parameters
    ----------
    set_path : str of pathlib.Path
        Path to distributed analysis set folder
    cleanup : bool, optional
        Delete distributed files after merging (default: False)

    """
    set_path = pathlib.Path(set_path)
    logger.debug("Merging set {}".format(set_path))

    set_stem = set_path.stem
    proc_paths = set_path.glob("{}_p*.h5".format(set_stem))
    proc_paths = natural_sort(proc_paths)
    joint_path = set_path.parent.joinpath("{}.h5".format(set_stem))

    # Create joint file, overwriting if it already exists
    with h5py.File(str(joint_path), mode='w') as joint_file:
        # Setup joint file based on first process file (arbitrary)
        merge_setup(joint_file, proc_paths[0])
        # Merge data from all process files
        for proc_path in proc_paths:
            merge_data(joint_file, proc_path)
    # Cleanup after completed merge, if directed
    if cleanup:
        for proc_path in proc_paths:
            proc_path.unlink()
        set_path.rmdir()


def merge_setup(joint_file, proc_path):
    """
    Merge HDF5 setup from part of a distributed analysis set into a joint file.

    Parameters
    ----------
    joint_file : HDF5 file
        Joint file
    proc_path : str or pathlib.Path
        Path to part of a distributed analysis set

    """
    proc_path = pathlib.Path(proc_path)
    logger.debug("Merging setup from {}".format(proc_path))

    with h5py.File(str(proc_path), mode='r') as proc_file:
        # File metadata
        try:
            joint_file.attrs['set_number'] = proc_file.attrs['set_number']
        except KeyError:
            joint_file.attrs['set_number'] = proc_file.attrs['file_number']
        joint_file.attrs['handler_name'] = proc_file.attrs['handler_name']
        try:
            joint_file.attrs['writes'] = writes = proc_file.attrs['writes']
        except KeyError:
            joint_file.attrs['writes'] = writes = len(proc_file['scales']['write_number'])
        # Copy scales (distributed files all have global scales)
        proc_file.copy('scales', joint_file)
        # Tasks
        joint_tasks = joint_file.create_group('tasks')
        proc_tasks = proc_file['tasks']
        for taskname in proc_tasks:
            # Setup dataset with automatic chunking
            proc_dset = proc_tasks[taskname]
            spatial_shape = proc_dset.attrs['global_shape']
            joint_shape = (writes,) + tuple(spatial_shape)
            joint_dset = joint_tasks.create_dataset(name=proc_dset.name,
                                                    shape=joint_shape,
                                                    dtype=proc_dset.dtype,
                                                    chunks=True)
            # Dataset metadata
            joint_dset.attrs['task_number'] = proc_dset.attrs['task_number']
            joint_dset.attrs['constant'] = proc_dset.attrs['constant']
            joint_dset.attrs['grid_space'] = proc_dset.attrs['grid_space']
            joint_dset.attrs['scales'] = proc_dset.attrs['scales']
            # Dimension scales
            for i, proc_dim in enumerate(proc_dset.dims):
                joint_dset.dims[i].label = proc_dim.label
                for scalename in proc_dim:
                    scale = joint_file['scales'][scalename]
                    joint_dset.dims.create_scale(scale, scalename)
                    joint_dset.dims[i].attach_scale(scale)


def merge_data(joint_file, proc_path):
    """
    Merge data from part of a distributed analysis set into a joint file.

    Parameters
    ----------
    joint_file : HDF5 file
        Joint file
    proc_path : str or pathlib.Path
        Path to part of a distributed analysis set

    """
    proc_path = pathlib.Path(proc_path)
    logger.debug("Merging data from {}".format(proc_path))

    with h5py.File(str(proc_path), mode='r') as proc_file:
        for taskname in proc_file['tasks']:
            joint_dset = joint_file['tasks'][taskname]
            proc_dset = proc_file['tasks'][taskname]
            # Merge across spatial distribution
            start = proc_dset.attrs['start']
            count = proc_dset.attrs['count']
            spatial_slices = tuple(slice(s, s+c) for (s,c) in zip(start, count))
            # Merge maintains same set of writes
            slices = (slice(None),) + spatial_slices
            joint_dset[slices] = proc_dset[:]


def merge_sets(joint_path, set_paths, cleanup=False, comm=MPI.COMM_WORLD):

    """
    Merge analysis sets.

    Parameters
    ----------
    joint_path : string or pathlib.Path
        Path for merged file.
    set_paths : list of strings or pathlib.Path objects
        Paths of all sets to be merged
    cleanup : bool, optional
        Delete set files after merging (default: False)
    comm : mpi4py.MPI.Intracomm, optional
        MPI communicator (default: COMM_WORLD)

    """

    # No parallelization
    if comm.rank > 0:
        return

    joint_path = pathlib.Path(joint_path)
    set_paths = [pathlib.Path(sp) for sp in set_paths]

    # Sort sets by minimum sim time
    set_starts = []
    for set_path in set_paths:
        with h5py.File(str(set_path), mode='r') as file:
            set_starts.append(np.min(file['scales']['sim_time'][:]))
    set_starts, set_paths = zip(*[pair for pair in sorted(zip(set_starts, set_paths))])

    # Find number of writes to extract from each set
    # (only extract writes from sim times before start of next set)
    set_lengths = []
    for n, set_path in enumerate(set_paths):
        with h5py.File(str(set_path), mode='r') as file:
            sim_time = file['scales']['sim_time'][:]
            if (n+1) < len(set_paths):
                set_lengths.append(np.sum(sim_time < set_starts[n+1]))
            else:
                set_lengths.append(len(sim_time))

    logger.info("Creating joint file {}".format(joint_path))
    with h5py.File(str(joint_path), mode='w') as joint_file:
        # Setup file
        logger.debug("Merging setup from {}".format(set_paths[0]))
        with h5py.File(str(set_paths[0]), mode='r') as set_file:
            # File metadata
            joint_file.attrs['handler_name'] = set_file.attrs['handler_name']
            joint_file.attrs['writes'] = writes = np.sum(set_lengths)
            # Copy scales
            set_file.copy('scales', joint_file)
            # Expand time scales
            for scale_name in ['sim_time', 'world_time', 'wall_time', 'iteration', 'write_number']:
                joint_dset = joint_file['scales'][scale_name]
                joint_dset.resize(writes, axis=0)
                joint_dset[:] = 0
            # # Copy tasks
            # set_file.copy('tasks', joint_file)
            # # Expand time axes
            # for task_name in joint_file['tasks']:
            #     joint_dset = joint_file['tasks'][task_name]
            #     joint_dset.resize(writes, axis=0)
            #     joint_dset[:] = 0
            # Tasks
            joint_tasks = joint_file.create_group('tasks')
            set_tasks = set_file['tasks']
            for task_name in set_tasks:
                # Setup dataset with automatic chunking
                set_dset = set_tasks[task_name]
                spatial_shape = set_dset.shape[1:]
                joint_shape = (writes,) + tuple(spatial_shape)
                joint_dset = joint_tasks.create_dataset(name=set_dset.name,
                                                        shape=joint_shape,
                                                        dtype=set_dset.dtype,
                                                        chunks=True)
                # Dataset metadata
                joint_dset.attrs['task_number'] = set_dset.attrs['task_number']
                joint_dset.attrs['constant'] = set_dset.attrs['constant']
                joint_dset.attrs['grid_space'] = set_dset.attrs['grid_space']
                joint_dset.attrs['scales'] = set_dset.attrs['scales']
                # Dimension scales
                for i, set_dim in enumerate(set_dset.dims):
                    joint_dset.dims[i].label = set_dim.label
                    for scale_name in set_dim:
                        scale = joint_file['scales'][scale_name]
                        joint_dset.dims.create_scale(scale, scale_name)
                        joint_dset.dims[i].attach_scale(scale)
        # Merge sets
        i0 = i1 = 0
        for n, set_path in enumerate(set_paths):
            logger.debug("Merging data from {}".format(set_path))
            length = set_lengths[n]
            i1 += length
            with h5py.File(str(set_path), mode='r') as set_file:
                # Copy scales
                for scale_name in ['sim_time', 'world_time', 'wall_time', 'iteration']:
                    set_dset = set_file['scales'][scale_name]
                    joint_dset = joint_file['scales'][scale_name]
                    joint_dset[i0:i1] = set_dset[:length]
                joint_file['scales']['write_number'][i0:i1] = np.arange(i0, i1)
                # Copy tasks
                for task_name in set_file['tasks']:
                    set_dset = set_file['tasks'][task_name]
                    joint_dset = joint_file['tasks'][task_name]
                    joint_dset[i0:i1] = set_dset[:length]
            i0 += length

    # Cleanup after completed merge, if directed
    if cleanup:
        for set_path in set_paths:
            set_path.unlink()


# Reference for backwards compatability
merge_analysis = merge_process_files

