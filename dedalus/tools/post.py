"""
Post-processing helpers.

"""

import pathlib
import hashlib
import h5py
import numpy as np
from mpi4py import MPI

from ..tools.general import natural_sort
from ..tools.parallel import Sync

import logging
logger = logging.getLogger(__name__.split('.')[-1])

MPI_RANK = MPI.COMM_WORLD.rank
MPI_SIZE = MPI.COMM_WORLD.size


def visit_writes(set_paths, function, **kw):
    """
    Apply function to writes from a list of analysis sets.

    Parameters
    ----------
    set_paths : list of str or pathlib.Path
        List of set paths
    function : function(set_path, start, count, **kw)
        A function on an HDF5 file, start index, and count.

    Other keyword arguments are passed on to `function`

    Notes
    -----
    This function is parallelized over writes, and so can be effectively
    parallelized up to the number of writes from all specified sets.

    """
    set_paths = natural_sort(str(sp) for sp in set_paths)
    arg_list = zip(set_paths, *get_assigned_writes(set_paths))
    for set_path, start, count in arg_list:
        if count:
            logger.info("Visiting set {} (start: {}, end: {})".format(set_path, start, start+count))
            function(set_path, start, count, **kw)


def get_assigned_writes(set_paths):
    """
    Divide writes from a list of analysis sets between MPI processes.

    Parameters
    ----------
    set_paths : list of str or pathlib.Path
        List of set paths

    """
    set_paths = natural_sort(str(sp) for sp in set_paths)
    # Distribute all writes in blocks
    writes = get_all_writes(set_paths)
    block = int(np.ceil(sum(writes) / MPI_SIZE))
    proc_start = MPI_RANK * block
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


def get_assigned_sets(base_path, distributed=False):
    """
    Divide analysis sets from a FileHandler between MPI processes.

    Parameters
    ----------
    base_path : str or pathlib.Path
        Base path of FileHandler output
    distributed : bool, optional
        Divide distributed sets instead of merged sets (default: False)

    """
    base_path = pathlib.Path(base_path)
    base_stem = base_path.stem
    if distributed:
        set_paths = base_path.glob("{}_*".format(base_stem))
        set_paths = filter(lambda path: path.is_dir(), set_paths)
    else:
        set_paths = base_path.glob("{}_*.h5".format(base_stem))
    set_paths = natural_sort(set_paths)
    return set_paths[MPI_RANK::MPI_SIZE]


def merge_analysis(base_path, cleanup=False):
    """
    Merge distributed analysis sets from a FileHandler.

    Parameters
    ----------
    base_path : str or pathlib.Path
        Base path of FileHandler output
    cleanup : bool, optional
        Delete distributed files after merging (default: False)

    Notes
    -----
    This function is parallelized over sets, and so can be effectively
    parallelized up to the number of distributed sets.

    """
    set_path = pathlib.Path(base_path)
    logger.info("Merging files from {}".format(base_path))

    set_paths = get_assigned_sets(base_path, distributed=True)
    for set_path in set_paths:
        merge_distributed_set(set_path, cleanup=cleanup)


def merge_distributed_set(set_path, cleanup=False):
    """
    Merge a distributed analysis set from a FileHandler.

    Parameters
    ----------
    set_path : str of pathlib.Path
        Path to distributed analysis set folder
    cleanup : bool, optional
        Delete distributed files after merging (default: False)

    """
    set_path = pathlib.Path(set_path)
    logger.info("Merging set {}".format(set_path))

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
    logger.info("Merging setup from {}".format(proc_path))

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
        #proc_file.copy('scales', joint_file)
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

            # TO FIX: MERGE DIMENSIONS DON'T WORK IN d3 YET!
            #         THEY ALSO NEED TO BE MERGED ACROSS PROCS
            # Dimension scales
            # for i, proc_dim in enumerate(proc_dset.dims):
            #     joint_dset.dims[i].label = proc_dim.label
            #     for scalename in proc_dim:
            #         scale = joint_file['scales'][scalename]
            #         joint_dset.dims.create_scale(scale, scalename)
            #         joint_dset.dims[i].attach_scale(scale)


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
    logger.info("Merging data from {}".format(proc_path))

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


def merge_virtual(joint_file, virtual_path):
    """
    Merge HDF5 setup from part of a distributed analysis set into a joint file.
    Parameters
    ----------
    joint_file : HDF5 file
        Joint file
    virtual_path : str or pathlib.Path
        Path to a joined virtual file 
    """
    virtual_path = pathlib.Path(virtual_path)
    logger.info("Merging setup from {}".format(virtual_path))
    with h5py.File(str(virtual_path), mode='r') as virtual_file:
        # File metadata
        try:
            joint_file.attrs['set_number'] = virtual_file.attrs['set_number']
        except KeyError:
            joint_file.attrs['set_number'] = virtual_file.attrs['file_number']
        joint_file.attrs['handler_name'] = virtual_file.attrs['handler_name']
        try:
            joint_file.attrs['writes'] = writes = virtual_file.attrs['writes']
        except KeyError:
            joint_file.attrs['writes'] = writes = len(virtual_file['scales']['write_number'])
        # Copy scales (distributed files all have global scales)
        virtual_file.copy('scales', joint_file)
        # Tasks
        virtual_tasks = virtual_file['tasks']

        joint_tasks = joint_file.create_group('tasks')
        for taskname in virtual_tasks:
            virtual_dset = virtual_tasks[taskname]
            joint_dset = joint_tasks.create_dataset(taskname, data=virtual_dset)

            # Dataset metadata
            joint_dset.attrs['task_number'] = virtual_dset.attrs['task_number']
            joint_dset.attrs['constant'] = virtual_dset.attrs['constant']
            joint_dset.attrs['grid_space'] = virtual_dset.attrs['grid_space']
            joint_dset.attrs['scales'] = virtual_dset.attrs['scales']



            # Dimension scales
            for i, virtual_dim in enumerate(virtual_dset.dims):
                joint_dset.dims[i].label = virtual_dim.label
                if joint_dset.dims[i].label == 't':
                    for sn in ['sim_time', 'world_time', 'wall_time', 'timestep', 'iteration', 'write_number']:
                        scale = joint_file['scales'][sn]
                        joint_dset.dims.create_scale(scale, sn)
                        joint_dset.dims[i].attach_scale(scale)
                else:
                    if virtual_dim.label == 'constant' or virtual_dim.label == '':
                        scalename = 'constant' 
                    else:
                        hashval = hashlib.sha1(np.array(virtual_dset.dims[i][0])).hexdigest()
                        scalename = 'hash_' + hashval
                    scale = joint_file['scales'][scalename]
                    joint_dset.dims.create_scale(scale, scalename)
                    joint_dset.dims[i].attach_scale(scale)

def merge_virtual_file_single_set(set_path, merged_path, cleanup=False):
    set_path = pathlib.Path(set_path)
    logger.info("Merging virtual file {}".format(set_path))

    set_stem = set_path.stem
    if 'merged' in set_stem:
        return
    joint_path = merged_path.joinpath("merged_{}.h5".format(set_stem))

    # Create joint file, overwriting if it already exists
    with h5py.File(str(joint_path), mode='w') as joint_file:
        # Setup joint file based on first process file (arbitrary)
        merge_virtual(joint_file, set_path)

    # Cleanup after completed merge, if directed
    if cleanup:
        folder = set_path.parent.joinpath("{}/".format(set_stem))
        logger.info("cleaning up {}".format(folder))
        if os.path.isdir(folder):
            partial_files = folder.glob('*.h5')
            for pf in partial_files:
                os.remove(pf)
            os.rmdir(folder)
        os.remove(set_path)
        os.rename(joint_path, set_path)

def merge_virtual_analysis(base_path, cleanup=False):
    set_path = pathlib.Path(base_path)
    set_paths = get_assigned_sets(set_path, distributed=False)

    handler_name = set_path.stem
    merged_path = set_path.parent.parent.joinpath("merged_{}/".format(handler_name))
    with Sync() as sync:
        if not merged_path.exists():
            if MPI_RANK == 0:
                merged_path.mkdir(exist_ok=True)

    for set_path in set_paths:
        print('set_path', set_path)
        merge_virtual_file_single_set(set_path, merged_path, cleanup=False)


