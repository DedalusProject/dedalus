"""
Post-processing helpers.

"""
import os
import pathlib
import hashlib
import shutil
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

def merge_virtual_analysis(base_path, cleanup=False):
    """
    Merge virtual files from a FileHandler into single files.

    Parameters
    ----------
    base_path : str or pathlib.Path
        Base path of FileHandler output
    cleanup : bool, optional
        Delete virtual and distributed files after merging (default: False)

    Notes
    -----
    This function is parallelized over sets, and so can be effectively
    parallelized up to the number of virtual files.

    Merged files are output into a 'merged_X' directory, where X is the
    stem name of the input FileHandler.
    """
    set_path = pathlib.Path(base_path)
    virtual_file_paths = get_assigned_sets(set_path, distributed=False)
    for virtual_file_path in virtual_file_paths:
        merge_virtual_file(virtual_file_path, cleanup=cleanup)

def merge_virtual_file(virtual_file_path, cleanup=False):
    """
    Merge a virtual file from a FileHandler.

    Parameters
    ----------
    virtual_file_path : str of pathlib.Path
        Path to a virtual .h5 file
    cleanup : bool, optional
        Delete distributed files and virtual file after merging (default: False)
    """
    merged_file_path = pathlib.Path(virtual_file_path)
    logger.info("Merging virtual file {}".format(merged_file_path))
    tmp_file_path = merged_file_path.parent.joinpath('tmp_{}.h5'.format(merged_file_path.stem))
    shutil.move(merged_file_path, tmp_file_path)

    # Create joint file, overwriting if it already exists
    with h5py.File(str(merged_file_path), mode='w') as merged_file:
        # Setup joint file based on first process file (arbitrary)
        merge_virtual(merged_file, tmp_file_path)
    os.remove(tmp_file_path)

    # Cleanup after completed merge, if directed
    if cleanup:
        folder = merged_file_path.parent.joinpath("{}/".format(virtual_file_path.stem))
        logger.info("cleaning up {}".format(folder))
        if os.path.isdir(folder):
            shutil.rmtree(folder)


def merge_virtual(joint_file, virtual_path):
    """
    Merge HDF5 setup from part of a distributed analysis set into a joint file.

    Parameters
    ----------
    joint_file : HDF5 file
        Joint file
    virtual_path : path to virtual file [str or pathlib.Path]
        Virtual file
    """
    virt_path = pathlib.Path(virtual_path)
    logger.info("Merging setup for {}".format(joint_file))
    with h5py.File(str(virt_path), mode='r') as virt_file:
        # Copy scales (distributed files all have global scales)
        virt_file.copy('scales', joint_file)

        # Tasks
        joint_tasks = joint_file.create_group('tasks')
        virt_tasks = virt_file['tasks']
        for taskname in virt_tasks:
            try:
                joint_file.attrs['writes'] = writes = virt_file.attrs['writes']
            except KeyError:
                joint_file.attrs['writes'] = writes = len(virt_file['scales']['write_number'])
            # Setup dataset with automatic chunking
            virt_dset = virt_tasks[taskname]
            joint_dset = joint_tasks.create_dataset(name=virt_dset.name, data=virt_dset)

            # Dataset metadata
            joint_dset.attrs['grid_space'] = virt_dset.attrs['grid_space']

            for i, virt_dim in enumerate(virt_dset.dims):
                joint_dset.dims[i].label = virt_dim.label
                if joint_dset.dims[i].label != 't':
                    if virt_dim.label == 'constant' or virt_dim.label == '':
                        scalename = 'constant' 
                    else:
                        hashval = hashlib.sha1(np.array(virt_dset.dims[i][0])).hexdigest()
                        scalename = virt_dim.label + '_hash_' + hashval

                    scale = joint_file['scales'][scalename]
                    joint_dset.dims.create_scale(scale, scalename)
                    joint_dset.dims[i].attach_scale(scale)

