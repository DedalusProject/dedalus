"""
Post-processing helpers.

"""

import pathlib
import glob
import h5py
import numpy as np
from mpi4py import MPI

from ..tools.general import natural_sort

import logging
logger = logging.getLogger(__name__.split('.')[-1])

MPI_RANK = MPI.COMM_WORLD.rank
MPI_SIZE = MPI.COMM_WORLD.size


def default_function(filename, start, count):
    print('Rank %i: Filename=%s, start=%i, count=%i' %(MPI_RANK, filename, start, count))


def visit(pattern, function=default_function, **kw):
    """Apply function to files/writes assigned to MPI process.

    Parameters
    ----------
    pattern : str
        Glob pattern for files
    function : function(file, start, count, **kw)
        A function on an HDF5 file, start index, and count.

    Other keyword arguments passed on to `function`

    """

    function_calls = zip(*get_assigned_writes(pattern))
    for filename, start, count in function_calls:
        if count:
            function(filename, start, count, **kw)


def get_assigned_writes(pattern):
    """
    Distribute files/writes matching a pattern between MPI processes.

    Parameters
    ----------
    pattern : str
        Glob pattern for files

    """

    # Distribute all writes in blocks
    filenames, writes = get_all_writes(pattern)
    block = int(np.ceil(sum(writes) / MPI_SIZE))
    proc_start = MPI_RANK * block
    # Find file start/end indices
    writes = np.array(writes)
    file_ends = np.cumsum(writes)
    file_starts = file_ends - writes
    # Find proc start indices and counts for each file
    starts = np.clip(proc_start, a_min=file_starts, a_max=file_ends)
    counts = np.clip(proc_start+block, a_min=file_starts, a_max=file_ends) - starts

    return filenames, starts-file_starts, counts


def get_all_writes(pattern):
    """
    Find all files/writes matching a pattern.

    Parameters
    ----------
    pattern : str
        Glob pattern for files

    """

    # Find matching files
    filenames = natural_sort(glob.glob(pattern))
    if not filenames:
        raise ValueError("No files match pattern %s" %pattern)
    # Get write numbers
    writes = list()
    for filename in filenames:
        logger.info("Found matching file %s" %filename)
        with h5py.File(filename, mode='r') as file:
            writes.append(file.attrs['writes'])

    return filenames, writes


def merge_analysis(base_path):
    """
    Merge distributed output files from a FileHandler.
    MPI parallelized up to number of distributed output files.

    Parameters
    ----------
    base_path : str or pathlib.Path
        Base path of FileHandler output

    """

    base_path = pathlib.Path(base_path)
    logger.info("Merging files from %s" %base_path)

    base_stem = base_path.stem
    folder_paths = base_path.glob("%s_f*" %base_stem)
    folder_paths = filter(lambda path: path.is_dir(), folder_paths)
    folder_paths = natural_sort(folder_paths)
    for folder_path in folder_paths[MPI_RANK::MPI_SIZE]:
        merge_folder(folder_path)


def merge_folder(folder_path):
    """
    Merge folder containing a distributed output file.

    Parameters
    ----------
    folder_path : str of pathlib.Path
        Path to folder containing a distributed output file

    """

    folder_path = pathlib.Path(folder_path)
    logger.info("Merging folder %s" %folder_path)

    folder_stem = folder_path.stem
    proc_paths = folder_path.glob("%s_p*.h5" %folder_stem)
    proc_paths = natural_sort(proc_paths)
    joint_path = folder_path.parent.joinpath("%s.h5" %folder_stem)

    # Create joint file, overwriting if it already exists
    with h5py.File(str(joint_path), mode='w') as joint_file:
        # Setup joint file based on first process file (arbitrary)
        merge_setup(joint_file, proc_paths[0])
        # Merge data from all process files
        for proc_path in proc_paths:
            merge_data(joint_file, proc_path)


def merge_setup(joint_file, proc_path):
    """
    Merge HDF5 setup from part of a distributed output file into a joint file.

    Parameters
    ----------
    joint_file : HDF5 file
        Joint file
    proc_path : str or pathlib.Path
        Path to part of a distributed output file

    """

    proc_path = pathlib.Path(proc_path)
    logger.info("Merging setup from %s" %proc_path)

    with h5py.File(str(proc_path), mode='r') as proc_file:
        # File metadata
        joint_file.attrs['file_number'] = proc_file.attrs['file_number']
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
            # Dimension scales
            for i, proc_dim in enumerate(proc_dset.dims):
                joint_dset.dims[i].label = proc_dim.label
                for scalename in proc_dim:
                    scale = joint_file['scales'][scalename]
                    joint_dset.dims.create_scale(scale, scalename)
                    joint_dset.dims[i].attach_scale(scale)


def merge_data(joint_file, proc_path):
    """
    Merge data from part of a distributed output file into a joint file.

    Parameters
    ----------
    joint_file : HDF5 file
        Joint file
    proc_path : str or pathlib.Path
        Path to part of a distributed output file

    """

    proc_path = pathlib.Path(proc_path)
    logger.info("Merging data from %s" %proc_path)

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

