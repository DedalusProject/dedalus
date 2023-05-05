"""Post-processing helpers."""

import os
import pathlib
import hashlib
import shutil
import h5py
import xarray
from xarray.backends import BackendEntrypoint
import numpy as np
from mpi4py import MPI
from ..tools.general import natural_sort

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
        merge_setup(joint_file, proc_paths)
        # Merge data from all process files
        for proc_path in proc_paths:
            merge_data(joint_file, proc_path)
    # Cleanup after completed merge, if directed
    if cleanup:
        for proc_path in proc_paths:
            proc_path.unlink()
        set_path.rmdir()


def merge_setup(joint_file, proc_paths, virtual=False):
    """
    Merge HDF5 setup from part of a distributed analysis set into a joint file.

    Parameters
    ----------
    joint_file : HDF5 file
        Joint file
    proc_paths : list of [str or pathlib.Path]
        List of files in a distributed analysis set
    virtual: bool, optional
        If True, merging a virtual file into a single file rather than distributed set

    """
    proc_path0 = pathlib.Path(proc_paths[0])
    logger.info("Merging setup for {}".format(joint_file))
    with h5py.File(str(proc_path0), mode='r') as proc_file:
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
        if virtual:
            proc_file.copy('scales', joint_file)
        else:
            needed_hashes = []
            joint_scales = joint_file.create_group('scales')
            for scalename in proc_file['scales']:
                if 'hash_' in scalename:
                    needed_hashes.append(scalename)
                else:
                    joint_scales.create_dataset(name=scalename, data=proc_file['scales'][scalename])
        # Tasks
        joint_tasks = joint_file.create_group('tasks')
        proc_tasks = proc_file['tasks']
        for taskname in proc_tasks:
            # Setup dataset with automatic chunking
            proc_dset = proc_tasks[taskname]
            if virtual:
                joint_dset = joint_tasks.create_dataset(name=proc_dset.name, data=proc_dset)
            else:
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

            for i, proc_dim in enumerate(proc_dset.dims):
                joint_dset.dims[i].label = proc_dim.label
                if joint_dset.dims[i].label == 't':
                    for scalename in ['sim_time', 'wall_time', 'timestep', 'iteration', 'write_number']:
                        scale = joint_file['scales'][scalename]
                        joint_dset.dims.create_scale(scale, scalename)
                        joint_dset.dims[i].attach_scale(scale)
                else:
                    if proc_dim.label == 'constant' or proc_dim.label == '':
                        scalename = 'constant'
                    else:
                        hashval = hashlib.sha1(np.array(proc_dset.dims[i][0])).hexdigest()
                        scalename = 'hash_' + hashval
                        if not virtual and scalename in needed_hashes:
                            if joint_shape[i] == 1:
                                scale_data = np.zeros(1)
                            else:
                                scale_data = np.zeros(joint_shape[i])
                                filled = np.zeros(joint_shape[i], dtype=bool)
                                for proc_path in proc_paths:
                                    with h5py.File(proc_path, 'r') as pf:
                                        start = pf['tasks'][taskname].attrs['start'][i-1]
                                        stop = start+pf['tasks'][taskname].attrs['count'][i-1]
                                        scale_data[start:stop] = pf['scales'][scalename]
                                        filled[start:stop] = 1
                                    if np.sum(filled) == scale_data.size: break #stop filling
                            joint_scales.create_dataset(name=scalename, data=scale_data)
                            needed_hashes.remove(scalename)

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


merge_virtual = lambda joint_file, virtual_path: merge_setup(joint_file, [virtual_path,], virtual=True)


def dedalus_h5_to_xarray(dset):
    """Convert Dedalus HDF5 dataset to an Xarray DataArray."""
    dims = [dim.label for dim in dset.dims]
    # Copy all coordinates
    coords = {}
    for dim in dset.dims:
        for name, coord in dim.items():
            coords[name] = (dim.label, coord[:])
    # Add sim_time as dimensional coordinate for time
    coords[dims[0]] = (dims[0], dset.dims[0]['sim_time'][:])
    # Build dataarray
    name = dset.name.split('/')[-1]
    return xarray.DataArray(dset[:], coords=coords, dims=dims, name=name)


def load_tasks_to_xarray(filename, tasks=None, squeeze_constant=True):
    """Load task from Dedalus HDF5 output to an Xarray DataArray."""
    with h5py.File(filename, 'r') as file:
        if tasks is None:
            tasks = list(file['tasks'].keys())
        dsets = [file['tasks'][task] for task in tasks]
        arrays = [dedalus_h5_to_xarray(dset) for dset in dsets]
    arrays = {array.name: array for array in arrays}
    # Drop constant dimensions
    if squeeze_constant:
        for task in arrays:
            array = arrays[task]
            constant_axes = [i for i, name in enumerate(array.dims) if name == 'constant']
            array = array.squeeze(axis=constant_axes, drop=True)
            arrays[task] = array
    return arrays


class DedalusXarrayBackend(BackendEntrypoint):
    """Xarray backend targeting Dedalus HDF5 outputs."""

    def open_dataset(self, filename_or_obj, *, drop_variables=None):
        with h5py.File(filename_or_obj, 'r') as file:
            data_arrays = {}
            for dset in file['tasks'].values():
                name = dset.name.split('/')[-1]
                if drop_variables is None or name not in drop_variables:
                    data_arrays[name] = dedalus_h5_to_xarray(dset)
        return xarray.Dataset(data_arrays)

