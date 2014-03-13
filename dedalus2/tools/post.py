"""
Post-processing helpers.

"""

import sys
import glob
from collections import defaultdict

import h5py
from mpi4py import MPI

mpi_rank = MPI.COMM_WORLD.rank
mpi_size = MPI.COMM_WORLD.size


def get_all_write_paths(fileroot):
    """Build global list of write paths."""

    filenames = glob.glob(fileroot)
    all_write_paths = []
    for filename in sorted(filenames):
        with h5py.File(filename, mode='r') as file:
            for writename in sorted(file['tasks']):
                all_write_paths.append([filename, writename])

    return all_write_paths


def get_local_write_paths(fileroot):
    """Build list of write paths assigned to process."""

    all_write_paths = get_all_write_paths(fileroot)
    local_write_paths = all_write_paths[mpi_rank::mpi_size]

    return local_write_paths


def get_local_write_path_dict(fileroot):
    """Build dictionary of write paths assigned to process."""

    local_write_paths = get_local_write_paths(fileroot)
    local_write_path_dict = defaultdict(list)
    for (filename, writename) in local_write_paths:
        local_write_path_dict[filename].append(writename)

    return local_write_path_dict


def visit_writes(fileroot, function):
    """Apply function to writes assigned to process."""

    local_write_path_dict = get_local_write_path_dict(fileroot)
    for filename in local_write_path_dict:
        with h5py.File(filename, mode='r') as file:
            for writename in local_write_path_dict[filename]:
                write = file['tasks'][writename]
                function(write)


if __name__ == "__main__":

    def default_function(group):
        print('Rank %i: %s' %(mpi_rank, group))

    fileroot = sys.argv[1]
    visit_writes(fileroot, default_function)

