"""
Tools for running in parallel.

"""

import pathlib
from mpi4py import MPI


class Sync:
    """
    Context manager for synchronizing MPI processes.

    Parameters
    ----------
    enter : boolean, optional
        Apply MPI barrier on entering context. Default: True
    exit : boolean, optional
        Apply MPI barrier on exiting context. Default: True

    """

    def __init__(self, comm=MPI.COMM_WORLD, enter=True, exit=True):
        self.comm = comm
        self.enter = enter
        self.exit = exit

    def __enter__(self):
        if self.enter:
            self.comm.Barrier()
        return self

    def __exit__(self, type, value, traceback):
        if self.exit:
            self.comm.Barrier()


class RotateProcesses:
    """
    Context manager for rotating execution between MPI processes.

    Parameters
    ----------
    comm : MPI communicator, optional

    """

    def __init__(self, comm=MPI.COMM_WORLD):
        self.comm = comm
        self.size = comm.size
        self.rank = comm.rank

    def __enter__(self):
        for i in range(self.rank):
            self.comm.Barrier()

    def __exit__(self, type, value, traceback):
        for i in range(self.size-self.rank):
            self.comm.Barrier()


class ProfileWrapper:
    """Pickleable wrapper for cProfile.Profile for use with pstats.Stats"""

    def __init__(self, stats):
        self.stats = stats

    def create_stats(self):
        pass


def parallel_mkdir(path, comm=MPI.COMM_WORLD):
    """Create a directory from root process."""
    path = pathlib.Path(path)
    with Sync(comm=comm, enter=False, exit=True) as sync:
        if sync.comm.rank == 0:
            if not path.exists():
                path.mkdir()

