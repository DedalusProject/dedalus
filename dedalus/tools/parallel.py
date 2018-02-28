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


def sync_glob(path, glob, comm=MPI.COMM_WORLD):
    """
    Syncronized pathlib globbing for consistent results across processes.

    Parameters
    ----------
    path : str or pathlib.Path
        Base path for globbing.
    pattern : str
        Glob pattern.
    comm : mpi4py communicator, optional
        MPI communicator. Default: MPI.COMM_WORLD

    """
    # Glob from rank 0 and broadcast
    # No barrier necessary on exit since broadcast is blocking
    with Sync(comm, enter=True, exit=False):
        if comm.rank == 0:
            result = tuple(pathlib.Path(path).glob(glob))
        else:
            result = None
        result = comm.bcast(result, root=0)
    return result

