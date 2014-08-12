"""
Tools for running in parallel.

"""

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

