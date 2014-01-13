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

    def __init__(self, enter=True, exit=True):
        self.enter = enter
        self.exit = exit

    def __enter__(self):
        if self.enter:
            MPI.COMM_WORLD.barrier()

    def __exit__(self, type, value, traceback):
        if self.exit:
            MPI.COMM_WORLD.barrier()

