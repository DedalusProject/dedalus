
try:
    from mpi4py import MPI
    from . import fftw_wrappers
    fftw_wrappers.fftw_mpi_init()
except ImportError:
    raise ImportError("Make sure to build using `python3 setup.py build_ext --inplace`")
