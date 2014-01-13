"""
Custom FFTW interface.

"""

import numpy as np
cimport numpy as cnp
from mpi4py.MPI cimport Comm as py_comm_t
from mpi4py.mpi_c cimport MPI_Comm as mpi_comm_t
from libc.stddef cimport ptrdiff_t as p_t
from cython.view cimport array as cy_array

cimport fftw_c_api as cfftw


fftw_flags = {'FFTW_ESTIMATE': cfftw.FFTW_ESTIMATE,
              'FFTW_EXHAUSTIVE': cfftw.FFTW_EXHAUSTIVE,
              'FFTW_MEASURE': cfftw.FFTW_MEASURE,
              'FFTW_PATIENT': cfftw.FFTW_PATIENT}


def fftw_mpi_init():
    """Run FFTW MPI initialization."""

    cfftw.fftw_mpi_init()


def create_buffer(size_t alloc_doubles):
    """Allocate memory using FFTW for SIMD alignment."""

    # Allocate using FFTW
    cdef double *c_data
    c_data = cfftw.fftw_alloc_real(alloc_doubles)

    # View as cython array with FFTW deallocation
    cdef cy_array cy_data = <double[:alloc_doubles]> c_data
    cy_data.callback_free_data = cfftw.fftw_free

    # View as numpy array
    np_data = np.asarray(cy_data)
    np_data.fill(0.)

    return np_data


cdef class Transpose:
    """
    FFTW distributed matrix transpose.  Interface can be used to direct a
    transpose of the first two dimensions of an N-dimensional array.

    Parameters
    ----------
    n0, n1 : int
        Global dimensions of matrix to be transposed, or alternately the first
        two dimensions of an N-dimensional array.
    howmany : int
        Number of array elements composing each matrix element, i.e the product
        of the remaining dimensions of an N-dimensional array.
    block0, block1 : int
        Block sizes for distribution amongst processes
    dtype : data type
        Matrix/array data type
    pycomm : MPI_Comm object
        Communicator
    flags : list of str, optional
        List of wrapped FFTW flags (default: ['FFTW_MEASURE'])

    Attributes
    ----------
    alloc_doubles : int
        Required buffer size (in doubles)
    local0, local1 : int
        Local distributed sizes
    start0, start1 : int
        Local distributed start indeces

    """

    cdef readonly p_t alloc_doubles
    cdef readonly p_t local0
    cdef readonly p_t start0
    cdef readonly p_t local1
    cdef readonly p_t start1
    cdef cfftw.fftw_plan gather_plan
    cdef cfftw.fftw_plan scatter_plan

    def __init__(self, p_t n0, p_t n1, p_t howmany, p_t block0, p_t block1,
                 dtype, py_comm_t pycomm, flags=['FFTW_MEASURE']):

        # Shape array
        cdef p_t *shape = [n0, n1]

        # Get item size from provided data type
        cdef p_t itemsize
        if dtype == np.float64:
            itemsize = 1
        elif dtype == np.complex128:
            itemsize = 2
        else:
            raise ValueError("Only np.float64 and np.complex128 arrays supported.")

        # Actual MPI communicator
        cdef mpi_comm_t comm = pycomm.ob_mpi

        # Build flags
        cdef unsigned intflags = 0
        for f in flags:
            intflags = intflags | fftw_flags[f]

        # Required buffer size (in doubles)
        self.alloc_doubles = cfftw.fftw_mpi_local_size_many_transposed(2,
                                                                       shape,
                                                                       howmany*itemsize,
                                                                       block0,
                                                                       block1,
                                                                       comm,
                                                                       &self.local0,
                                                                       &self.start0,
                                                                       &self.local1,
                                                                       &self.start1)

        # Create plans using a temporary memory allocation
        cdef double *data
        data = cfftw.fftw_alloc_real(self.alloc_doubles)
        self.scatter_plan = cfftw.fftw_mpi_plan_many_transpose(n1,
                                                               n0,
                                                               howmany*itemsize,
                                                               block1,
                                                               block0,
                                                               data,
                                                               data,
                                                               comm,
                                                               intflags | cfftw.FFTW_MPI_TRANSPOSED_IN)
        self.gather_plan = cfftw.fftw_mpi_plan_many_transpose(n0,
                                                              n1,
                                                              howmany*itemsize,
                                                              block0,
                                                              block1,
                                                              data,
                                                              data,
                                                              comm,
                                                              intflags | cfftw.FFTW_MPI_TRANSPOSED_OUT)
        cfftw.fftw_free(data)

        # Check that plan creation succeeded
        if (self.gather_plan == NULL) or (self.scatter_plan == NULL):
            raise RuntimeError("FFTW could not create plans.")

    def __dealloc__(self):
        """Destroy plans on deallocation."""

        cfftw.fftw_destroy_plan(self.gather_plan)
        cfftw.fftw_destroy_plan(self.scatter_plan)

    def gather(self, cnp.ndarray data):
        """Gather along first axis (0), scattering from second axis (1)."""

        # Execute plan using new-array interface
        cfftw.fftw_mpi_execute_r2r(self.gather_plan,
                                   <double *> data.data,
                                   <double *> data.data)

    def scatter(self, cnp.ndarray data):
        """Scatter from first axis (0), gathering along second axis (1)."""

        # Execute plan using new-array interface
        cfftw.fftw_mpi_execute_r2r(self.scatter_plan,
                                   <double *> data.data,
                                   <double *> data.data)

