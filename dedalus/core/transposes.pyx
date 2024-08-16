cimport cython
cimport numpy as cnp
import numpy as np
import math
from math import prod

import logging
logger = logging.getLogger(__name__.split('.')[-1])

from mpi4py import MPI
from mpi4py.MPI cimport Comm as py_comm_t
from mpi4py.libmpi cimport MPI_Comm as mpi_comm_t
from libc.stddef cimport ptrdiff_t as p_t

from ..libraries.fftw import fftw_wrappers as fftw
from ..libraries.fftw cimport fftw_c_api as cfftw
from ..tools.config import config
IN_PLACE = config['parallelism-fftw'].getboolean('IN_PLACE')
PLANNING_RIGOR = config['parallelism-fftw'].get('PLANNING_RIGOR')


cdef class FFTWTranspose:
    """
    FFTW distributed array transpose, for redistributing a block-distributed
    multidimensional array across adjacent axes.

    Parameters
    ----------
    global_shape : ndarray of np.int32
        Global array shape
    dtype : data type
        Data type
    axis : int
        Column axis of transposition plan (row axis is the next axis)
    pycomm : mpi4py communicator
        Communicator

    """

    cdef readonly py_comm_t pycomm
    cdef readonly int datasize, axis
    cdef readonly int N0, N1, N2, N3
    cdef readonly int[::1] global_shape
    cdef readonly int[::1] chunk_shape
    cdef readonly int[::1] col_starts
    cdef readonly int[::1] row_starts
    cdef readonly int[::1] col_ends
    cdef readonly int[::1] row_ends
    cdef readonly int[::1] col_counts
    cdef readonly int[::1] row_counts
    cdef readonly int[::1] CL_reduced_shape
    cdef readonly int[::1] RL_reduced_shape
    cdef readonly double[::1] CL_buffer
    cdef readonly double[::1] RL_buffer
    cdef readonly cnp.ndarray CL_view
    cdef readonly cnp.ndarray RL_view
    cdef cfftw.fftw_plan CL_to_RL_plan
    cdef cfftw.fftw_plan RL_to_CL_plan

    def __init__(self, global_shape, chunk_shape, dtype, axis, pycomm):
        logger.debug("Building FFTW transpose plan for (dtype, gshape, axis) = (%s, %s, %s)" %(dtype, global_shape, axis))
        # Attributes
        self.global_shape = global_shape = np.array(global_shape, dtype=np.int32)
        self.chunk_shape = chunk_shape = np.array(chunk_shape, dtype=np.int32)
        self.datasize = {np.float64: 1, np.complex128: 2}[np.dtype(dtype).type]
        self.axis = axis
        self.pycomm = pycomm
        # Reduced global shape (4d array)
        self.N0 = N0 = prod(global_shape[:axis])
        self.N1 = N1 = global_shape[axis]
        self.N2 = N2 = global_shape[axis+1]
        self.N3 = N3 = prod(global_shape[axis+2:])
        # Chunks
        C1 = chunk_shape[axis]
        C2 = chunk_shape[axis+1]
        # Global number of chunks
        CG1 = -(-global_shape[axis] // C1)  # ceil
        CG2 = -(-global_shape[axis+1] // C2)  # ceil
        # Local number of chunks
        CL1 = -(-CG1 // pycomm.size)
        CL2 = -(-CG2 // pycomm.size)
        # Local number of elements
        B1 = C1 * CL1
        B2 = C2 * CL2
        # Starting indices
        ranks = np.arange(pycomm.size, dtype=np.int32)
        self.col_starts = col_starts = np.minimum(B2*ranks, global_shape[axis+1])
        self.row_starts = row_starts = np.minimum(B1*ranks, global_shape[axis])
        # Ending indices
        self.col_ends = col_ends = np.minimum(B2*(ranks+1), global_shape[axis+1])
        self.row_ends = row_ends = np.minimum(B1*(ranks+1), global_shape[axis])
        # Counts
        self.col_counts = col_counts = col_ends - col_starts
        self.row_counts = row_counts = row_ends - row_starts
        local_col_count = col_counts[<int> pycomm.rank]
        local_row_count = row_counts[<int> pycomm.rank]
        # Local reduced shapes
        self.CL_reduced_shape = np.array([N0, N1, local_col_count, N3], dtype=np.int32)
        self.RL_reduced_shape = np.array([N0, local_row_count, N2, N3], dtype=np.int32)
        # Build plans (sets up buffers)
        howmany = N0 * N3 * self.datasize
        flags = ['FFTW_'+PLANNING_RIGOR.upper()]
        self.build_plans(N1, N2, howmany, B1, B2, pycomm, IN_PLACE, flags=flags)
        # Transposed array views (contiguous)
        # For interfacing with FFTW transpose on first two dimensions
        CL_tran_shape = [N1, local_col_count, N0, N3]
        RL_tran_shape = [local_row_count, N2, N0, N3]
        CL_tran_view = np.ndarray(shape=CL_tran_shape, dtype=dtype, buffer=self.CL_buffer)
        RL_tran_view = np.ndarray(shape=RL_tran_shape, dtype=dtype, buffer=self.RL_buffer)
        # Anti-transposed array views (discontiguous)
        # For direct copying to and from original data
        self.CL_view = np.transpose(CL_tran_view, (2, 0, 1, 3))
        self.RL_view = np.transpose(RL_tran_view, (2, 0, 1, 3))

        # def _fftw_setup(self, scales):
        #     logger.debug("Building FFTW transpose plan for (scales, axis, in_place) = (%s, %s, %s)" %(scales, axis, IN_PLACE))

    def build_plans(self, p_t n0, p_t n1, p_t howmany, p_t block0, p_t block1,
                    py_comm_t pycomm, in_place, flags=['FFTW_MEASURE']):
        """
        FFTW distributed matrix transpose.  Interface can be used to direct a
        transpose of the first two dimensions of an N-dimensional array.

        Parameters
        ----------
        n0, n1 : int
            Global dimensions of matrix to be transposed, or alternately the first
            two dimensions of an N-dimensional array.
        howmany : int
            Number of doubles composing each matrix element, i.e the product
            of the remaining dimensions of an N-dimensional array and the dtype itemsize.
        block0, block1 : int
            Block sizes for distribution amongst processes
        pycomm : MPI_Comm object
            Communicator
        flags : list of str, optional
            List of wrapped FFTW flags (default: ['FFTW_MEASURE'])

        """

        # Shape array
        cdef p_t *shape = [n0, n1]
        # C MPI communicator
        cdef mpi_comm_t comm = pycomm.ob_mpi
        # Build flags
        cdef unsigned intflags = 0
        for f in flags:
            intflags = intflags | fftw.fftw_flags[f]

        # Build buffers
        cdef p_t alloc_doubles, local0, start0, local1, start1
        alloc_doubles = cfftw.fftw_mpi_local_size_many_transposed(
                2,
                shape,
                howmany,
                block0,
                block1,
                comm,
                &local0,
                &start0,
                &local1,
                &start1)
        cdef cnp.ndarray buffer0 = fftw.create_buffer(alloc_doubles)
        cdef cnp.ndarray buffer1 = fftw.create_buffer(alloc_doubles)
        if in_place:
            self.CL_buffer = buffer0
            self.RL_buffer = buffer0
        else:
            self.CL_buffer = buffer0
            self.RL_buffer = buffer1

        # Build plans
        self.CL_to_RL_plan = cfftw.fftw_mpi_plan_many_transpose(
                n1,
                n0,
                howmany,
                block1,
                block0,
                &self.CL_buffer[0],
                &self.RL_buffer[0],
                comm,
                intflags | cfftw.FFTW_MPI_TRANSPOSED_IN)
        self.RL_to_CL_plan = cfftw.fftw_mpi_plan_many_transpose(
                n0,
                n1,
                howmany,
                block0,
                block1,
                &self.RL_buffer[0],
                &self.CL_buffer[0],
                comm,
                intflags | cfftw.FFTW_MPI_TRANSPOSED_OUT)

        # Check that plan creation succeeded
        if (self.CL_to_RL_plan == NULL) or (self.RL_to_CL_plan == NULL):
            raise RuntimeError("FFTW could not create plans.")

    def __dealloc__(self):
        """Destroy FFTW plans on deallocation."""
        cfftw.fftw_destroy_plan(self.CL_to_RL_plan)
        cfftw.fftw_destroy_plan(self.RL_to_CL_plan)

    def make_views(self, RL, CL):
        CL_reduced = np.ndarray(shape=self.CL_reduced_shape, dtype=CL.dtype, buffer=CL)
        RL_reduced = np.ndarray(shape=self.RL_reduced_shape, dtype=RL.dtype, buffer=RL)
        return CL_reduced, RL_reduced

    def fftw_CL_to_RL(self):
        cfftw.fftw_execute(self.CL_to_RL_plan)

    def localize_rows(self, CL, RL):
        """Transpose from column-local to row-local data distribution."""
        # Create reduced views of data arrays
        CL_reduced, RL_reduced = self.make_views(RL, CL)
        # Transpose from input array to buffer
        self.copy_in_CL_fftw(CL_reduced)
        # Communicate between buffers
        self.fftw_CL_to_RL()
        # Transpose from buffer to output array
        self.copy_out_RL_fftw(RL_reduced)

    def fftw_RL_to_CL(self):
        cfftw.fftw_execute(self.RL_to_CL_plan)

    def copy_in_RL_fftw(self, RL_reduced):
        np.copyto(self.RL_view, RL_reduced)

    def copy_out_RL_fftw(self, RL_reduced):
        np.copyto(RL_reduced, self.RL_view)

    def copy_in_CL_fftw(self, CL_reduced):
        np.copyto(self.CL_view, CL_reduced)

    def copy_out_CL_fftw(self, CL_reduced):
        np.copyto(CL_reduced, self.CL_view)

    def localize_columns(self, RL, CL):
        """Transpose from row-local to column-local data distribution."""
        # Create reduced views of data arrays
        CL_reduced, RL_reduced = self.make_views(RL, CL)
        # Transpose from input array to buffer
        self.copy_in_RL_fftw(RL_reduced)
        # Communicate between buffers
        self.fftw_RL_to_CL()
        # Transpose from buffer to output array
        self.copy_out_CL_fftw(CL_reduced)

cdef class AlltoallvTranspose:
    """
    MPI Alltoallv-based distributed array transpose, for redistributing
    a block-distributed multidimensional array across adjacent axes.

    Parameters
    ----------
    global_shape : ndarray of np.int32
        Global array shape
    dtype : data type
        Data type
    axis : int
        Column axis of transposition plan (row axis is the next axis)
    pycomm : mpi4py communicator
        Communicator

    """

    cdef readonly py_comm_t pycomm
    cdef readonly int datasize, axis
    cdef readonly int N0, N1, N2, N3
    cdef readonly int[::1] global_shape
    cdef readonly int[::1] col_starts
    cdef readonly int[::1] row_starts
    cdef readonly int[::1] col_ends
    cdef readonly int[::1] row_ends
    cdef readonly int[::1] col_counts
    cdef readonly int[::1] row_counts
    cdef readonly int[::1] CL_reduced_shape
    cdef readonly int[::1] RL_reduced_shape
    cdef readonly int[::1] CL_displs
    cdef readonly int[::1] RL_displs
    cdef readonly int[::1] CL_counts
    cdef readonly int[::1] RL_counts
    cdef readonly double[::1] CL_buffer
    cdef readonly double[::1] RL_buffer
    cdef readonly int local_col_count
    cdef readonly int local_row_count

    def __init__(self, global_shape, dtype, axis, pycomm):
        logger.debug("Building MPI transpose plan for (dtype, gshape, axis) = (%s, %s, %s)" %(dtype, global_shape, axis))
        # Attributes
        self.global_shape = global_shape = global_shape.astype(np.int32)
        self.datasize = {np.float64: 1, np.complex128: 2}[np.dtype(dtype).type]
        self.axis = axis
        self.pycomm = pycomm
        # Reduced global shape (4d array)
        self.N0 = N0 = prod(global_shape[:axis])
        self.N1 = N1 = global_shape[axis]
        self.N2 = N2 = global_shape[axis+1]
        self.N3 = N3 = prod(global_shape[axis+2:]) * self.datasize
        # Blocks
        B1 = math.ceil(global_shape[axis] / pycomm.size)
        B2 = math.ceil(global_shape[axis+1] / pycomm.size)
        # Starting indices
        ranks = np.arange(pycomm.size, dtype=np.int32)
        self.col_starts = col_starts = np.minimum(B2*ranks, global_shape[axis+1])
        self.row_starts = row_starts = np.minimum(B1*ranks, global_shape[axis])
        # Ending indices
        self.col_ends = col_ends = np.minimum(B2*(ranks+1), global_shape[axis+1])
        self.row_ends = row_ends = np.minimum(B1*(ranks+1), global_shape[axis])
        # Counts
        self.col_counts = col_counts = col_ends - col_starts
        self.row_counts = row_counts = row_ends - row_starts
        self.local_col_count = local_col_count = col_counts[<int> pycomm.rank]
        self.local_row_count = local_row_count = row_counts[<int> pycomm.rank]
        # Local reduced shapes
        self.CL_reduced_shape = np.array([N0, N1, local_col_count, N3], dtype=np.int32)
        self.RL_reduced_shape = np.array([N0, local_row_count, N2, N3], dtype=np.int32)
        # Alltoallv displacements
        self.CL_displs = (N0 * local_col_count * N3) * row_starts
        self.RL_displs = (N0 * local_row_count * N3) * col_starts
        # Alltoallv counts
        self.CL_counts = (N0 * local_col_count * N3) * row_counts
        self.RL_counts = (N0 * local_row_count * N3) * col_counts
        # Buffers
        CL_size = N0 * N1 * local_col_count * N3
        RL_size = N0 * local_row_count * N2 * N3
        self.CL_buffer = np.zeros(CL_size, dtype=np.float64)
        self.RL_buffer = np.zeros(RL_size, dtype=np.float64)

    def localize_rows(self, CL, RL):
        """Transpose from column-local to row-local data distribution."""
        # Create reduced views of data arrays
        CL_reduced = np.ndarray(shape=self.CL_reduced_shape, dtype=np.float64, buffer=CL)
        RL_reduced = np.ndarray(shape=self.RL_reduced_shape, dtype=np.float64, buffer=RL)
        # Rearrange from input array to buffer
        if self.local_col_count > 0:
            self.split_rows(CL_reduced, self.CL_buffer)
        # Communicate between buffers
        self.pycomm.Alltoallv([self.CL_buffer, self.CL_counts, self.CL_displs, MPI.DOUBLE],
                              [self.RL_buffer, self.RL_counts, self.RL_displs, MPI.DOUBLE])
        # Rearrange from buffer to output array
        if self.local_row_count > 0:
            self.combine_columns(self.RL_buffer, RL_reduced)

    def localize_columns(self, RL, CL):
        """Transpose from row-local to column-local data distribution."""
        # Create reduced views of data arrays
        CL_reduced = np.ndarray(shape=self.CL_reduced_shape, dtype=np.float64, buffer=CL)
        RL_reduced = np.ndarray(shape=self.RL_reduced_shape, dtype=np.float64, buffer=RL)
        # Rearrange from input array to buffer
        if self.local_row_count > 0:
            self.split_columns(RL_reduced, self.RL_buffer)
        # Communicate between buffers
        self.pycomm.Alltoallv([self.RL_buffer, self.RL_counts, self.RL_displs, MPI.DOUBLE],
                              [self.CL_buffer, self.CL_counts, self.CL_displs, MPI.DOUBLE])
        # Rearrange from buffer to output array
        if self.local_col_count > 0:
            self.combine_rows(self.CL_buffer, CL_reduced)

    @cython.boundscheck(False)
    cdef void split_rows(self, double[:,:,:,::1] A, double[::1] B):
        """Reorder column-local dataset into sending buffer of rows."""
        # Create local copies of fixed loop bounds
        cdef unsigned int N0 = self.N0
        cdef unsigned int col_count = self.local_col_count
        cdef unsigned int N3 = self.N3
        # Allocate loop variables
        cdef unsigned int proc, row_start, row_end
        cdef unsigned int n0, n1, n2, n3
        cdef unsigned int i = 0
        # Copy contiguously into buffer
        for proc in range(self.pycomm.size):
            row_start = self.row_starts[proc]
            row_end = self.row_ends[proc]
            for n0 in range(N0):
                for n1 in range(row_start, row_end):
                    for n2 in range(col_count):
                        for n3 in range(N3):
                            B[i] = A[n0, n1, n2, n3]
                            i = i + 1

    @cython.boundscheck(False)
    cdef void combine_rows(self, double[::1] B, double[:,:,:,::1] A):
        """Reorder receiving buffer of rows into column-local dataset."""
        # Create local copies of fixed loop bounds
        cdef unsigned int N0 = self.N0
        cdef unsigned int col_count = self.local_col_count
        cdef unsigned int N3 = self.N3
        # Allocate loop variables
        cdef unsigned int proc, row_start, row_end
        cdef unsigned int n0, n1, n2, n3
        cdef unsigned int i = 0
        # Copy contiguously from buffer
        for proc in range(self.pycomm.size):
            row_start = self.row_starts[proc]
            row_end = self.row_ends[proc]
            for n0 in range(N0):
                for n1 in range(row_start, row_end):
                    for n2 in range(col_count):
                        for n3 in range(N3):
                            A[n0, n1, n2, n3] = B[i]
                            i = i + 1

    @cython.boundscheck(False)
    cdef void split_columns(self, double[:,:,:,::1] A, double[::1] B):
        """Reorder row-local dataset into sending buffer of columns."""
        # Create local copies of fixed loop bounds
        cdef unsigned int N0 = self.N0
        cdef unsigned int row_count = self.local_row_count
        cdef unsigned int N3 = self.N3
        # Allocate loop variables
        cdef unsigned int proc, col_start, col_end
        cdef unsigned int n0, n1, n2, n3
        cdef unsigned int i = 0
        # Copy contiguously into buffer
        for proc in range(self.pycomm.size):
            col_start = self.col_starts[proc]
            col_end = self.col_ends[proc]
            for n0 in range(N0):
                for n1 in range(row_count):
                    for n2 in range(col_start, col_end):
                        for n3 in range(N3):
                            B[i] = A[n0, n1, n2, n3]
                            i = i + 1

    @cython.boundscheck(False)
    cdef void combine_columns(self, double[::1] B, double[:,:,:,::1] A):
        """Reorder receiving buffer of columns into row-local dataset."""
        # Create local copies of fixed loop bounds
        cdef unsigned int N0 = self.N0
        cdef unsigned int row_count = self.local_row_count
        cdef unsigned int N3 = self.N3
        # Allocate loop variables
        cdef unsigned int proc, col_start, col_end
        cdef unsigned int n0, n1, n2, n3
        cdef unsigned int i = 0
        # Copy contiguously from buffer
        for proc in range(self.pycomm.size):
            col_start = self.col_starts[proc]
            col_end = self.col_ends[proc]
            for n0 in range(N0):
                for n1 in range(row_count):
                    for n2 in range(col_start, col_end):
                        for n3 in range(N3):
                            A[n0, n1, n2, n3] = B[i]
                            i = i + 1


cdef class ColDistributor(AlltoallvTranspose):
    """
    MPI Allgatherv-Scatterv-based distributed array duplication and
    condensation.

    Parameters
    ----------
    global_shape : ndarray of np.int32
        Global array shape
    dtype : data type
        Data type
    axis : int
        Column axis of transposition plan (row axis is the next axis)
    pycomm : mpi4py communicator
        Communicator

    """

    cdef readonly int local_row_start
    cdef readonly int local_row_end
    cdef readonly int local_RL_count

    def __init__(self, global_shape, dtype, axis, pycomm):
        logger.debug("Building MPI transpose plan for (dtype, gshape, axis) = (%s, %s, %s)" %(dtype, global_shape, axis))
        # Attributes
        self.global_shape = global_shape = global_shape.astype(np.int32)
        self.datasize = {np.float64: 1, np.complex128: 2}[np.dtype(dtype).type]
        self.axis = axis
        self.pycomm = pycomm
        # Reduced global shape (4d array)
        self.N0 = N0 = prod(global_shape[:axis])
        self.N1 = N1 = global_shape[axis]
        self.N2 = N2 = global_shape[axis+1]
        self.N3 = N3 = prod(global_shape[axis+2:]) * self.datasize
        # Blocks
        B1 = math.ceil(global_shape[axis] / pycomm.size)
        B2 = N2
        # Starting indices
        ranks = np.arange(pycomm.size, dtype=np.int32)
        self.col_starts = col_starts = 0*ranks
        self.row_starts = row_starts = np.minimum(B1*ranks, global_shape[axis])
        self.local_row_start = row_starts[<int> pycomm.rank]
        # Ending indices
        self.col_ends = col_ends = 0*ranks + B2
        self.row_ends = row_ends = np.minimum(B1*(ranks+1), global_shape[axis])
        self.local_row_end = row_ends[<int> pycomm.rank]
        # Counts
        self.col_counts = col_counts = col_ends - col_starts
        self.row_counts = row_counts = row_ends - row_starts
        self.local_row_count = local_row_count = row_counts[<int> pycomm.rank]
        self.local_col_count = local_col_count = col_counts[<int> pycomm.rank]
        # Local reduced shapes
        self.CL_reduced_shape = np.array([N0, N1, local_col_count, N3], dtype=np.int32)
        self.RL_reduced_shape = np.array([N0, local_row_count, N2, N3], dtype=np.int32)
        # Alltoallv displacements
        self.CL_displs = (N0 * local_col_count * N3) * row_starts
        self.RL_displs = (N0 * local_row_count * N3) * col_starts
        # Alltoallv counts
        self.CL_counts = (N0 * local_col_count * N3) * row_counts
        self.RL_counts = (N0 * local_row_count * N3) * col_counts
        self.local_RL_count = self.RL_counts[<int> pycomm.rank]
        # Buffers
        CL_size = N0 * N1 * local_col_count * N3
        RL_size = N0 * local_row_count * N2 * N3
        self.CL_buffer = np.zeros(CL_size, dtype=np.float64)
        self.RL_buffer = np.zeros(RL_size, dtype=np.float64)

    def localize_rows(self, CL, RL):
        """Transpose from column-local to row-local data distribution."""
        # Create reduced views of data arrays
        CL_reduced = np.ndarray(shape=self.CL_reduced_shape, dtype=np.float64, buffer=CL)
        RL_reduced = np.ndarray(shape=self.RL_reduced_shape, dtype=np.float64, buffer=RL)
        # Restrict to local columns
        start = self.local_row_start
        end = self.local_row_end
        np.copyto(RL_reduced, CL_reduced[:,start:end,:,:])

    def localize_columns(self, RL, CL):
        """Transpose from row-local to column-local data distribution."""
        # Create reduced views of data arrays
        CL_reduced = np.ndarray(shape=self.CL_reduced_shape, dtype=np.float64, buffer=CL)
        RL_reduced = np.ndarray(shape=self.RL_reduced_shape, dtype=np.float64, buffer=RL)
        # Copy from input array to buffer
        cdef double[::1] temp = RL_reduced.ravel()
        self.RL_buffer[:] = temp
        # Communicate between buffers
        self.pycomm.Allgatherv([self.RL_buffer, self.local_RL_count, MPI.DOUBLE],
                               [self.CL_buffer, self.CL_counts, self.CL_displs, MPI.DOUBLE])
        # Rearrange from buffer to output array
        self.combine_rows(self.CL_buffer, CL_reduced)


cdef class RowDistributor(AlltoallvTranspose):
    """
    MPI Allgatherv-Scatterv-based distributed array duplication and
    condensation.

    Parameters
    ----------
    global_shape : ndarray of np.int32
        Global array shape
    dtype : data type
        Data type
    axis : int
        Column axis of transposition plan (row axis is the next axis)
    pycomm : mpi4py communicator
        Communicator

    """

    cdef readonly int local_col_start
    cdef readonly int local_col_end
    cdef readonly int local_CL_count

    def __init__(self, global_shape, dtype, axis, pycomm):
        logger.debug("Building MPI transpose plan for (dtype, gshape, axis) = (%s, %s, %s)" %(dtype, global_shape, axis))
        # Attributes
        self.global_shape = global_shape = global_shape.astype(np.int32)
        self.datasize = {np.float64: 1, np.complex128: 2}[np.dtype(dtype).type]
        self.axis = axis
        self.pycomm = pycomm
        # Reduced global shape (4d array)
        self.N0 = N0 = prod(global_shape[:axis])
        self.N1 = N1 = global_shape[axis]
        self.N2 = N2 = global_shape[axis+1]
        self.N3 = N3 = prod(global_shape[axis+2:]) * self.datasize
        # Blocks
        B1 = N1
        B2 = math.ceil(global_shape[axis+1] / pycomm.size)
        # Starting indices
        ranks = np.arange(pycomm.size, dtype=np.int32)
        self.col_starts = col_starts = np.minimum(B2*ranks, global_shape[axis+1])
        self.row_starts = row_starts = 0*ranks
        self.local_col_start = col_starts[<int> pycomm.rank]
        # Ending indices
        self.col_ends = col_ends = np.minimum(B2*(ranks+1), global_shape[axis+1])
        self.row_ends = row_ends = 0*ranks + B1
        self.local_col_end = col_ends[<int> pycomm.rank]
        # Counts
        self.col_counts = col_counts = col_ends - col_starts
        self.row_counts = row_counts = row_ends - row_starts
        self.local_col_count = local_col_count = col_counts[<int> pycomm.rank]
        self.local_row_count = local_row_count = row_counts[<int> pycomm.rank]
        # Local reduced shapes
        self.CL_reduced_shape = np.array([N0, N1, local_col_count, N3], dtype=np.int32)
        self.RL_reduced_shape = np.array([N0, local_row_count, N2, N3], dtype=np.int32)
        # Alltoallv displacements
        self.CL_displs = (N0 * local_col_count * N3) * row_starts
        self.RL_displs = (N0 * local_row_count * N3) * col_starts
        # Alltoallv counts
        self.CL_counts = (N0 * local_col_count * N3) * row_counts
        self.RL_counts = (N0 * local_row_count * N3) * col_counts
        self.local_CL_count = self.CL_counts[<int> pycomm.rank]
        # Buffers
        CL_size = N0 * N1 * local_col_count * N3
        RL_size = N0 * local_row_count * N2 * N3
        self.CL_buffer = np.zeros(CL_size, dtype=np.float64)
        self.RL_buffer = np.zeros(RL_size, dtype=np.float64)

    def localize_rows(self, CL, RL):
        """Transpose from column-local to row-local data distribution."""
        # Create reduced views of data arrays
        CL_reduced = np.ndarray(shape=self.CL_reduced_shape, dtype=np.float64, buffer=CL)
        RL_reduced = np.ndarray(shape=self.RL_reduced_shape, dtype=np.float64, buffer=RL)
        # Copy from input array to buffer
        cdef double[::1] temp = CL_reduced.ravel()
        self.CL_buffer[:] = temp
        # Communicate between buffers
        self.pycomm.Allgatherv([self.CL_buffer, self.local_CL_count, MPI.DOUBLE],
                               [self.RL_buffer, self.RL_counts, self.RL_displs, MPI.DOUBLE])
        # Rearrange from buffer to output array
        self.combine_columns(self.RL_buffer, RL_reduced)

    def localize_columns(self, RL, CL):
        """Transpose from row-local to column-local data distribution."""
        # Create reduced views of data arrays
        CL_reduced = np.ndarray(shape=self.CL_reduced_shape, dtype=np.float64, buffer=CL)
        RL_reduced = np.ndarray(shape=self.RL_reduced_shape, dtype=np.float64, buffer=RL)
        # Restrict to local columns
        start = self.local_col_start
        end = self.local_col_end
        np.copyto(CL_reduced, RL_reduced[:,:,start:end,:])
