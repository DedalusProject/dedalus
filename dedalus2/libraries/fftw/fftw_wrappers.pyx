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
from ...tools.array import axslice

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


def create_array(shape, dtype):
    """Create array using FFTW-aligned buffer."""

    if dtype == np.float64:
        alloc_doubles = np.prod(shape)
    elif dtype == np.complex128:
        alloc_doubles = 2 * np.prod(shape)
    else:
        raise ValueError("Unsupported dtype: %s" %str(dtype))

    buffer = create_buffer(alloc_doubles)
    return np.ndarray(shape=shape, dtype=dtype, buffer=buffer)


def create_copy(array):
    """Create copy using FFTW-aligned array."""

    copy = create_array(array.shape, array.dtype)
    np.copyto(copy, array)
    return copy


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
    cdef public cnp.ndarray buffer0
    cdef public cnp.ndarray buffer1

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

        # Create in-place plans using a single buffer
        cdef cnp.ndarray in_place_buffer = create_buffer(self.alloc_doubles)
        self.buffer0 = in_place_buffer
        self.buffer1 = in_place_buffer
        self.scatter_plan = cfftw.fftw_mpi_plan_many_transpose(n1,
                                                               n0,
                                                               howmany*itemsize,
                                                               block1,
                                                               block0,
                                                               <double *> cnp.PyArray_DATA(self.buffer1),
                                                               <double *> cnp.PyArray_DATA(self.buffer0),
                                                               comm,
                                                               intflags | cfftw.FFTW_MPI_TRANSPOSED_IN)
        self.gather_plan = cfftw.fftw_mpi_plan_many_transpose(n0,
                                                              n1,
                                                              howmany*itemsize,
                                                              block0,
                                                              block1,
                                                              <double *> cnp.PyArray_DATA(self.buffer0),
                                                              <double *> cnp.PyArray_DATA(self.buffer1),
                                                              comm,
                                                              intflags | cfftw.FFTW_MPI_TRANSPOSED_OUT)

        # Check that plan creation succeeded
        if (self.gather_plan == NULL) or (self.scatter_plan == NULL):
            raise RuntimeError("FFTW could not create plans.")

    def __dealloc__(self):
        """Destroy plans on deallocation."""

        cfftw.fftw_destroy_plan(self.gather_plan)
        cfftw.fftw_destroy_plan(self.scatter_plan)

    def gather(self):
        """Gather along first axis (0), scattering from second axis (1)."""

        # Execute plan
        cfftw.fftw_execute(self.gather_plan)

    def scatter(self):
        """Scatter from first axis (0), gathering along second axis (1)."""

        # Execute plan
        cfftw.fftw_execute(self.scatter_plan)


cdef class FourierTransform:
    """
    FFTW-based discrete Fourier transform along one axis of an N-dimensional array.

    Parameters
    ----------
    grid_dtype : dtype
        Grid space data type
    grid_shape : tuple of ints, array of ints
        Array shape in grid space
    axis : int
        Axis over which to compute the FFT
    flags : list of str, optional
        List of wrapped FFTW flags

    """

    cdef readonly cshape
    cdef readonly forward
    cdef readonly backward
    cdef cfftw.fftw_plan forward_plan
    cdef cfftw.fftw_plan backward_plan

    def __init__(self, grid_dtype, grid_shape, int axis, flags=[]):

        # Dispatch transform methods
        if grid_dtype == np.float64:
            self.forward = self._forward_real
            self.backward = self._backward_real
        elif grid_dtype == np.complex128:
            self.forward = self._forward_complex
            self.backward = self._backward_complex
        else:
            raise ValueError("Unsupported dtype: %s" %str(grid_dtype))

        # Compute coeff shape based on dtype
        if grid_dtype == np.float64:
            coeff_shape = list(grid_shape)
            coeff_shape[axis] = grid_shape[axis]//2 + 1
        elif grid_dtype == np.complex128:
            coeff_shape = list(grid_shape)
        gshape = np.array(grid_shape, dtype=int)
        cshape = np.array(coeff_shape, dtype=int)
        self.cshape = cshape

        # Build FFTW guru transform structure
        cdef int trans_rank = 1
        cdef cfftw.fftw_iodim trans
        # Transform along `axis`
        trans.n = gshape[axis]
        trans.in_stride  = np.prod(gshape[axis+1:])
        trans.out_stride = np.prod(cshape[axis+1:])
        cdef cfftw.fftw_iodim *trans_struct = [trans]

        # Build FFTW guru vector structures
        cdef int vec_rank = 2
        cdef cfftw.fftw_iodim vec0, vec1f, vec1b
        # Loop over higher axes
        vec0.n = np.prod(gshape[axis+1:])
        vec0.in_stride = 1
        vec0.out_stride = 1
        # Loop over lower axes
        vec1f.n = vec1b.n = np.prod(gshape[:axis])
        vec1f.in_stride  = vec1b.out_stride = np.prod(gshape[axis:])
        vec1f.out_stride = vec1b.in_stride  = np.prod(cshape[axis:])
        cdef cfftw.fftw_iodim *vec_struct_f = [vec0, vec1f]
        cdef cfftw.fftw_iodim *vec_struct_b = [vec0, vec1b]

        # Build flags
        cdef unsigned intflags = 0
        for f in flags:
            intflags = intflags | fftw_flags[f]

        # Create out-of-place plans using temporary memory allocations
        cdef double *rg_temp
        cdef complex *cg_temp
        cdef complex *cc_temp
        if grid_dtype == np.float64:
            rg_temp = cfftw.fftw_alloc_real(np.prod(gshape))
            cc_temp = cfftw.fftw_alloc_complex(np.prod(cshape))
            self.forward_plan = cfftw.fftw_plan_guru_dft_r2c(trans_rank,
                                                             trans_struct,
                                                             vec_rank,
                                                             vec_struct_f,
                                                             rg_temp,
                                                             cc_temp,
                                                             intflags | cfftw.FFTW_DESTROY_INPUT)
            self.backward_plan = cfftw.fftw_plan_guru_dft_c2r(trans_rank,
                                                              trans_struct,
                                                              vec_rank,
                                                              vec_struct_b,
                                                              cc_temp,
                                                              rg_temp,
                                                              intflags | cfftw.FFTW_DESTROY_INPUT)
            cfftw.fftw_free(rg_temp)
            cfftw.fftw_free(cc_temp)
        elif grid_dtype == np.complex128:
            cg_temp = cfftw.fftw_alloc_complex(np.prod(gshape))
            cc_temp = cfftw.fftw_alloc_complex(np.prod(cshape))
            self.forward_plan = cfftw.fftw_plan_guru_dft(trans_rank,
                                                         trans_struct,
                                                         vec_rank,
                                                         vec_struct_f,
                                                         cg_temp,
                                                         cc_temp,
                                                         cfftw.FFTW_FORWARD,
                                                         intflags | cfftw.FFTW_DESTROY_INPUT)
            self.backward_plan = cfftw.fftw_plan_guru_dft(trans_rank,
                                                          trans_struct,
                                                          vec_rank,
                                                          vec_struct_b,
                                                          cc_temp,
                                                          cg_temp,
                                                          cfftw.FFTW_BACKWARD,
                                                          intflags | cfftw.FFTW_DESTROY_INPUT)
            cfftw.fftw_free(cg_temp)
            cfftw.fftw_free(cc_temp)

        # Check that plan creation succeeded
        if (self.forward_plan == NULL) or (self.backward_plan == NULL):
            raise RuntimeError("FFTW could not create plans.")

    def __dealloc__(self):
        """Destroy plans on deallocation."""

        cfftw.fftw_destroy_plan(self.forward_plan)
        cfftw.fftw_destroy_plan(self.backward_plan)

    def _forward_real(self, cnp.ndarray gdata, cnp.ndarray cdata):
        """Real grid-to-coefficient transform."""

        # Execute plan using new-array interface
        cfftw.fftw_execute_dft_r2c(self.forward_plan,
                                   <double *> cnp.PyArray_DATA(gdata),
                                   <complex *> cnp.PyArray_DATA(cdata))

    def _forward_complex(self, cnp.ndarray gdata, cnp.ndarray cdata):
        """Complex grid-to-coefficient transform."""

        # Execute plan using new-array interface
        cfftw.fftw_execute_dft(self.forward_plan,
                               <complex *> cnp.PyArray_DATA(gdata),
                               <complex *> cnp.PyArray_DATA(cdata))

    def _backward_real(self, cnp.ndarray cdata, cnp.ndarray gdata):
        """Real coefficient-to-grid transform."""

        # Execute plan using new-array interface
        cfftw.fftw_execute_dft_c2r(self.backward_plan,
                                   <complex *> cnp.PyArray_DATA(cdata),
                                   <double *> cnp.PyArray_DATA(gdata))

    def _backward_complex(self, cnp.ndarray cdata, cnp.ndarray gdata):
        """Complex coefficient-to-grid transform."""

        # Execute plan using new-array interface
        cfftw.fftw_execute_dft(self.backward_plan,
                               <complex *> cnp.PyArray_DATA(cdata),
                               <complex *> cnp.PyArray_DATA(gdata))


cdef class DiscreteCosineTransform:
    """
    FFTW-based discrete cosine transform along one axis of an N-dimensional array.

    Parameters
    ----------
    grid_dtype : dtype
        Grid space data type
    grid_shape : tuple of ints, array of ints
        Array shape in grid space
    axis : int
        Axis over which to compute the DFT
    flags : list of str, optional
        List of wrapped FFTW flags

    """

    cdef cfftw.fftw_plan forward_plan
    cdef cfftw.fftw_plan backward_plan

    def __init__(self, grid_dtype, grid_shape, int axis, flags=[]):

        # Compute data shapes based on dtype
        if np.dtype(grid_dtype) == np.float64:
            real_shape = list(grid_shape)
        elif np.dtype(grid_dtype) == np.complex128:
            # Treat as interleaved real data
            real_shape = list(grid_shape) + [2]
        else:
            raise ValueError("Unsupported dtype: %s" %str(grid_dtype))
        gshape = np.array(real_shape, dtype=int)
        cshape = np.array(real_shape, dtype=int)

        # Build FFTW guru transform structure
        cdef int trans_rank = 1
        cdef cfftw.fftw_iodim trans
        # Transform along `axis`
        trans.n = gshape[axis]
        trans.in_stride  = np.prod(gshape[axis+1:])
        trans.out_stride = np.prod(cshape[axis+1:])
        cdef cfftw.fftw_iodim *trans_struct = [trans]

        # Build FFTW guru vector structures
        cdef int vec_rank = 2
        cdef cfftw.fftw_iodim vec0, vec1f, vec1b
        # Loop over higher axes
        vec0.n = np.prod(gshape[axis+1:])
        vec0.in_stride = 1
        vec0.out_stride = 1
        # Loop over lower axes
        vec1f.n = vec1b.n = np.prod(gshape[:axis])
        vec1f.in_stride  = vec1b.out_stride = np.prod(gshape[axis:])
        vec1f.out_stride = vec1b.in_stride  = np.prod(cshape[axis:])
        cdef cfftw.fftw_iodim *vec_struct_f = [vec0, vec1f]
        cdef cfftw.fftw_iodim *vec_struct_b = [vec0, vec1b]

        # Build flags
        cdef unsigned intflags = 0
        for f in flags:
            intflags = intflags | fftw_flags[f]

        # Select DCT types
        cdef int *kind_f = [cfftw.FFTW_REDFT10]
        cdef int *kind_b = [cfftw.FFTW_REDFT01]

        # Create out-of-place plans using temporary memory allocations
        cdef double *gdata
        cdef double *cdata
        gdata = cfftw.fftw_alloc_real(np.prod(gshape))
        cdata = cfftw.fftw_alloc_real(np.prod(cshape))
        self.forward_plan = cfftw.fftw_plan_guru_r2r(trans_rank,
                                                     trans_struct,
                                                     vec_rank,
                                                     vec_struct_f,
                                                     gdata,
                                                     cdata,
                                                     kind_f,
                                                     intflags | cfftw.FFTW_DESTROY_INPUT)
        self.backward_plan = cfftw.fftw_plan_guru_r2r(trans_rank,
                                                      trans_struct,
                                                      vec_rank,
                                                      vec_struct_b,
                                                      cdata,
                                                      gdata,
                                                      kind_b,
                                                      intflags | cfftw.FFTW_DESTROY_INPUT)
        cfftw.fftw_free(gdata)
        cfftw.fftw_free(cdata)

        # Check that plan creation succeeded
        if (self.forward_plan == NULL) or (self.backward_plan == NULL):
            raise RuntimeError("FFTW could not create plans.")

    def __dealloc__(self):
        """Destroy plans on deallocation."""

        cfftw.fftw_destroy_plan(self.forward_plan)
        cfftw.fftw_destroy_plan(self.backward_plan)

    def forward(self, cnp.ndarray gdata, cnp.ndarray cdata):
        """Grid-to-coefficient transform."""

        # Execute plan using new-array interface
        cfftw.fftw_execute_r2r(self.forward_plan,
                               <double *> cnp.PyArray_DATA(gdata),
                               <double *> cnp.PyArray_DATA(cdata))

    def backward(self, cnp.ndarray cdata, cnp.ndarray gdata):
        """Coefficient-to-grid transform."""

        # Execute plan using new-array interface
        cfftw.fftw_execute_r2r(self.backward_plan,
                               <double *> cnp.PyArray_DATA(cdata),
                               <double *> cnp.PyArray_DATA(gdata))


cdef class DiscreteSineTransform:
    """
    FFTW-based discrete sine transform along one axis of an N-dimensional array.

    Parameters
    ----------
    grid_dtype : dtype
        Grid space data type
    grid_shape : tuple of ints, array of ints
        Array shape in grid space
    axis : int
        Axis over which to compute the DFT
    flags : list of str, optional
        List of wrapped FFTW flags

    """

    cdef cfftw.fftw_plan forward_plan
    cdef cfftw.fftw_plan backward_plan

    def __init__(self, grid_dtype, grid_shape, int axis, flags=[]):

        # Compute data shapes based on dtype
        if np.dtype(grid_dtype) == np.float64:
            real_shape = list(grid_shape)
        elif np.dtype(grid_dtype) == np.complex128:
            # Treat as interleaved real data
            real_shape = list(grid_shape) + [2]
        else:
            raise ValueError("Unsupported dtype: %s" %str(grid_dtype))
        gshape = np.array(real_shape, dtype=int)
        cshape = np.array(real_shape, dtype=int)

        # Build FFTW guru transform structure
        cdef int trans_rank = 1
        cdef cfftw.fftw_iodim trans
        # Transform along `axis`
        trans.n = gshape[axis]
        trans.in_stride  = np.prod(gshape[axis+1:])
        trans.out_stride = np.prod(cshape[axis+1:])
        cdef cfftw.fftw_iodim *trans_struct = [trans]

        # Build FFTW guru vector structures
        cdef int vec_rank = 2
        cdef cfftw.fftw_iodim vec0, vec1f, vec1b
        # Loop over higher axes
        vec0.n = np.prod(gshape[axis+1:])
        vec0.in_stride = 1
        vec0.out_stride = 1
        # Loop over lower axes
        vec1f.n = vec1b.n = np.prod(gshape[:axis])
        vec1f.in_stride  = vec1b.out_stride = np.prod(gshape[axis:])
        vec1f.out_stride = vec1b.in_stride  = np.prod(cshape[axis:])
        cdef cfftw.fftw_iodim *vec_struct_f = [vec0, vec1f]
        cdef cfftw.fftw_iodim *vec_struct_b = [vec0, vec1b]

        # Build flags
        cdef unsigned intflags = 0
        for f in flags:
            intflags = intflags | fftw_flags[f]

        # Select DCT types
        cdef int *kind_f = [cfftw.FFTW_RODFT10]
        cdef int *kind_b = [cfftw.FFTW_RODFT01]

        # Create out-of-place plans using temporary memory allocations
        cdef double *gdata
        cdef double *cdata
        gdata = cfftw.fftw_alloc_real(np.prod(gshape))
        cdata = cfftw.fftw_alloc_real(np.prod(cshape))
        self.forward_plan = cfftw.fftw_plan_guru_r2r(trans_rank,
                                                     trans_struct,
                                                     vec_rank,
                                                     vec_struct_f,
                                                     gdata,
                                                     cdata,
                                                     kind_f,
                                                     intflags | cfftw.FFTW_DESTROY_INPUT)
        self.backward_plan = cfftw.fftw_plan_guru_r2r(trans_rank,
                                                      trans_struct,
                                                      vec_rank,
                                                      vec_struct_b,
                                                      cdata,
                                                      gdata,
                                                      kind_b,
                                                      intflags | cfftw.FFTW_DESTROY_INPUT)
        cfftw.fftw_free(gdata)
        cfftw.fftw_free(cdata)

        # Check that plan creation succeeded
        if (self.forward_plan == NULL) or (self.backward_plan == NULL):
            raise RuntimeError("FFTW could not create plans.")

    def __dealloc__(self):
        """Destroy plans on deallocation."""

        cfftw.fftw_destroy_plan(self.forward_plan)
        cfftw.fftw_destroy_plan(self.backward_plan)

    def forward(self, cnp.ndarray gdata, cnp.ndarray cdata):
        """Grid-to-coefficient transform."""

        # Execute plan using new-array interface
        cfftw.fftw_execute_r2r(self.forward_plan,
                               <double *> cnp.PyArray_DATA(gdata),
                               <double *> cnp.PyArray_DATA(cdata))

    def backward(self, cnp.ndarray cdata, cnp.ndarray gdata):
        """Coefficient-to-grid transform."""

        # Execute plan using new-array interface
        cfftw.fftw_execute_r2r(self.backward_plan,
                               <double *> cnp.PyArray_DATA(cdata),
                               <double *> cnp.PyArray_DATA(gdata))

