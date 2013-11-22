

from mpi4py.mpi_c cimport MPI_Comm as mpi_comm_t
from libc.stddef cimport ptrdiff_t


# Make C99 complex types available to FFTW
cdef extern from "complex.h":
    pass


cdef extern from "fftw3.h":

    # Using plans (4.2)
    # Plan type as opaque pointer
    ctypedef void *fftw_plan
    void fftw_destroy_plan(fftw_plan plan)

    # Memory allocation (4.1.3)
    double *fftw_alloc_real(size_t size)
    complex *fftw_alloc_complex(size_t size)
    void fftw_free(void *data)


cdef extern from "fftw3-mpi.h":

    # MPI initialization (6.12.2)
    void fftw_mpi_init()

    # Using MPI plans (6.12.3)
    # Distributed transposes count as rank-zero r2r plans
    void fftw_mpi_execute_r2r(fftw_plan plan,
                              double *in_,
                              double *out)

    # MPI data distribution functions (6.12.4)
    ptrdiff_t fftw_mpi_local_size_many_transposed(int rank,
                                                  ptrdiff_t *shape,
                                                  ptrdiff_t itemsize,
                                                  ptrdiff_t block0,
                                                  ptrdiff_t block1,
                                                  mpi_comm_t comm,
                                                  ptrdiff_t *local0,
                                                  ptrdiff_t *start0,
                                                  ptrdiff_t *local1,
                                                  ptrdiff_t *start1)

    # MPI plan creation (6.12.5)
    fftw_plan fftw_mpi_plan_many_transpose(ptrdiff_t shape0,
                                           ptrdiff_t shape1,
                                           ptrdiff_t itemsize,
                                           ptrdiff_t block0,
                                           ptrdiff_t block1,
                                           double *in_,
                                           double *out,
                                           mpi_comm_t comm,
                                           unsigned flags)


cdef enum:

    FFTW_ESTIMATE = (1 << 6)
    FFTW_EXHAUSTIVE = (1 << 3)
    FFTW_MEASURE = 0
    FFTW_MPI_TRANSPOSED_IN = (1 << 29)
    FFTW_MPI_TRANSPOSED_OUT = (1 << 30)
    FFTW_PATIENT = (1 << 5)

