"""
FFTW Cython declarations.  Comments refer to corresponding sections of the
FFTW3 documentation.

"""

from mpi4py.libmpi cimport MPI_Comm as mpi_comm_t
from libc.stddef cimport ptrdiff_t


# Make C99 complex types available to FFTW
cdef extern from "complex.h":
    pass


cdef extern from "fftw3.h":

    # Memory allocation (4.1.3)
    double *fftw_alloc_real(size_t size)
    complex *fftw_alloc_complex(size_t size)
    void fftw_free(void *data)

    # Using plans (4.2)
    # Use opaque pointer as plan type
    ctypedef void *fftw_plan
    void fftw_execute(fftw_plan plan)
    void fftw_destroy_plan(fftw_plan plan)

    # Advanced complex DFTs (4.4.1)
    fftw_plan fftw_plan_many_dft(int rank,
                                 int *n,
                                 int howmany,
                                 complex *in_,
                                 int *in_embed,
                                 int in_stride,
                                 int in_dist,
                                 complex *out,
                                 int *out_embed,
                                 int out_stride,
                                 int out_dist,
                                 int sign,
                                 unsigned flags)

    # Guru vector and transform sizes (4.5.2)
    ctypedef struct fftw_iodim:
        int n
        int in_stride "is"
        int out_stride "os"

    # Guru complex DFTs (4.5.3)
    fftw_plan fftw_plan_guru_dft(int rank,
                                 fftw_iodim *dims,
                                 int howmany_rank,
                                 fftw_iodim *howmany_dims,
                                 complex *in_,
                                 complex *out,
                                 int sign,
                                 unsigned flags)

    # Guru real-data DFTs (4.5.4)
    fftw_plan fftw_plan_guru_dft_r2c(int rank,
                                     fftw_iodim *dims,
                                     int howmany_rank,
                                     fftw_iodim *howmany_dims,
                                     double *in_,
                                     complex *out,
                                     unsigned flags)
    fftw_plan fftw_plan_guru_dft_c2r(int rank,
                                     fftw_iodim *dims,
                                     int howmany_rank,
                                     fftw_iodim *howmany_dims,
                                     complex *in_,
                                     double *out,
                                     unsigned flags)

    # Guru real-to-real transforms (4.5.5)
    ctypedef int fftw_r2r_kind
    fftw_plan fftw_plan_guru_r2r(int rank,
                                 fftw_iodim *dims,
                                 int howmany_rank,
                                 fftw_iodim *howmany_dims,
                                 double *in_,
                                 double *out,
                                 fftw_r2r_kind *kind,
                                 unsigned flags)

    # New-array execute functions (4.6)
    void fftw_execute_dft(fftw_plan plan,
                          complex *in_,
                          complex *out)
    void fftw_execute_dft_r2c(fftw_plan plan,
                              double *in_,
                              complex *out)
    void fftw_execute_dft_c2r(fftw_plan plan,
                              complex *in_,
                              double *out)
    void fftw_execute_r2r(fftw_plan plan,
                          double *in_,
                          double *out)


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

    # FFTW flags, defined in FFTW headers
    FFTW_BACKWARD = 1
    FFTW_FORWARD = -1

    FFTW_ESTIMATE = (1 << 6)
    FFTW_EXHAUSTIVE = (1 << 3)
    FFTW_MEASURE = 0
    FFTW_PATIENT = (1 << 5)

    FFTW_DESTROY_INPUT = (1 << 0)
    FFTW_PRESERVE_INPUT = (1 << 4)

    FFTW_R2HC = 0
    FFTW_HC2R = 1
    FFTW_DHT = 2
    FFTW_REDFT00 = 3
    FFTW_REDFT01 = 4
    FFTW_REDFT10 = 5
    FFTW_REDFT11 = 6
    FFTW_RODFT00 = 7
    FFTW_RODFT01 = 8
    FFTW_RODFT10 = 9
    FFTW_RODFT11 = 10

    FFTW_MPI_TRANSPOSED_IN = (1 << 29)
    FFTW_MPI_TRANSPOSED_OUT = (1 << 30)
