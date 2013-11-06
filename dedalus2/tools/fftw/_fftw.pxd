"""Custom FFTW wrappers for Dedalus

Author: J. S. Oishi <jsoishi@gmail.com>
Affiliation: KIPAC/SLAC/Stanford
License:
  Copyright (C) 2011 J. S. Oishi.  All Rights Reserved.

  This file is part of dedalus.

  dedalus is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from mpi4py cimport MPI
cdef extern from "complex.h":
    pass

cdef extern from "fftw3.h":
    ctypedef struct fftw_plan_s:
        pass
    ctypedef fftw_plan_s *fftw_plan
    ctypedef struct fftw_iodim:
        int n
        int ins "is"
        int ous "os"
        
    fftw_plan fftw_plan_many_dft(int rank,
                                 int *n,
                                 int howmany,
                                 complex *in_,
                                 int *inembed,
                                 int istride,
                                 int idist,
                                 complex *out,
                                 int *onembed,
                                 int ostride,
                                 int odist,
                                 int sign,
                                 unsigned flags)
    fftw_plan fftw_plan_many_dft_r2c(int rank, 
                                     int *n, 
                                     int howmany,
                                     double *in_, 
                                     int *inembed,
                                     int istride, 
                                     int idist,
                                     complex *out, 
                                     int *onembed,
                                     int ostride, 
                                     int odist,
                                     unsigned flags)            
    fftw_plan fftw_plan_many_dft_c2r(int rank, 
                                     int *n, 
                                     int howmany,
                                     complex *out,
                                     int *inembed,
                                     int istride, 
                                     int idist, 
                                     double *in_,
                                     int *onembed,
                                     int ostride, 
                                     int odist,
                                     unsigned flags)

    fftw_plan fftw_plan_dft_2d(int n0,
                               int n1,
                               complex* in_,
                               complex* out_,
                               int sign,
                               unsigned flags)
    fftw_plan fftw_plan_dft_3d(int n0,
                               int n1,
                               int n2,
                               complex* in_,
                               complex* out_,
                               int sign,
                               unsigned flags)

    fftw_plan fftw_plan_guru_dft(int rank,
                                 fftw_iodim *dims,
                                 int howmany_rank,
                                 fftw_iodim *howmany_dims,
                                 complex *in_,
                                 complex *out,
                                 int sign,
                                 unsigned flags)

    fftw_plan fftw_plan_dft_r2c_2d(int n0,
                                   int n1,
                                   double* in_,
                                   complex* out_,
                                   unsigned flags)
    fftw_plan fftw_plan_dft_r2c_3d(int n0,
                                   int n1,
                                   int n2,
                                   double* in_,
                                   complex* out_,
                                   unsigned flags)
    fftw_plan fftw_plan_dft_c2r_2d(int n0,
                                   int n1,
                                   complex* in_,
                                   double* out_,
                                   unsigned flags)
    fftw_plan fftw_plan_dft_c2r_3d(int n0,
                                   int n1,
                                   int n2,
                                   complex* in_,
                                   double* out_,
                                   unsigned flags)

    double* fftw_alloc_real(size_t n)
    complex* fftw_alloc_complex(size_t n)
    void fftw_execute(fftw_plan plan)
    void fftw_destroy_plan(fftw_plan plan)
    void fftw_free(void *mem)

cdef extern from "fftw3-mpi.h":
    size_t fftw_mpi_local_size_2d(size_t n0, size_t n1, MPI.MPI_Comm comm,
                                     size_t *local_n0, size_t *local_0_start)
    size_t fftw_mpi_local_size_3d(size_t n0, size_t n1, size_t n2, MPI.MPI_Comm comm,
                                     size_t *local_n0, size_t *local_0_start)
    size_t fftw_mpi_local_size_2d_transposed(size_t n0, size_t n1, MPI.MPI_Comm comm,
                                             size_t *local_n0, size_t *local_0_start,
                                             size_t *local_n1, size_t *local_1_start)
    size_t fftw_mpi_local_size_3d_transposed(size_t n0, size_t n1, size_t n2, MPI.MPI_Comm comm,
                                             size_t *local_n0, size_t *local_0_start,
                                             size_t *local_n1, size_t *local_1_start)


    fftw_plan fftw_mpi_plan_dft_r2c_2d(int n0,
                                       int n1,
                                       double *in_,
                                       complex *out,
                                       MPI.MPI_Comm comm,
                                       unsigned flags)
    fftw_plan fftw_mpi_plan_dft_c2r_2d(int n0,
                                       int n1,
                                       complex *in_,
                                       double *out,
                                       MPI.MPI_Comm comm,
                                       unsigned flags)
    fftw_plan fftw_mpi_plan_dft_r2c_3d(int n0,
                                       int n1,
                                       int n2,
                                       double *in_,
                                       complex *out,
                                       MPI.MPI_Comm comm,
                                       unsigned flags)
    fftw_plan fftw_mpi_plan_dft_c2r_3d(int n0,
                                       int n1,
                                       int n2,
                                       complex *in_,
                                       double *out,
                                       MPI.MPI_Comm comm,
                                       unsigned flags)
                                       
    fftw_plan fftw_mpi_plan_many_dft(int rank, 
                                     size_t *n,
                                     size_t howmany, 
                                     size_t block0, 
                                     size_t block1,
                                     complex *in_, 
                                     complex *out,
                                     MPI.MPI_Comm comm, 
                                     int sign, 
                                     unsigned flags)
    fftw_plan fftw_mpi_plan_many_transpose(size_t n0, 
                                           size_t n1,
                                           size_t howmany,
                                           size_t block0,
                                           size_t block1,
                                           double *in_, 
                                           double *out,
                                           MPI.MPI_Comm comm, 
                                           unsigned flags)

    void fftw_mpi_init()
    void fftw_mpi_cleanup()

cdef enum:
    FFTW_FORWARD = -1
    FFTW_BACKWARD = +1
    FFTW_MEASURE = 0
    FFTW_DESTROY_INPUT =  (1 << 0)
    FFTW_UNALIGNED = (1 << 1)
    FFTW_CONSERVE_MEMORY = (1 << 2)
    FFTW_EXHAUSTIVE = (1 << 3) # /* NO_EXHAUSTIVE is default */
    FFTW_PRESERVE_INPUT = (1 << 4) # /* cancels FFTW_DESTROY_INPUT */
    FFTW_PATIENT = (1 << 5) # /* IMPATIENT is default */
    FFTW_ESTIMATE = (1 << 6)
    FFTW_MPI_TRANSPOSED_IN = (1U << 29)
    FFTW_MPI_TRANSPOSED_OUT = (1U << 30)
    FFTW_MPI_DEFAULT_BLOCK = 0



    
