"""
Custom FFTW wrappers for Dedalus.

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

from _fftw cimport fftw_iodim, FFTW_FORWARD, FFTW_BACKWARD, \
        FFTW_MEASURE, FFTW_DESTROY_INPUT, FFTW_UNALIGNED, \
        FFTW_CONSERVE_MEMORY, FFTW_EXHAUSTIVE, FFTW_PRESERVE_INPUT, \
        FFTW_PATIENT, FFTW_ESTIMATE, FFTW_MPI_TRANSPOSED_IN, \
        FFTW_MPI_TRANSPOSED_OUT, fftw_plan, FFTW_MPI_DEFAULT_BLOCK
cimport _fftw as fftw
cimport libc
cimport numpy as np
import numpy as np
from mpi4py cimport MPI
from mpi4py.mpi_c cimport *
from cpython cimport Py_INCREF
              
fftw_flags = {'FFTW_CONSERVE_MEMORY': FFTW_CONSERVE_MEMORY,
              'FFTW_DESTROY_INPUT': FFTW_DESTROY_INPUT,
              'FFTW_ESTIMATE': FFTW_ESTIMATE,
              'FFTW_EXHAUSTIVE': FFTW_EXHAUSTIVE,
              'FFTW_MEASURE': FFTW_MEASURE,
              'FFTW_MPI_TRANSPOSED_IN': FFTW_MPI_TRANSPOSED_IN,
              'FFTW_MPI_TRANSPOSED_OUT': FFTW_MPI_TRANSPOSED_OUT,
              'FFTW_PATIENT': FFTW_PATIENT,
              'FFTW_PRESERVE_INPUT': FFTW_PRESERVE_INPUT,  
              'FFTW_UNALIGNED': FFTW_UNALIGNED}

cdef extern from "numpy/arrayobject.h":
    object PyArray_NewFromDescr(object subtype, np.dtype descr,
                                int nd, np.npy_intp* dims, np.npy_intp* strides,
                                void* data, int flags, object obj)

cdef extern from "stdlib.h":
    void free(void *ptr)

DTYPEi = np.int64
ctypedef np.int64_t DTYPEi_t

cdef class fftwMemoryReleaser:
    cdef void* memory

    def __cinit__(self):
        self.memory = NULL
                  
    def __dealloc__(self):
        if self.memory:
            #release memory
            fftw.fftw_free(self.memory)
            #print "memory released", hex(<long>self.memory)

cdef fftwMemoryReleaser MemoryReleaserFactory(void* ptr):
    cdef fftwMemoryReleaser mr = fftwMemoryReleaser.__new__(fftwMemoryReleaser)
    mr.memory = ptr
    return mr

def fftw_mpi_init():
    fftw.fftw_mpi_init()

def create_data(np.ndarray[DTYPEi_t, ndim=1] shape not None, com_sys):
    """
    Allocate data using fftw's allocation routines. We allocate
    the complex part directly, shaping it with the local_n1 returned
    by FFTW's parallel interface.

    Parameters
    ----------
    shape : int64 ndarray
        Numpy array giving the global logical shape in x-space grid points. 
    com_sys

    """
    
    np.import_array()
    cdef np.ndarray array
    cdef np.dtype kdtype = np.dtype('complex128')
    cdef np.dtype xdtype = np.dtype('float64')
    Py_INCREF(kdtype)
    Py_INCREF(kdtype)
    Py_INCREF(xdtype)

    cdef complex *data
    cdef size_t n, local_x0, local_x0_start, local_k1, local_k1_start
    cdef MPI.Comm comm = com_sys.comm
    cdef MPI_Comm c_comm = comm.ob_mpi
    
    # Global mixed-shape (shape after r2c transform on last dimension)
    cdef np.ndarray[DTYPEi_t, ndim=1] gmshape = shape.copy()
    gmshape[-1] = gmshape[-1] / 2 + 1

    # Get required allocation space and local portion information
    if shape.size == 2:
        n = fftw.fftw_mpi_local_size_2d_transposed(<size_t> gmshape[0],
                                                   <size_t> gmshape[1],
                                                   c_comm,
                                                   &local_x0, 
                                                   &local_x0_start,
                                                   &local_k1, 
                                                   &local_k1_start)
    elif shape.size == 3:
        n = fftw.fftw_mpi_local_size_3d_transposed(<size_t> gmshape[0],
                                                   <size_t> gmshape[1],
                                                   <size_t> gmshape[2],
                                                   c_comm,
                                                   &local_x0, 
                                                   &local_x0_start,
                                                   &local_k1, 
                                                   &local_k1_start)
    else:
        raise ValueError("Only 2D and 3D arrays are supported.")
        
    # Local x-shape
    cdef np.ndarray[DTYPEi_t, ndim=1] xshape = shape.copy()
    xshape[0] = local_x0
    
    # Padded local x-shape
    cdef np.ndarray[DTYPEi_t, ndim=1] pxshape = xshape.copy()
    pxshape[-1] = 2 * gmshape[-1]
    
    # Local mixed-shape
    cdef np.ndarray[DTYPEi_t, ndim=1] mshape = gmshape.copy()
    mshape[0] = local_x0

    # Local k-shape
    cdef np.ndarray[DTYPEi_t, ndim=1] kshape = gmshape.copy()
    kshape[1] = kshape[0]
    kshape[0] = local_k1

    # Allocate and zero required space
    data = fftw.fftw_alloc_complex(n)
    cdef int i
    for i in range(n):
        data[i] = 0j

    # Construct strides
    cdef np.ndarray[DTYPEi_t, ndim=1] np_xstrides = np.array((1,) + tuple(pxshape[1:][::-1])).cumprod()[::-1]
    cdef np.ndarray[DTYPEi_t, ndim=1] np_mstrides = np.array((1,) + tuple(mshape[1:][::-1])).cumprod()[::-1]
    cdef np.ndarray[DTYPEi_t, ndim=1] np_kstrides = np.array((1,) + tuple(kshape[1:][::-1])).cumprod()[::-1]
    
    np_xstrides *= xdtype.itemsize
    np_mstrides *= kdtype.itemsize
    np_kstrides *= kdtype.itemsize

    rank = shape.size
    xstrides = <size_t *> libc.stdlib.malloc(sizeof(size_t) * rank)
    mstrides = <size_t *> libc.stdlib.malloc(sizeof(size_t) * rank)
    kstrides = <size_t *> libc.stdlib.malloc(sizeof(size_t) * rank)
    for i in range(rank):
        xstrides[i] = np_xstrides[i]
        mstrides[i] = np_mstrides[i]
        kstrides[i] = np_kstrides[i]
        
    # Create arrays using same "data"
    xarray = PyArray_NewFromDescr(np.ndarray, xdtype, rank, <np.npy_intp *> xshape.data, <np.npy_intp *> xstrides, <void *> data, np.NPY_DEFAULT, None)
    marray = PyArray_NewFromDescr(np.ndarray, kdtype, rank, <np.npy_intp *> mshape.data, <np.npy_intp *> mstrides, <void *> data, np.NPY_DEFAULT, None)
    karray = PyArray_NewFromDescr(np.ndarray, kdtype, rank, <np.npy_intp *> kshape.data, <np.npy_intp *> kstrides, <void *> data, np.NPY_DEFAULT, None)
    
    # Garbage
    np.set_array_base(karray, MemoryReleaserFactory(data))
    libc.stdlib.free(xstrides)
    libc.stdlib.free(mstrides)
    libc.stdlib.free(kstrides)

    return karray, marray, xarray, local_x0, local_x0_start, local_k1, local_k1_start

def fftw_mpi_allocate(comm):
    """Allocates memory for local data block. fftw provides the load
    balancing, and so this needs to be done before arrays of local k
    and local x are created.

    """
    pass
    
cdef class Plan:
    cdef fftw_plan _fftw_plan
    cdef np.ndarray _data
    cdef int direction
    cdef int flags
    def __init__(self, data, forward=True, flags=['FFTW_MEASURE']):
        """A custom wrapping of FFTW for use in Dedalus.

        """
        if forward:
            self.direction = FFTW_FORWARD
        else:
            self.direction = FFTW_BACKWARD
        for f in flags:
            self.flags = self.flags | fftw_flags[f]
        self._data = data
        if len(data.shape) == 2:
            self._fftw_plan = fftw.fftw_plan_dft_2d(data.shape[0],
                                               data.shape[1],
                                               <complex *> self._data.data,
                                               <complex *> self._data.data,
                                               self.direction,
                                               self.flags)
        elif len(data.shape) == 3:
            self._fftw_plan = fftw.fftw_plan_dft_3d(data.shape[0],
                                               data.shape[1],
                                               data.shape[2],
                                               <complex *> self._data.data,
                                               <complex *> self._data.data,
                                               self.direction,
                                               self.flags)
        else:
            raise RuntimeError("Only 2D and 3D arrays are supported.")

        if self._fftw_plan == NULL:
            raise RuntimeError("FFTW could not create plan.")

    def __call__(self):
        if self._fftw_plan != NULL:
            fftw.fftw_execute(self._fftw_plan)

    def __dealloc__(self):
        fftw.fftw_destroy_plan(self._fftw_plan)

    property data:
        def __get__(self):
            return self._data
    
        def __set__(self, inp):
            self._data[:]=inp

cdef class rPlan(Plan):
    cdef np.ndarray _xdata, _kdata
    def __init__(self, xdata, kdata, com_sys, shape, forward=True, 
            flags=['FFTW_MEASURE']):
        """
        Constructs a FFTW plan that will take a parallelized, full R2C DFT of a 
        2D or 3D row-major data array.

        """
        
        # MPI communicators
        cdef MPI.Comm comm = com_sys.comm
        cdef MPI_Comm c_comm = comm.ob_mpi
        
        # Store input arrays
        self._xdata = xdata
        self._kdata = kdata
            
        # Build flags
        for f in flags:
            self.flags = self.flags | fftw_flags[f]

        # Create plan
        if len(shape) == 2:
            if forward:
                self.flags = self.flags | fftw_flags['FFTW_MPI_TRANSPOSED_OUT']
                self._fftw_plan = fftw.fftw_mpi_plan_dft_r2c_2d(shape[0],
                                                                shape[1],
                                                                <double *> self._xdata.data,
                                                                <complex *> self._kdata.data,
                                                                c_comm,
                                                                self.flags)
            else:
                self.flags = self.flags | fftw_flags['FFTW_MPI_TRANSPOSED_IN']
                self._fftw_plan = fftw.fftw_mpi_plan_dft_c2r_2d(shape[0],
                                                                shape[1],
                                                                <complex *> self._kdata.data,
                                                                <double *> self._xdata.data,
                                                                c_comm,
                                                                self.flags)
        elif len(shape) == 3:
            if forward:
                self.flags = self.flags | fftw_flags['FFTW_MPI_TRANSPOSED_OUT']
                self._fftw_plan = fftw.fftw_mpi_plan_dft_r2c_3d(shape[0],
                                                                shape[1],
                                                                shape[2],
                                                                <double *> self._xdata.data,
                                                                <complex *> self._kdata.data,
                                                                c_comm,
                                                                self.flags)
            else:
                self.flags = self.flags | fftw_flags['FFTW_MPI_TRANSPOSED_IN']
                self._fftw_plan = fftw.fftw_mpi_plan_dft_c2r_3d(shape[0],
                                                                shape[1],
                                                                shape[2],
                                                                <complex *> self._kdata.data,
                                                                <double *> self._xdata.data,
                                                                c_comm,
                                                                self.flags)
        else:
            raise RuntimeError("Only 2D and 3D arrays are supported.")

        if self._fftw_plan == NULL:
            raise RuntimeError("FFTW could not create plan.")
            
cdef class PencilPlan(Plan):
    cdef np.ndarray _xdata, _kdata
    def __init__(self, xdata, kdata, forward=True, flags=['FFTW_MEASURE']):
        """
        Constructs a FFTW plan that will take a 1D C2C DFT along the last
        dimension of a 2D or 3D row-major data array.
        
        """

        # Store input arrays
        self._xdata = xdata
        self._kdata = kdata
            
        # Build flags
        for f in flags:
            self.flags = self.flags | fftw_flags[f]
            
        # Get array size
        cdef int nx = xdata.shape[-1]
        cdef int ny = xdata.shape[-2]
        cdef int nz = 1
        if len(xdata.shape) == 3:
            nz = xdata.shape[-3]
        
        # Construct plan inputs
        cdef int rank = 1
        cdef int sign = 0
        cdef int howmany = nz * ny
        cdef int xstride = 1
        cdef int kstride = 1
        cdef int xdist = nx
        cdef int kdist = nx

        # Create plan
        if forward:
            sign = FFTW_FORWARD
            self._fftw_plan = fftw.fftw_plan_many_dft(rank, 
                                                      &nx,
                                                      howmany,
                                                      <complex *> self._xdata.data, 
                                                      NULL,
                                                      xstride, 
                                                      xdist,
                                                      <complex *> self._kdata.data, 
                                                      NULL,
                                                      kstride, 
                                                      kdist,
                                                      sign,
                                                      self.flags)
        else:
            sign = FFTW_BACKWARD
            self._fftw_plan = fftw.fftw_plan_many_dft(rank, 
                                                      &nx,
                                                      howmany,
                                                      <complex *> self._kdata.data,
                                                      NULL,
                                                      kstride, 
                                                      kdist,
                                                      <complex *> self._xdata.data,
                                                      NULL,
                                                      xstride, 
                                                      xdist,
                                                      sign,
                                                      self.flags)
            
        if self._fftw_plan == NULL:
            raise RuntimeError("FFTW could not create plan.")
            
cdef class rPencilPlan(Plan):
    cdef np.ndarray _xdata, _kdata
    def __init__(self, xdata, kdata, forward=True, flags=['FFTW_MEASURE']):
        """
        Constructs a FFTW plan that will take a 1D R2C DFT along the last
        dimension of a 2D or 3D row-major data array.
        
        """

        # Store input arrays
        self._xdata = xdata
        self._kdata = kdata
            
        # Build flags
        for f in flags:
            self.flags = self.flags | fftw_flags[f]
            
        # Get local array size
        cdef int nx = xdata.shape[-1]
        cdef int ny = xdata.shape[-2]
        cdef int nz = 1
        if len(xdata.shape) == 3:
            nz = xdata.shape[-3]
        
        # Construct plan inputs
        cdef int rank = 1
        cdef int howmany = nz * ny
        cdef int xstride = 1
        cdef int kstride = 1
        cdef int xdist = 2 * (nx / 2 + 1)
        cdef int kdist = nx / 2 + 1

        # Create plan
        if forward:
            self._fftw_plan = fftw.fftw_plan_many_dft_r2c(rank, 
                                                          &nx,
                                                          howmany,
                                                          <double *> self._xdata.data, 
                                                          NULL,
                                                          xstride, 
                                                          xdist,
                                                          <complex *> self._kdata.data, 
                                                          NULL,
                                                          kstride, 
                                                          kdist,
                                                          self.flags)
        else:
            self._fftw_plan = fftw.fftw_plan_many_dft_c2r(rank, 
                                                          &nx,
                                                          howmany,
                                                          <complex *> self._kdata.data,
                                                          NULL,
                                                          kstride, 
                                                          kdist,
                                                          <double *> self._xdata.data,
                                                          NULL,
                                                          xstride, 
                                                          xdist,
                                                          self.flags)
            
        if self._fftw_plan == NULL:
            raise RuntimeError("FFTW could not create plan.")
            
cdef class TransposePlan(Plan):
    cdef np.ndarray _xdata, _kdata
    def __init__(self, xdata, kdata, com_sys, shape, forward=True,
            flags=['FFTW_MEASURE']):
        """
        Constructs a FFTW plan that will perform a parallelized transpose of
        the first two dimensions of a 2D or 3D row-major complex data array.
        
        """
        
        # MPI communicators
        cdef MPI.Comm comm = com_sys.comm
        cdef MPI_Comm c_comm = comm.ob_mpi

        # Store input arrays
        self._xdata = xdata
        self._kdata = kdata
        
        # Build flags
        for f in flags:
            self.flags = self.flags | fftw_flags[f]
            
        # Get global array size
        cdef size_t n0 = shape[0]
        cdef size_t n1 = shape[1]
        cdef size_t n2 = 1
        if len(shape) == 3:
            n2 = shape[2]
            
        # Items to be transformed are tuples of n2 complex numbers
        cdef size_t howmany = 2 * n2  
             
        # Create plan
        if forward:
            self._fftw_plan = fftw.fftw_mpi_plan_many_transpose(n0,
                                                                n1,
                                                                howmany,
                                                                <size_t> FFTW_MPI_DEFAULT_BLOCK,
                                                                <size_t> FFTW_MPI_DEFAULT_BLOCK,
                                                                <double *> self._xdata.data, 
                                                                <double *> self._kdata.data,
                                                                c_comm, 
                                                                self.flags)
        else:
            self._fftw_plan = fftw.fftw_mpi_plan_many_transpose(n1,
                                                                n0,
                                                                howmany,
                                                                <size_t> FFTW_MPI_DEFAULT_BLOCK,
                                                                <size_t> FFTW_MPI_DEFAULT_BLOCK,
                                                                <double *> self._kdata.data, 
                                                                <double *> self._xdata.data,
                                                                c_comm, 
                                                                self.flags)                                                    
        
        if self._fftw_plan == NULL:
            raise RuntimeError("FFTW could not create plan.")
            
cdef class PlanePlan(Plan):
    cdef np.ndarray _xdata, _kdata
    def __init__(self, xdata, kdata, com_sys, shape=None, forward=True, 
            flags=['FFTW_MEASURE']):
        """
        Constructs a FFTW plan that will take a parallelized 2D C2C DFT along 
        the first two dimensions of a 3D, row-major data array (including an
        MPI transpose of those dimensions).
        
        """
        
        # MPI communicators
        cdef MPI.Comm comm = com_sys.comm
        cdef MPI_Comm c_comm = comm.ob_mpi

        # Store input arrays
        self._xdata = xdata
        self._kdata = kdata
            
        # Build flags
        for f in flags:
            self.flags = self.flags | fftw_flags[f]
            
        # Get global array size
        cdef size_t n0 = shape[0]
        cdef size_t n1 = shape[1]
        cdef size_t n2 = shape[2]
        
        # Construct plan inputs
        cdef int rank = 2
        cdef int sign = 0
        cdef size_t n[2]
        n[0] = n0
        n[1] = n1
            
        # Create plan
        if forward:
            sign = FFTW_FORWARD
            self.flags = self.flags | fftw_flags['FFTW_MPI_TRANSPOSED_OUT']
            self._fftw_plan = fftw.fftw_mpi_plan_many_dft(rank, 
                                                          n,
                                                          n2,
                                                          <size_t> FFTW_MPI_DEFAULT_BLOCK,
                                                          <size_t> FFTW_MPI_DEFAULT_BLOCK,
                                                          <complex *> self._xdata.data, 
                                                          <complex *> self._kdata.data, 
                                                          c_comm,
                                                          sign,
                                                          self.flags)
        else:
            sign = FFTW_BACKWARD
            self.flags = self.flags | fftw_flags['FFTW_MPI_TRANSPOSED_IN']
            self._fftw_plan = fftw.fftw_mpi_plan_many_dft(rank, 
                                                          n,
                                                          n2,
                                                          <size_t> FFTW_MPI_DEFAULT_BLOCK,
                                                          <size_t> FFTW_MPI_DEFAULT_BLOCK,
                                                          <complex *> self._kdata.data, 
                                                          <complex *> self._xdata.data, 
                                                          c_comm,
                                                          sign,
                                                          self.flags)
            
        if self._fftw_plan == NULL:
            raise RuntimeError("FFTW could not create plan.")

