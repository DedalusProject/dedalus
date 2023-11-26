"""Cythonized linear algebra routines."""

cimport cython
from cython.parallel cimport prange


ctypedef Py_ssize_t index
ctypedef fused dtype:
    float
    double
    float complex
    double complex


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def solve_upper_csr_mid(
    const int [::1] Ap,
    const int [::1] Aj,
    const dtype [::1] Ax,
          dtype [:,:,::1] x):
    """Solve upper triangular CSR matrix along middle axis of 3D array."""
    cdef int n_before = x.shape[0]
    cdef int n_row = x.shape[1]
    cdef int n_after = x.shape[2]
    cdef index h, i, jj, j, k
    cdef dtype a
    for h in range(n_before):
        for i in range(n_row-1, -1, -1):
            # Subtract off-diagonal entries
            for jj in range(Ap[i+1]-1, Ap[i], -1):
                j = Aj[jj]
                a = Ax[jj]
                for k in range(n_after):
                    x[h,i,k] = x[h,i,k] - a * x[h,j,k]
            # Divide by diagonal entry
            a = Ax[Ap[i]]
            for k in range(n_after):
                x[h,i,k] = x[h,i,k] / a


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def solve_upper_csr_mid_omp(
    const int [::1] Ap,
    const int [::1] Aj,
    const dtype [::1] Ax,
          dtype [:,:,::1] x,
    const int num_threads):
    """Solve upper triangular CSR matrix along middle axis of 3D array. Threaded with OpenMP."""
    cdef int n_before = x.shape[0]
    cdef int n_row = x.shape[1]
    cdef int n_after = x.shape[2]
    cdef index h, i, jj, j, k
    cdef dtype a
    for h in prange(n_before, nogil=True, num_threads=num_threads, schedule='static'):
        for i in range(n_row-1, -1, -1):
            # Subtract off-diagonal entries
            for jj in range(Ap[i+1]-1, Ap[i], -1):
                j = Aj[jj]
                a = Ax[jj]
                for k in range(n_after):
                    x[h,i,k] = x[h,i,k] - a * x[h,j,k]
            # Divide by diagonal entry
            a = Ax[Ap[i]]
            for k in range(n_after):
                x[h,i,k] = x[h,i,k] / a


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def solve_upper_csr_last(
    const int [::1] Ap,
    const int [::1] Aj,
    const dtype [::1] Ax,
          dtype [:,::1] x):
    """Solve upper triangular CSR matrix along last axis of 2D array."""
    cdef int n_before = x.shape[0]
    cdef int n_row = x.shape[1]
    cdef index h, i, jj
    cdef dtype sum
    for h in range(n_before):
        for i in range(n_row-1, -1, -1):
            sum = x[h,i]
            # Subtract off-diagonal entries
            for jj in range(Ap[i+1]-1, Ap[i], -1):
                sum = sum - Ax[jj] * x[h,Aj[jj]]
            # Divide by diagonal entry
            x[h,i] = sum / Ax[Ap[i]]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def solve_upper_csr_last_omp(
    const int [::1] Ap,
    const int [::1] Aj,
    const dtype [::1] Ax,
          dtype [:,::1] x,
    const int num_threads):
    """Solve upper triangular CSR matrix along last axis of 2D array. Threaded with OpenMP."""
    cdef int n_before = x.shape[0]
    cdef int n_row = x.shape[1]
    cdef index h, i, jj
    cdef dtype sum
    for h in prange(n_before, nogil=True, num_threads=num_threads, schedule='static'):
        for i in range(n_row-1, -1, -1):
            sum = x[h,i]
            # Subtract off-diagonal entries
            for jj in range(Ap[i+1]-1, Ap[i], -1):
                sum = sum - Ax[jj] * x[h,Aj[jj]]
            # Divide by diagonal entry
            x[h,i] = sum / Ax[Ap[i]]

