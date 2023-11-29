"""Cythonized linear algebra routines."""

cimport cython
cimport numpy as cnp
cnp.import_array()
import numpy as np
from cython.parallel cimport prange


ctypedef Py_ssize_t index
ctypedef fused dtype:
    float
    double
    float complex
    double complex


@cython.boundscheck(False)
@cython.wraparound(False)
def solve_upper_csr(
    const int [::1] indptr,
    const int [::1] indices,
    const dtype [::1] entries,
    cnp.ndarray array,
    const int axis,
    const int num_threads):
    """Solve upper triangular CSR matrix along specified axis of an array."""
    cdef int ndim = array.ndim
    cdef int N1 = 1
    cdef int N2 = 1
    cdef int N3 = 1
    cdef int [::1] shape = np.zeros(ndim, dtype=np.int32)
    for i in range(ndim):
        shape[i] = array.shape[i]
    if ndim == 1:
        if axis == 0:
            solve_upper_csr_vec(indptr, indices, entries, array, num_threads)
    elif ndim == 2:
        if axis == 0:
            if shape[1] == 1:
                solve_upper_csr_vec(indptr, indices, entries, array[:,0], num_threads)
            else:
                solve_upper_csr_first(indptr, indices, entries, array, num_threads)
        elif axis == 1:
            if shape[0] == 1:
                solve_upper_csr_vec(indptr, indices, entries, array[0,:], num_threads)
            else:
                solve_upper_csr_last(indptr, indices, entries, array, num_threads)
    else:
        # Treat as 3D array with specified axis in the middle
        # Compute equivalent shape (N1, N2, N3)
        if ndim == 3 and axis == 1:
            N1 = shape[0]
            N2 = shape[1]
            N3 = shape[2]
        else:
            if axis != 0:
                for i in range(axis):
                    N1 = N1 * shape[i]
            N2 = shape[axis]
            if axis != ndim - 1:
                for i in range(axis+1, ndim):
                    N3 = N3 * shape[i]
        # Dispatch to cython routines
        if N1 == 1:
            if N3 == 1:
                # (1, N2, 1) -> (N2,)
                x1 = array.reshape((N2,))
                solve_upper_csr_vec(indptr, indices, entries, x1, num_threads)
            else:
                # (1, N2, N3) -> (N2, N3)
                x2 = array.reshape((N2, N3))
                solve_upper_csr_first(indptr, indices, entries, x2, num_threads)
        else:
            if N3 == 1:
                # (N1, N2, 1) -> (N1, N2)
                x2 = array.reshape((N1, N2))
                solve_upper_csr_last(indptr, indices, entries, x2, num_threads)
            else:
                # (N1, N2, N3)
                x3 = array.reshape((N1, N2, N3))
                solve_upper_csr_mid(indptr, indices, entries, x3, num_threads)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def solve_upper_csr_vec(
    const int [::1] Ap,
    const int [::1] Aj,
    const dtype [::1] Ax,
          dtype [::1] x,
    const int num_threads):
    """Solve upper triangular CSR matrix along a vector."""
    cdef int n_row = x.shape[0]
    cdef index i, jj
    cdef dtype sum
    for i in range(n_row-1, -1, -1):
        # Subtract off-diagonal entries
        sum = x[i]
        for jj in range(Ap[i+1]-1, Ap[i], -1):
            sum = sum - Ax[jj] * x[Aj[jj]]
        # Divide by diagonal entry
        x[i] = sum / Ax[Ap[i]]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def solve_upper_csr_first(
    const int [::1] Ap,
    const int [::1] Aj,
    const dtype [::1] Ax,
          dtype [:,::1] x,
    const int num_threads):
    """Solve upper triangular CSR matrix along first axis of 2D array."""
    cdef int n_row = x.shape[0]
    cdef int n_after = x.shape[1]
    cdef index i, jj, j, k
    cdef dtype a
    for i in range(n_row-1, -1, -1):
        # Subtract off-diagonal entries
        for jj in range(Ap[i+1]-1, Ap[i], -1):
            j = Aj[jj]
            a = Ax[jj]
            for k in range(n_after):
                x[i,k] = x[i,k] - a * x[j,k]
        # Divide by diagonal entry
        a = Ax[Ap[i]]
        for k in range(n_after):
            x[i,k] = x[i,k] / a


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def solve_upper_csr_last(
    const int [::1] Ap,
    const int [::1] Aj,
    const dtype [::1] Ax,
          dtype [:,::1] x,
    const int num_threads):
    """Solve upper triangular CSR matrix along last axis of 2D array."""
    cdef int n_before = x.shape[0]
    cdef int n_row = x.shape[1]
    cdef index h, i, jj
    cdef dtype sum
    for h in prange(n_before, nogil=True, num_threads=num_threads, schedule='static'):
        for i in range(n_row-1, -1, -1):
            # Subtract off-diagonal entries
            sum = x[h,i]
            for jj in range(Ap[i+1]-1, Ap[i], -1):
                sum = sum - Ax[jj] * x[h,Aj[jj]]
            # Divide by diagonal entry
            x[h,i] = sum / Ax[Ap[i]]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def solve_upper_csr_mid(
    const int [::1] Ap,
    const int [::1] Aj,
    const dtype [::1] Ax,
          dtype [:,:,::1] x,
    const int num_threads):
    """Solve upper triangular CSR matrix along middle axis of 3D array."""
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
def apply_csr(
    const int [::1] indptr,
    const int [::1] indices,
    const dtype [::1] entries,
    cnp.ndarray array,
    cnp.ndarray out,
    const int axis,
    const int num_threads):
    """Apply CSR matrix to array along specified axis."""
    # Special-case based on dimension
    # This avoids overhead from reshaping small arrays
    cdef int ndim = array.ndim
    cdef int N1 = 1
    cdef int N2_in = 1
    cdef int N2_out = 1
    cdef int N3 = 1
    if ndim == 1:
        if axis == 0:
            apply_csr_vec(indptr, indices, entries, array, out, num_threads)
    elif ndim == 2:
        if axis == 0:
            if array.shape[1] == 1:
                apply_csr_vec(indptr, indices, entries, array[:,0], out[:,0], num_threads)
            else:
                apply_csr_first(indptr, indices, entries, array, out, num_threads)
        elif axis == 1:
            if array.shape[0] == 1:
                apply_csr_vec(indptr, indices, entries, array[0,:], out[0,:], num_threads)
            else:
                apply_csr_last(indptr, indices, entries, array, out, num_threads)
    else:
        # Treat as 3D array with specified axis in the middle
        # Compute equivalent shape (N1, N2, N3)
        if ndim == 3 and axis == 1:
            N1 = array.shape[0]
            N2_in = array.shape[1]
            N2_out = out.shape[1]
            N3 = array.shape[2]
        else:
            if axis != 0:
                for i in range(axis):
                    N1 = N1 * array.shape[i]
            N2_in = array.shape[axis]
            N2_out = out.shape[axis]
            if axis != ndim - 1:
                for i in range(axis+1, ndim):
                    N3 = N3 * array.shape[i]
        # Dispatch to cython routines
        if N1 == 1:
            if N3 == 1:
                # (1, N2, 1) -> (N2,)
                x1 = array.reshape((N2_in,))
                y1 = out.reshape((N2_out,))
                apply_csr_vec(indptr, indices, entries, x1, y1, num_threads)
            else:
                # (1, N2, N3) -> (N2, N3)
                x2 = array.reshape((N2_in, N3))
                y2 = out.reshape((N2_out, N3))
                apply_csr_first(indptr, indices, entries, x2, y2, num_threads)
        else:
            if N3 == 1:
                # (N1, N2, 1) -> (N1, N2)
                x2 = array.reshape((N1, N2_in))
                y2 = out.reshape((N1, N2_out))
                apply_csr_last(indptr, indices, entries, x2, y2, num_threads)
            else:
                # (N1, N2, N3)
                x3 = array.reshape((N1, N2_in, N3))
                y3 = out.reshape((N1, N2_out, N3))
                apply_csr_mid(indptr, indices, entries, x3, y3, num_threads)



@cython.boundscheck(False)
@cython.wraparound(False)
def apply_csr_vec(
    const int [::1] Ap,
    const int [::1] Aj,
    const dtype [::1] Ax,
    const dtype [:] x,
          dtype [::1] y,
    const int num_threads):
    """Apply CSR matrix along a vector."""
    cdef int n_row = y.shape[0]
    cdef index i, jj
    cdef dtype sum
    for i in prange(n_row, nogil=True, num_threads=num_threads, schedule='static'):
        sum = 0
        for jj in range(Ap[i], Ap[i+1]):
            sum = sum + Ax[jj] * x[Aj[jj]]
        y[i] = sum


@cython.boundscheck(False)
@cython.wraparound(False)
def apply_csr_first(
    const int [::1] Ap,
    const int [::1] Aj,
    const dtype [::1] Ax,
    const dtype [:,:] x,
          dtype [:,::1] y,
    const int num_threads):
    """Apply CSR matrix along first axis of 2D array."""
    cdef int n_row = y.shape[0]
    cdef int n_after = x.shape[1]
    cdef index i, jj, j, k
    cdef dtype a
    for i in prange(n_row, nogil=True, num_threads=num_threads, schedule='static'):
        for k in range(n_after):
            y[i,k] = 0
        for jj in range(Ap[i], Ap[i+1]):
            j = Aj[jj]
            a = Ax[jj]
            for k in range(n_after):
                y[i,k] = y[i,k] + a * x[j,k]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def apply_csr_last(
    const int [::1] Ap,
    const int [::1] Aj,
    const dtype [::1] Ax,
    const dtype [:,:] x,
          dtype [:,::1] y,
    const int num_threads):
    """Apply CSR matrix along last axis of 2D array."""
    cdef int n_before = x.shape[0]
    cdef int n_row = y.shape[1]
    cdef int hi
    cdef index h, i, jj
    cdef dtype sum
    for hi in prange(n_before*n_row, nogil=True, num_threads=num_threads, schedule='static'):
        h = hi / n_row
        i = hi % n_row
        sum = 0
        for jj in range(Ap[i], Ap[i+1]):
            sum = sum + Ax[jj] * x[h,Aj[jj]]
        y[h,i] = sum


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def apply_csr_mid(
    const int [::1] Ap,
    const int [::1] Aj,
    const dtype [::1] Ax,
    const dtype [:,:,:] x,
          dtype [:,:,::1] y,
    const int num_threads):
    """Apply CSR matrix along middle axis of 3D array."""
    cdef int n_before = x.shape[0]
    cdef int n_row = y.shape[1]
    cdef int n_after = x.shape[2]
    cdef int hi
    cdef index h, i, jj, j, k
    cdef dtype a
    for hi in prange(n_before*n_row, nogil=True, num_threads=num_threads, schedule='static'):
        h = hi / n_row
        i = hi % n_row
        for k in range(n_after):
            y[h,i,k] = 0
        for jj in range(Ap[i], Ap[i+1]):
            j = Aj[jj]
            a = Ax[jj]
            for k in range(n_after):
                y[h,i,k] = y[h,i,k] + a * x[h,j,k]

