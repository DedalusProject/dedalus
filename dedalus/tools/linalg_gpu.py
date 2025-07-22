
import numpy as np
try:
    import cupy as cp
    import cupyx.scipy.sparse as csp
    HAVE_CUPY = True
except ImportError:
    HAVE_CUPY = False


def cupy_apply_csr(matrix, array, axis, out):
    """Apply CSR matrix to arbitrary axis of array."""
    if not HAVE_CUPY:
        raise ImportError("cupy must be installed to use GPU linear algebra")
    # Check matrix format
    if not isinstance(matrix, csp.csr_matrix):
        raise ValueError("Matrix must be in CSR format.")
    # Switch by dimension
    ndim = array.ndim
    if ndim == 1:
        if axis == 0:
            out[:] = cupy_apply_csr_vec(matrix, array)
        else:
            raise ValueError("axis must be 0 for 1D arrays")
    elif ndim == 2:
        if axis == 0:
            if array.shape[1] == 1:
                out[:,0] = cupy_apply_csr_vec(matrix, array[:,0])
            else:
                out[:] = cupy_apply_csr_first(matrix, array)
        elif axis == 1:
            if array.shape[0] == 1:
                out[0,:] = cupy_apply_csr_vec(matrix, array[0,:])
            else:
                out[:] = cupy_apply_csr_last(matrix, array)
        else:
            raise ValueError("axis must be 0 or 1 for 2D arrays")
    else:
        # Treat as 3D array with specified axis in the middle
        # Compute equivalent shape (N1, N2, N3)
        if ndim == 3 and axis == 1:
            N1 = array.shape[0]
            N2 = array.shape[1]
            N3 = array.shape[2]
        else:
            N1 = int(np.prod(array.shape[:axis]))
            N2 = array.shape[axis]
            N3 = int(np.prod(array.shape[axis+1:]))
        # Dispatch to cupy routines
        if N1 == 1:
            if N3 == 1:
                # (1, N2, 1) -> (N2,)
                x1 = array.reshape((N2,))
                temp = cupy_apply_csr_vec(matrix, x1)
                out[:] = temp.reshape(out.shape)
            else:
                # (1, N2, N3) -> (N2, N3)
                x2 = array.reshape((N2, N3))
                temp = cupy_apply_csr_first(matrix, x2)
                out[:] = temp.reshape(out.shape)
        else:
            if N3 == 1:
                # (N1, N2, 1) -> (N1, N2)
                x2 = array.reshape((N1, N2))
                temp = cupy_apply_csr_last(matrix, x2)
                out[:] = temp.reshape(out.shape)
            else:
                # (N1, N2, N3)
                x3 = array.reshape((N1, N2, N3))
                y3 = out.reshape(((N1, matrix.shape[0], N3)))
                for n1 in range(N1):
                    y3[n1] = cupy_apply_csr_first(matrix, x3[n1])


def cupy_apply_csr_vec(matrix, vec):
    return matrix.dot(vec)

def cupy_apply_csr_first(matrix, array):
    return matrix.dot(array)

def cupy_apply_csr_last(matrix, array):
    return matrix.dot(array.T).T


