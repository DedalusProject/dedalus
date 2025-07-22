"""Linear algebra routines using cupy."""

import numpy as np
import math
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
        # TODO: avoid this explicit conversion
        matrix = csp.csr_matrix(matrix)
        #raise ValueError("Matrix must be in CSR format.")
    # Switch by dimension
    ndim = array.ndim
    if ndim == 1:
        if axis == 0:
            out[:] = matrix.dot(array)
        else:
            raise ValueError("axis must be 0 for 1D arrays")
    elif ndim == 2:
        if axis == 0:
            if array.shape[1] == 1:
                out[:,0] = matrix.dot(array[:,0])
            else:
                out[:] = matrix.dot(array)
        elif axis == 1:
            if array.shape[0] == 1:
                out[0,:] = matrix.dot(array[0,:])
            else:
                out[:] = matrix.dot(array.T).T
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
                temp = matrix.dot(x1)
                out[:] = temp.reshape(out.shape)
            else:
                # (1, N2, N3) -> (N2, N3)
                x2 = array.reshape((N2, N3))
                temp = matrix.dot(x2)
                out[:] = temp.reshape(out.shape)
        else:
            if N3 == 1:
                # (N1, N2, 1) -> (N1, N2)
                x2 = array.reshape((N1, N2))
                temp = matrix.dot(x2.T).T
                out[:] = temp.reshape(out.shape)
            else:
                # (N1, N2, N3)
                x3 = array.reshape((N1, N2, N3))
                y3 = out.reshape(((N1, matrix.shape[0], N3)))
                cupy_apply_csr_mid(matrix, x3, y3)


# Kernel for applying CSR matrix with parallelization over n1 and n3
apply_csr_mid_kernel = cp.RawKernel(
    r'''
    extern "C" __global__ void apply_csr_mid_kernel(
        const double* data,     // CSR data of shape (nnz,)
        const int* indices,    // CSR column indices (nnz,)
        const int* indptr,     // CSR row pointers (N2o + 1,)
        const double* input,    // shape (N1, N2i, N3)
        double* output,         // shape (N1, N2o, N3)
        int N1, int N2i, int N2o, int N3)
    {
        int n1 = blockIdx.x * blockDim.x + threadIdx.x ;  // batch index
        int n3 = blockIdx.y * blockDim.y + threadIdx.y;  // output column index

        if (n1 >= N1 || n3 >= N3) return;

        // Loop over output rows = CSR matrix rows
        for (int i = 0; i < N2o; ++i) {
            double acc = 0;
            int start = indptr[i];
            int end   = indptr[i + 1];

            for (int k = start; k < end; ++k) {
                int j = indices[k];  // input column
                double val = data[k];
                acc += val * input[n1 * N2i * N3 + j * N3 + n3];
            }

            output[n1 * N2o * N3 + i * N3 + n3] = acc;
        }
    }
    ''',
    'apply_csr_mid_kernel')


def cupy_apply_csr_mid(matrix, array, out):
    N1, N2i, N3 = array.shape
    N2o = matrix.shape[0]
    # Choose thread/block config
    threads_y = min(1024, N3) # maximize concurrency along n3
    threads_x = 1024 // threads_y # make block have 1024 threads
    blockdim = (threads_x, threads_y)
    blocks_x = (N1 + threads_x - 1) // threads_x
    blocks_y = (N3 + threads_y - 1) // threads_y
    griddim = (blocks_x, blocks_y)
    # Launch kernel
    apply_csr_mid_kernel(griddim, blockdim, (matrix.data, matrix.indices, matrix.indptr, array, out, N1, N2i, N2o, N3))

