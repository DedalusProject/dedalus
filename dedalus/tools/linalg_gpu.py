"""Linear algebra routines using cupy."""

import numpy as np
import math
try:
    import cupy as cp
    import cupyx.scipy.sparse as csp
    import cupyx.scipy.sparse.linalg as cupy_spla
    cupy_available = True
except ImportError:
    cupy_available = False


def cupy_apply_csr(matrix, array, axis, out):
    """Apply CSR matrix to arbitrary axis of array."""
    if not cupy_available:
        raise ImportError("cupy must be installed to use GPU linear algebra")
    # Check matrix format
    if not isinstance(matrix, csp.csr_matrix):
        # TODO: avoid this explicit conversion
        print('WARNING: converting matrix to CSR format')
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


def custom_spsm(a, b, alpha=1.0, lower=True, unit_diag=False, transa=False, spsm_descr=None):
    """Custom spsm wrapper to save spsm_descr, since spsm_analysis takes lots of time."""
    """Solves a sparse triangular linear system op(a) * x = alpha * op(b).

    Args:
        a (cupyx.scipy.sparse.csr_matrix or cupyx.scipy.sparse.coo_matrix):
            Sparse matrix with dimension ``(M, M)``.
        b (cupy.ndarray): Dense matrix with dimension ``(M, K)``.
        alpha (float or complex): Coefficient.
        lower (bool):
            True: ``a`` is lower triangle matrix.
            False: ``a`` is upper triangle matrix.
        unit_diag (bool):
            True: diagonal part of ``a`` has unit elements.
            False: diagonal part of ``a`` has non-unit elements.
        transa (bool or str): True, False, 'N', 'T' or 'H'.
            'N' or False: op(a) == ``a``.
            'T' or True: op(a) == ``a.T``.
            'H': op(a) == ``a.conj().T``.
    """
    import cupyx
    from cupyx import cusparse
    import cupy as _cupy
    import numpy as _numpy
    from cupy._core import _dtype
    from cupy_backends.cuda.libs import cusparse as _cusparse
    from cupy.cuda import device as _device
    from cupyx.cusparse import SpMatDescriptor, DnMatDescriptor
    if not cusparse.check_availability('spsm'):
        raise RuntimeError('spsm is not available.')

    # Canonicalise transa
    if transa is False:
        transa = 'N'
    elif transa is True:
        transa = 'T'
    elif transa not in 'NTH':
        raise ValueError(f'Unknown transa (actual: {transa})')

    # Check A's type and sparse format
    if cupyx.scipy.sparse.isspmatrix_csr(a):
        pass
    elif cupyx.scipy.sparse.isspmatrix_csc(a):
        if transa == 'N':
            a = a.T
            transa = 'T'
        elif transa == 'T':
            a = a.T
            transa = 'N'
        elif transa == 'H':
            a = a.conj().T
            transa = 'N'
        lower = not lower
    elif cupyx.scipy.sparse.isspmatrix_coo(a):
        pass
    else:
        raise ValueError('a must be CSR, CSC or COO sparse matrix')
    assert a.has_canonical_format

    # Check B's ndim
    if b.ndim == 1:
        is_b_vector = True
        b = b.reshape(-1, 1)
    elif b.ndim == 2:
        is_b_vector = False
    else:
        raise ValueError('b.ndim must be 1 or 2')

    # Check shapes
    if not (a.shape[0] == a.shape[1] == b.shape[0]):
        raise ValueError('mismatched shape')

    # Check dtypes
    dtype = a.dtype
    if dtype.char not in 'fdFD':
        raise TypeError('Invalid dtype (actual: {})'.format(dtype))
    if dtype != b.dtype:
        raise TypeError('dtype mismatch')

    # Prepare fill mode
    if lower is True:
        fill_mode = _cusparse.CUSPARSE_FILL_MODE_LOWER
    elif lower is False:
        fill_mode = _cusparse.CUSPARSE_FILL_MODE_UPPER
    else:
        raise ValueError('Unknown lower (actual: {})'.format(lower))

    # Prepare diag type
    if unit_diag is False:
        diag_type = _cusparse.CUSPARSE_DIAG_TYPE_NON_UNIT
    elif unit_diag is True:
        diag_type = _cusparse.CUSPARSE_DIAG_TYPE_UNIT
    else:
        raise ValueError('Unknown unit_diag (actual: {})'.format(unit_diag))

    # Prepare op_a
    if transa == 'N':
        op_a = _cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE
    elif transa == 'T':
        op_a = _cusparse.CUSPARSE_OPERATION_TRANSPOSE
    else:  # transa == 'H'
        if dtype.char in 'fd':
            op_a = _cusparse.CUSPARSE_OPERATION_TRANSPOSE
        else:
            op_a = _cusparse.CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE

    # Prepare op_b
    if b._f_contiguous:
        op_b = _cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE
    elif b._c_contiguous:
        if _cusparse.get_build_version() < 11701:  # earlier than CUDA 11.6
            raise ValueError('b must be F-contiguous.')
        b = b.T
        op_b = _cusparse.CUSPARSE_OPERATION_TRANSPOSE
    else:
        raise ValueError('b must be F-contiguous or C-contiguous.')

    # Allocate space for matrix C. Note that it is known cusparseSpSM requires
    # the output matrix zero initialized.
    m, _ = a.shape
    if op_b == _cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE:
        _, n = b.shape
    else:
        n, _ = b.shape
    c_shape = m, n
    c = _cupy.zeros(c_shape, dtype=a.dtype, order='f')

    # Prepare descriptors and other parameters
    handle = _device.get_cusparse_handle()
    mat_a = SpMatDescriptor.create(a)
    mat_b = DnMatDescriptor.create(b)
    mat_c = DnMatDescriptor.create(c)
    if spsm_descr is None:
        spsm_descr = _cusparse.spSM_createDescr()
        new_spsm_descr = True
    else:
        spsm_descr, buff = spsm_descr
        new_spsm_descr = False
    alpha = _numpy.array(alpha, dtype=c.dtype).ctypes
    cuda_dtype = _dtype.to_cuda_dtype(c.dtype)
    algo = _cusparse.CUSPARSE_SPSM_ALG_DEFAULT

    try:
        # Specify Lower|Upper fill mode
        mat_a.set_attribute(_cusparse.CUSPARSE_SPMAT_FILL_MODE, fill_mode)

        # Specify Unit|Non-Unit diagonal type
        mat_a.set_attribute(_cusparse.CUSPARSE_SPMAT_DIAG_TYPE, diag_type)

        # Allocate the workspace needed by the succeeding phases
        if new_spsm_descr:
            buff_size = _cusparse.spSM_bufferSize(
                handle, op_a, op_b, alpha.data, mat_a.desc, mat_b.desc,
                mat_c.desc, cuda_dtype, algo, spsm_descr)
            buff = _cupy.empty(buff_size, dtype=_cupy.int8)

        # Perform the analysis phase
        if new_spsm_descr:
            _cusparse.spSM_analysis(
                handle, op_a, op_b, alpha.data, mat_a.desc, mat_b.desc,
                mat_c.desc, cuda_dtype, algo, spsm_descr, buff.data.ptr)

        # Executes the solve phase
        _cusparse.spSM_solve(
            handle, op_a, op_b, alpha.data, mat_a.desc, mat_b.desc,
            mat_c.desc, cuda_dtype, algo, spsm_descr, buff.data.ptr)

        # Reshape back if B was a vector
        if is_b_vector:
            c = c.reshape(-1)

        return c, (spsm_descr, buff)

    finally:
        # Destroy matrix/vector descriptors
        #_cusparse.spSM_destroyDescr(spsm_descr)
        pass


def custom_SuperLU_solve(self, rhs, trans='N', spsm_descr=None):
    """Custom SuperLU solve wrapper to save spsm_descr, since spsm_analysis takes lots of time."""
    """Solves linear system of equations with one or several right-hand sides.

    Args:
        rhs (cupy.ndarray): Right-hand side(s) of equation with dimension
            ``(M)`` or ``(M, K)``.
        trans (str): 'N', 'T' or 'H'.
            'N': Solves ``A * x = rhs``.
            'T': Solves ``A.T * x = rhs``.
            'H': Solves ``A.conj().T * x = rhs``.

    Returns:
        cupy.ndarray:
            Solution vector(s)
    """  # NOQA
    from cupyx import cusparse
    import cupy
    from cupyx.scipy.sparse.linalg._solve import _should_use_spsm

    if not isinstance(rhs, cupy.ndarray):
        raise TypeError('ojb must be cupy.ndarray')
    if rhs.ndim not in (1, 2):
        raise ValueError('rhs.ndim must be 1 or 2 (actual: {})'.
                            format(rhs.ndim))
    if rhs.shape[0] != self.shape[0]:
        raise ValueError('shape mismatch (self.shape: {}, rhs.shape: {})'
                            .format(self.shape, rhs.shape))
    if trans not in ('N', 'T', 'H'):
        raise ValueError('trans must be \'N\', \'T\', or \'H\'')

    if cusparse.check_availability('spsm') and _should_use_spsm(rhs):
        def spsm(A, B, lower, transa, spsm_descr):
            return custom_spsm(A, B, lower=lower, transa=transa, spsm_descr=spsm_descr)
        sm = spsm
    else:
        raise NotImplementedError

    x = rhs.astype(self.L.dtype)
    if trans == 'N':
        if self.perm_r is not None:
            if x.ndim == 2 and x._f_contiguous:
                x = x.T[:, self._perm_r_rev].T  # want to keep f-order
            else:
                x = x[self._perm_r_rev]
        x, self.spsm_L_descr = sm(self.L, x, lower=True, transa=trans, spsm_descr=self.spsm_L_descr)
        x, self.spsm_U_descr = sm(self.U, x, lower=False, transa=trans, spsm_descr=self.spsm_U_descr)
        if self.perm_c is not None:
            x = x[self.perm_c]
    else:
        if self.perm_c is not None:
            if x.ndim == 2 and x._f_contiguous:
                x = x.T[:, self._perm_c_rev].T  # want to keep f-order
            else:
                x = x[self._perm_c_rev]
        x, self.spsm_U_descr = sm(self.U, x, lower=False, transa=trans, spsm_descr=self.spsm_U_descr)
        x, self.spsm_L_descr = sm(self.L, x, lower=True, transa=trans, spsm_descr=self.spsm_L_descr)
        if self.perm_r is not None:
            x = x[self.perm_r]

    if not x._f_contiguous:
        # For compatibility with SciPy
        x = x.copy(order='F')
    return x


class CustomCupyUpperTriangularSolver:
    """Hacky class to save spsm_descr for reuse in spsm for triangular solves."""

    def __init__(self, matrix):
        # Check matrix format
        if not isinstance(matrix, csp.csr_matrix):
            # TODO: avoid this explicit conversion
            matrix = csp.csr_matrix(matrix)
            print('WARNING: converting matrix to CSR format')
            #raise ValueError("Matrix must be in CSR format.")
        self.matrix = matrix
        self.spsm_descr = None

    def solve(self, b, lower=True, overwrite_A=False, overwrite_b=False,
                        unit_diagonal=False):
        """Solves a sparse triangular system ``A x = b``.

        Args:
            A (cupyx.scipy.sparse.spmatrix):
                Sparse matrix with dimension ``(M, M)``.
            b (cupy.ndarray):
                Dense vector or matrix with dimension ``(M)`` or ``(M, K)``.
            lower (bool):
                Whether ``A`` is a lower or upper triangular matrix.
                If True, it is lower triangular, otherwise, upper triangular.
            overwrite_A (bool):
                (not supported)
            overwrite_b (bool):
                Allows overwriting data in ``b``.
            unit_diagonal (bool):
                If True, diagonal elements of ``A`` are assumed to be 1 and will
                not be referenced.

        Returns:
            cupy.ndarray:
                Solution to the system ``A x = b``. The shape is the same as ``b``.
        """
        from cupyx import cusparse
        from cupyx.scipy import sparse
        import cupy
        from cupyx.scipy.sparse.linalg._solve import _should_use_spsm

        A = self.matrix

        if not (cusparse.check_availability('spsm') or
                cusparse.check_availability('csrsm2')):
            raise NotImplementedError

        if not sparse.isspmatrix(A):
            raise TypeError('A must be cupyx.scipy.sparse.spmatrix')
        if not isinstance(b, cupy.ndarray):
            raise TypeError('b must be cupy.ndarray')
        if A.shape[0] != A.shape[1]:
            raise ValueError(f'A must be a square matrix (A.shape: {A.shape})')
        if b.ndim not in [1, 2]:
            raise ValueError(f'b must be 1D or 2D array (b.shape: {b.shape})')
        if A.shape[0] != b.shape[0]:
            raise ValueError('The size of dimensions of A must be equal to the '
                            'size of the first dimension of b '
                            f'(A.shape: {A.shape}, b.shape: {b.shape})')
        if A.dtype.char not in 'fdFD':
            raise TypeError(f'unsupported dtype (actual: {A.dtype})')

        if cusparse.check_availability('spsm') and _should_use_spsm(b):
            if not (sparse.isspmatrix_csr(A) or
                    sparse.isspmatrix_csc(A) or
                    sparse.isspmatrix_coo(A)):
                warnings.warn('CSR, CSC or COO format is required. Converting to '
                            'CSR format.', sparse.SparseEfficiencyWarning)
                A = A.tocsr()
            A.sum_duplicates()
            x, self.spsm_descr = custom_spsm(A, b, lower=lower, unit_diag=unit_diagonal, spsm_descr=self.spsm_descr)
        elif cusparse.check_availability('csrsm2'):
            if not (sparse.isspmatrix_csr(A) or sparse.isspmatrix_csc(A)):
                warnings.warn('CSR or CSC format is required. Converting to CSR '
                            'format.', sparse.SparseEfficiencyWarning)
                A = A.tocsr()
            A.sum_duplicates()

            if (overwrite_b and A.dtype == b.dtype and
                    (b._c_contiguous or b._f_contiguous)):
                x = b
            else:
                x = b.astype(A.dtype, copy=True)

            cusparse.csrsm2(A, x, lower=lower, unit_diag=unit_diagonal)
        else:
            assert False

        if x.dtype.char in 'fF':
            # Note: This is for compatibility with SciPy.
            dtype = numpy.promote_types(x.dtype, 'float64')
            x = x.astype(dtype)
        return x


def cupy_solve_upper_csr(matrix, array, axis, out):
    """Solve upper triangular CSR matrix along specified axis of an array."""
    # Switch by dimension
    ndim = array.ndim
    if ndim == 1:
        if axis == 0:
            cupy_solve_upper_csr_vec(matrix, array, out)
        else:
            raise ValueError("axis must be 0 for 1D arrays")
    elif ndim == 2:
        if axis == 0:
            if array.shape[1] == 1:
                cupy_solve_upper_csr_vec(matrix, array[:,0], out[:,0])
            else:
                cupy_solve_upper_csr_first(matrix, array, out)
        elif axis == 1:
            if array.shape[0] == 1:
                cupy_solve_upper_csr_vec(matrix, array[0,:], out[0,:])
            else:
                cupy_solve_upper_csr_last(matrix, array, out)
        else:
            raise ValueError("axis must be 0 or 1 for 2D arrays")
    else:
        # Treat as 3D array with specified axis in the middle
        # Compute equivalent shape (N1, N2, N3)
        if ndim == 3 and axis == 1:
            N1 = shape[0]
            N2 = shape[1]
            N3 = shape[2]
        else:
            N1 = int(np.prod(array.shape[:axis]))
            N2 = array.shape[axis]
            N3 = int(np.prod(array.shape[axis+1:]))
        # Dispatch to cupy routines
        if N1 == 1:
            if N3 == 1:
                # (1, N2, 1) -> (N2,)
                x1 = array.reshape((N2,))
                y1 = out.reshape((N2,))
                cupy_solve_upper_csr_vec(matrix, x1, y1)
            else:
                # (1, N2, N3) -> (N2, N3)
                x2 = array.reshape((N2, N3))
                y2 = out.reshape((N2, N3))
                cupy_solve_upper_csr_first(matrix, x2, y2)
        else:
            if N3 == 1:
                # (N1, N2, 1) -> (N1, N2)
                x2 = array.reshape((N1, N2))
                y2 = out.reshape((N1, N2))
                cupy_solve_upper_csr_last(matrix, x2, y2)
            else:
                # (N1, N2, N3)
                x3 = array.reshape((N1, N2, N3))
                y3 = out.reshape((N1, N2, N3))
                cupy_solve_upper_csr_mid(matrix, x3, y3)


def cupy_solve_upper_csr_vec(matrix, vec, out):
    """Solve upper triangular CSR matrix along a vector."""
    out[:] = matrix.solve(vec, lower=False)


def cupy_solve_upper_csr_first(matrix, array, out):
    """Solve upper triangular CSR matrix along first axis of 2D array."""
    out[:] = matrix.solve(array, lower=False)


def cupy_solve_upper_csr_last(matrix, array, out):
    """Solve upper triangular CSR matrix along last axis of 2D array."""
    out.T[:] = matrix.solve(array.T, lower=False)


def cupy_solve_upper_csr_mid(matrix, array, out):
    """Solve upper triangular CSR matrix along middle axis of 3D array."""
    raise NotImplementedError
