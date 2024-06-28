"""Tools for array manipulations."""

import numpy as np
from scipy import sparse
import scipy.sparse as sp
from scipy.sparse import _sparsetools
from scipy.sparse import linalg as spla
from math import prod

from .config import config
from . import linalg as cython_linalg

SPLIT_CSR_MATVECS = config['linear algebra'].getboolean('SPLIT_CSR_MATVECS')
OLD_CSR_MATVECS = config['linear algebra'].getboolean('OLD_CSR_MATVECS')


def interleaved_view(data):
    """
    View n-dim complex array as (n+1)-dim real array, where the last axis
    separates real and imaginary parts.
    """
    # Check datatype
    if data.dtype != np.complex128:
        raise ValueError("Complex array required.")
    # Create view array
    iv_shape = data.shape + (2,)
    iv = np.ndarray(iv_shape, dtype=np.float64, buffer=data.data)
    return iv


def reshape_vector(data, dim=2, axis=-1):
    """Reshape 1-dim array as a multidimensional vector."""
    # Build multidimensional shape
    shape = [1] * dim
    shape[axis] = data.size
    return data.reshape(shape)


def axindex(axis, index):
    """Index array along specified axis."""
    if axis < 0:
        raise ValueError("`axis` must be positive")
    # Add empty slices for leading axes
    return (slice(None),)*axis + (index,)


def axslice(axis, start, stop, step=None):
    """Slice array along a specified axis."""
    return axindex(axis, slice(start, stop, step))


def zeros_with_pattern(*args):
    """Create sparse matrix with the combined pattern of other sparse matrices."""
    # Join individual patterns in COO format
    coo = [A.tocoo() for A in args]
    rows = np.concatenate([A.row for A in coo])
    cols = np.concatenate([A.col for A in coo])
    shape = coo[0].shape
    # Create new COO matrix with zeroed data and combined pattern
    data = np.concatenate([A.data*0 for A in coo])
    return sparse.coo_matrix((data, (rows, cols)), shape=shape)


def expand_pattern(input, pattern):
    """Return copy of sparse matrix with extended pattern."""
    # Join input and pattern in COO format
    A = input.tocoo()
    P = pattern.tocoo()
    rows = np.concatenate((A.row, P.row))
    cols = np.concatenate((A.col, P.col))
    shape = A.shape
    # Create new COO matrix with expanded data and combined pattern
    data = np.concatenate((A.data, P.data*0))
    return sparse.coo_matrix((data, (rows, cols)), shape=shape)


def apply_matrix(matrix, array, axis, **kw):
    """Apply matrix along any axis of an array."""
    if sparse.isspmatrix(matrix):
        return apply_sparse(matrix, array, axis, **kw)
    else:
        return apply_dense(matrix, array, axis, **kw)


def apply_dense_einsum(matrix, array, axis, optimize=True, **kw):
    """Apply dense matrix along any axis of an array."""
    dim = len(array.shape)
    # Build Einstein signatures
    mat_sig = [dim, axis]
    arr_sig = list(range(dim))
    out_sig = list(range(dim))
    out_sig[axis] = dim
    out = np.einsum(matrix, mat_sig, array, arr_sig, out_sig, optimize=optimize, **kw)
    return out


def move_single_axis(a, source, destination):
    """Similar to np.moveaxis but faster for just a single axis."""
    order = [n for n in range(a.ndim) if n != source]
    order.insert(destination, source)
    return a.transpose(order)


def apply_dense(matrix, array, axis, out=None):
    """Apply dense matrix along any axis of an array."""
    dim = array.ndim
    # Resolve wraparound axis
    axis = axis % dim
    # Move axis to 0
    if axis != 0:
        array = move_single_axis(array, axis, 0) # May allocate copy
    # Flatten later axes
    if dim > 2:
        array_shape = array.shape
        array = array.reshape((array_shape[0], -1)) # May allocate copy
    # Apply matmul
    temp = np.matmul(matrix, array) # Allocates temp
    # Unflatten later axes
    if dim > 2:
        temp = temp.reshape((temp.shape[0],) + array_shape[1:]) # View
    # Move axis back from 0
    if axis != 0:
        temp = move_single_axis(temp, 0, axis) # View
    # Return
    if out is None:
        return temp
    else:
        out[:] = temp # Copy
        return out


def splu_inverse(matrix, permc_spec="NATURAL", **kw):
    """Create LinearOperator implicitly acting as a sparse matrix inverse."""
    splu = spla.splu(matrix.tocsc(), permc_spec=permc_spec, **kw)
    def solve(x):
        if np.iscomplexobj(x) and matrix.dtype == np.float64:
            return splu.solve(x.real) + 1j*splu.solve(x.imag)
        else:
            return splu.solve(x)
    return spla.LinearOperator(shape=matrix.shape, dtype=matrix.dtype, matvec=solve, matmat=solve)


def apply_sparse_dot(matrix, array, axis, out=None):
    """Apply sparse matrix along any axis of an array."""
    dim = array.ndim
    # Resolve wraparound axis
    axis = axis % dim
    # Move axis to 0
    if axis != 0:
        array = move_single_axis(array, axis, 0) # May allocate copy
    # Flatten later axes
    if dim > 2:
        array_shape = array.shape
        array = array.reshape((array_shape[0], -1)) # May allocate copy
    # Apply matmul
    temp = matrix.dot(array) # Allocates temp
    # Unflatten later axes
    if dim > 2:
        temp = temp.reshape((temp.shape[0],) + array_shape[1:]) # View
    # Move axis back from 0
    if axis != 0:
        temp = move_single_axis(temp, 0, axis) # View
    # Return
    if out is None:
        return temp
    else:
        out[:] = temp # Copy
        return out


def apply_sparse(matrix, array, axis, out=None, check_shapes=False, num_threads=1):
    """
    Apply sparse matrix along any axis of an array.
    Must be out of place if ouptut is specified.
    """
    # Check matrix
    if not isinstance(matrix, sparse.csr_matrix):
        raise ValueError("Matrix must be in CSR format.")
    # Check output
    if out is None:
        out_shape = list(array.shape)
        out_shape[axis] = matrix.shape[0]
        out = np.empty(out_shape, dtype=array.dtype)
    elif out is array:
        raise ValueError("Cannot apply in place")
    # Check shapes
    if check_shapes:
        if not (0 <= axis < array.ndim):
            raise ValueError("Axis out of bounds.")
        if matrix.shape[1] != array.shape[axis] or matrix.shape[0] != out.shape[axis]:
            raise ValueError("Matrix shape mismatch.")
    # Old way if requested
    if OLD_CSR_MATVECS and array.ndim == 2 and axis == 0:
        out.fill(0)
        return csr_matvecs(matrix, array, out)
    # Promote datatypes
    # TODO: find way to optimize this with fused types
    matrix_data = matrix.data
    if matrix_data.dtype != out.dtype:
        matrix_data = matrix_data.astype(out.dtype)
    # Call cython routine
    cython_linalg.apply_csr(matrix.indptr, matrix.indices, matrix_data, array, out, axis, num_threads)
    return out


def solve_upper_sparse(matrix, rhs, axis, out=None, check_shapes=False, num_threads=1):
    """
    Solve upper triangular sparse matrix along any axis of an array.
    Matrix assumed to be nonzero on the diagonals.
    """
    # Check matrix
    if not isinstance(matrix, sparse.csr_matrix):
        raise ValueError("Matrix must be in CSR format.")
    if not matrix._has_canonical_format: # avoid property hook (without underscore)
        matrix.sum_duplicates()
    # Setup output = rhs
    if out is None:
        out = np.copy(rhs)
    elif out is not rhs:
        np.copyto(out, rhs)
    # Promote datatypes
    matrix_data = matrix.data
    if matrix_data.dtype != rhs.dtype:
        matrix_data = matrix_data.astype(rhs.dtype)
    # Check shapes
    if check_shapes:
        if not (0 <= axis < rhs.ndim):
            raise ValueError("Axis out of bounds.")
        if not (matrix.shape[0] == matrix.shape[1] == rhs.shape[axis]):
            raise ValueError("Matrix shape mismatch.")
    # Call cython routine
    cython_linalg.solve_upper_csr(matrix.indptr, matrix.indices, matrix_data, out, axis, num_threads)


def csr_matvec(A_csr, x_vec, out_vec):
    """
    Fast CSR matvec with dense vector skipping output allocation. The result is
    added to the specificed output array, so the output should be manually
    zeroed prior to calling this routine, if necessary.
    """
    # Check format but don't convert
    if A_csr.format != "csr":
        raise ValueError("Matrix must be in CSR format.")
    # Check shapes
    M, N = A_csr.shape
    m, n = out_vec.size, x_vec.size
    if x_vec.ndim > 1 or out_vec.ndim > 1:
        raise ValueError("Only vectors allowed for input and output.")
    if M != m or N != n:
        raise ValueError(f"Matrix shape {(M,N)} does not match input {(n,)} and output {(m,)} shapes.")
    # Apply matvec
    _sparsetools.csr_matvec(M, N, A_csr.indptr, A_csr.indices, A_csr.data, x_vec, out_vec)
    return out_vec


def csr_matvecs(A_csr, x_vec, out_vec):
    """
    Fast CSR matvec with dense vector skipping output allocation. The result is
    added to the specificed output array, so the output should be manually
    zeroed prior to calling this routine, if necessary.
    """
    # Check format but don't convert
    if A_csr.format != "csr":
        raise ValueError("Matrix must be in CSR format.")
    # Check shapes
    M, N = A_csr.shape
    if x_vec.ndim != 2 or out_vec.ndim != 2:
        raise ValueError("Only matrices allowed for input and output.")
    n, kx = x_vec.shape
    m, ko = out_vec.shape
    if M != m or N != n:
        raise ValueError(f"Matrix shape {(M,N)} does not match input {(n,)} and output {(m,)} shapes.")
    if kx != ko:
        raise ValueError("Output size does not match input size.")
    # Apply matvecs
    if SPLIT_CSR_MATVECS:
        for k in range(kx):
            _sparsetools.csr_matvec(M, N, A_csr.indptr, A_csr.indices, A_csr.data, x_vec[:,k], out_vec[:,k])
    else:
        _sparsetools.csr_matvecs(M, N, kx, A_csr.indptr, A_csr.indices, A_csr.data, x_vec, out_vec)
    return out_vec


def add_sparse(A, B):
    """Add sparse matrices, promoting scalars to multiples of the identity."""
    A_is_scalar = np.isscalar(A)
    B_is_scalar = np.isscalar(B)
    if A_is_scalar and B_is_scalar:
        return A + B
    elif A_is_scalar:
        I = sparse.eye(*B.shape, dtype=B.dtype, format=B.format)
        return A*I + B
    elif B_is_scalar:
        I = sparse.eye(*A.shape, dtype=A.dtype, format=A.format)
        return A + B*I
    else:
        return A + B


def sparse_block_diag(blocks, shape=None):
    """Build a block diagonal sparse matrix allowing size 0 matrices."""
    # Collect subblocks
    data, rows, cols = [], [], []
    i0, j0 = 0, 0
    for block in blocks:
        block = sparse.coo_matrix(block)
        if block.nnz > 0:
            data.append(block.data)
            rows.append(block.row + i0)
            cols.append(block.col + j0)
        i0 += block.shape[0]
        j0 += block.shape[1]
    # Build full matrix
    if shape is None:
        shape = (i0, j0)
    if data:
        data = np.concatenate(data)
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        return sparse.coo_matrix((data, (rows, cols)), shape=shape).tocsr()
    else:
        return sparse.csr_matrix(shape)


def kron(*factors):
    if factors:
        out = factors[0]
        for f in factors[1:]:
            out = np.kron(out, f)
    else:
        out = np.identity(1)
    return out


def nkron(factor, n):
    return kron(*[factor for i in range(n)])


def permute_axis(array, axis, permutation, out=None):
    # OPTIMIZE: currently creates a temporary
    slices = [slice(None) for i in array.shape]
    slices[axis] = permutation
    perm = array[tuple(slices)]
    if out is None:
        return perm
    else:
        np.copyto(out, perm)
        return out


def copyto(dest, src):
    # Seems to be faster than np.copyto
    dest[:] = src


def perm_matrix(perm, M=None, source_index=False, sparse=True):
    """
    Build sparse permutation matrix from permutation vector.

    Parameters
    ----------
    perm : ndarray
        Permutation vector.
    M : int, optional
        Output dimension. Default: len(perm).
    source_index : bool, optional
        False (default) if perm entries indicate destination index:
            output[i,j] = (i == perm[j])
        True if perm entires indicate source index:
            output[i,j] = (j == perm[i])
    sparse : bool, optional
        Whether to return sparse matrix or dense array (default: True).
    """
    N = len(perm)
    if M is None:
        M = N
    if source_index:
        row = np.arange(N)
        col = np.array(perm)
    else:
        row = np.array(perm)
        col = np.arange(N)
    if sparse:
        data = np.ones(N, dtype=int)
        return sp.coo_matrix((data, (row, col)), shape=(M, N))
    else:
        output = np.zeros((M, N), dtype=int)
        output[row, col] = 1
        return output


def drop_empty_rows(mat):
    mat = sparse.csr_matrix(mat)
    nonempty_rows = (np.diff(mat.indptr) > 0)
    return mat[nonempty_rows]


def scipy_sparse_eigs(A, B, left, N, target, matsolver, **kw):
    """
    Perform targeted eigenmode search using the scipy/ARPACK sparse solver
    for the reformulated generalized eigenvalue problem

        A.x = λ B.x  ==>  (A - σB)^I B.x = (1/(λ-σ)) x

    for eigenvalues λ near the target σ.

    Parameters
    ----------
    A, B : scipy sparse matrices
        Sparse matrices for generalized eigenvalue problem
    N : int
        Number of eigenmodes to return
    left: boolean
        Whether to solve for the left eigenvectors or not
    target : complex
        Target σ for eigenvalue search
    matsolver : matrix solver class
        Class implementing solve method for solving sparse systems.

    Other keyword options passed to scipy.sparse.linalg.eigs.
    """
    # Build sparse linear operator representing (A - σB)^I B = C^I B = D
    C = A - target * B
    solver = matsolver(C)
    def matvec(x):
        return solver.solve(B.dot(x))
    D = spla.LinearOperator(dtype=A.dtype, shape=A.shape, matvec=matvec)
    # Solve using scipy sparse algorithm
    evals, evecs = spla.eigs(D, k=N, which='LM', sigma=None, **kw)
    # Rectify eigenvalues
    evals = 1 / evals + target
    if left:
        # Build sparse linear operator representing (A^H - conj(σ)B^H)^I B^H = C^-H B^H = left_D
        # Note: left_D is not equal to D^H
        def matvec_left(x):
            return solver.solve_H(B.conj().T.dot(x))
        left_D = spla.LinearOperator(dtype=A.dtype, shape=A.shape, matvec=matvec_left)
        # Solve using scipy sparse algorithm
        left_evals, left_evecs = spla.eigs(left_D, k=N, which='LM', sigma=None, **kw)
        # Rectify left eigenvalues
        left_evals = 1 / left_evals + np.conj(target)
        return evals, evecs, left_evals, left_evecs
    else:
        return evals, evecs


def interleave_matrices(matrices):
    N = len(matrices)
    if N == 1:
        return matrices[0]
    sum = 0
    P = sparse.lil_matrix((N, N))
    for i, matrix in enumerate(matrices):
        P[i, i] = 1
        sum += sparse.kron(matrix, P)
        P[i, i] = 0
    return sum


def sparse_allclose(A, B):
    A = A.tocsr()
    B = B.tocsr()
    return (np.allclose(A.data, B.data) and
            np.allclose(A.indices, B.indices) and
            np.allclose(A.indptr, B.indptr))

def assert_sparse_pinv(A, B):
    if not sparse_allclose(A @ B @ A, A):
        raise AssertionError("Not a pseudoinverse")
    if not sparse_allclose(B @ A @ B, B):
        raise AssertionError("Not a pseudoinverse")
    if not sparse_allclose((A @ B).conj().T, A @ B):
        raise AssertionError("Not a pseudoinverse")
    if not sparse_allclose((B @ A).conj().T, B @ A):
        raise AssertionError("Not a pseudoinverse")

