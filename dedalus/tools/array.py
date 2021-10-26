"""
Tools for array manipulations.

"""

import numpy as np
from scipy import sparse


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


def axslice(axis, start, stop, step=None):
    """Slice array along a specified axis."""
    if axis < 0:
        raise ValueError("`axis` must be positive")
    slicelist = [slice(None)] * axis
    slicelist.append(slice(start, stop, step))
    return tuple(slicelist)


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


def move_single_axis(a, source, destination):
    """Similar to np.moveaxis but faster for just a single axis."""
    order = [n for n in range(a.ndim) if n != source]
    order.insert(destination, source)
    return a.transpose(order)


def apply_sparse(matrix, array, axis, out=None):
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
