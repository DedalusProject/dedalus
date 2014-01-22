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

    return slicelist


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

