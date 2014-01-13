"""
Tools for array manipulations.

"""

import numpy as np


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

