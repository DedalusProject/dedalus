"""Tools for random array generation."""

import numpy as np
from numpy.random import default_rng


def chunked_rng(seed, chunk_size, distribution, **kw):
    """RNG stream as a chunked generator."""
    rng = default_rng(seed)
    distribution = getattr(rng, distribution)
    # Yield from distribution in chunks
    chunk_index = 0
    while True:
        chunk_data = distribution(size=chunk_size, **kw)
        yield chunk_index, chunk_data
        chunk_index += 1


def rng_element(index, seed, chunk_size, distribution, **kw):
    """Get a specified element from a RNG stream."""
    # Trim chunk size if index is in first chunk
    chunk_size = min(1 + index, chunk_size)
    # Iterate over RNG until reaching proper chunk
    rng = chunked_rng(seed, chunk_size, distribution, **kw)
    div, mod = divmod(index, chunk_size)
    for chunk, data in rng:
        if chunk == div:
            break
    return data[mod]


def rng_elements(indices, seed, chunk_size, distribution, **kw):
    """Get an array of elements from a RNG stream."""
    indices = np.array(indices)
    # Early quit on empty samples
    if indices.size == 0:
        data = rng_element(0, seed, chunk_size, distribution, **kw)
        return np.zeros(indices.shape, dtype=data.dtype)
    # Trim chunk size if all indices are in first chunk
    chunk_size = min(1 + indices.max(), chunk_size)
    # Iterate over RNG until reaching all chunks
    rng = chunked_rng(seed, chunk_size, distribution, **kw)
    div, mod = np.divmod(indices, chunk_size)
    div_set = set(np.unique(div))
    max_div = max(div_set)
    for chunk, data in rng:
        if chunk == 0:
            values = np.zeros(indices.shape, dtype=data.dtype)
        if chunk in div_set:
            selected = (chunk == div)
            values[selected] = data[mod[selected]]
        if chunk == max_div:
            break
    return values


class IndexArray:
    """
    mgrid-like object that returns an array of flat indices when sliced.

    Parameters
    ----------
    shape : tuple of ints
        Shape of array.
    order : {'C', 'F'}, optional
        Determines whether the returned flat indices should correspond to
        a row-major (C-style) or column-major (Fortran-style) order.
    """

    def __init__(self, shape, order='C'):
        self.shape = shape
        self.ndim = len(shape)
        self.order = order

    def __getitem__(self, key):
        # Cast single selection to tuple
        if not isinstance(key, tuple):
            key = (key,)
        # Check length
        if len(key) < self.ndim:
            # Add trailing empty slices
            key = key + (slice(None),) * (self.ndim - len(key))
        elif len(key) > self.ndim:
            raise ValueError("Too many selections.")
        # Concretize slices using specified shape
        key = tuple(slice(*k.indices(s)) for k, s in zip(key, self.shape))
        # Get indices via mgrid
        indices = np.mgrid.__getitem__(key)
        indices = np.ravel_multi_index(indices, self.shape, order=self.order)
        return indices


class ChunkedRandomArray(IndexArray):
    """
    Random array with elements that are deterministically produced when indexed.
    This allows robustly constructing and selecting from a random array without
    requiring enough local memory to hold the entire array.

    Parameters
    ----------
    shape : tuple of ints
        Shape of array.
    seed : int, optional
        RNG seed. Default: None.
    chunk_size : int, optional
        Chunk size for drawing from distribution. Should be less than locally
        available memory. Default: 2**20, corresponding to 8 MB of float64.
    distribution : str, optional
        Distribution name, corresponding to numpy random Generator method.
        Default: 'uniform'.
    **kw : dict
        Other keywords passed to the distribution method.
    """

    def __init__(self, shape, seed=None, chunk_size=2**20, distribution='uniform', **kw):
        super().__init__(shape)
        self.seed = seed
        self.chunk_size = chunk_size
        self.distribution = distribution
        self.kw = kw

    def __getitem__(self, key):
        indices = super().__getitem__(key)
        return rng_elements(indices, self.seed, self.chunk_size, self.distribution, **self.kw)

