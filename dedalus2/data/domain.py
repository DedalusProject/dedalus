

import numpy as np


class Domain:
    """Problem domain."""

    def __init__(self, bases, dtype=np.complex128):

        # Inputs
        self.bases = bases
        self.dtype = dtype

        # Iteratively set basis data types
        for b in self.bases:
            dtype = b.set_dtype(dtype)

        # Dimension
        self.dim = len(bases)




        # Build shape
        self.shape = [b.grid_size for b in bases]
        self.dim = len(self.shape)

        # Grid
        self.grids = [b.grid for b in bases]
        if self.dim > 1:
            self.grids = np.meshgrid(*self.grids, indexing='ij')

        # Pencil slices
        num_pencils = np.prod(self.shape) / self.shape[-1]
        self.slices = [slice(None)]

