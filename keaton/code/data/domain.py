

import numpy as np


class Domain(object):
    """Problem domain."""

    def __init__(self, bases):

        # Inputs
        self.bases = bases

        # Build shape
        self.shape = [b.size for b in bases]
        self.dim = len(self.shape)

        # Grid
        self.grids = [b.grid for b in bases]
        if self.dim > 1:
            self.grids = np.meshgrid(*self.grids, indexing='ij')

        # Pencil slices
        num_pencils = np.prod(self.shape) / self.shape[-1]
        self.slices = [slice(None)]

