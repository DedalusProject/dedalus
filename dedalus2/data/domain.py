

import numpy as np

from .distributor import Distributor
from .field import Field
from .pencil import Pencil
from ..tools.cache import CachedMethod
from ..tools.array import reshape_vector


class Domain:
    """Global domain composed of orthogonal bases."""

    def __init__(self, bases, grid_dtype=np.complex128, mesh=None):

        # Initial attributes
        self.bases = bases
        self.dim = len(bases)
        self.grid_dtype = grid_dtype

        # Iteratively set basis data types
        # (Transforms from grid-to-coeff proceed in the listed order)
        for b in self.bases:
            grid_dtype = b.set_dtypes(grid_dtype)

        # Field management
        self._field_list = list()
        self._field_count = 0

        # Create distributor
        self.distributor = Distributor(self, mesh)

    @CachedMethod
    def grid(self, axis):

        # Get local part of grid
        grid = self.bases[axis].grid
        if not self.distributor.grid_layout.local[axis]:
            start = self.distributor.grid_layout.start[axis]
            size = self.distributor.grid_layout.shape[axis]
            grid = grid[start:start+size]

        # Reshape as multidimensional vector
        grid = reshape_vector(grid, self.dim, axis)

        return grid

    def _collect_field(self, field):

        # Clean field
        field.layout = self.distributor.grid_layout
        field.data.fill(0)

        # Add to field list
        self._field_list.append(field)

    def new_field(self):

        # Return a free field if available
        if self._field_list:
            field = self._field_list.pop()
        # Otherwise instantiate another field
        else:
            field = Field(self)

        return field

