"""
Class for problem domain.

"""

import numpy as np

from .distributor import Distributor
from .field import Field
from .operators import create_diff_operator
from ..tools.logging import logger
from ..tools.cache import CachedMethod
from ..tools.array import reshape_vector


class Domain:
    """
    Problem domain composed of orthogonal bases.

    Parameters
    ----------
    bases : list of basis objects
        Bases composing the domain
    grid_dtype : dtype
        Grid data type
    mesh : tuple of ints, optional
        Process mesh for parallelization (default: 1-D mesh of available processes)

    Attributes
    ----------
    dim : int
        Dimension of domain, equal to length of bases list
    distributor : distributor object
        Data distribution controller

    """

    def __init__(self, bases, grid_dtype=np.complex128, mesh=None):

        # Initial attributes
        self.bases = bases
        self.dim = len(bases)
        self.grid_dtype = grid_dtype

        # Iteratively set basis data types
        # (Grid-to-coefficient transforms proceed in the listed order)
        for b in bases:
            grid_dtype = b.set_transforms(grid_dtype)

        # Get global grid and coefficient shapes
        self.global_grid_shape = np.array([b.grid_size for b in bases], dtype=int)
        self.global_coeff_shape = np.array([b.coeff_size for b in bases], dtype=int)
        logger.debug('Global grid shape: %s' %str(self.global_grid_shape))
        logger.debug('Global coeff shape: %s' %str(self.global_coeff_shape))

        # Manage field allocation
        self._field_cache = list()
        self._field_count = 0

        # Create distributor
        self.distributor = Distributor(self, mesh)
        self.dist = self.distributor
        self.local_grid_shape = self.distributor.grid_layout.shape
        self.local_coeff_shape = self.distributor.coeff_layout.shape

        # Create differential operators
        self.diff_ops = [create_diff_operator(b,i) for (i,b) in enumerate(self.bases)]

    def get_basis_object(self, basis_like):
        """Return basis from a related object."""

        if basis_like in self.bases:
            return basis_like
        if basis_like in self.diff_ops:
            axis = self.diff_ops.index(basis_like)
            return self.bases[axis]
        if isinstance(basis_like, str):
            for b in self.bases:
                if basis_like == b.name:
                    return b
            raise ValueError("No matching basis name.")
        else:
            return self.bases[basis_like]

    @CachedMethod
    def grid(self, axis):
        """Return local grid along specified axis."""

        # Get local part of basis grid
        start = self.distributor.grid_layout.start[axis]
        size = self.distributor.grid_layout.shape[axis]
        grid = self.bases[axis].grid[start:start+size]

        # Reshape as multidimensional vector
        grid = reshape_vector(grid, self.dim, axis)

        return grid

    def grids(self):
        """Return list of local grids along each axis."""

        return [self.grid(i) for i in range(self.dim)]

    @CachedMethod
    def grid_spacing(self, axis):
        """Return minimum grid spacing along specified axis."""

        # Compute spacing on basis grid to include non-local spaces
        grid = self.bases[axis].grid
        diff = np.abs(np.diff(grid))
        spacing = np.zeros_like(grid)
        spacing[0] = diff[0]
        for i in range(1, grid.size-1):
            spacing[i] = min_nonzero(diff[i], diff[i-1])
        spacing[-1] = diff[-1]

        # Get local part of basis spacing
        start = self.distributor.grid_layout.start[axis]
        size = self.distributor.grid_layout.shape[axis]
        spacing = spacing[start:start+size]

        # Reshape as multidimensional vector
        spacing = reshape_vector(spacing, self.dim, axis)

        return spacing

    def _collect_field(self, field):
        """Cache free field."""

        # Add cleaned field to cache
        field.clean()
        self._field_cache.append(field)

    def new_field(self, name=None, constant=None):
        """Return a free field."""

        # Return a previously allocated field, if available
        if self._field_cache:
            field = self._field_cache.pop()
        # Otherwise instantiate a new field
        else:
            field = Field(self)

        # Set attributes
        field.name = name
        field.constant[:] = constant

        return field


def min_nonzero(*args):
    nonzero = lambda x: x != 0
    return min(filter(nonzero, args))

