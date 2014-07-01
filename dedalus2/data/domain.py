"""
Class for problem domain.

"""

import logging
import numpy as np

from .distributor import Distributor
from .field import Field
from .operators import create_diff_operator
from ..tools.cache import CachedMethod
from ..tools.array import reshape_vector

logger = logging.getLogger(__name__.split('.')[-1])


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

        # Iteratively set basis data types
        # (Grid-to-coefficient transforms proceed in the listed order)
        for basis in bases:
            grid_dtype = basis.set_dtype(grid_dtype)

        self.bases = bases
        self.dim = len(bases)
        self.dealias = tuple(basis.dealias for basis in bases)
        self.global_coeff_shape = np.array([b.coeff_size for b in bases], dtype=int)
        logger.debug('Global coeff shape: %s' %str(self.global_coeff_shape))

        # Manage field allocation
        self._field_cache = list()
        self._field_count = 0

        # Create distributor
        self.distributor = self.dist = Distributor(self, mesh)
        self.local_coeff_shape = self.dist.coeff_layout.local_shape(self.remedy_scales(None))

        # Create differential operators
        self.diff_ops = [create_diff_operator(b,i) for (i,b) in enumerate(self.bases)]

    def global_grid_shape(self, scales=None):

        scales = self.remedy_scales(scales)
        return np.array([b.grid_size(s) for (s, b) in zip(scales, self.bases)])

    def local_grid_shape(self, scales=None):

        scales = self.remedy_scales(scales)
        return self.dist.grid_layout.local_shape(scales)

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

    def grids(self, scales=None):
        """Return list of local grids along each axis."""

        return [self.grid(i, scales) for i in range(self.dim)]

    @CachedMethod
    def grid(self, axis, scales=None):
        """Return local grid along one axis."""

        scales = self.remedy_scales(scales)
        # Get local part of global basis grid
        slices = self.distributor.grid_layout.slices(scales)
        grid = self.bases[axis].grid(scales[axis])
        grid = grid[slices[axis]]
        # Reshape as multidimensional vector
        grid = reshape_vector(grid, self.dim, axis)

        return grid

    @CachedMethod
    def grid_spacing(self, axis, scales=None):
        """Compute grid spacings along one axis."""

        scales = self.remedy_scales(scales)
        # Compute spacing on global basis grid
        # This includes inter-process spacings
        grid = self.bases[axis].grid(scales[axis])
        spacing = np.gradient(grid)
        # Restrict to local part of global spacing
        slices = self.dist.grid_layout.slices(scales)
        spacing = spacing[slices[axis]]
        # Reshape as multidimensional vector
        spacing = reshape_vector(spacing, self.dim, axis)

        return spacing

    def _collect_field(self, field):
        """Cache free field."""

        self._field_cache.append(field)

    def new_field(self):
        """Return a free field."""

        # Return a previously allocated field, if available
        if self._field_cache:
            return self._field_cache.pop()
        # Otherwise instantiate a new field
        else:
            return Field(self)

    def remedy_scales(self, scales):

        # Default to 1.
        if scales is None:
            return tuple([1.] * self.dim)
        # Repeat scalars
        elif np.isscalar(scales):
            return tuple([scales] * self.dim)
        # Cast others as tuple
        else:
            return tuple(scales)

