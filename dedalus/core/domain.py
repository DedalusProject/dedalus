"""
Domain class definition.
"""

import logging
import numpy as np
from collections import OrderedDict

from ..tools.cache import CachedMethod, CachedClass, CachedAttribute
from ..tools.general import unify_attributes, unify, OrderedSet
from ..tools.array import reshape_vector
from .coords import Coordinate, CartesianCoordinates

logger = logging.getLogger(__name__.split('.')[-1])


class Domain(metaclass=CachedClass):
    """
    The direct product of a set of bases.

    Parameters
    ----------
    dist : Distributor object
        Distributor for an operand/field.
    bases : collection of Basis objects
        Bases comprising the direct product domain.
    """

    @classmethod
    def _preprocess_args(cls, dist, bases):
        # Drop None bases
        bases = [b for b in bases if b is not None]
        # Drop duplicate bases
        bases = tuple(OrderedSet(bases))
        # Make sure coordsystems don't overlap
        cs = [b.coordsystem for b in bases]
        if len(set(cs)) < len(cs):
            raise ValueError("Overlapping bases specified.")
        # Sort by first axis
        key = lambda basis: basis.first_axis
        bases = tuple(sorted(bases, key=key))
        return (dist, bases), {}

    def __init__(self, dist, bases):
        self.dist = dist
        self.bases = bases  # Preprocessed to remove Nones and duplicates
        self.dim = sum(basis.dim for basis in self.bases)

    @CachedAttribute
    def bases_by_axis(self):
        bases_by_axis = OrderedDict()
        for basis in self.bases:
            for axis in range(basis.first_axis, basis.first_axis+basis.dim):
                bases_by_axis[axis] = basis
        return bases_by_axis

    @CachedAttribute
    def full_bases(self):
        full_bases = [None for i in range(self.dist.dim)]
        for basis in self.bases:
            for axis in range(basis.first_axis, basis.first_axis+basis.dim):
                full_bases[axis] = basis
        return tuple(full_bases)

    @CachedAttribute
    def bases_by_coord(self):
        bases_by_coord = OrderedDict()
        for coord in self.dist.coords:
            if type(coord.cs) in [type(None), CartesianCoordinates]:
                bases_by_coord[coord] = None
            else:
                bases_by_coord[coord.cs] = None
        for basis in self.bases:
            bases_by_coord[basis.coords] = basis
            #bases_by_coord[basis.coordsystem] = basis
        return bases_by_coord

    @CachedAttribute
    def dealias(self):
        dealias = [1] * self.dist.dim
        for basis in self.bases:
            for subaxis in range(basis.dim):
                dealias[basis.first_axis+subaxis] = basis.dealias[subaxis]
        return tuple(dealias)

    def substitute_basis(self, old_basis, new_basis):
        new_bases = list(self.bases)
        if old_basis in new_bases:
            new_bases.remove(old_basis)
        new_bases.append(new_basis)
        return Domain(self.dist, new_bases)

    def get_basis(self, coords):
        if isinstance(coords, int):
            axis = coords
        else:
            axis = coords.axis
        return self.full_bases[axis]

    def get_basis_subaxis(self, coord):
        axis = coord.axis
        for basis in self.bases:
            if (axis >= basis.axis) and (axis <= basis.axis + basis.dim):
                return axis - basis.axis

    def get_coord(self, name):
        for basis in self.bases:
            # This is hacky...
            if isinstance(basis.coords, Coordinate):
                if name == basis.coords.name:
                    return basis.coords
            else:
                for basis_coord in basis.coords.coords:
                    if name == basis_coord.name:
                        return basis_coord
        raise ValueError("Coordinate name not in domain")

    def enumerate_unique_bases(self):
        axes = []
        unique_bases = []
        for axis, basis in enumerate(self.full_bases):
            if (basis is None) or (basis not in unique_bases):
                axes.append(axis)
                unique_bases.append(basis)
        return zip(axes, unique_bases)

    @CachedAttribute
    def constant(self):
        """Tuple of constant flags."""
        const = np.ones(self.dist.dim, dtype=bool)
        for basis in self.bases:
            for subaxis in range(basis.dim):
                const[basis.axis+subaxis] = basis.constant[subaxis]
        return tuple(const)

    @CachedAttribute
    def nonconstant(self):
        return tuple(~c for c in self.constant)

    @CachedAttribute
    def mode_dependence(self):
        """Tuple of dependence flags."""
        dep = np.zeros(self.dist.dim, dtype=bool)
        for basis in self.bases:
            for subaxis in range(basis.dim):
                dep[basis.axis+subaxis] = basis.subaxis_dependence[subaxis]
        return tuple(dep)

    @CachedAttribute
    def dim(self):
        return sum(self.nonconstant)

    @CachedAttribute
    def coeff_shape(self):
        """Compute coefficient shape."""
        scales = tuple(1 for i in range(self.dist.dim))
        return self.global_shape(layout=self.dist.coeff_layout, scales=scales)

    def grid_shape(self, scales):
        """Compute grid shape."""
        # Remedy scales before calling cached method
        scales = self.dist.remedy_scales(scales)
        return self._grid_shape(scales)

    def global_shape(self, layout, scales):
        shape = np.ones(self.dist.dim, dtype=int)
        for basis in self.bases:
            basis_scales = scales[basis.first_axis:basis.last_axis+1]
            shape[basis.first_axis:basis.last_axis+1] = basis.global_shape(layout, basis_scales)
        return shape

    def chunk_shape(self, layout):
        """Compute group shape."""
        shape = np.ones(self.dist.dim, dtype=int)
        for basis in self.bases:
            shape[basis.first_axis:basis.last_axis+1] = basis.chunk_shape(layout)
        return tuple(shape)

    @CachedMethod
    def _grid_shape(self, scales):
        """Cached grid shape computation."""
        shape = np.ones(self.dist.dim, dtype=int)
        for basis in self.bases:
            subscales = scales[basis.axis:basis.axis+basis.dim]
            subshape = basis.grid_shape(subscales)
            shape[basis.axis:basis.axis+basis.dim] = subshape
        return tuple(shape)

    # def expand_bases(self, bases):
    #     exp_bases = [None] * self.domain.dim
    #     for basis in bases:
    #         if basis is not None:
    #             if exp_bases[basis.space.axis] is not None:
    #                 raise ValueError("Degenerate bases.")
    #             exp_bases[basis.space.axis] = basis
    #     return tuple(exp_bases)

    # def __contains__(self, item):
    #     if isinstance(item, Subdomain):
    #         for axis in range(self.domain.dim):
    #             if item.spaces[axis] not in {None, self.spaces[axis]}:
    #                 return False
    #         return True
    #     else:
    #         space = self.domain.get_space_object(item)
    #         return (space in self.spaces)

    @CachedMethod
    def grid_spacing(self, axis, scales=None):
        """Compute grid spacings along one axis."""
        from .basis import MultidimensionalBasis
        scales = self.dist.remedy_scales(scales)
        # Compute spacing on global basis grid
        # This includes inter-process spacings
        basis = self.full_bases[axis]
        if issubclass(type(basis), MultidimensionalBasis):
            spacing = basis.grid_spacing(axis - basis.axis, scales=scales)
        else:
            spacing = basis.grid_spacing(scale=scales[axis])
        # Slice out local portion
        slices = self.dist.grid_layout.slices(self, scales)
        if len(spacing.shape) > 1:
            spacing = spacing[slices]
        else:
            spacing = spacing[slices[axis]]
            # Reshape as multidimensional vector
            spacing = reshape_vector(spacing, self.dim, axis)
        return spacing
