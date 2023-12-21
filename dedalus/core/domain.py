"""
Domain class definition.
"""

import logging
import numpy as np
from collections import OrderedDict
from math import prod

from ..tools.cache import CachedMethod, CachedClass, CachedAttribute
from ..tools.general import unify_attributes, unify, OrderedSet
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
        cs = [b.coordsys for b in bases]
        if len(set(cs)) < len(cs):
            raise ValueError("Overlapping bases specified.")
        # Sort by first axis
        key = lambda basis: dist.get_basis_axis(basis)
        bases = tuple(sorted(bases, key=key))
        return (dist, bases), {}

    def __init__(self, dist, bases):
        self.dist = dist
        self.bases = bases  # Preprocessed to remove Nones and duplicates
        self.dim = sum(basis.dim for basis in self.bases)

    @CachedAttribute
    def volume(self):
        return prod([basis.volume for basis in self.bases])

    @CachedAttribute
    def bases_by_axis(self):
        bases_by_axis = OrderedDict()
        for basis in self.bases:
            for axis in range(self.dist.get_basis_axis(basis), self.dist.get_basis_axis(basis)+basis.dim):
                bases_by_axis[axis] = basis
        return bases_by_axis

    @CachedAttribute
    def full_bases(self):
        full_bases = [None for i in range(self.dist.dim)]
        for basis in self.bases:
            for axis in range(self.dist.get_basis_axis(basis), self.dist.get_basis_axis(basis)+basis.dim):
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
            #bases_by_coord[basis.coordsys] = basis
        return bases_by_coord

    @CachedAttribute
    def dealias(self):
        dealias = [1] * self.dist.dim
        for basis in self.bases:
            for subaxis in range(basis.dim):
                dealias[self.dist.get_basis_axis(basis)+subaxis] = basis.dealias[subaxis]
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
            axis = self.dist.get_axis(coords)
        return self.full_bases[axis]

    def get_basis_subaxis(self, coord):
        axis = self.dist.get_axis(coord)
        for basis in self.bases:
            basis_axis = self.dist.get_basis_axis(basis)
            if basis_axis <= axis < basis_axis + basis.dim:
                return axis - basis_axis

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
            first_axis = self.dist.get_basis_axis(basis)
            for subaxis in range(basis.dim):
                const[first_axis+subaxis] = basis.constant[subaxis]
        return tuple(const)

    @CachedAttribute
    def nonconstant(self):
        return tuple(~c for c in self.constant)

    @CachedAttribute
    def mode_dependence(self):
        """Tuple of dependence flags."""
        dep = np.zeros(self.dist.dim, dtype=bool)
        for basis in self.bases:
            first_axis = self.dist.get_basis_axis(basis)
            for subaxis in range(basis.dim):
                dep[first_axis+subaxis] = basis.subaxis_dependence[subaxis]
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
            first_axis = self.dist.get_basis_axis(basis)
            basis_axes = slice(first_axis, first_axis+basis.dim)
            shape[basis_axes] = basis.global_shape(layout.grid_space[basis_axes], scales[basis_axes])
        return tuple(shape)

    @CachedMethod
    def chunk_shape(self, layout):
        """Compute chunk shape."""
        shape = np.ones(self.dist.dim, dtype=int)
        for basis in self.bases:
            first_axis = self.dist.get_basis_axis(basis)
            basis_axes = slice(first_axis, first_axis+basis.dim)
            shape[basis_axes] = basis.chunk_shape(layout.grid_space[basis_axes])
        return tuple(shape)

    def group_shape(self, layout):
        """Compute group shape."""
        group_shape = np.ones(self.dist.dim, dtype=int)
        for basis in self.bases:
            first_axis = self.dist.get_basis_axis(basis)
            basis_axes = slice(first_axis, first_axis+basis.dim)
            group_shape[basis_axes] = basis.group_shape
        group_shape[layout.grid_space] = 1
        return group_shape

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
