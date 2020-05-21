"""
Domain class definition.
"""

import logging
import numpy as np

from ..tools.cache import CachedMethod, CachedClass, CachedAttribute
from ..tools.general import unify_attributes, unify
from .coords import Coordinate

logger = logging.getLogger(__name__.split('.')[-1])


def constant_spaces(dist):
    """Construct constant spaces for each axis of a distributor."""
    from .spaces import Constant
    return [Constant(dist=dist, axis=axis) for axis in range(dist.dim)]


def expand_spaces(spaces):
    """Expand list of spaces to tuple including constant spaces."""
    # Verify same distributor
    dist = unify_attributes(spaces, 'dist')
    # Verify spaces are non-overlapping
    if len(spaces) > 1:
        axes_sets = [set(space.axes) for space in spaces]
        if set.intersection(*axes_sets):
            raise ValueError("Overlapping spaces specified.")
    # Build full space tuple
    full_spaces = constant_spaces(dist)
    for space in spaces:
        for axis in space.axes:
            full_spaces[axis] = space
    return tuple(full_spaces)


class Domain(metaclass=CachedClass):
    """Object representing the direct product of a set of spaces."""

    # @classmethod
    # def _preprocess_args(cls, *args, **kw):
    #     # Expand spaces for proper caching
    #     args = list(args)
    #     args[0] = expand_spaces(args[0])
    #     return tuple(args), kw

    def __init__(self, dist, bases):
        self.dist = dist
        self.bases, self.full_bases = self._check_bases(bases)
        # self.dim = sum(space.dim for space in self.spaces)
        self.dealias = [1] * dist.dim
        for basis in self.bases:
            for ax in range(basis.dim):
                self.dealias[basis.axis + ax] = basis.dealias[ax]
        self.dealias = tuple(self.dealias)
        self.bases_dict = {basis.coords: basis for basis in self.bases}

    def __add__(self, other):
        dist = unify_attributes([self, other], 'dist')
        full_bases = [b1+b2 for b1, b2 in zip(self.full_bases, other.full_bases)]
        raise

    # def reduce_bases(self, bases):m

    def substitute_basis(self, inbasis, outbasis):
        bases_dict = self.bases_dict.copy()
        if inbasis is None:
            if outbasis.coords in bases_dict:
                raise ValueError("Basis already specified for coords: %s" %coords)
            else:
                bases_dict[outbasis.coords] = outbasis
        else:
            bases_dict[inbasis.coords] = outbasis
        bases = tuple(bases_dict.values())
        return Domain(self.dist, bases)

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

    def _check_bases(self, bases):
        # Drop Nones
        bases = [basis for basis in bases if basis is not None]
        # Drop duplicates
        bases_unique = []
        for basis in bases:
            repeat = False
            for basis_unique in bases_unique:
                if basis == basis_unique:
                    repeat = True
            if not repeat:
                bases_unique.append(basis)
        # Sort by axis
        key = lambda b: b.axis
        bases = sorted(bases_unique, key=key)
        # Check for overlap
        full_bases = [None for i in range(self.dist.dim)]
        for basis in bases:
            for subaxis in range(basis.dim):
                axis = basis.axis + subaxis
                if full_bases[axis] is not None:
                    raise ValueError("Overlapping bases specified.")
                else:
                    full_bases[axis] = basis
        return tuple(bases), tuple(full_bases)

    def enumerate_unique_bases(self):
        axes = []
        unique_bases = []
        for axis, basis in enumerate(self.full_bases):
            if (basis is None) or (basis not in unique_bases):
                axes.append(axis)
                unique_bases.append(basis)
        return zip(axes, unique_bases)

    # @classmethod
    # def from_dist(cls, dist):
    #     """Build constant domain from distributor."""
    #     return cls(constant_spaces(dist))

    # @classmethod
    # def from_bases(cls, dist, bases):
    #     """Build domain from bases."""
    #     return cls(dist, [basis.space for basis in bases])

    # @CachedAttribute
    # def dealias(self):
    #     """Tuple of dealias flags."""
    #     return tuple(space.dealias for space in self.spaces)

    @CachedAttribute
    def constant(self):
        """Tuple of constant flags."""
        return tuple(basis is None for basis in self.full_bases)

    @CachedAttribute
    def coeff_group_shape(self):
        """Compute group shape."""
        shape = np.ones(self.dist.dim, dtype=int)
        for basis in self.bases:
            for subaxis in range(basis.dim):
                shape[basis.axis+subaxis] = basis.group_shape[subaxis]
        return tuple(shape)

    @CachedAttribute
    def coeff_shape(self):
        """Compute coefficient shape."""
        shape = np.ones(self.dist.dim, dtype=int)
        for basis in self.bases:
            for subaxis in range(basis.dim):
                shape[basis.axis+subaxis] = basis.shape[subaxis]
        return tuple(shape)

    def grid_shape(self, scales):
        """Compute grid shape."""
        # Remedy scales before calling cached method
        scales = self.dist.remedy_scales(scales)
        return self._grid_shape(scales)

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

    # @CachedMethod
    # def grid_spacing(self, axis, scales=None):
    #     """Compute grid spacings along one axis."""
    #     scales = self.remedy_scales(scales)
    #     # Compute spacing on global basis grid
    #     # This includes inter-process spacings
    #     grid = self.bases[axis].grid(scales[axis])
    #     spacing = np.gradient(grid)
    #     # Restrict to local part of global spacing
    #     slices = self.dist.grid_layout.slices(scales)
    #     spacing = spacing[slices[axis]]
    #     # Reshape as multidimensional vector
    #     spacing = reshape_vector(spacing, self.dim, axis)
    #     return spacing

