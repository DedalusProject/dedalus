"""
Class for problem domain.

"""

import logging
import numpy as np

from .distributor import Distributor
from .field import Field
#from .operators import create_diff_operator
from ..tools.cache import CachedMethod
from ..tools.array import reshape_vector
from ..tools.general import unify

logger = logging.getLogger(__name__.split('.')[-1])


class Domain:
    """
    Problem domain.

    Parameters
    ----------
    dim : int
        Dimensionality of domain
    dtype : dtype
        Domain data dtype
    mesh : tuple of ints, optional
        Process mesh for parallelization (default: 1-D mesh of available processes)

    Attributes
    ----------
    distributor : distributor object
        Data distribution controller

    """

    def __init__(self, dim, dtype, mesh=None):
        self.dim = dim
        self.dtype = np.dtype(dtype)
        #self.spaces = [set() for i in range(dim)]
        self.space_dict = {}
        self.spaces = [[] for i in range(dim)]
        self.distributor = self.dist = Distributor(self, mesh)

    def add_space(self, space, axis):
        if hasattr(space, 'domain'):
            raise ValueError("Space already assigned to a domain.")
        space.domain = self
        space.axis = axis
        for subaxis in range(space.dim):
            self.spaces[axis+subaxis].append(space)
        self.space_dict[space.name] = space

    def get_space_object(self, basis_like):
        """Return basis from a related object."""
        if basis_like in self.space_dict.items():
            return basis_like
        elif basis_like in self.space_dict:
            return self.space_dict[basis_like]
        else:
            raise ValueError()

    def subspace(self, axis_flags):
        spaces = []
        for axis in range(self.dim):
            if axis_flags[axis]:
                spaces.append(self.spaces[axis][0])
            else:
                spaces.append(None)
        if all(space is None for space in spaces):
            spaces = self
        return Subdomain(spaces)

        # # Objects
        # if basis_like in self.bases:
        #     return basis_like
        # # Indices
        # elif isinstance(basis_like, int):
        #     return self.bases[basis_like]
        # # Names
        # if isinstance(basis_like, str):
        #     for basis in self.bases:
        #         if basis_like == basis.name:
        #             return basis
        # # Otherwise
        # else:
        #     return None




    # def grids(self, scales=None):
    #     """Return list of local grids along each axis."""
    #     return [self.grid(i, scales) for i in range(self.dim)]

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

    # def new_data(self, type, **kw):
    #     return type(domain=self, **kw)

    # def new_field(self, **kw):
    #     return Field(domain=self, **kw)

    def remedy_scales(self, scales):
        if scales is None:
            scales = 1
        if np.isscalar(scales):
            scales = [scales] * self.dim
        if 0 in scales:
            raise ValueError("Scales must be nonzero.")
        return tuple(scales)


class Subdomain:

    def __init__(self, spaces):
        if isinstance(spaces, Domain):
            self.domain = spaces
            self.spaces = [None] * self.domain.dim
        else:

            spaces = [space for space in spaces if (space is not None)]
            self.domain = unify(space.domain for space in spaces)
            self.spaces = [None] * self.domain.dim
            for space in spaces:
                for subaxis in range(space.dim):
                    if self.spaces[space.axis+subaxis] is not None:
                        raise ValueError("Multiple spaces specified for axis {}".format(space.axis+subaxis))
                    self.spaces[space.axis+subaxis] = space
        self.spaces = tuple(self.spaces)

    def __contains__(self, item):
        for axis in range(self.domain.dim):
            if item.spaces[axis] not in {None, self.spaces[axis]}:
                return False
        return True

    @classmethod
    def from_spaces(cls, spaces):
        # Filter Nones
        spaces = [s for s in spaces if (s is not None)]
        domain = unify(space.domain for space in spaces)
        subdomain = Subdomain(domain)

    @classmethod
    def from_bases(cls, bases):
        if isinstance(bases, Domain):
            subdomain = Subdomain(bases)
            new_bases = [None] * subdomain.domain.dim
        else:
            bases = [basis for basis in bases if (basis is not None)]
            if len(bases) == 0:
                raise ValueError("No bases")
            spaces = [basis.space for basis in bases]
            subdomain = Subdomain(spaces)
            new_bases = [None] * subdomain.domain.dim
            for basis in bases:
                new_bases[basis.space.axis] = basis
        return subdomain, tuple(new_bases)

    @property
    def constant(self):
        return np.array([space is None for space in self.spaces])

    @property
    def dealias(self):
        dealias = []
        for space in self.spaces:
            if space is None:
                dealias.append(1)
            else:
                dealias.append(space.dealias)
        return dealias

    @property
    def global_coeff_shape(self):
        shape = np.zeros(self.domain.dim, dtype=int)
        for axis, space in enumerate(self.spaces):
            if space is None:
                shape[axis] = 1
            else:
                shape[axis] = space.coeff_size
        return shape

    def global_grid_shape(self, scales):
        scales = self.domain.remedy_scales(scales)
        shape = np.zeros(self.domain.dim, dtype=int)
        for axis, space in enumerate(self.spaces):
            if space is None:
                shape[axis] = 1
            else:
                shape[axis] = space.grid_size(scales[axis])
        return shape

    def local_groups(self):
        coeff_layout = self.domain.dist.coeff_layout
        blocks = coeff_layout.blocks(self, scales=1)
        group_start = coeff_layout.start(self, scales=1) / blocks
        group_count = coeff_layout.local_shape(self, scales=1) / blocks
        for group_offset in np.ndindex(*group_count.astype(int)):
            group = []
            for axis, space in enumerate(self.spaces):
                if space is None:
                    group.append(None)
                else:
                    group.append(group_start[axis] + group_offset[axis])
            yield group


