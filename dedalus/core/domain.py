"""
Class for problem domain.

"""

import logging
import numpy as np

from .distributor import Distributor
from .field import Field
#from .operators import create_diff_operator
from ..tools.cache import CachedMethod, CachedClass
from ..tools.array import reshape_vector
from ..tools.general import unify, unify_attributes

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
        if basis_like in self.space_dict.values():
            return basis_like
        elif basis_like in self.space_dict:
            return self.space_dict[basis_like]
        else:
            raise ValueError()

    def subdomain(self, axis_flags):
        spaces = []
        for axis in range(self.dim):
            if axis_flags[axis]:
                spaces.append(self.spaces[axis][0])
            else:
                spaces.append(None)
        if any(spaces):
            return Subdomain.from_spaces(spaces)
        else:
            return Subdomain.from_domain(self)

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


class Subdomain(metaclass=CachedClass):

    def __init__(self, domain, spaces):
        self.domain = domain
        self.spaces = spaces

    @classmethod
    def from_spaces(cls, spaces):
        spaces = [s for s in spaces if (s is not None)]
        domain = unify_attributes(spaces, 'domain')
        full_spaces = [None] * domain.dim
        for space in spaces:
            for subaxis in range(space.dim):
                if full_spaces[space.axis+subaxis] is not None:
                    raise ValueError("Multiple spaces specified for axis {}".format(space.axis+subaxis))
                full_spaces[space.axis+subaxis] = space
        return Subdomain(domain, tuple(full_spaces))

    @classmethod
    def from_domain(cls, domain):
        spaces = (None,) * domain.dim
        return Subdomain(domain, spaces)

    @classmethod
    def from_bases(cls, bases):
        bases = [b for b in bases if (b is not None)]
        spaces = [b.space for b in bases]
        return cls.from_spaces(spaces)

    def expand_bases(self, bases):
        exp_bases = [None] * self.domain.dim
        for basis in bases:
            if basis is not None:
                if exp_bases[basis.space.axis] is not None:
                    raise ValueError("Degenerate bases.")
                exp_bases[basis.space.axis] = basis
        return tuple(exp_bases)

    def __contains__(self, item):
        if isinstance(item, Subdomain):
            for axis in range(self.domain.dim):
                if item.spaces[axis] not in {None, self.spaces[axis]}:
                    return False
            return True
        else:
            space = self.domain.get_space_object(item)
            return (space in self.spaces)

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



