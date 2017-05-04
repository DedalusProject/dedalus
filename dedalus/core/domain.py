"""
Class for problem domain.

"""

import logging
import numpy as np

from .metadata import Metadata
from .distributor import Distributor
from .field import Scalar, Field
#from .operators import create_diff_operator
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

    def __init__(self, bases, grid_dtype=np.complex128, comm=None, mesh=None):

        # Iteratively set basis data types
        # (Grid-to-coefficient transforms proceed in the listed order)
        for basis in bases:
            grid_dtype = basis.set_dtype(grid_dtype)

        self.bases = bases
        self.dim = len(bases)
        self.dealias = tuple(basis.dealias for basis in bases)
        self.hypervolume = np.prod([(basis.interval[1]-basis.interval[0]) for basis in bases])
        self.global_coeff_shape = np.array([b.coeff_size for b in bases], dtype=int)
        logger.debug('Global coeff shape: %s' %str(self.global_coeff_shape))

        # Create distributor
        self.distributor = self.dist = Distributor(self, comm, mesh)
        self.local_coeff_shape = self.dist.coeff_layout.local_shape(self.remedy_scales(None))
        self.dealias_buffer_size = self.dist.buffer_size(self.dealias)

        # Create differential operators
        #self.diff_ops = [create_diff_operator(b,i) for (i,b) in enumerate(self.bases)]

    def global_grid_shape(self, scales=None):

        scales = self.remedy_scales(scales)
        return np.array([b.grid_size(s) for (s, b) in zip(scales, self.bases)])

    def local_grid_shape(self, scales=None):

        scales = self.remedy_scales(scales)
        return self.dist.grid_layout.local_shape(scales)

    def get_basis_object(self, basis_like):
        """Return basis from a related object."""

        # Objects
        if basis_like in self.bases:
            return basis_like
        # Indices
        elif isinstance(basis_like, int):
            return self.bases[basis_like]
        # Names
        if isinstance(basis_like, str):
            for basis in self.bases:
                if basis_like == basis.name:
                    return basis
        # Otherwise
        else:
            return None

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
    def elements(self, axis):
        """Return local elements along one axis."""
        scales = self.remedy_scales(None)
        # Get local part of global basis elements
        slices = self.distributor.coeff_layout.slices(scales)
        elements = self.bases[axis].elements
        elements = elements[slices[axis]]
        # Reshape as multidimensional vector
        elements = reshape_vector(elements, self.dim, axis)
        return elements

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

    def new_data(self, type, **kw):
        return type(domain=self, **kw)

    def new_field(self, **kw):
        return Field(domain=self, **kw)

    def new_fields(self, nfields, **kw):
        return [self.new_field(**kw) for n in range(nfields)]

    def remedy_scales(self, scales):

        # Default to 1.
        if scales is None:
            return tuple([1.] * self.dim)
        # Repeat scalars
        elif np.isscalar(scales):
            if scales == 0:
                raise ValueError("Cannot request zero scales.")
            return tuple([scales] * self.dim)
        # Cast others as tuple
        else:
            return tuple(scales)


class EmptyDomain:

    def __init__(self, grid_dtype=np.complex128):
        self.bases = []
        self.dim = 0
        self.dealias = (1,)

    def get_basis_object(self, basis_like):
        """Return basis from a related object."""
        return None

    def new_data(self, type, **kw):
        if type != Scalar:
            raise ValueError()
        return type(self, **kw)


def combine_domains(*domains):
    # Drop Nones
    domains = [domain for domain in domains if domain]
    # Drop Emptys
    domains = [domain for domain in domains if not isinstance(domain, EmptyDomain)]
    # Get set
    domain_set = set(domains)
    if len(domain_set) == 0:
        return EmptyDomain()
    if len(domain_set) > 1:
        raise ValueError("Non-unique domains")
    else:
        return list(domain_set)[0]
