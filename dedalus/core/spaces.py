"""
Space class definitions.
"""

import numpy as np

from ..tools import jacobi
from ..tools.array import reshape_vector
from ..tools.cache import CachedMethod, CachedAttribute


class AffineCOV:
    """
    Class for affine change-of-variables for remapping space bounds.

    Parameters
    ----------
    native_bounds : tuple of floats
        Native bounds given as (lower, upper)
    problem_bounds : tuple of floats
        New bounds given as (lower, upper)
    """

    def __init__(self, native_bounds, problem_bounds):
        self.native_bounds = native_bounds
        self.problem_bounds = problem_bounds
        self.native_left, self.native_right = native_bounds
        self.native_length = self.native_right - self.native_left
        self.native_center = (self.native_left + self.native_right) / 2
        self.problem_left, self.problem_right = problem_bounds
        self.problem_length = self.problem_right - self.problem_left
        self.problem_center = (self.problem_left + self.problem_right) / 2
        self.stretch = self.problem_length / self.native_length

    def problem_coord(self, native_coord):
        """Convert native coordinates to problem coordinates."""
        if isinstance(native_coord, str):
            if native_coord in ('left', 'lower'):
                return self.problem_left
            elif native_coord in ('right', 'upper'):
                return self.problem_right
            elif native_coord in ('center', 'middle'):
                return self.problem_center
            else:
                raise ValueError("String coordinate '%s' not recognized." %native_coord)
        else:
            neutral_coord = (native_coord - self.native_left) / self.native_length
            return self.problem_left + neutral_coord * self.problem_length

    def native_coord(self, problem_coord):
        """Convert problem coordinates to native coordinates."""
        if isinstance(problem_coord, str):
            if problem_coord in ('left', 'lower'):
                return self.native_left
            elif problem_coord in ('right', 'upper'):
                return self.native_right
            elif problem_coord in ('center', 'middle'):
                return self.native_center
            else:
                raise ValueError("String coordinate '%s' not recognized." %problem_coord)
        else:
            neutral_coord = (problem_coord - self.problem_left) / self.problem_length
            return self.native_left + neutral_coord * self.native_length


class Space:
    """Base class for spaces."""

    @classmethod
    def check_shape(cls, space_shape):
        """Check compatibility between space shape and group shape."""
        for ss, gs in zip(space_shape, cls.group_shape):
            if (ss % gs) != 0:
                raise ValueError("Space shape must be multiple of group shape.")

    def grid_shape(self, scales):
        """Scaled grid shape."""
        scales = self.dist.remedy_scales(scales)
        subscales = scales[self.axis:self.axis+self.dim]
        return tuple(int(s*n) for s,n in zip(subscales, self.shape))

    def grids(self, scales):
        """Flat global grids."""
        raise NotImplementedError()

    def local_grids(self, scales):
        """Local grid vectors by axis."""
        scales = self.dist.remedy_scales(scales)
        # Get grid slices for relevant axes
        slices = self.dist.grid_layout.slices(self.domain, scales)
        subslices = slices[self.axis:self.axis+self.dim]
        # Select local portion of global grids
        grids = self.grids(scales)
        local_grids = tuple(g[s] for g,s in zip(grids, subslices))
        # Reshape as vectors
        return tuple(reshape_vector(g, self.dist.dim, i) for g,i in zip(local_grids, self.axes))

    # @CachedMethod
    # def grid_field(self, scales=None):
    #     """Return field object representing grid."""
    #     from .field import Field
    #     grid = Field(name=self.name, dist=self.dist, bases=[self.grid_basis])
    #     grid.set_scales(scales)
    #     grid['g'] = self.local_grid(scales)
    #     return grid

    # @CachedAttribute
    # def operators(self):
    #     from .operators import prefixes
    #     return {prefix+self.name: partial(op, **{self.name:1}) for prefix, op in prefixes.items()}


class Constant(Space):
    """Constant spaces."""

    constant = True
    dim = 1
    group_shape = (1,)
    shape = (1,)
    dealias = 1

    def __init__(self, dist, axis):
        self.dist = dist
        self.axes = (axis,)
        self.grid_basis = self.Constant

    def grid_shape(self, scale):
        """Compute scaled grid size."""
        # No scaling for constant spaces
        return self.shape

    def grids(self, scales):
        # No scaling for constant spaces
        return (np.array([0.]),)

    def Constant(self):
        return basis.Constant(self)


class Interval(Space):
    """Base class for 1D intervals."""

    coord_type = Coordinate
    dim = 1

    def __init__(self, coord, size, bounds, dealias=1):
        self.coord = coord
        self.coords = (coord,)
        self.size = size
        self.shape = (size,)
        self.bounds = bounds
        self.dealias = dealias
        self.COV = AffineCOV(self.native_bounds, bounds)

    def grids(self, scales):
        """Flat global grids."""
        native_grid = self.native_grid(scales)
        problem_grid = self.COV.problem_coord(native_grid)
        return (problem_grid,)


class PeriodicInterval(Interval):
    """Periodic interval for Fourier series."""

    native_bounds = (0, 2*np.pi)

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        # Maximum native k, dispensing Nyquist mode
        self.kmax = (self.size - 1) // 2

    def native_grid(self, scales):
        """Evenly spaced endpoint grid: sin(N*x/2) = 0"""
        N = self.grid_shape(scales)[0]
        return (2 * np.pi / N) * np.arange(N)

    def Fourier(self):
        return basis.Fourier(self)

    def grid_basis(self):
        return self.Fourier()


class ParityInterval(Interval):
    """Definite-parity periodic interval for Sine and Cosine series."""

    native_bounds = (0, np.pi)

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.kmax = self.size - 1

    def native_grid(self, scales):
        """Evenly spaced interior grid: cos(N*x) = 0"""
        N = self.grid_shape(scales)[0]
        return (np.pi / N) * (np.arange(N) + 1/2)

    def Sine(self):
        return basis.Sine(self)

    def Cosine(self):
        return basis.Cosine(self)

    def grid_basis(self,):
        return self.Cosine()


class FiniteInterval(Interval):
    """
    Affine transformation of the interval [-1, 1], under the weight:
        w(x) = (1-x)^a (1+x)^b
    """

    native_bounds = (-1, 1)

    def __init__(self, a, b, *args, **kw):
        super().__init__(*args, **kw)
        self.a = float(a)
        self.b = float(b)

    def native_grid(self, scales):
        """Gauss-Jacobi grid."""
        N = self.grid_shape(scales)[0]
        return jacobi.build_grid(N, self.a, self.b)

    def weights(self, scales):
        """Gauss-Jacobi weights."""
        N = self.grid_shape(scales)[0]
        return jacobi.build_weights(N, self.a, self.b)

    def Jacobi(self, *args, **kw):
        return basis.Jacobi(self, *args, **kw)

    def grid_basis(self, *args, **kw):
        return self.Jacobi(da=0, db=0, *args, **kw)

    def Legendre(self, **kw):
        if (self.a != 0) or (self.b != 0):
            raise ValueError("Must use a0 = b0 = 0 for Legendre polynomials.")
        return self.Jacobi(da=0, db=0, **kw)

    def Ultraspherical(self, d, **kw):
        if (self.a != -1/2) or (self.b != -1/2):
            raise ValueError("Must use a0 = b0 = -1/2 for Ultraspherical polynomials.")
        return self.Jacobi(da=d, db=d, **kw)

    def ChebyshevT(self, **kw):
        return self.Ultraspherical(d=0, **kw)

    def ChebyshevU(self, **kw):
        return self.Ultraspherical(d=1, **kw)

    def ChebyshevV(self, **kw):
        return self.Ultraspherical(d=2, **kw)

    def ChebyshevW(self, **kw):
        return self.Ultraspherical(d=3, **kw)

class Disk(Domain):

    coord_type = PolarCoords
    dim = 2

    def __init__(self, coords, radius):
        self.coords = coords
        self.radius = radius
        if radius <= 0:
            raise ValueError("Radius must be positive.")
        self._check_coords()


class Annulus(Domain):

    coord_type = PolarCoords
    dim = 2

    def __init__(self, coords, radial_interval):
        self.coords = coords
        self.radial_interval = radial_interval
        self.r0, self.r1 = radial_interval
        if self.r0 <= 0:
            raise ValueError("Inner radius must be positive.")
        if self.r1 <= self.r0:
            raise ValueError("Outer radius must be larger than inner radius.")
        self._check_coords()


class Sphere(Domain):
    coord_type = (SphericalCoords2D, SphericalCoords3D)

    def __init__(self, coords, radius):
        self.coords = coords
        self.radius = radius
        if radius <= 0:
            raise ValueError("Radius must be positive.")
        self._check_coords()


class Ball(Domain):

    coord_type = SphericalCoords3D
    dim = 3

    def __init__(self, coords, radius):
        self.coords = coords
        self.radius = radius
        if radius <= 0:
            raise ValueError("Radius must be positive.")
        self._check_coords()


class SphericalShell(Domain):

    coord_type = SphericalCoords3D
    dim = 3

    def __init__(self, coords, radial_interval):
        self.coords = coords
        self.radial_interval = radial_interval
        self.r0, self.r1 = radial_interval
        if self.r0 <= 0:
            raise ValueError("Inner radius must be positive.")
        if self.r1 <= self.r0:
            raise ValueError("Outer radius must be larger than inner radius.")
        self._check_coords()









# class Sheet(Space):
#     """Base class for 1-dimensional spaces."""

#     dim = 1

#     def __init__(self, names, shape, bounds, domain, axis, dealias=1):
#         self.name = name
#         self.base_grid_size = base_grid_size
#         self.domain = domain
#         self.axis = axis
#         self.axes = np.arange(axis, axis+self.dim)
#         self.dealias = dealias

#         for index in range(self.dim):
#             domain.spaces[axis+index].append(self)
#         domain.space_dict[name] = self

#     def grid_size(self, scale):
#         """Compute scaled grid size."""
#         grid_size = float(scale) * self.base_grid_size
#         if not grid_size.is_integer():
#             raise ValueError("Scaled grid size is not an integer: %f" %grid_size)
#         return int(grid_size)

#     @CachedAttribute
#     def subdomain(self):
#         from .domain import Subdomain
#         return Subdomain.from_spaces([self])

#     @CachedMethod
#     def local_grid(self, scales=None):
#         """Return local grid along one axis."""
#         scales = self.domain.remedy_scales(scales)
#         axis = self.axis
#         # Get local part of global basis grid
#         elements = np.ix_(*self.domain.dist.grid_layout.local_elements(self.subdomain, scales))
#         grid = self.grid(scales[axis])
#         local_grid = grid[elements[axis]]
#         # Reshape as multidimensional vector
#         #local_grid = reshape_vector(local_grid, self.domain.dim, axis)

#         return local_grid

#     @CachedMethod
#     def grid_field(self, scales=None):
#         """Return field object representing grid."""
#         from .field import Field
#         grid = Field(name=self.name, domain=self.domain, bases=[self.grid_basis])
#         grid.set_scales(scales)
#         grid.set_local_data(self.local_grid(scales))
#         return grid

# class Sphere:

#     def __init__(self, name, Lmax, Mmax, domain, axis, dealias=1):
#         pass

# class FiniteInterval(Interval):
#     a = b = -1/2  # default Chebyshev
#     def Jacobi(self, a, b):
#         return basis.JacobiFactory(a, b)(self)

# class SemiInfiniteInterval(Interval):
#     def Laguerre(self, a):
#         return basis.LaguerreFactory(a)(self)

# class InfiniteInterval(Interval):
#     @property
#     def Hermite(self):
#         return basis.Hermite(self)

# class RadialInterval(Interval):

#     pass
# class ShearingSheet:
#     dim = 2

# class Sphere:
#     dim = 2
#     def SWSH(self, s):
#         return SWSHFactory(s)(self)

