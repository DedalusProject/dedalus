

import numpy as np
from functools import partial

from . import basis
from . import transforms
from ..tools.array import reshape_vector
from ..tools.general import unify
from ..tools.cache import CachedMethod, CachedAttribute


class AffineCOV:

    def __init__(self, native_bounds, problem_bounds):
        self.native_bounds = native_bounds
        self.problem_bounds = problem_bounds
        self.native_left, self.native_right = native_bounds
        self.native_length = self.native_right - self.native_left
        self.problem_left, self.problem_right = problem_bounds
        self.problem_length = self.problem_right - self.problem_left

        self.jacobian = self.native_length / self.problem_length

        self.problem_center = (self.problem_left + self.problem_right) / 2
        self.native_center = (self.native_left + self.native_right) / 2
        self.stretch = self.problem_length / self.native_length

    def problem_coord(self, native_coord):
        if native_coord == 'left':
            return self.problem_left
        elif native_coord == 'right':
            return self.problem_right
        elif native_coord == 'center':
            return self.problem_center
        else:
            neutral_coord = (native_coord - self.native_left) / self.native_length
            return self.problem_left + neutral_coord * self.problem_length

    def native_coord(self, problem_coord):
        if problem_coord == 'left':
            return self.native_left
        elif problem_coord == 'right':
            return self.native_right
        elif problem_coord == 'center':
            return self.native_center
        else:
            neutral_coord = (problem_coord - self.problem_left) / self.problem_length
            return self.native_left + neutral_coord * self.native_length

    def native_jacobian(self, problem_coord):
        return (self.native_length / self.problem_length)


class Space:

    def __repr__(self):
        return '<{} {}>'.format(self.__class__.__name__, id(self))

    def __str__(self):
        if self.name:
            return self.name
        else:
            return self.__repr__()


class Interval(Space):
    """Base class for 1-dimensional spaces."""

    dim = 1

    def __init__(self, name, base_grid_size, bounds, dealias=1):
        self.name = name
        self.base_grid_size = base_grid_size
        self.bounds = bounds
        self.dealias = dealias
        self.COV = AffineCOV(self.native_bounds, bounds)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    @CachedAttribute
    def operators(self):
        from .operators import prefixes
        return {prefix+self.name: partial(op, **{self.name:1}) for prefix, op in prefixes.items()}

    def grid_size(self, scale):
        """Compute scaled grid size."""
        grid_size = float(scale) * self.base_grid_size
        if not grid_size.is_integer():
            raise ValueError("Scaled grid size is not an integer: %f" %grid_size)
        return int(grid_size)

    @CachedAttribute
    def subdomain(self):
        from .domain import Subdomain
        return Subdomain.from_spaces([self])

    @CachedMethod
    def local_grid(self, scales=None):
        """Return local grid along one axis."""
        scales = self.domain.remedy_scales(scales)
        axis = self.axis
        # Get local part of global basis grid
        elements = np.ix_(*self.domain.dist.grid_layout.local_elements(self.subdomain, scales))
        grid = self.grid(scales[axis])
        local_grid = grid[elements[axis]]
        # Reshape as multidimensional vector
        #local_grid = reshape_vector(local_grid, self.domain.dim, axis)

        return local_grid

    @CachedMethod
    def grid_array(self, scales=None):
        """Return Array object representing grid."""
        from .field import Array
        grid = Array([self.grid_basis], name=self.name)
        grid.set_scales(scales)
        grid.set_local_data(self.local_grid(scales))
        return grid


class PeriodicInterval(Interval):

    native_bounds = (0, 2*np.pi)
    group_size = 2

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        if (self.base_grid_size % 2) != 0:
            raise ValueError("Periodic interval base grid size must be even.")
        self.kmax = (self.base_grid_size - 1) // 2
        self.coeff_size = 2 * (self.kmax + 1)

    @CachedMethod
    def grid(self, scale=1):
        """Evenly spaced endpoint grid: sin(N*x/2) = 0"""
        grid_size = self.grid_size(scale)
        native_grid = np.linspace(0, 2*np.pi, grid_size, endpoint=False)
        return self.COV.problem_coord(native_grid)

    @CachedAttribute
    def Fourier(self):
        return basis.Fourier(self)

    @CachedAttribute
    def grid_basis(self):
        return self.Fourier


class ParityInterval(Interval):

    native_bounds = (0, np.pi)
    group_size = 1

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.kmax = self.base_grid_size - 1
        self.coeff_size = self.base_grid_size

    @CachedMethod
    def grid(self, scale=1):
        """Evenly spaced interior grid: cos(N*x) = 0"""
        N = self.grid_size(scale)
        native_grid = np.pi * (np.arange(N) + 1/2) / N
        return self.COV.problem_coord(native_grid)

    @CachedAttribute
    def Sine(self):
        return basis.Sine(self)

    @CachedAttribute
    def Cosine(self):
        return basis.Cosine(self)

    @CachedAttribute
    def grid_basis(self):
        return self.Cosine


class FiniteInterval(Interval):

    native_bounds = (-1, 1)
    group_size = 1

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.coeff_size = self.base_grid_size

    @CachedMethod
    def grid(self, scale=1):
        """Chebyshev interior roots grid: cos(N*acos(x)) = 0"""
        N = self.grid_size(scale)
        theta = np.pi * (np.arange(N) + 1/2) / N
        native_grid = -np.cos(theta)
        return self.COV.problem_coord(native_grid)

    @CachedAttribute
    def ChebyshevT(self):
        return basis.ChebyshevT(self)

    @CachedAttribute
    def ChebyshevU(self):
        return basis.ChebyshevU(self)

    @CachedAttribute
    def grid_basis(self):
        return self.ChebyshevT


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

