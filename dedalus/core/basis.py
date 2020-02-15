"""
Abstract and built-in classes for spectral bases.

"""

import math
import numpy as np
from scipy import sparse

from . import operators
from ..tools.array import axslice
from ..tools.array import apply_matrix
from ..tools.cache import CachedAttribute
from ..tools.cache import CachedMethod
from ..tools.cache import CachedClass
from ..tools import jacobi
from ..tools import clenshaw
from ..tools.array import reshape_vector

from .spaces import ParityInterval, Disk
from .coords import Coordinate, S2Coordinates, SphericalCoordinates
from .domain import Domain
import dedalus_sphere
#from . import transforms

import logging
logger = logging.getLogger(__name__.split('.')[-1])

from ..tools.config import config
DEFAULT_LIBRARY = config['transforms'].get('DEFAULT_LIBRARY')
DEFAULT_LIBRARY = 'scipy'


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


class Basis:
    """Abstract base class for spectral bases."""

    constant = False

    def __init__(self, coords):
        self.coords = coords
        self.dist = coords[0].dist
        self.axis = self.dist.coords.index(coords[0])
        self.domain = Domain(self.dist, bases=(self,))

    # def __repr__(self):
    #     return '<%s %i>' %(self.__class__.__name__, id(self))

    # def __str__(self):
    #     return '%s.%s' %(self.space.name, self.__class__.__name__)

    # def __radd__(self, other):
    #     return self.__add__(other)

    # def __rmul__(self, other):
    #     return self.__mul__(other)

    def grid_shape(self, scales):
        return tuple(int(np.ceil(s*n)) for s, n in zip(scales, self.shape))

    def global_grids(self, scales):
        """Global grids."""
        # Subclasses must implement
        # Returns tuple of global grids along each subaxis
        raise NotImplementedError

    def local_grids(self, scales):
        """Local grids."""
        # Subclasses must implement
        # Returns tuple of local grids along each subaxis
        raise NotImplementedError

    def forward_transform(self, field, axis, gdata, cdata):
        """Grid-to-coefficient transform."""
        # Subclasses must implement
        # Performs transform along specified axis, must be in-place-safe.
        raise NotImplementedError

    def backward_transform(self, field, axis, cdata, gdata):
        """Coefficient-to-grid transform."""
        # Subclasses must implement
        # Performs transform along specified axis, must be in-place-safe.
        raise NotImplementedError

    # def __getitem__(self, mode):
    #     """Return field populated by one mode."""
    #     # if not self.compute_mode(mode):
    #     #     raise ValueError("Basis does not contain specified mode.")
    #     from .field import Field
    #     axis = self.space.axis
    #     out = Field(bases=[self], layout='c')
    #     data = np.zeros(out.global_shape, dtype=out.dtype)
    #     if mode < 0:
    #         mode += self.space.coeff_size
    #     data[axslice(axis, mode, mode+1)] = 1
    #     out.set_global_data(data)
    #     return out

    # @classmethod
    # def _check_coord(cls, coord):
    #     if not isinstance(coord, cls.coord_type):
    #         raise ValueError("Invalid coord type.")

    # @CachedAttribute
    # def inclusion_flags(self):
    #     return np.array([self.include_mode(i) for i in range(self.space.coeff_size)])

    # @CachedAttribute
    # def inclusion_matrix(self):
    #     diag = self.inclusion_flags.astype(float)
    #     return sparse.diags(diag, 0, format='csr')

    # @CachedAttribute
    # def modes(self):
    #     return np.arange(self.space.coeff_size)[self.inclusion_flags]

    # @CachedAttribute
    # def n_modes(self):
    #     return self.modes.size

    # def mode_map(self, group):
    #     flags = self.inclusion_flags
    #     matrix = self.inclusion_matrix
    #     # Restrict to group elements
    #     if group is not None:
    #         n0 = group * self.space.group_size
    #         n1 = n0 + self.space.group_size
    #         matrix = matrix[n0:n1, n0:n1]
    #         flags = flags[n0:n1]
    #     # Discard empty rows
    #     return matrix[flags, :]

#     def ncc_matrix(self, arg_basis, coeffs, cutoff=1e-6):
#         """Build NCC matrix via direct summation."""
#         N = len(coeffs)
#         for i in range(N):
#             coeff = coeffs[i]
#             # Build initial matrix
#             if i == 0:
#                 matrix = self.product_matrix(arg_basis, i)
#                 total = 0 * sparse.kron(matrix, coeff)
#                 total.eliminate_zeros()
#             if len(coeff.shape) or (abs(coeff) > cutoff):
#                 matrix = self.product_matrix(arg_basis, i)
#                 total = total + sparse.kron(matrix, coeff)
#         return total

#     def product_matrix(self, arg_basis, i):
#         if arg_basis is None:
#             N = self.space.coeff_size
#             return sparse.coo_matrix(([1],([i],[0])), shape=(N,1)).tocsr()
#         else:
#             raise NotImplementedError()


# class Constant(Basis, metaclass=CachedClass):
#     """Constant basis."""

#     def __add__(self, other):
#         if other is self:
#             return self
#         else:
#             return NotImplemented

#     def __mul__(self, other):
#         if other is self:
#             return self
#         else:
#             return NotImplemented



class IntervalBasis(Basis):

    dim = 1

    def __init__(self, coord, size, bounds):
        super().__init__((coord,))
        self.coord = coord
        self.size = size
        self.shape = (size,)
        self.bounds = bounds
        self.COV = AffineCOV(self.native_bounds, bounds)

    def global_grids(self, scales):
        """Global grids."""
        return (self.global_grid(scales[0]),)

    def global_grid(self, scale):
        """Global grid."""
        native_grid = self._native_grid(scale)
        problem_grid = self.COV.problem_coord(native_grid)
        return reshape_vector(problem_grid, dim=self.dist.dim, axis=self.axis)

    def local_grids(self, scales):
        """Local grids."""
        return (self.local_grid(scales[0]),)

    def local_grid(self, scale):
        """Local grid."""
        local_elements = self.dist.grid_layout.local_elements(self.domain, scales=scale)[self.axis]
        native_grid = self._native_grid(scale)[local_elements]
        problem_grid = self.COV.problem_coord(native_grid)
        return reshape_vector(problem_grid, dim=self.dist.dim, axis=self.axis)

    def _native_grid(self, scale):
        """Native flat global grid."""
        # Subclasses must implement
        raise NotImplementedError

    def forward_transform(self, field, axis, gdata, cdata):
        """Forward transform field data."""
        data_axis = len(field.tensorsig) + axis
        grid_size = gdata.shape[data_axis]
        plan = self.transform_plan(grid_size)
        plan.forward(gdata, cdata, data_axis)

    def backward_transform(self, field, axis, cdata, gdata):
        """Backward transform field data."""
        data_axis = len(field.tensorsig) + axis
        grid_size = gdata.shape[data_axis]
        plan = self.transform_plan(grid_size)
        plan.backward(cdata, gdata, data_axis)

    def transform_plan(self, grid_size):
        # Subclasses must implement
        raise NotImplementedError


class Jacobi(IntervalBasis, metaclass=CachedClass):
    """Jacobi polynomial basis."""

    group_shape = (1,)
    native_bounds = (-1, 1)
    transforms = {}

    def __init__(self, coord, size, bounds, a, b, a0=None, b0=None, dealias=1, library='matrix'):
        super().__init__(coord, size, bounds)
        # Default grid parameters
        if a0 is None:
            a0 = a
        if b0 is None:
            b0 = b
        self.a = float(a)
        self.b = float(b)
        self.a0 = float(a0)
        self.b0 = float(b0)
        self.dealias = dealias
        self.library = library
        self.grid_params = (coord, bounds, a0, b0)
        #self.const = 1 / np.sqrt(jacobi.mass(self.a, self.b))

    def _new_a_b(self, a, b):
        return Jacobi(self.coord, self.size, self.bounds, a, b, a0=self.a0, b0=self.b0, dealias=self.dealias, library=self.library)

    def _native_grid(self, scale):
        """Native flat global grid."""
        N, = self.grid_shape((scale,))
        return jacobi.build_grid(N, a=self.a, b=self.b)

    @CachedMethod
    def transform_plan(self, grid_size):
        """Build transform plan."""
        return self.transforms[self.library](grid_size, self.size, self.a, self.b, self.a0, self.b0)

    # def weights(self, scales):
    #     """Gauss-Jacobi weights."""
    #     N = self.grid_shape(scales)[0]
    #     return jacobi.build_weights(N, a=self.a, b=self.b)

    # def __str__(self):
    #     space = self.space
    #     cls = self.__class__
    #     return '%s.%s(%s,%s)' %(space.name, cls.__name__, self.a, self.b)

    def __add__(self, other):
        if other is None:
            return self
        if other is self:
            return self
        if isinstance(other, Jacobi):
            if self.grid_params == other.grid_params:
                size = max(self.size, other.size)
                a = max(self.a, other.a)
                b = max(self.b, other.b)
                dealias = max(self.dealias, other.dealias)
                return Jacobi(self.coord, size, self.bounds, a, b, a0=self.a0, b0=self.b0, dealias=dealias, library=self.library)
        return NotImplemented

    def __mul__(self, other):
        if other is None:
            return self
        if other is self:
            return self
        if isinstance(other, Jacobi):
            if self.grid_params == other.grid_params:
                size = max(self.size, other.size)
                a = self.a0
                b = self.b0
                dealias = max(self.dealias, other.dealias)
                return Jacobi(self.coord, size, self.bounds, a, b, a0=self.a0, b0=self.b0, dealias=dealias, library=self.library)
        return NotImplemented

    # def include_mode(self, mode):
    #     return (0 <= mode < self.space.coeff_size)

    # def ncc_matrix(self, arg_basis, coeffs, cutoff=1e-6):
    #     """Build NCC matrix via Clenshaw algorithm."""
    #     if arg_basis is None:
    #         return super().ncc_matrix(arg_basis, coeffs)
    #     # Kronecker Clenshaw on argument Jacobi matrix
    #     N = self.space.coeff_size
    #     J = jacobi.jacobi_matrix(N, arg_basis.a, arg_basis.b)
    #     A, B = clenshaw.jacobi_recursion(N, self.a, self.b, J)
    #     f0 = self.const * sparse.identity(N)
    #     total = clenshaw.kronecker_clenshaw(coeffs, A, B, f0, cutoff=cutoff)
    #     # Conversion matrix
    #     input_basis = arg_basis
    #     output_basis = (self * arg_basis)
    #     conversion = ConvertJacobiJacobi._subspace_matrix(self.space, input_basis, output_basis)
    #     # Kronecker with identity for matrix coefficients
    #     coeff_size = total.shape[0] // conversion.shape[0]
    #     if coeff_size > 1:
    #         conversion = sparse.kron(conversion, sparse.identity(coeff_size))
    #     return (conversion @ total)


def Legendre(*args, **kw):
    return Jacobi(*args, a=0, b=0, **kw)


def Ultraspherical(*args, alpha, alpha0=None, **kw):
    # Default grid parameter
    if alpha0 is None:
        alpha0 = alpha
    a = b = alpha - 1/2
    a0 = b0 = alpha0 - 1/2
    return Jacobi(*args, a=a, b=b, a0=a0, b0=b0, **kw)


def ChebyshevT(*args, **kw):
    return Ultraspherical(*args, alpha=0, **kw)


def ChebyshevU(*args, **kw):
    return Ultraphserical(*args, alpha=1, **kw)


class ConvertJacobiJacobi(operators.Convert1D):
    """Jacobi polynomial conversion."""

    input_basis_type = Jacobi
    output_basis_type = Jacobi
    separable = False

    @staticmethod
    def _subspace_matrix(input_basis, output_basis):
        N = input_basis.size
        a0, b0 = input_basis.a, input_basis.b
        a1, b1 = output_basis.a, output_basis.b
        matrix = jacobi.conversion_matrix(N, a0, b0, a1, b1)
        return matrix.tocsr()


class DifferentiateJacobi(operators.Differentiate):
    """Jacobi polynomial differentiation."""

    input_basis_type = Jacobi
    separable = False

    @staticmethod
    def output_basis(input_basis):
        a = input_basis.a + 1
        b = input_basis.b + 1
        return input_basis._new_a_b(a, b)

    @staticmethod
    def _subspace_matrix(input_basis):
        N = input_basis.size
        a, b = input_basis.a, input_basis.b
        matrix = jacobi.differentiation_matrix(N, a, b)
        return (matrix.tocsr() / input_basis.COV.stretch)


# class InterpolateJacobi(operators.Interpolate):
#     """Jacobi polynomial interpolation."""

#     input_basis_type = Jacobi

#     @staticmethod
#     def _subspace_matrix(space, input_basis, position):
#         N = space.coeff_size
#         a, b = input_basis.a, input_basis.b
#         x = space.COV.native_coord(position)
#         return jacobi.interpolation_vector(N, a, b, x)


# class IntegrateJacobi(operators.Integrate):
#     """Jacobi polynomial integration."""

#     input_basis_type = Jacobi

#     @staticmethod
#     def _subspace_matrix(space, input_basis):
#         N = space.coeff_size
#         a, b = input_basis.a, input_basis.b
#         vector = jacobi.integration_vector(N, a, b)
#         return (vector * space.COV.stretch)


# class Fourier(Basis, metaclass=CachedClass):
#     """Fourier cosine/sine basis."""
#     #space_type = PeriodicInterval
#     const = 1

#     def __add__(self, other):
#         space = self.space
#         if other is None:
#             return space.Fourier
#         elif other is space.Fourier:
#             return space.Fourier
#         else:
#             return NotImplemented

#     def __mul__(self, other):
#         space = self.space
#         if other is None:
#             return space.Fourier
#         elif other is space.Fourier:
#             return space.Fourier
#         else:
#             return NotImplemented

#     def __pow__(self, other):
#         return self.space.Fourier

#     @CachedAttribute
#     def wavenumbers(self):
#         kmax = self.space.kmax
#         return np.concatenate((np.arange(0, kmax+1), np.arange(-kmax, 0)))

#     def include_mode(self, mode):
#         k = mode // 2
#         if (mode % 2) == 0:
#             # Cosine modes: drop Nyquist mode
#             return (0 <= k <= self.space.kmax)
#         else:
#             # Sine modes: drop k=0 and Nyquist mode
#             return (1 <= k <= self.space.kmax)


class ComplexFourier(IntervalBasis):
    """Fourier complex exponential basis."""

    group_shape = (1,)
    native_bounds = (0, 2*np.pi)
    transforms = {}

    def __add__(self, other):
        if other is None:
            return self
        elif other is self:
            return self
        else:
            return NotImplemented

    def __mul__(self, other):
        if other is None:
            return self
        elif other is self:
            return self
        else:
            return NotImplemented

    # def __pow__(self, other):
    #     return self.space.Fourier

    def __init__(self, coord, size, bounds, dealias=1, library='matrix'):
        super().__init__(coord, size, bounds)
        self.dealias = dealias
        self.library = library
        self.kmax = kmax = (size - 1) // 2
        self.wavenumbers = np.concatenate((np.arange(0, kmax+2), np.arange(-kmax, 0)))  # Includes Nyquist mode

    def _native_grid(self, scale):
        """Native flat global grid."""
        N, = self.grid_shape((scale,))
        return (2 * np.pi / N) * np.arange(N)

    @CachedMethod
    def transform_plan(self, grid_size):
        """Build transform plan."""
        return self.transforms[self.library](grid_size, self.size)

    # def include_mode(self, mode):
    #     k = mode // 2
    #     if (mode % 2) == 0:
    #         # Cosine modes: drop Nyquist mode
    #         return (0 <= k <= self.space.kmax)
    #     else:
    #         # Sine modes: drop k=0 and Nyquist mode
    #         return (1 <= k <= self.space.kmax)



class DifferentiateComplexFourier(operators.Differentiate):
    """Complex Fourier differentiation."""

    input_basis_type = ComplexFourier
    bands = [0]
    separable = True

    @staticmethod
    def output_basis(input_basis):
        return input_basis

    @staticmethod
    def _subspace_entry(i, j, input_basis, *args):
        # dx(cos(n*x)) = -n*sin(n*x)
        # dx(sin(n*x)) = n*cos(n*x)
        if i == j:
            k = input_basis.wavenumbers[i]
            return 1j*k / input_basis.COV.stretch
        else:
            raise


# class InterpolateFourier(operators.Interpolate):
#     """Fourier series interpolation."""

#     input_basis_type = Fourier

#     @staticmethod
#     def _build_subspace_entry(j, space, input_basis, position):
#         # cos(n*x)
#         # sin(n*x)
#         n = j // 2
#         x = space.COV.native_coord(position)
#         if (j % 2) == 0:
#             return math.cos(n*x)
#         else:
#             return math.sin(n*x)


# class IntegrateFourier(operators.Integrate):
#     """Fourier series integration."""

#     input_basis_type = Fourier

#     @staticmethod
#     def _build_subspace_entry(j, space, input_basis):
#         # integral(cos(n*x), 0, 2*pi) = 2 * pi * δ(n, 0)
#         # integral(sin(n*x), 0, 2*pi) = 0
#         if j == 0:
#             return 2 * np.pi * space.COV.stretch
#         else:
#             return 0


# class DifferentiateFourier(operators.Differentiate):
#     """Fourier series differentiation."""

#     input_basis_type = Fourier
#     bands = [-1, 1]
#     separable = True

#     @staticmethod
#     def output_basis(space, input_basis):
#         return space.Fourier

#     @staticmethod
#     def _build_subspace_entry(i, j, space, input_basis):
#         # dx(cos(n*x)) = -n*sin(n*x)
#         # dx(sin(n*x)) = n*cos(n*x)
#         n = j // 2
#         if n == 0:
#             return 0
#         elif (j % 2) == 0:
#             # dx(cos(n*x)) = -n*sin(n*x)
#             if i == (j + 1):
#                 return (-n) / space.COV.stretch
#             else:
#                 return 0
#         else:
#             # dx(sin(n*x)) = n*cos(n*x)
#             if i == (j - 1):
#                 return n / space.COV.stretch
#             else:
#                 return 0


# class HilbertTransformFourier(operators.HilbertTransform):
#     """Fourier series Hilbert transform."""

#     input_basis_type = Fourier
#     bands = [-1, 1]
#     separable = True

#     @staticmethod
#     def output_basis(space, input_basis):
#         return space.Fourier

#     @staticmethod
#     def _build_subspace_entry(i, j, space, input_basis):
#         # Hx(cos(n*x)) = sin(n*x)
#         # Hx(sin(n*x)) = -cos(n*x)
#         n = j // 2
#         if n == 0:
#             return 0
#         elif (j % 2) == 0:
#             # Hx(cos(n*x)) = sin(n*x)
#             if i == (j + 1):
#                 return 1
#             else:
#                 return 0
#         else:
#             # Hx(sin(n*x)) = -cos(n*x)
#             if i == (j - 1):
#                 return (-1)
#             else:
#                 return 0


# class Sine(Basis, metaclass=CachedClass):
#     """Sine series basis."""
#     space_type = ParityInterval
#     const = None
#     supported_dtypes = {np.float64, np.complex128}

#     def __add__(self, other):
#         space = self.space
#         if other is space.Sine:
#             return space.Sine
#         else:
#             return NotImplemented

#     def __mul__(self, other):
#         space = self.space
#         if other is None:
#             return space.Sine
#         elif other is space.Sine:
#             return space.Cosine
#         elif other is space.Cosine:
#             return space.Sine
#         else:
#             return NotImplemented

#     def __pow__(self, other):
#         space = self.space
#         if (other % 2) == 0:
#             return space.Cosine
#         elif (other % 2) == 1:
#             return space.Sine
#         else:
#             return NotImplemented

#     def include_mode(self, mode):
#         # Drop k=0 and Nyquist mode
#         k = mode
#         return (1 <= k <= self.space.kmax)


# class Cosine(Basis, metaclass=CachedClass):
#     """Cosine series basis."""
#     space_type = ParityInterval
#     const = 1

#     def __add__(self, other):
#         space = self.space
#         if other is None:
#             return space.Cosine
#         elif other is space.Cosine:
#             return space.Cosine
#         else:
#             return NotImplemented

#     def __mul__(self, other):
#         space = self.space
#         if other is None:
#             return space.Cosine
#         elif other is space.Sine:
#             return space.Sine
#         elif other is space.Cosine:
#             return space.Cosine
#         else:
#             return NotImplemented

#     def __pow__(self, other):
#         return self.space.Cosine

#     def include_mode(self, mode):
#         # Drop Nyquist mode
#         k = mode
#         return (0 <= k <= self.space.kmax)


# class InterpolateSine(operators.Interpolate):
#     """Sine series interpolation."""

#     input_basis_type = Sine

#     @staticmethod
#     def _build_subspace_entry(j, space, input_basis, position):
#         # sin(n*x)
#         x = space.COV.native_coord(position)
#         return math.sin(j*x)


# class InterpolateCosine(operators.Interpolate):
#     """Cosine series interpolation."""

#     input_basis_type = Cosine

#     @staticmethod
#     def _build_subspace_entry(j, space, input_basis, position):
#         # cos(n*x)
#         x = space.COV.native_coord(position)
#         return math.cos(j*x)


# class IntegrateSine(operators.Integrate):
#     """Sine series integration."""

#     input_basis_type = Sine

#     @staticmethod
#     def _build_subspace_entry(j, space, input_basis):
#         # integral(sin(n*x), 0, pi) = (2 / n) * (n % 2)
#         if (j % 2):
#             return 0
#         else:
#             return (2 / j) * space.COV.stretch


# class IntegrateCosine(operators.Integrate):
#     """Cosine series integration."""

#     input_basis_type = Cosine

#     @staticmethod
#     def _build_subspace_entry(j, space, input_basis):
#         # integral(cos(n*x), 0, pi) = pi * δ(n, 0)
#         if j == 0:
#             return np.pi * space.COV.stretch
#         else:
#             return 0


# class DifferentiateSine(operators.Differentiate):
#     """Sine series differentiation."""

#     input_basis_type = Sine
#     bands = [0]
#     separable = True

#     @staticmethod
#     def output_basis(space, input_basis):
#         return space.Cosine

#     @staticmethod
#     def _build_subspace_entry(i, j, space, input_basis):
#         # dx(sin(n*x)) = n*cos(n*x)
#         if i == j:
#             return j / space.COV.stretch
#         else:
#             return 0


# class DifferentiateCosine(operators.Differentiate):
#     """Cosine series differentiation."""

#     input_basis_type = Cosine
#     bands = [0]
#     separable = True

#     @staticmethod
#     def output_basis(space, input_basis):
#         return space.Sine

#     @staticmethod
#     def _build_subspace_entry(i, j, space, input_basis):
#         # dx(cos(n*x)) = -n*sin(n*x)
#         if i == j:
#             return (-j) / space.COV.stretch
#         else:
#             return 0


# class HilbertTransformSine(operators.HilbertTransform):
#     """Sine series Hilbert transform."""

#     input_basis_type = Sine
#     bands = [0]
#     separable = True

#     @staticmethod
#     def output_basis(space, input_basis):
#         return space.Cosine

#     @staticmethod
#     def _build_subspace_entry(i, j, space, input_basis):
#         # Hx(sin(n*x)) = -cos(n*x)
#         if i == j:
#             return (-1)
#         else:
#             return 0


# class HilbertTransformCosine(operators.HilbertTransform):
#     """Cosine series Hilbert transform."""

#     input_basis_type = Cosine
#     bands = [0]
#     separable = True

#     @staticmethod
#     def output_basis(space, input_basis):
#         return space.Sine

#     @staticmethod
#     def _build_subspace_entry(i, j, space, input_basis):
#         # Hx(cos(n*x)) = sin(n*x)
#         if i == j:
#             return 1
#         else:
#             return 0


class MultidimensionalBasis(Basis):

    def forward_transform(self, field, axis, gdata, cdata):
        subaxis = axis - self.axis
        return self.forward_transforms[subaxis](field, axis, gdata, cdata)

    def backward_transform(self, field, axis, cdata, gdata):
        subaxis = axis - self.axis
        return self.backward_transforms[subaxis](field, axis, cdata, gdata)


class SpinBasis(MultidimensionalBasis):

    def __init__(self, coordsystem, shape, azimuth_library='matrix'):
        super().__init__(coordsystem.coords)
        self.coordsystem = coordsystem
        self.shape = shape
        self.azimuth_library = azimuth_library
        self.mmax = (shape[0] - 1) // 2
        self.azimuth_basis = ComplexFourier(self.coords[0], shape[0], bounds=(0, 2*np.pi), library=azimuth_library)
        self.global_grid_azimuth = self.azimuth_basis.global_grid
        self.local_grid_azimuth = self.azimuth_basis.local_grid
        self.forward_transform_azimuth = self.azimuth_basis.forward_transform
        self.backward_transform_azimuth = self.azimuth_basis.backward_transform

    @CachedAttribute
    def local_m(self):
        layout = self.dist.coeff_layout
        local_m_elements = layout.local_elements(self.domain, scales=1)[self.axis]
        return tuple(self.azimuth_basis.wavenumbers[local_m_elements])

    @CachedMethod
    def spin_weights(self, tensorsig):
        # Spin-component ordering: [-, +]
        Ss = np.array([-1, 1], dtype=int)
        S = np.zeros([vs.dim for vs in tensorsig], dtype=int)
        for i, vs in enumerate(tensorsig):
            if self.coordsystem is vs: # kludge before we decide how compound coordinate systems work
                S[axslice(i, 0, self.dim)] += reshape_vector(Ss, dim=len(tensorsig), axis=i)
            elif self.coordsystem in vs:
                n = vs.get_index(self.coordsystem)
                S[axslice(i, n, n+self.dim)] += reshape_vector(Ss, dim=len(tensorsig), axis=i)
        return S

    @CachedMethod
    def spin_recombination_matrices(self, tensorsig):
        """Build matrices for appling spin recombination to each tensor rank."""
        # Setup unitary spin recombination
        # [azimuth, colatitude] -> [-, +]
        Us = np.array([[-1j, 1], [1j, 1]]) / np.sqrt(2)
        # Perform unitary spin recombination along relevant tensor indeces
        U = []
        for i, vs in enumerate(tensorsig):
            if self.coordsystem is vs: # kludge before we decide how compound coordinate systems work
                Ui = np.identity(vs.dim, dtype=np.complex128)
                Ui[:self.dim, :self.dim] = Us
                U.append(Ui)
            elif self.coordsystem in vs.spaces:
                n = vector_space.get_index(self.space)
                Ui = np.identity(vector_space.dim, dtype=np.complex128)
                Ui[n:n+self.dim, n:n+self.dim] = Us
                U.append(Ui)
            else:
                U.append(None)
        return U

    def forward_spin_recombination(self, tensorsig, gdata):
        """Apply component-to-spin recombination in place."""
        U = self.spin_recombination_matrices(tensorsig)
        for i, Ui in enumerate(U):
            if Ui is not None:
                # Directly apply U
                apply_matrix(Ui, gdata, axis=i, out=gdata)

    def backward_spin_recombination(self, tensorsig, gdata):
        """Apply spin-to-component recombination in place."""
        U = self.spin_recombination_matrices(tensorsig)
        for i, Ui in enumerate(U):
            if Ui is not None:
                # Apply U^H (inverse of U)
                apply_matrix(Ui.T.conj(), gdata, axis=i, out=gdata)


class RegularityBasis(MultidimensionalBasis):

    def __init__(self, coordsystem, shape, azimuth_library='matrix', colatitude_library='matrix'):
        super().__init__(coordsystem.coords)
        self.coordsystem = coordsystem
        self.shape = shape
        self.azimuth_library = azimuth_library
        self.colatitude_library = colatitude_library
        self.sphere_basis = SWSH(coordsystem, shape[:2], azimuth_library=azimuth_library, colatitude_library=colatitude_library)
        self.mmax = self.sphere_basis.mmax
        self.Lmax = self.sphere_basis.Lmax
        self.global_grid_azimuth = self.sphere_basis.global_grid_azimuth
        self.global_grid_colatitude = self.sphere_basis.global_grid_colatitude
        self.local_grid_azimuth = self.sphere_basis.local_grid_azimuth
        self.local_grid_colatitude = self.sphere_basis.local_grid_colatitude
        self.forward_transform_azimuth = self.sphere_basis.forward_transform_azimuth
        self.forward_transform_colatitude = self.sphere_basis.forward_transform_colatitude
        self.backward_transform_azimuth = self.sphere_basis.backward_transform_azimuth
        self.backward_transform_colatitude = self.sphere_basis.backward_transform_colatitude

    @CachedAttribute
    def local_l(self):
        layout = self.dist.coeff_layout
        local_l_elements = layout.local_elements(self.domain, scales=1)[self.axis+1]
        return tuple(self.sphere_basis.degrees[local_l_elements])

    @CachedMethod
    def xi(self,mu,l):
        import dedalus_sphere
        return dedalus_sphere.intertwiner.xi(mu,l)

    @CachedMethod
    def regularity_allowed(self,l,regularity):
        import dedalus_sphere
        Rb = np.array([-1, 1, 0], dtype=int)
        if regularity == (): return True
        return not dedalus_sphere.intertwiner.forbidden_regularity(l,Rb[np.array(regularity)])

    @CachedMethod
    def radial_recombinations(self, tensorsig):
        import dedalus_sphere
        # For now only implement recombinations for Ball-only tensors
        for vs in tensorsig:
            if self.coordsystem is not vs:
                raise ValueError("Only supports tensors over ball.")
        order = len(tensorsig)

        Q_matrices = np.zeros((len(self.local_l),3**order,3**order))
        for i, l in enumerate(self.local_l):
            for j in range(3**order):
                for k in range(3**order):
                    s = dedalus_sphere.intertwiner.index2tuple(j,order,indexing=(-1,+1,0))
                    r = dedalus_sphere.intertwiner.index2tuple(k,order,indexing=(-1,+1,0))
                    Q_matrices[i,j,k] = dedalus_sphere.intertwiner.regularity2spinMap(l,s,r)
        return Q_matrices

    @CachedMethod
    def regularity_classes(self, tensorsig):
        # Regularity-component ordering: [-, +, 0]
        Rb = np.array([-1, 1, 0], dtype=int)
        R = np.zeros([vs.dim for vs in tensorsig], dtype=int)
        for i, vs in enumerate(tensorsig):
            if self.coordsystem is vs: # kludge before we decide how compound coordinate systems work
                R[axslice(i, 0, self.dim)] += reshape_vector(Rb, dim=len(tensorsig), axis=i)
            elif self.space in vs.spaces:
                n = vs.get_index(self.space)
                R[axslice(i, n, n+self.dim)] += reshape_vector(Rb, dim=len(tensorsig), axis=i)
        return R

    def forward_regularity_recombination(self, field, axis, gdata):
        # Apply radial recombinations
        order = len(field.tensorsig)
        if order > 0:
            Q = self.radial_recombinations(field.tensorsig)
            # Flatten tensor axes
            shape = gdata.shape
            order = len(field.tensorsig)
            temp = gdata.reshape((-1,)+shape[order:])
            # Apply Q transformations for each l to flattened tensor data
            for l_index, Q_l in enumerate(Q):
                # Here the l axis is 'axis' instead of 'axis-1' since we have one tensor axis prepended
                l_view = temp[axslice(axis, l_index, l_index+1)]
                apply_matrix(Q_l.T, l_view, axis=0, out=l_view)

    def backward_regularity_recombination(self, field, axis, gdata):
        # Apply radial recombinations
        order = len(field.tensorsig)
        if order > 0:
            Q = self.radial_recombinations(field.tensorsig)
            # Flatten tensor axes
            shape = gdata.shape
            order = len(field.tensorsig)
            temp = gdata.reshape((-1,)+shape[order:])
            # Apply Q transformations for each l to flattened tensor data
            for l_index, Q_l in enumerate(Q):
                # Here the l axis is 'axis' instead of 'axis-1' since we have one tensor axis prepended
                l_view = temp[axslice(axis, l_index, l_index+1)]
                apply_matrix(Q_l, l_view, axis=0, out=l_view)

class DiskBasis(SpinBasis):

    space_type = Disk
    dim = 2

    def __init__(self, space, dk=0):
        self._check_space(space)
        self.space = space
        self.dk = dk
        self.k = space.k0 + dk
        self.axis = space.axis
        self.azimuth_basis = Fourier(self.space.azimuth_space)
        self.forward_transforms = [self.forward_transform_azimuth,
                                   self.forward_transform_radius]
        self.backward_transforms = [self.backward_transform_azimuth,
                                    self.backward_transform_radius]
        #self.forward_transform_azimuth = self.azimuth_basis.forward_transform
        #self.backward_transform_azimuth = self.azimuth_basis.backward_transform

    def forward_transform_radius(self, field, axis, gdata, cdata):
        # Apply spin recombination
        self.forward_spin_recombination(field.tensorsig, gdata)
        # Perform transforms component-by-component
        S = self.spin_weights(field.tensorsig)
        k0, k = self.k0, self.k
        local_m = self.local_m
        for i, s in np.ndenumerate(S):
            transforms.forward_disk(gdata[i], cdata[i], axis=axis, k0=k0, k=k, s=s, local_m=local_m)

    def backward_transform_radius(self, field, axis, cdata, gdata):
        # Perform transforms component-by-component
        S = self.spin_weights(field.tensorsig)
        k0, k = self.k0, self.k
        local_m = self.local_m
        for i, s in np.ndenumerate(S):
            transforms.backward_disk(cdata[i], gdata[i], axis=axis, k0=k0, k=k, s=s, local_m=local_m)
        # Apply spin recombination
        self.backward_spin_recombination(field.tensorsig, gdata)


class SpinWeightedSphericalHarmonics(SpinBasis):

    dim = 2
    dims = ['azimuth', 'colatitude']
    group_shape = (1, 1)
    transforms = {}

    def __init__(self, coordsystem, shape, radius=1, colatitude_library='matrix', **kw):
        super().__init__(coordsystem, shape, **kw)
        if radius <= 0:
            raise ValueError("Radius must be positive.")
        self.radius = radius
        self.colatitude_library = colatitude_library
        self.Lmax = shape[1] - 1
        if self.mmax > self.Lmax:
            raise ValueError("shape[0] cannot be more than twice shape[1].")
        self.degrees = np.arange(shape[1])
        self.forward_transforms = [self.forward_transform_azimuth,
                                   self.forward_transform_colatitude]
        self.backward_transforms = [self.backward_transform_azimuth,
                                    self.backward_transform_colatitude]

    def global_grids(self, scales):
        return (self.global_grid_azimuth(scales[0]),
                self.global_grid_colatitude(scales[1]))

    def global_grid_colatitude(self, scale):
        theta = self._native_colatitude_grid(scale)
        return reshape_vector(theta, dim=self.dist.dim, axis=self.axis+1)

    def local_grids(self, scales):
        return (self.local_grid_azimuth(scales[0]),
                self.local_grid_colatitude(scales[1]))

    def local_grid_colatitude(self, scale):
        local_elements = self.dist.grid_layout.local_elements(self.domain, scales=scale)[self.axis+1]
        theta = self._native_colatitude_grid(scale)[local_elements]
        return reshape_vector(theta, dim=self.dist.dim, axis=self.axis+1)

    def _native_colatitude_grid(self, scale):
        N = int(np.ceil(scale * self.shape[1]))
        cos_theta, weights = dedalus_sphere.sphere.quadrature(Lmax=N-1)
        theta = np.arccos(cos_theta).astype(np.float64)
        return theta

    @CachedMethod
    def transform_plan(self, grid_size, s):
        """Build transform plan."""
        return self.transforms[self.colatitude_library](grid_size, self.Lmax+1, self.local_m, s)

    def forward_transform_colatitude(self, field, axis, gdata, cdata):
        data_axis = len(field.tensorsig) + axis
        grid_size = gdata.shape[data_axis]
        # Apply spin recombination
        self.forward_spin_recombination(field.tensorsig, gdata)
        # Perform transforms component-by-component
        S = self.spin_weights(field.tensorsig)
        for i, s in np.ndenumerate(S):
            plan = self.transform_plan(grid_size, s)
            plan.forward(gdata[i], cdata[i], axis)

    def backward_transform_colatitude(self, field, axis, cdata, gdata):
        data_axis = len(field.tensorsig) + axis
        grid_size = gdata.shape[data_axis]
        # Perform transforms component-by-component
        S = self.spin_weights(field.tensorsig)
        for i, s in np.ndenumerate(S):
            plan = self.transform_plan(grid_size, s)
            plan.backward(cdata[i], gdata[i], axis)
        # Apply spin recombination
        self.backward_spin_recombination(field.tensorsig, gdata)

    @CachedMethod
    def k_vector(self,mu,m,s,local_l):
        import dedalus_sphere
        vector = np.zeros(len(local_l))
        Lmin = max(abs(m),abs(s),abs(s+mu))
        for i,l in enumerate(local_l):
            if l < Lmin: vector[i] = 0
            else: vector[i] = dedalus_sphere.sphere.k_element(mu,l,s,self.radius)
        return vector

SWSH = SpinWeightedSphericalHarmonics


class BallBasis(RegularityBasis):

    dim = 3
    dims = ['azimuth', 'colatitude', 'radius']
    group_shape = (1, 1, 1)
    transforms = {}

    def __init__(self, coordsystem, shape, radius=1, k=0, alpha=0, azimuth_library='matrix', colatitude_library='matrix', radius_library='matrix'):
        super().__init__(coordsystem, shape, azimuth_library=azimuth_library, colatitude_library=colatitude_library)
        if radius <= 0:
            raise ValueError("Radius must be positive.")
        self.radius = radius
        self.alpha = alpha
        self.k = k
        self.radius_library = radius_library
        self.radial_COV = AffineCOV((0, 1), (0, radius))
        self.Nmax = shape[2] - 1
        if self.Lmax + 1 > 2 * (self.Nmax + 1):
            raise ValueError("shape[1] cannot be more than twice shape[2]")
        self.forward_transforms = [self.forward_transform_azimuth,
                                   self.forward_transform_colatitude,
                                   self.forward_transform_radius]
        self.backward_transforms = [self.backward_transform_azimuth,
                                    self.backward_transform_colatitude,
                                    self.backward_transform_radius]

    def _new_k(self, k):
        return BallBasis(self.coordsystem, self.shape, radius = self.radius, k=k, alpha=self.alpha,
                         azimuth_library=self.azimuth_library, colatitude_library=self.colatitude_library,
                         radius_library=self.radius_library)

    def global_grids(self, scales):
        return (self.global_grid_azimuth(scales[0]),
                self.global_grid_colatitude(scales[1]),
                self.global_grid_radius(scales[2]))

    def local_grids(self, scales):
        return (self.local_grid_azimuth(scales[0]),
                self.local_grid_colatitude(scales[1]),
                self.local_grid_radius(scales[2]))

    def global_grid_radius(self, scale):
        native_grid = self._native_radius_grid(scale)[local_elements]
        problem_grid = self.radial_COV.problem_coord(native_grid)
        return reshape_vector(problem_grid, dim=self.dist.dim, axis=self.axis+2)

    def local_grid_radius(self, scale):
        local_elements = self.dist.grid_layout.local_elements(self.domain, scales=scale)[self.axis+2]
        native_grid = self._native_radius_grid(scale)[local_elements]
        problem_grid = self.radial_COV.problem_coord(native_grid)
        return reshape_vector(problem_grid, dim=self.dist.dim, axis=self.axis+2)

    def _native_radius_grid(self, scale):
        N = int(np.ceil(scale * self.shape[2]))
        z, weights = dedalus_sphere.ball.quadrature(3, N-1, niter=3, alpha=self.alpha)
        r = np.sqrt((z + 1) / 2)
        return r.astype(np.float64)

    @CachedMethod
    def transform_plan(self, grid_size, regularity, deg, k, alpha):
        """Build transform plan."""
        return self.transforms[self.radius_library](grid_size, self.Lmax+1, self.local_l, regularity, deg, k, alpha)

    def forward_transform_radius(self, field, axis, gdata, cdata):
        data_axis = len(field.tensorsig) + axis
        grid_size = gdata.shape[data_axis]
        # Apply regularity recombination
        self.forward_regularity_recombination(field, axis, gdata)
        # Perform radial transforms component-by-component
        R = self.regularity_classes(field.tensorsig)
        for i, r in np.ndenumerate(R):
           plan = self.transform_plan(grid_size, i, r, self.k, self.alpha)
           plan.forward(gdata[i], cdata[i], axis)

    def backward_transform_radius(self, field, axis, cdata, gdata):
        data_axis = len(field.tensorsig) + axis
        grid_size = gdata.shape[data_axis]
        # Perform radial transforms component-by-component
        R = self.regularity_classes(field.tensorsig)
        for i, r in np.ndenumerate(R):
           plan = self.transform_plan(grid_size, i, r, self.k, self.alpha)
           plan.backward(cdata[i], gdata[i], axis)
        # Apply regularity recombinations
        self.backward_regularity_recombination(field, axis, gdata)

    @CachedMethod
    def operator_matrix(self,op,l,deg):
        import dedalus_sphere
        return dedalus_sphere.ball.operator(3,op,self.Nmax,self.k,l,deg,radius=self.radius,alpha=self.alpha).astype(np.float64)

    def regtotal(self, regindex):
        regorder = np.array([-1, 1, 0])
        regsig = regorder[np.array(regindex)]
        return regsig.sum()

    def n_limits(self, regindex, ell):
        if not self.regularity_allowed(ell, regindex):
            return None
        regtotal = self.regtotal(regindex)
        nmin = dedalus_sphere.ball.Nmin(ell, regtotal)
        return (nmin, self.Nmax)

    def n_slice(self, regindex, ell):
        if not self.regularity_allowed(ell, regindex):
            return None
        nmin, nmax = self.n_limits(regindex, ell)
        return slice(nmin, nmax+1)



class GradientBall(operators.SphericalGradient):
    """Gradient operator on the ball."""

    input_basis_type = BallBasis
    separable = False

    @classmethod
    def _check_args(cls, operand, cs, out=None):
        # Dispatch by operand basis
        #if isinstance(operand, Operand):
        basis = operand.get_basis(cs)
        if isinstance(basis, cls.input_basis_type):
            return True
        return False

    @staticmethod
    def output_basis(input_basis):
        out = input_basis._new_k(input_basis.k + 1)
        return out



from . import transforms
