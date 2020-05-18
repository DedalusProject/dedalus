"""
Abstract and built-in classes for spectral bases.

"""

import math
import numpy as np
from scipy import sparse
from functools import reduce
import operator

from . import operators
from ..tools.array import axslice
from ..tools.array import apply_matrix
from ..tools.cache import CachedAttribute
from ..tools.cache import CachedMethod
from ..tools.cache import CachedClass
from ..tools import jacobi
from ..tools import clenshaw
from ..tools.array import reshape_vector, axindex, axslice
from ..tools.dispatch import MultiClass

from .spaces import ParityInterval, Disk
from .coords import Coordinate, S2Coordinates, SphericalCoordinates
from .domain import Domain
from .field  import Operand
import dedalus_sphere
#from . import transforms

import logging
logger = logging.getLogger(__name__.split('.')[-1])

from ..tools.config import config
#DEFAULT_LIBRARY = config['transforms'].get('DEFAULT_LIBRARY')
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
        self.dist = coords.dist
        self.axis = coords.axis
        self.domain = Domain(self.dist, bases=(self,))

    # def __repr__(self):
    #     return '<%s %i>' %(self.__class__.__name__, id(self))

    # def __str__(self):
    #     return '%s.%s' %(self.space.name, self.__class__.__name__)

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    @property
    def first_axis(self):
        return self.axis

    @property
    def last_axis(self):
        return self.axis + self.dim - 1

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

    def ncc_matrix(self, arg_basis, coeffs, cutoff=1e-6):
        """Build NCC matrix via direct summation."""
        N = len(coeffs)
        for i in range(N):
            coeff = coeffs[i]
            # Build initial matrix
            if i == 0:
                matrix = self.product_matrix(arg_basis, i)
                total = 0 * sparse.kron(matrix, coeff)
                total.eliminate_zeros()
            if len(coeff.shape) or (abs(coeff) > cutoff):
                matrix = self.product_matrix(arg_basis, i)
                total = total + sparse.kron(matrix, coeff)
        return total

    def product_matrix(self, arg_basis, i):
        if arg_basis is None:
            N = self.size
            return sparse.coo_matrix(([1],([i],[0])), shape=(N,1)).tocsr()
        else:
            raise NotImplementedError()


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

    def __init__(self, coord, size, bounds, dealias):
        self.coord = coord
        coord.check_bounds(bounds)
        self.size = size
        self.shape = (size,)
        self.bounds = bounds
        self.dealias = (dealias,)
        self.COV = AffineCOV(self.native_bounds, bounds)
        super().__init__(coord)

    # Why do we need this?
    def global_grids(self, scales=None):
        """Global grids."""
        if scales == None: scales = (1,)
        return (self.global_grid(scales[0]),)

    def global_grid(self, scale=None):
        """Global grid."""
        if scale == None: scale = 1
        native_grid = self._native_grid(scale)
        problem_grid = self.COV.problem_coord(native_grid)
        return reshape_vector(problem_grid, dim=self.dist.dim, axis=self.axis)

    # Why do we need this?
    def local_grids(self, scales=None):
        """Local grids."""
        if scales == None: scales = (1,)
        return (self.local_grid(scales[0]),)

    def local_grid(self, scale=None):
        """Local grid."""
        if scale == None: scale = 1
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

    def local_groups(self, basis_coupling):
        coupling, = basis_coupling
        if coupling:
            return [[None]]
        else:
            local_chunks = self.dist.coeff_layout.local_chunks(self.domain, scales=1)[self.axis]
            return [[group] for group in local_chunks]

    def local_group_slices(self, basis_group):
        group, = basis_group
        # Return slices
        if group is None:
            # Return all coefficients
            return [slice(None)]
        else:
            # Get local groups
            local_chunks = self.dist.coeff_layout.local_chunks(self.domain, scales=1)[self.axis]
            # Groups are stored sequentially
            local_index = list(local_chunks).index(group)
            group_size = self.group_shape[0]
            return [slice(local_index*group_size, (local_index+1)*group_size)]


class Jacobi(IntervalBasis, metaclass=CachedClass):
    """Jacobi polynomial basis."""

    group_shape = (1,)
    native_bounds = (-1, 1)
    transforms = {}

    def __init__(self, coord, size, bounds, a, b, a0=None, b0=None, dealias=1, library='matrix'):
        super().__init__(coord, size, bounds, dealias)
        # Default grid parameters
        if a0 is None:
            a0 = a
        if b0 is None:
            b0 = b
        self.a = float(a)
        self.b = float(b)
        self.a0 = float(a0)
        self.b0 = float(b0)
        self.library = library
        self.grid_params = (coord, bounds, a0, b0)
        #self.const = 1 / np.sqrt(jacobi.mass(self.a, self.b))

    def _new_a_b(self, a, b):
        return Jacobi(self.coord, self.size, self.bounds, a, b, a0=self.a0, b0=self.b0, dealias=self.dealias[0], library=self.library)

    def _native_grid(self, scale):
        """Native flat global grid."""
        N, = self.grid_shape((scale,))
        return jacobi.build_grid(N, a=self.a0, b=self.b0)

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
                dealias = max(self.dealias[0], other.dealias[0])
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
                dealias = max(self.dealias[0], other.dealias[0])
                return Jacobi(self.coord, size, self.bounds, a, b, a0=self.a0, b0=self.b0, dealias=dealias, library=self.library)
        if isinstance(other, SpinWeightedSphericalHarmonics):
            return other.__mul__(self)
        return NotImplemented

    # def include_mode(self, mode):
    #     return (0 <= mode < self.space.coeff_size)

    def ncc_matrix(self, arg_basis, coeffs, cutoff=1e-6):
        """Build NCC matrix via Clenshaw algorithm."""
        if arg_basis is None:
            return super().ncc_matrix(arg_basis, coeffs)
        # Kronecker Clenshaw on argument Jacobi matrix
        elif isinstance(arg_basis, Jacobi):
            N = self.size
            J = jacobi.jacobi_matrix(N, arg_basis.a, arg_basis.b)
            A, B = clenshaw.jacobi_recursion(N, self.a, self.b, J)
            f0 = self.const * sparse.identity(N)
            total = clenshaw.kronecker_clenshaw(coeffs, A, B, f0, cutoff=cutoff)
            # Conversion matrix
            input_basis = arg_basis
            output_basis = (self * arg_basis)
            conversion = ConvertJacobi._subspace_matrix(input_basis, output_basis)
            # Kronecker with identity for matrix coefficients
            coeff_size = total.shape[0] // conversion.shape[0]
            if coeff_size > 1:
                conversion = sparse.kron(conversion, sparse.identity(coeff_size))
            return (conversion @ total)
        else:
            raise ValueError("Jacobi ncc_matrix not implemented for basis type: %s" %type(arg_basis))


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
    return Ultraspherical(*args, alpha=1, **kw)


def ChebyshevV(*args, **kw):
    return Ultraspherical(*args, alpha=2, **kw)


class ConvertJacobi(operators.Convert, operators.SpectralOperator1D):
    """Jacobi polynomial conversion."""

    input_basis_type = Jacobi
    output_basis_type = Jacobi
    subaxis_dependence = [True]
    subaxis_coupling = [True]

    @staticmethod
    def _subspace_matrix(input_basis, output_basis):
        N = input_basis.size
        a0, b0 = input_basis.a, input_basis.b
        a1, b1 = output_basis.a, output_basis.b
        matrix = jacobi.conversion_matrix(N, a0, b0, a1, b1)
        return matrix.tocsr()


class ConvertConstantJacobi(operators.Convert, operators.SpectralOperator1D):

    input_basis_type = type(None)
    output_basis_type = Jacobi
    subaxis_dependence = [True]
    subaxis_coupling = [False]

    @staticmethod
    def _subspace_matrix(input_basis, output_basis):
        basis = output_basis
        MMT = basis.transforms['matrix'](grid_size=1, coeff_size=basis.size, a=basis.a, b=basis.b, a0=basis.a0, b0=basis.b0)
        return MMT.forward_matrix


class DifferentiateJacobi(operators.Differentiate):
    """Jacobi polynomial differentiation."""

    input_basis_type = Jacobi
    subaxis_dependence = [True]
    subaxis_coupling = [True]

    @staticmethod
    def _output_basis(input_basis):
        a = input_basis.a + 1
        b = input_basis.b + 1
        return input_basis._new_a_b(a, b)

    @staticmethod
    def _subspace_matrix(input_basis):
        N = input_basis.size
        a, b = input_basis.a, input_basis.b
        matrix = jacobi.differentiation_matrix(N, a, b)
        return (matrix.tocsr() / input_basis.COV.stretch)


class InterpolateJacobi(operators.Interpolate, operators.SpectralOperator1D):
    """Jacobi polynomial interpolation."""

    input_basis_type = Jacobi
    subaxis_dependence = [True]
    subaxis_coupling = [True]

    @staticmethod
    def _subspace_matrix(input_basis, position):
        N = input_basis.size
        a, b = input_basis.a, input_basis.b
        x = input_basis.COV.native_coord(position)
        interp_vector = jacobi.interpolation_vector(N, a, b, x)
        # Return as 1*N array
        return interp_vector[None,:]

    @staticmethod
    def _output_basis(input_basis, position):
        return None


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
        super().__init__(coord, size, bounds, dealias)
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


class ConvertConstantComplexFourier(operators.Convert, operators.SpectralOperator1D):

    input_basis_type = type(None)
    output_basis_type = ComplexFourier
    subaxis_dependence = [True]
    subaxis_coupling = [False]

    @staticmethod
    def _subspace_matrix(input_basis, output_basis):
        basis = output_basis
        MMT = basis.transforms['matrix'](grid_size=1, coeff_size=output_basis.size)
        return MMT.forward_matrix


class DifferentiateComplexFourier(operators.Differentiate):
    """Complex Fourier differentiation."""

    input_basis_type = ComplexFourier
    bands = [0]
    subaxis_dependence = [True]
    subaxis_coupling = [False]

    @staticmethod
    def _output_basis(input_basis):
        return input_basis

    @staticmethod
    def _subspace_entry(i, j, input_basis, *args):
        if i == j:
            k = input_basis.wavenumbers[i]
            return 1j*k / input_basis.COV.stretch
        else:
            raise


class InterpolateComplexFourier(operators.Interpolate, operators.SpectralOperator1D):
    """Complex Fourier interpolation."""

    input_basis_type = ComplexFourier
    subaxis_dependence = [True]
    subaxis_coupling = [True]

    @staticmethod
    def _subspace_matrix(input_basis, position):
        N = input_basis.size
        x = input_basis.COV.native_coord(position)
        interp_vector = np.array([np.exp(1j*k*x) for k in input_basis.wavenumbers])
        # Return as 1*N array
        return interp_vector[None,:]

    @staticmethod
    def _output_basis(input_basis, position):
        return None


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


# These are common for S2 and D2
class SpinBasis(MultidimensionalBasis):

    def __init__(self, coordsystem, shape, dealias, azimuth_library='matrix'):
        self.coordsystem = coordsystem
        self.shape = shape
        if np.isscalar(dealias):
            self.dealias = (dealias,) * 2
        elif len(dealias) != 2:
            raise ValueError("dealias must either be a number or a tuple of two numbers")
        else:
            self.dealias = dealias
        self.azimuth_library = azimuth_library
        self.mmax = (shape[0] - 1) // 2
        self.azimuth_basis = ComplexFourier(coordsystem.coords[0], shape[0], bounds=(0, 2*np.pi), library=azimuth_library)
        self.global_grid_azimuth = self.azimuth_basis.global_grid
        self.local_grid_azimuth = self.azimuth_basis.local_grid
        self.forward_transform_azimuth = self.azimuth_basis.forward_transform
        self.backward_transform_azimuth = self.azimuth_basis.backward_transform
        super().__init__(coordsystem)

    @CachedAttribute
    def local_m(self):
        layout = self.dist.coeff_layout
        local_m_elements = layout.local_elements(self.domain, scales=1)[self.axis]
        return tuple(self.azimuth_basis.wavenumbers[local_m_elements])

    def local_groups(self, basis_coupling):
        m_coupling, ell_coupling = basis_coupling
        if (not m_coupling) and (not ell_coupling):
            local_chunks = self.dist.coeff_layout.local_chunks(self.domain, scales=1)
            m_chunks = local_chunks[self.first_axis]
            ell_chunks = local_chunks[self.first_axis+1]
            groups = []
            # Add groups satisfying triangular truncation
            for m_chunk in m_chunks:
                m = self.azimuth_basis.wavenumbers[m_chunk]
                for ell_chunk in ell_chunks:
                    ell = ell_chunk
                    if ell >= np.abs(m):
                        groups.append([m_chunk, ell_chunk])
            return groups
        else:
            raise NotImplementedError()

    def local_group_slices(self, basis_group):
        m_group, ell_group = basis_group
        if (m_group is not None) and (ell_group is not None):
            local_chunks = self.dist.coeff_layout.local_chunks(self.domain, scales=1)
            m_chunks = local_chunks[self.first_axis]
            m_index = list(m_chunks).index(m_group)
            m_gs = self.group_shape[0]
            m_slice = slice(m_index*m_gs, (m_index+1)*m_gs)
            ell_chunks = local_chunks[self.last_axis]
            ell_index = list(ell_chunks).index(ell_group)
            ell_gs = self.group_shape[1]
            ell_slice = slice(ell_index*ell_gs, (ell_index+1)*ell_gs)
            return [m_slice, ell_slice]
        else:
            raise NotImplementedError()

    @CachedMethod
    def spin_weights(self, tensorsig):
        # Spin-component ordering: [-, +, 0]
        Ss = {2:np.array([-1, 1], dtype=int), 3:np.array([-1, 1, 0], dtype=int)}
        S = np.zeros([cs.dim for cs in tensorsig], dtype=int)
        for i, cs in enumerate(tensorsig):
            if self.coordsystem == cs or (type(cs) is SphericalCoordinates and cs.S2coordsys == self.coordsystem):
                S[axslice(i, 0, cs.dim)] += reshape_vector(Ss[cs.dim], dim=len(tensorsig), axis=i)
            #if self.coordsystem is vs: # kludge before we decide how compound coordinate systems work
            #    S[axslice(i, 0, self.dim)] += reshape_vector(Ss, dim=len(tensorsig), axis=i)
            #elif self.coordsystem in vs:
            #    n = vs.get_index(self.coordsystem)
            #    S[axslice(i, n, n+self.dim)] += reshape_vector(Ss, dim=len(tensorsig), axis=i)
        return S

    @CachedMethod
    def spin_recombination_matrices(self, tensorsig):
        """Build matrices for appling spin recombination to each tensor rank."""
        # Setup unitary spin recombination
        # [azimuth, colatitude] -> [-, +]
        Us = {2:np.array([[-1j, 1], [1j, 1]]) / np.sqrt(2),
              3:np.array([[-1j, 1, 0], [1j, 1, 0], [0, 0, np.sqrt(2)]]) / np.sqrt(2)}
        # Perform unitary spin recombination along relevant tensor indeces
        U = []
        for i, cs in enumerate(tensorsig):
            if self.coordsystem == cs or (type(cs) is SphericalCoordinates and cs.S2coordsys == self.coordsystem):
                U.append(Us[cs.dim])
            #if self.coordsystem is vs: # kludge before we decide how compound coordinate systems work
            #    Ui = np.identity(vs.dim, dtype=np.complex128)
            #    Ui[:self.dim, :self.dim] = Us
            #    U.append(Ui)
            #elif self.coordsystem in vs.spaces:
            #    n = vector_space.get_index(self.space)
            #    Ui = np.identity(vector_space.dim, dtype=np.complex128)
            #    Ui[n:n+self.dim, n:n+self.dim] = Us
            #    U.append(Ui)
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


# This does not work right now... Look at SWSH for a better idea of what the basis class should look like
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

    def __init__(self, coordsystem, shape, radius=1, dealias=(1,1), colatitude_library='matrix', **kw):
        super().__init__(coordsystem, shape, dealias, **kw)
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
        self.grid_params = (coordsystem, radius, dealias)

    def __eq__(self, other):
        if isinstance(other, SpinWeightedSphericalHarmonics):
            if self.grid_params == other.grid_params:
                if self.shape == other.shape:
                    return True
        return False

    def __hash__(self):
        return id(self)

    def __add__(self, other):
        if other is None:
            return self
        if other is self:
            return self
        if isinstance(other, SpinWeightedSphericalHarmonics):
            if self.radius == other.radius:
                shape = tuple(np.maximum(self.shape, other.shape))
                return SpinWeightedSphericalHarmonics(self.coordsystem, shape, radius=self.radius, dealias=self.dealias)
        return NotImplemented

    def __mul__(self, other):
        if other is None:
            return self
        if other is self:
            return self
        if isinstance(other, SpinWeightedSphericalHarmonics):
            if self.radius == other.radius:
                shape = tuple(np.maximum(self.shape, other.shape))
                return SpinWeightedSphericalHarmonics(self.coordsystem, shape, radius=self.radius, dealias=self.dealias)
        if isinstance(other, Jacobi):
            if isinstance(other.coord.cs, coords.SphericalCoordinates):
                spherical_coords = other.coord.cs
                if self.coordsystem == spherical_coords.S2coordsys and other.coord == spherical_coords.radius:
                    if other.bounds[0] == 0:
                        raise ValueError("Cannot multiply a radial function starting at r=0 by an angular function")
                    else:
                        shape = (self.shape[0], self.shape[1], other.shape)
                        dealias = (self.dealias[0], self.dealias[1], other.dealias)
                        return SphericalShellBasis(spherical_coords, shape, radii=other.bounds, alpha=(other.a0, other.b0), dealias=dealias,
                                                   azimuth_library=self.azimuth_library, colatitude_library=self.colatitude_library,
                                                   radius_library=other.library)
        return NotImplemented

    def coeff_subshape(self, groups):
        subshape = []
        for subaxis, group in enumerate(groups):
            if group is None:
                subshape.append(self.shape[subaxis])
            else:
                subshape.append(self.group_shape[subaxis])
        return subshape

    @CachedAttribute
    def local_l(self):
        layout = self.dist.coeff_layout
        local_l_elements = layout.local_elements(self.domain, scales=1)[self.axis+1]
        return tuple(self.degrees[local_l_elements])

    def global_grids(self, scales=None):
        if scales == None: scales = (1, 1)
        return (self.global_grid_azimuth(scales[0]),
                self.global_grid_colatitude(scales[1]))

    def global_grid_colatitude(self, scale):
        theta = self._native_colatitude_grid(scale)
        return reshape_vector(theta, dim=self.dist.dim, axis=self.axis+1)

    def local_grids(self, scales=None):
        if scales == None: scales = (1, 1)
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

    def global_colatitude_weights(self, scale=None):
        if scale == None: scale = 1
        N = int(np.ceil(scale * self.shape[1]))
        cos_theta, weights = dedalus_sphere.sphere.quadrature(Lmax=N-1)
        return reshape_vector(weights.astype(np.float64), dim=self.dist.dim, axis=self.axis+1)

    def local_colatitude_weights(self, scale=None):
        if scale == None: scale = 1
        local_elements = self.dist.grid_layout.local_elements(self.domain, scales=scale)[self.axis+1]
        N = int(np.ceil(scale * self.shape[1]))
        cos_theta, weights = dedalus_sphere.sphere.quadrature(Lmax=N-1)
        return reshape_vector(weights.astype(np.float64)[local_elements], dim=self.dist.dim, axis=self.axis+1)

    @CachedMethod
    def transform_plan(self, grid_shape, axis, s):
        """Build transform plan."""
        return self.transforms[self.colatitude_library](grid_shape, self.Lmax+1, axis, self.local_m, s)

    def forward_transform_colatitude(self, field, axis, gdata, cdata):
        # Apply spin recombination
        self.forward_spin_recombination(field.tensorsig, gdata)
        # Perform transforms component-by-component
        S = self.spin_weights(field.tensorsig)
        # HACK -- don't want to make a new array every transform
        temp = np.copy(cdata)
        for i, s in np.ndenumerate(S):
            grid_shape = gdata[i].shape
            plan = self.transform_plan(grid_shape, axis, s)
            plan.forward(gdata[i], temp[i], axis)
        np.copyto(cdata, temp)

    def backward_transform_colatitude(self, field, axis, cdata, gdata):
        # Perform transforms component-by-component
        S = self.spin_weights(field.tensorsig)
        # HACK -- don't want to make a new array every transform
        temp = np.copy(gdata)
        for i, s in np.ndenumerate(S):
            grid_shape = gdata[i].shape
            plan = self.transform_plan(grid_shape, axis, s)
            plan.backward(cdata[i], temp[i], axis)
        # Apply spin recombination
        np.copyto(gdata, temp)
        self.backward_spin_recombination(field.tensorsig, gdata)

    @CachedMethod
    def k_vector(self,mu,m,s,local_l):
        vector = np.zeros(len(local_l))
        Lmin = max(abs(m),abs(s),abs(s+mu))
        for i,l in enumerate(local_l):
            if l < Lmin: vector[i] = 0
            else: vector[i] = dedalus_sphere.sphere.k_element(mu,l,s,self.radius)
        return vector

    @CachedMethod
    def vector_slice(self, m, ell):
        if m > ell:
            return None
        mi = self.local_m.index(m)
        li = self.local_l.index(ell)
        return (mi, li)

    def vector_3(self, comp, m, ell):
        slices = self.vector_slice(m, ell)
        if slices is None:
            return None
        comp5 = reduced_view(comp, axis=self.axis, dim=self.dist.dim)
        return comp5[(slice(None),) + slices + (slice(None),)]


SWSH = SpinWeightedSphericalHarmonics


# These are common for RadialBallBasis and RadialSphericalShellBasis
class RegularityBasis(Basis):

    @CachedAttribute
    def local_l(self):
        return (0,)

    @CachedAttribute
    def local_m(self):
        return (0,)

    def global_grid(self, scale):
        problem_grid = self._radius_grid(scale)
        return reshape_vector(problem_grid, dim=self.dist.dim, axis=self.axis)

    def local_grid(self, scale):
        local_elements = self.dist.grid_layout.local_elements(self.domain, scales=scale)[self.axis]
        problem_grid = self._radius_grid(scale)[local_elements]
        return reshape_vector(problem_grid, dim=self.dist.dim, axis=self.axis)

    def global_weights(self, scale=None):
        if scale == None: scale = 1
        weights = self._radius_weights(scale)
        return reshape_vector(weights.astype(np.float64), dim=self.dist.dim, axis=self.axis)

    def local_weights(self, scale=None):
        if scale == None: scale = 1
        local_elements = self.dist.grid_layout.local_elements(self.domain, scales=scale)[self.axis]
        weights = self._radius_weights(scale)
        return reshape_vector(weights.astype(np.float64)[local_elements], dim=self.dist.dim, axis=self.axis)

    @CachedMethod
    def regularity_allowed(self,l,regularity):
        Rb = np.array([-1, 1, 0], dtype=int)
        if regularity == (): return True
        return not dedalus_sphere.intertwiner.forbidden_regularity(l,Rb[np.array(regularity)])

    @CachedMethod
    def regtotal(self, regindex):
        regorder = [-1, 1, 0]
        reg = lambda index: regorder[index]
        return sum(reg(index) for index in regindex)

    @CachedMethod
    def xi(self,mu,l):
        return dedalus_sphere.intertwiner.xi(mu,l)

    @CachedMethod
    def radial_recombinations(self, tensorsig, ell_list=None):
        if ell_list == None: ell_list = self.local_l
        # For now only implement recombinations for Ball-only tensors
        for cs in tensorsig:
            if self.coordsystem.cs is not cs:
                raise ValueError("Only supports tensors over ball.")
        order = len(tensorsig)

        Q_matrices = np.zeros((len(ell_list),3**order,3**order))
        for i, l in enumerate(ell_list):
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
        R = np.zeros([cs.dim for cs in tensorsig], dtype=int)
        for i, cs in enumerate(tensorsig):
            if self.coordsystem.cs is cs: # kludge before we decide how compound coordinate systems work
                R[axslice(i, 0, cs.dim)] += reshape_vector(Rb, dim=len(tensorsig), axis=i)
            #elif self.space in vs.spaces:
            #    n = vs.get_index(self.space)
            #    R[axslice(i, n, n+self.dim)] += reshape_vector(Rb, dim=len(tensorsig), axis=i)
        return R

    def forward_regularity_recombination(self, tensorsig, axis, gdata):
        rank = len(tensorsig)
        # Apply radial recombinations
        if rank > 0:
            Q = self.radial_recombinations(tensorsig)
            # Flatten tensor axes
            shape = gdata.shape
            temp = gdata.reshape((-1,)+shape[rank:])
            # Apply Q transformations for each l to flattened tensor data
            for l_index, Q_l in enumerate(Q):
                # Here the l axis is 'axis' instead of 'axis-1' since we have one tensor axis prepended
                l_view = temp[axslice(axis, l_index, l_index+1)]
                apply_matrix(Q_l.T, l_view, axis=0, out=l_view)

    def backward_regularity_recombination(self, tensorsig, axis, gdata):
        rank = len(tensorsig)
        # Apply radial recombinations
        if rank > 0:
            Q = self.radial_recombinations(tensorsig)
            # Flatten tensor axes
            shape = gdata.shape
            temp = gdata.reshape((-1,)+shape[rank:])
            # Apply Q transformations for each l to flattened tensor data
            for l_index, Q_l in enumerate(Q):
                # Here the l axis is 'axis' instead of 'axis-1' since we have one tensor axis prepended
                l_view = temp[axslice(axis, l_index, l_index+1)]
                apply_matrix(Q_l, l_view, axis=0, out=l_view)

    def radial_vector_3(self, comp, m, ell, regindex):
        slices = self.radial_vector_slices(m, ell, regindex)
        if slices is None:
            return None
        comp5 = reduced_view(comp, axis=self.axis-2, dim=3)
        return comp5[(slice(None),) + slices + (slice(None),)]

    @CachedMethod
    def radial_vector_slices(self, m, ell, regindex):
        if m > ell:
            return None
        if not self.regularity_allowed(ell, regindex):
            return None
        mi = self.local_m.index(m)
        li = self.local_l.index(ell)
        return (mi, li, self.n_slice(ell))

    def local_groups(self, basis_coupling):
        coupling, = basis_coupling
        if coupling:
            return [[None]]
        else:
            raise NotImplementedError()

    def local_group_slices(self, basis_group):
        group, = basis_group
        if group is None:
            n_slice = self.n_slice((), ell=0)
            return [n_slice]
        else:
            raise NotImplementedError()

    def dot_product_ncc(self, arg_basis, coeffs, ncc_ts, arg_ts, out_ts, subproblem, ncc_first, indices, cutoff=1e-6):
        Gamma = dedalus_sphere.intertwiner.GammaDotProduct(indices, ncc_first=ncc_first)
        return self._spin_op_ncc(arg_basis, coeffs, ncc_ts, arg_ts, out_ts, subproblem, Gamma, cutoff)

    def tensor_product_ncc(self, arg_basis, coeffs, ncc_ts, arg_ts, out_ts, subproblem, ncc_first, cutoff=1e-6):
        Gamma = dedalus_sphere.intertwiner.GammaTensorProduct(ncc_first=ncc_first)
        return self._spin_op_ncc(arg_basis, coeffs, ncc_ts, arg_ts, out_ts, subproblem, Gamma, cutoff)

    def _spin_op_ncc(self, arg_basis, coeffs, ncc_ts, arg_ts, out_ts, subproblem, Gamma, cutoff):
        # Don't really understand what this is doing...
        #if arg_basis is None:
        #    return super().ncc_matrix(arg_basis, coeffs)

        ell = subproblem.ell

        R_in = self.regularity_classes(arg_ts)
        R_out = self.regularity_classes(out_ts)

        submatrices = []
        for regindex_out, regtotal_out in np.ndenumerate(R_out):
            submatrix_row = []
            for regindex_in, regtotal_in in np.ndenumerate(R_in):
                submatrix_row.append(self._spin_op_matrix(ell, arg_basis, coeffs, regindex_in, regtotal_in, regindex_out, regtotal_out, ncc_ts, arg_ts, Gamma, cutoff))
            submatrices.append(submatrix_row)
        return sparse.bmat(submatrices)

    def _spin_op_matrix(self, ell, basis_in, coeffs, regindex_in, regtotal_in, regindex_out, regtotal_out, ncc_ts, input_ts, Gamma, cutoff, gamma_threshold=1e-10):
        # here self is the ncc
        R_ncc = self.regularity_classes(ncc_ts)
        S_ncc = self.sphere_basis.spin_weights(ncc_ts)

        S_in = self.sphere_basis.spin_weights(input_ts)
        diff_regtotal = regtotal_out - regtotal_in

        # jacobi parameters
        a_ncc = self.alpha + self.k

        N = self.n_size((),ell)
        matrix = 0 * sparse.identity(N)

        for regindex_ncc, regtotal_ncc in np.ndenumerate(R_ncc):
            b_ncc = sum(regindex_ncc) + 1/2
            d = regtotal_ncc - abs(diff_regtotal)
            if (d >= 0) and (d % 2 == 0):
                gamma = Gamma(ell, S_ncc, S_in, regindex_ncc, regindex_in, regindex_out)
                if abs(gamma) > gamma_threshold:
                    coeffs_filter = coeffs[regindex_ncc][:N]
                    J = basis_in.operator_matrix('Z',ell,regtotal_in)
                    R2 = (J + basis_in.operator_matrix('I',ell,regtotal_in))/2
                    A, B = clenshaw.jacobi_recursion(N, a_ncc, b_ncc, J)
                    f0 = 1/np.sqrt(jacobi.mass(a_ncc, b_ncc)) * sparse.identity(N)
                    prefactor = basis_in.radius_multiplication_matrix(ell, regtotal_in, diff_regtotal)
                    if np.max(np.abs(coeffs_filter)) > 1e-5:
                        print(regindex_ncc, regindex_in, regindex_out, d, diff_regtotal)
                    for i in range(d//2):
                        prefactor = prefactor @ R2
                    matrix += gamma * prefactor @ clenshaw.matrix_clenshaw(coeffs_filter, A, B, f0, cutoff=cutoff)

        return matrix

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


class SphericalShellRadialBasis(RegularityBasis):

    dim = 1
    dims = ['radius']
    group_shape = (1,)
    transforms = {}

    def __init__(self, coordsystem, shape, radii=(1,2), alpha=(-0.5,-0.5), dealias=(1,), k=0, radius_library='matrix'):
        self.coordsystem = coordsystem
        if radii[0] <= 0:
            raise ValueError("Inner radius must be positive.")
        self.radii = radii
        self.dR = self.radii[1] - self.radii[0]
        self.rho = (self.radii[1] + self.radii[0])/self.dR
        self.alpha = alpha
        self.dealias = dealias
        self.k = k
        self.radius_library = radius_library
        self.Nmax = shape - 1
        self.shape = shape
        self.grid_params = (coordsystem, radii, alpha, dealias)
        Basis.__init__(self, coordsystem)

    def __eq__(self, other):
        if isinstance(other, SphericalShellRadialBasis):
            if self.coordsystem == other.coordsystem:
                if self.grid_params == other.grid_params:
                    if self.k == other.k:
                        return True
        return False

    def __hash__(self):
        return id(self)

    def __add__(self, other):
        if other is None:
            return self
        if other is self:
            return self
        if isinstance(other, SphericalShellRadialBasis):
            if self.grid_params == other.grid_params:
                shape = max(self.shape, other.shape)
                k = max(self.k, other.k)
                return SphericalShellRadialBasis(self.coordsystem, shape, radii=self.radii, alpha=self.alpha, dealias=self.dealias, k=k, radius_library=self.radius_library)
        return NotImplemented

    def __mul__(self, other):
        if other is None:
            return self
        if other is self:
            return self
        if isinstance(other, SphericalShellRadialBasis):
            if self.grid_params == other.grid_params:
                shape = max(self.shape, other.shape)
                k = 0
                return SphericalShellRadialBasis(self.coordsystem, shape, radii=self.radii, alpha=self.alpha, dealias=self.dealias, k=k, radius_library=self.radius_library)
        return NotImplemented

    def _new_k(self, k):
        return SphericalShellRadialBasis(self.coordsystem, self.shape, radii = self.radii, alpha=self.alpha, dealias=self.dealias, k=k,
                                         radius_library=self.radius_library)

    @CachedMethod
    def _radius_grid(self, scale):
        N = int(np.ceil(scale * self.shape))
        z, weights = dedalus_sphere.annulus.quadrature(N-1, alpha=self.alpha, niter=3)
        r = self.dR/2*(z + self.rho)
        return r.astype(np.float64)

    @CachedMethod
    def _radius_weights(self, scale):
        N = int(np.ceil(scale * self.shape))
        z_proj, weights_proj = dedalus_sphere.annulus.quadrature(N-1, alpha=self.alpha, niter=3)
        z0, weights0 = dedalus_sphere.jacobi128.quadrature(N-1, 0, 0)
        init0 = dedalus_sphere.jacobi128.envelope(-1/2,-1/2,-1/2,-1/2,z0)
        Q0 = dedalus_sphere.jacobi128.recursion(N-1,-1/2,-1/2,z0,init0)
        init_proj = dedalus_sphere.jacobi128.envelope(-1/2,-1/2,-1/2,-1/2,z_proj)
        Q_proj = dedalus_sphere.jacobi128.recursion(N-1,-1/2,-1/2,z_proj,init_proj)
        normalization = self.dR/2
        return normalization*((Q0.dot(weights0)).T).dot(weights_proj*Q_proj)

    @CachedMethod
    def radial_transform_factor(self, scale, data_axis, dk):
        r = reshape_vector(self._radius_grid(scale), dim=data_axis, axis=data_axis-1)
        return (self.dR/r)**dk

    @CachedMethod
    def transform_plan(self, grid_size, k):
        """Build transform plan."""
        a = self.alpha[0] + k
        b = self.alpha[1] + k
        a0 = self.alpha[0]
        b0 = self.alpha[1]
        return self.transforms[self.radius_library](grid_size, self.Nmax+1, a, b, a0, b0)

    def forward_transform(self, field, axis, gdata, cdata):
        data_axis = len(field.tensorsig) + axis
        grid_size = gdata.shape[data_axis]
        # Multiply by radial factor
        if self.k > 0:
            gdata *= self.radial_transform_factor(field.scales[axis], data_axis, -self.k)
        # Apply regularity recombination
        self.forward_regularity_recombination(field.tensorsig, axis, gdata)
        # Perform radial transforms component-by-component
        R = self.regularity_classes(field.tensorsig)
        # HACK -- don't want to make a new array every transform
        temp = np.copy(cdata)
        for regindex, regtotal in np.ndenumerate(R):
           plan = self.transform_plan(grid_size, self.k)
           plan.forward(gdata[regindex], temp[regindex], axis)
        np.copyto(cdata, temp)

    def backward_transform(self, field, axis, cdata, gdata):
        data_axis = len(field.tensorsig) + axis
        grid_size = gdata.shape[data_axis]
        # Perform radial transforms component-by-component
        R = self.regularity_classes(field.tensorsig)
        # HACK -- don't want to make a new array every transform
        temp = np.copy(gdata)
        for i, r in np.ndenumerate(R):
           plan = self.transform_plan(grid_size, self.k)
           plan.backward(cdata[i], temp[i], axis)
        np.copyto(gdata, temp)
        # Apply regularity recombinations
        self.backward_regularity_recombination(field.tensorsig, axis, gdata)
        # Multiply by radial factor
        if self.k > 0:
            gdata *= self.radial_transform_factor(field.scales[axis], data_axis, self.k)

    @CachedMethod
    def operator_matrix(self, op, l, regtotal, dk=0):
        return dedalus_sphere.annulus.operator(3,op,self.Nmax,self.k+dk,l+regtotal,self.radii,alpha=self.alpha).astype(np.float64)

    @CachedMethod
    def conversion_matrix(self, l, regtotal, dk):
        for dki in range(dk):
            Ek = dedalus_sphere.annulus.operator(3, 'E', self.Nmax, self.k+dki, l+regtotal, self.radii, alpha=self.alpha)
            if dki == 0:
                E = Ek
            else:
                E = Ek @ E
        return E.astype(np.float64)

    def n_size(self, ell):
        return self.Nmax + 1

    def n_slice(self, ell):
        return slice(0, self.Nmax + 1)

    def start(self, groups):
        return 0


class BallRadialBasis(RegularityBasis):

    dim = 1
    dims = ['radius']
    group_shape = (1,)
    transforms = {}

    def __init__(self, coordsystem, shape, radius=1, k=0, alpha=0, dealias=(1,), radius_library='matrix'):
        self.coordsystem = coordsystem
        if radius <= 0:
            raise ValueError("Radius must be positive.")
        self.radius = radius
        self.k = k
        self.alpha = alpha
        self.dealias = dealias
        self.radial_COV = AffineCOV((0, 1), (0, radius))
        self.radius_library = radius_library
        self.Nmax = shape - 1
        self.shape = shape
        self.grid_params = (coordsystem, radius, alpha, dealias)
        Basis.__init__(self, coordsystem)

    def __eq__(self, other):
        if isinstance(other, BallRadialBasis):
            if self.coordsystem == other.coordsystem:
                if self.grid_params == other.grid_params:
                    if self.k == other.k:
                        return True
        return False

    def __hash__(self):
        return id(self)

    def __add__(self, other):
        if other is None:
            return self
        if other is self:
            return self
        if isinstance(other, BallRadialBasis):
            if self.grid_params == other.grid_params:
                shape = max(self.shape, other.shape)
                k = max(self.k, other.k)
                return BallRadialBasis(self.coordsystem, shape, radius=self.radius, k=k, alpha=self.alpha, dealias=self.dealias, radius_library=self.radius_library)
        return NotImplemented

    def __mul__(self, other):
        if other is None:
            return self
        if other is self:
            return self
        if isinstance(other, BallRadialBasis):
            if self.grid_params == other.grid_params:
                shape = max(self.shape, other.shape)
                k = 0
                return BallRadialBasis(self.coordsystem, shape, radius=self.radius, k=k, alpha=self.alpha, dealias=self.dealias, radius_library=self.radius_library)
        return NotImplemented

    def _new_k(self, k):
        return BallRadialBasis(self.coordsystem, self.shape, radius=self.radius, k=k, alpha=self.alpha, dealias=self.dealias, radius_library=self.radius_library)

    @CachedMethod
    def _radius_grid(self, scale):
        N = int(np.ceil(scale * self.shape))
        z, weights = dedalus_sphere.ball.quadrature(3, N-1, niter=3, alpha=self.alpha)
        r = np.sqrt((z + 1) / 2)
        return self.radial_COV.problem_coord(r)

    @CachedMethod
    def _radius_weights(self, scale):
        N = int(np.ceil(scale * self.shape))
        z, weights = dedalus_sphere.ball.quadrature(3, N-1, alpha=self.alpha, niter=3)
        return weights

    @CachedMethod
    def transform_plan(self, grid_size, k):
        """Build transform plan."""
        a = self.alpha[0] + k
        b = self.alpha[1] + k
        a0 = self.alpha[0]
        b0 = self.alpha[1]
        return self.transforms[self.radius_library](grid_size, self.Nmax+1, a, b, a0, b0)

    @CachedMethod
    def transform_plan(self, grid_shape, regindex, axis, regtotal, k, alpha):
        """Build transform plan."""
        return self.transforms[self.radius_library](grid_shape, self.Nmax+1, axis, self.local_l, regindex, regtotal, k, alpha)

    def forward_transform(self, field, axis, gdata, cdata):
        # Apply regularity recombination
        self.forward_regularity_recombination(field.tensorsig, axis, gdata)
        # Perform radial transforms component-by-component
        R = self.regularity_classes(field.tensorsig)
        # HACK -- don't want to make a new array every transform
        temp = np.copy(cdata)
        for regindex, regtotal in np.ndenumerate(R):
           grid_shape = gdata[regindex].shape
           plan = self.transform_plan(grid_shape, regindex, axis, regtotal, self.k, self.alpha)
           plan.forward(gdata[regindex], temp[regindex], axis)
        np.copyto(cdata, temp)

    def backward_transform(self, field, axis, cdata, gdata):
        # Perform radial transforms component-by-component
        R = self.regularity_classes(field.tensorsig)
        # HACK -- don't want to make a new array every transform
        temp = np.copy(gdata)
        for regindex, regtotal in np.ndenumerate(R):
           grid_shape = gdata[regindex].shape
           plan = self.transform_plan(grid_shape, regindex, axis, regtotal, self.k, self.alpha)
           plan.backward(cdata[regindex], temp[regindex], axis)
        np.copyto(gdata, temp)
        # Apply regularity recombinations
        self.backward_regularity_recombination(field.tensorsig, axis, gdata)

    @CachedMethod
    def operator_matrix(self, op, l, regtotal, dk=0):
        return dedalus_sphere.ball.operator(3,op,self.Nmax,self.k+dk,l,regtotal,radius=self.radius,alpha=self.alpha).astype(np.float64)

    @CachedMethod
    def conversion_matrix(self, l, regtotal, dk):
        for dki in range(dk):
            Ek = dedalus_sphere.ball.operator(3, 'E', self.Nmax, self.k+dki, l, regtotal, radius=self.radius, alpha=self.alpha)
            if dki == 0:
                E = Ek
            else:
                E = Ek @ E
        return E.astype(np.float64)

    @CachedMethod
    def radius_multiplication_matrix(self, ell, regtotal, order):
        R = dedalus_sphere.ball.operator(3, 'I', self.Nmax, self.k, ell, regtotal, radius=self.radius, alpha=self.alpha)
        if order < 0:
            op = 'R-'
            sign = -1
        if order > 0:
            op = 'R+'
            sign = +1
        for order_i in range(abs(order)):
            Ri = dedalus_sphere.ball.operator(3, op, self.Nmax, self.k, ell, regtotal+sign*order_i, radius=self.radius, alpha=self.alpha)
            R = Ri @ R
        return R.astype(np.float64)

    def _n_limits(self, ell):
        nmin = dedalus_sphere.ball.Nmin(ell, 0)
        return (nmin, self.Nmax)

    def n_size(self, ell):
        nmin, nmax = self._n_limits(ell)
        return nmax - nmin + 1

    def n_slice(self, ell):
        nmin, nmax = self._n_limits(ell)
        return slice(nmin, nmax+1)

    def start(self, groups):
        ell = groups[1]
        (nmin, Nmax) = self._n_limits(ell)
        return nmin


class Spherical3DBasis(MultidimensionalBasis):

    dim = 3
    dims = ['azimuth', 'colatitude', 'radius']
    group_shape = (1, 1, 1)
    transforms = {}

    def __init__(self, coordsystem, shape_angular, dealias_angular, radial_basis, azimuth_library='matrix', colatitude_library='matrix'):
        self.coordsystem = coordsystem
        self.shape = tuple( (*shape_angular, radial_basis.shape ) )
        if np.isscalar(dealias_angular):
            self.dealias = ( (dealias_angular, dealias_angular, *radial_basis.dealias) )
        elif len(dealias_angular) != 2:
            raise ValueError("dealias_angular must either be a number or a tuple of two numbers")
        else:
            self.dealias = tuple( (*dealias_angular, *radial_basis.dealias) )
        self.radial_basis = radial_basis
        self.k = radial_basis.k
        self.azimuth_library = azimuth_library
        self.colatitude_library = colatitude_library
        self.sphere_basis = self.S2_basis()
        self.mmax = self.sphere_basis.mmax
        self.Lmax = self.sphere_basis.Lmax
        self.radial_basis.local_m = self.sphere_basis.local_m
        self.radial_basis.local_l = self.sphere_basis.local_l
        self.global_grid_azimuth = self.sphere_basis.global_grid_azimuth
        self.global_grid_colatitude = self.sphere_basis.global_grid_colatitude
        self.local_grid_azimuth = self.sphere_basis.local_grid_azimuth
        self.local_grid_colatitude = self.sphere_basis.local_grid_colatitude
        self.global_colatitude_weights = self.sphere_basis.global_colatitude_weights
        self.local_colatitude_weights = self.sphere_basis.local_colatitude_weights
        self.forward_transform_azimuth = self.sphere_basis.forward_transform_azimuth
        self.forward_transform_colatitude = self.sphere_basis.forward_transform_colatitude
        self.forward_transform_radius = self.radial_basis.forward_transform
        self.backward_transform_azimuth = self.sphere_basis.backward_transform_azimuth
        self.backward_transform_colatitude = self.sphere_basis.backward_transform_colatitude
        self.backward_transform_radius = self.radial_basis.backward_transform
        self.forward_transforms = [self.forward_transform_azimuth,
                                   self.forward_transform_colatitude,
                                   self.forward_transform_radius]
        self.backward_transforms = [self.backward_transform_azimuth,
                                    self.backward_transform_colatitude,
                                    self.backward_transform_radius]
        Basis.__init__(self, coordsystem)
        self.radial_basis.domain = self.domain

    def global_grid_radius(self, scale=None):
        return self.radial_basis.global_grid(scale)

    def local_grid_radius(self, scale=None):
        return self.radial_basis.local_grid(scale)

    def global_radial_weights(self, scale=None):
        return self.radial_basis.global_weights(scale)

    def local_radial_weights(self, scale=None):
        return self.radial_basis.local_weights(scale)

    def global_grids(self, scales=None):
        if scales == None: scales = (1,1,1)
        return (self.global_grid_azimuth(scales[0]),
                self.global_grid_colatitude(scales[1]),
                self.global_grid_radius(scales[2]))

    def local_grids(self, scales=None):
        if scales == None: scales = (1,1,1)
        return (self.local_grid_azimuth(scales[0]),
                self.local_grid_colatitude(scales[1]),
                self.local_grid_radius(scales[2]))

    def S2_basis(self,radius=1):
        return SWSH(self.coordsystem.S2coordsys, self.shape[:2], radius=radius, dealias=self.dealias[:2],
                    azimuth_library=self.azimuth_library, colatitude_library=self.colatitude_library)

    @CachedMethod
    def operator_matrix(self, op, l, regtotal, dk=0):
        return self.radial_basis.operator_matrix(op, l, regtotal, dk=dk)

    @CachedMethod
    def conversion_matrix(self, l, regtotal, dk):
        return self.radial_basis.conversion_matrix(l, regtotal, dk)

    def n_size(self, ell):
        return self.radial_basis.n_size(ell)

    def n_slice(self, ell):
        return self.radial_basis.n_slice(ell)

    def start(self, groups):
        return self.radial_basis.start(groups)

    def local_groups(self, basis_coupling):
        m_coupling, ell_coupling, n_coupling = basis_coupling
        if (not m_coupling) and (not ell_coupling) and (n_coupling):
            local_chunks = self.dist.coeff_layout.local_chunks(self.domain, scales=1)
            m_chunks = local_chunks[self.first_axis]
            ell_chunks = local_chunks[self.first_axis+1]
            groups = []
            # Add groups satisfying triangular truncation
            for m_chunk in m_chunks:
                m = self.sphere_basis.azimuth_basis.wavenumbers[m_chunk]
                for ell_chunk in ell_chunks:
                    ell = ell_chunk
                    if ell >= np.abs(m):
                        groups.append([m_chunk, ell_chunk, None])
            return groups
        else:
            raise NotImplementedError()

    def local_group_slices(self, basis_group):
        m_group, ell_group, n_group = basis_group
        if (m_group is not None) and (ell_group is not None) and (n_group is None):
            local_chunks = self.dist.coeff_layout.local_chunks(self.domain, scales=1)
            m_chunks = local_chunks[self.first_axis]
            m_index = list(m_chunks).index(m_group)
            m_gs = self.group_shape[0]
            m_slice = slice(m_index*m_gs, (m_index+1)*m_gs)
            ell_chunks = local_chunks[self.first_axis+1]
            ell_index = list(ell_chunks).index(ell_group)
            ell_gs = self.group_shape[1]
            ell_slice = slice(ell_index*ell_gs, (ell_index+1)*ell_gs)
            n_slice = self.radial_basis.n_slice(ell=ell_group)
            return [m_slice, ell_slice, n_slice]
        else:
            raise NotImplementedError()


class SphericalShellBasis(Spherical3DBasis):

    def __init__(self, coordsystem, shape, radii=(1,2), alpha=(-0.5,-0.5), dealias=(1,1,1), k=0, azimuth_library='matrix', colatitude_library='matrix', radius_library='matrix'):
        self.radial_basis = SphericalShellRadialBasis(coordsystem.radius, shape[2], radii=radii, alpha=alpha, dealias=(dealias[2],), k=k, radius_library=radius_library)
        Spherical3DBasis.__init__(self, coordsystem, shape[:2], dealias[:2], self.radial_basis, azimuth_library=azimuth_library, colatitude_library=colatitude_library)
        self.grid_params = (coordsystem, radii, alpha, dealias)

    def __eq__(self, other):
        if isinstance(other, SphericalShellBasis):
            if self.coordsystem == other.coordsystem:
                if self.grid_params == other.grid_params:
                    if self.k == other.k:
                        return True
        return False

    def __hash__(self):
        return id(self)

    def __add__(self, other):
        if other is None:
            return self
        if other is self:
            return self
        if isinstance(other, SphericalShellBasis):
            if self.grid_params == other.grid_params:
                shape = tuple(np.maximum(self.shape, other.shape))
                k = max(self.k, other.k)
                return SphericalShellBasis(self.coordsystem, shape, radii=self.radial_basis.radii, alpha=self.radial_basis.alpha, dealias=self.dealias, k=k,
                                           azimuth_library=self.azimuth_library, colatitude_library=self.colatitude_library,
                                           radius_library=self.radial_basis.radius_library)
        return NotImplemented

    def __mul__(self, other):
        if other is None:
            return self
        if other is self:
            return self
        if isinstance(other, SphericalShellBasis):
            if self.grid_params == other.grid_params:
                shape = tuple(np.maximum(self.shape, other.shape))
                k = 0
                return SphericalShellBasis(self.coordsystem, shape, radii=self.radial_basis.radii, alpha=self.radial_basis.alpha, dealias=self.dealias, k=k,
                                           azimuth_library=self.azimuth_library, colatitude_library=self.colatitude_library,
                                           radius_library=self.radial_basis.radius_library)
        return NotImplemented

    def _new_k(self, k):
        return SphericalShellBasis(self.coordsystem, self.shape, radii=self.radial_basis.radii, alpha=self.radial_basis.alpha, dealias=self.dealias, k=k,
                                   azimuth_library=self.azimuth_library, colatitude_library=self.colatitude_library,
                                   radius_library=self.radial_basis.radius_library)


class BallBasis(Spherical3DBasis):

    def __init__(self, coordsystem, shape, radius=1, k=0, alpha=0, dealias=(1,1,1), azimuth_library='matrix', colatitude_library='matrix', radius_library='matrix'):
        self.radial_basis = BallRadialBasis(coordsystem.radius, shape[2], radius=radius, k=k, alpha=alpha, dealias=(dealias[2],), radius_library=radius_library)
        Spherical3DBasis.__init__(self, coordsystem, shape[:2], dealias[:2], self.radial_basis, azimuth_library=azimuth_library, colatitude_library=colatitude_library)
        self.grid_params = (coordsystem, radius, alpha, dealias)

    def __eq__(self, other):
        if isinstance(other, BallBasis):
            if self.coordsystem == other.coordsystem:
                if self.grid_params == other.grid_params:
                    if self.k == other.k:
                        return True
        return False

    def __hash__(self):
        return id(self)

    def __add__(self, other):
        if other is None:
            return self
        if other is self:
            return self
        if isinstance(other, BallBasis):
            if self.grid_params == other.grid_params:
                shape = tuple(np.maximum(self.shape, other.shape))
                k = max(self.k, other.k)
                return BallBasis(self.coordsystem, shape, radius=self.radial_basis.radius, k=k, alpha=self.radial_basis.alpha, dealias=self.dealias,
                                 azimuth_library=self.azimuth_library, colatitude_library=self.colatitude_library,
                                 radius_library=self.radial_basis.radius_library)
        return NotImplemented

    def __mul__(self, other):
        if other is None:
            return self
        if other is self:
            return self
        if isinstance(other, BallBasis):
            if self.grid_params == other.grid_params:
                shape = tuple(np.maximum(self.shape, other.shape))
                k = 0
                return BallBasis(self.coordsystem, shape, radius=self.radial_basis.radius, k=k, alpha=self.radial_basis.alpha, dealias=self.dealias,
                                 azimuth_library=self.azimuth_library, colatitude_library=self.colatitude_library,
                                 radius_library=self.radial_basis.radius_library)
        return NotImplemented

    def _new_k(self, k):
        return BallBasis(self.coordsystem, self.shape, radius = self.radial_basis.radius, k=k, alpha=self.radial_basis.alpha, dealias=self.dealias,
                         azimuth_library=self.azimuth_library, colatitude_library=self.colatitude_library,
                         radius_library=self.radial_basis.radius_library)


def prod(arg):
    if arg:
        return reduce(operator.mul, arg)
    else:
        return 1


def reduced_view(data, axis, dim):
    shape = data.shape
    Na = (int(prod(shape[:axis])),)
    Nb = shape[axis:axis+dim]
    Nc = (int(prod(shape[axis+dim:])),)
    return data.reshape(Na+Nb+Nc)


class ConvertRegularity(operators.Convert, operators.SphericalEllOperator):

    input_basis_type = Spherical3DBasis
    output_basis_type = Spherical3DBasis

    def __init__(self, operand, output_basis, out=None):
        super().__init__(operand, output_basis, out=out)
        self.radial_basis = self.input_basis.radial_basis

    def regindex_out(self, regindex_in):
        return (regindex_in,)

    def radial_matrix(self, regindex_in, regindex_out, ell):
        basis = self.radial_basis
        regtotal = basis.regtotal(regindex_in)
        dk = self.output_basis.k - basis.k
        if regindex_in == regindex_out:
            return basis.conversion_matrix(ell, regtotal, dk)
        else:
            return basis.operator_matrix('0', ell, 0)


class BallRadialInterpolate(operators.Interpolate, operators.SphericalEllOperator):

    basis_type = BallBasis
    basis_subaxis = 2

    @classmethod
    def _check_args(cls, operand, coord, position, out=None):
        if isinstance(operand, Operand):
            if isinstance(operand.domain.get_basis(coord), cls.basis_type):
                if operand.domain.get_basis_subaxis(coord) == cls.basis_subaxis:
                    return True
        return False

    @staticmethod
    def _output_basis(input_basis, position):
        return input_basis.S2_basis(radius=position)

    def subproblem_matrix(self, subproblem):
        ell = subproblem.group[self.last_axis - 1]
        matrix = super().subproblem_matrix(subproblem)
        if self.tensorsig != ():
            Q = self.input_basis.radial_basis.radial_recombinations(self.tensorsig, ell_list=(ell,))
            matrix = Q[0] @ matrix
        return matrix

    def operate(self, out):
        """Perform operation."""
        operand = self.args[0]
        basis_in = self.input_basis.radial_basis
        basis_out = self.output_basis
        # Set output layout
        out.set_layout(operand.layout)
        # Apply operator
        R = basis_in.regularity_classes(operand.tensorsig)
        for regindex, regtotal in np.ndenumerate(R):
           comp_in = operand.data[regindex]
           comp_out = out.data[regindex]
           for m in basis_in.local_m:
               for ell in basis_in.local_l:
                   vec3_in = basis_in.radial_vector_3(comp_in, m, ell, regindex)
                   vec3_out = basis_out.vector_3(comp_out, m, ell)
                   if (vec3_in is not None) and (vec3_out is not None):
                       A = self.radial_matrix(regindex, regindex, ell)
                       apply_matrix(A, vec3_in, axis=1, out=vec3_out)
        # Q matrix
        basis_in.backward_regularity_recombination(operand.tensorsig, self.basis_subaxis, out.data)

    def radial_matrix(self, regindex_in, regindex_out, ell):
        position = self.position
        basis = self.input_basis.radial_basis
        if regindex_in == regindex_out:
            return self._radial_matrix(basis, 'r=R', ell, basis.regtotal(regindex_in))
        else:
            return np.zeros((1,basis.n_size((), ell)))

    @staticmethod
    @CachedMethod
    def _radial_matrix(basis, op, ell, regtotal):
        return reshape_vector(basis.operator_matrix(op, ell, regtotal), dim=2, axis=1)


class SphericalShellRadialInterpolate(operators.Interpolate, operators.SphericalEllOperator):

    basis_type = SphericalShellBasis
    basis_subaxis = 2

    @classmethod
    def _check_args(cls, operand, coord, position, out=None):
        # Dispatch by operand basis
        if isinstance(operand, Operand):
            if isinstance(operand.domain.get_basis(coord), cls.basis_type):
                if operand.domain.get_basis_subaxis(coord) == cls.basis_subaxis:
                    return True
        return False

    @staticmethod
    def _output_basis(input_basis, position):
        return input_basis.S2_basis(radius=position)

    def subproblem_matrix(self, subproblem):
        ell = subproblem.group[self.last_axis - 1]
        basis_in = self.input_basis.radial_basis
        matrix = super().subproblem_matrix(subproblem)
        if self.tensorsig != ():
            Q = basis_in.radial_recombinations(self.tensorsig, ell_list=(ell,))
            matrix = Q[0] @ matrix
        # Radial rescaling
        return matrix * (basis_in.dR/self.position)**basis_in.k

    def operate(self, out):
        """Perform operation."""
        operand = self.args[0]
        basis_in = self.input_basis.radial_basis
        basis_out = self.output_basis
        # Set output layout
        out.set_layout(operand.layout)
        # Apply operator
        R = basis_in.regularity_classes(operand.tensorsig)
        for regindex, regtotal in np.ndenumerate(R):
           comp_in = operand.data[regindex]
           comp_out = out.data[regindex]
           for m in basis_in.local_m:
               for ell in basis_in.local_l:
                   vec3_in = basis_in.radial_vector_3(comp_in, m, ell, regindex)
                   vec3_out = basis_out.vector_3(comp_out, m, ell)
                   if (vec3_in is not None) and (vec3_out is not None):
                       A = self.radial_matrix(regindex, regindex, ell)
                       apply_matrix(A, vec3_in, axis=1, out=vec3_out)
        # Q matrix
        basis_in.backward_regularity_recombination(operand.tensorsig, self.basis_subaxis, out.data)
        # Radial rescaling
        out.data *= (basis_in.dR/self.position)**basis_in.k

    def radial_matrix(self, regindex_in, regindex_out, ell):
        position = self.position
        basis = self.input_basis.radial_basis
        if regindex_in == regindex_out:
            if position == basis.radii[0]:
                return self._radial_matrix(basis, 'r=Ri', ell, basis.regtotal(regindex_in))
            elif position == basis.radii[1]:
                return self._radial_matrix(basis, 'r=Ro', ell, basis.regtotal(regindex_in))
            else:
                raise ValueError("Right now can only interpolate to inner or outer radii")
        else:
            return np.zeros((1,basis.n_size((), ell)))

    @staticmethod
    @CachedMethod
    def _radial_matrix(basis, op, ell, regtotal):
        return reshape_vector(basis.operator_matrix(op, ell, regtotal), dim=2, axis=1)


class SphericalTransposeComponents(operators.TransposeComponents):

    basis_type = Spherical3DBasis

    def __init__(self, operand, indices=(0,1), out=None):
        super().__init__(operand, indices=indices, out=out)
        self.radius_axis = self.coordsys.coords[2].axis

    def check_conditions(self):
        """Can always take the transpose"""
        return True

    def enforce_conditions(self):
        """Can always take the transpose"""
        pass

    def subproblem_matrix(self, subproblem):
        operand = self.args[0]
        input_basis = self.domain.get_basis(self.coordsys)
        basis = input_basis.radial_basis
        R = basis.regularity_classes(self.tensorsig)

        ell = subproblem.group[self.radius_axis - 1]

        indices = self.indices
        rank = len(self.tensorsig)
        neworder = np.arange(rank)
        neworder[indices[0]] = indices[1]
        neworder[indices[1]] = indices[0]

        matrix = []
        for regindex_out, regtotal_out in np.ndenumerate(R):
            regindex_out = np.array(regindex_out)
            matrix_row = []
            for regindex_in, regtotal_in in np.ndenumerate(R):
                if tuple(regindex_out[neworder]) == regindex_in:
                    matrix_row.append( 1 )
                else:
                    matrix_row.append( 0 )
            matrix.append(matrix_row)
        transpose = np.array(matrix)

        Q = basis.radial_recombinations(self.tensorsig,ell_list=(ell,))
        transpose = Q[0].T @ transpose @ Q[0]

        # assume all regularities have the same n_size
        eye = sparse.identity(basis.n_size((), ell), self.dtype, format='csr')
        matrix = sparse.kron( transpose, eye)
        return matrix

    def operate(self, out):
        """Perform operation."""
        operand = self.args[0]
        input_basis = self.domain.get_basis(self.coordsys)
        basis = input_basis.radial_basis
        # Set output layout
        layout = operand.layout
        out.set_layout(layout)
        indices = self.indices
        np.copyto(out.data, operand.data)

        if not layout.grid_space[self.radius_axis]: # in regularity componentsinput
            basis.backward_regularity_recombination(operand.tensorsig, self.radius_axis, out.data)

        axes_list = np.arange(len(out.data.shape))
        axes_list[indices[0]] = indices[1]
        axes_list[indices[1]] = indices[0]
        np.copyto(out.data,np.transpose(out.data,axes=axes_list))

        if not layout.grid_space[self.radius_axis]: # in regularity components
            basis.forward_regularity_recombination(operand.tensorsig, self.radius_axis, out.data)


class S2RadialComponent(operators.RadialComponent):

    basis_type = SWSH
    name = 'Radial'

    def subproblem_matrix(self, subproblem):
        operand = self.args[0]
        basis = self.domain.get_basis(self.coordsys)
        S_in = basis.spin_weights(operand.tensorsig)
        S_out = basis.spin_weights(self.tensorsig)

        matrix = []
        for spinindex_out, spintotal_out in np.ndenumerate(S_out):
            matrix_row = []
            for spinindex_in, spintotal_in in np.ndenumerate(S_in):
                if tuple(spinindex_in[:self.index] + spinindex_in[self.index+1:]) == spinindex_out and spinindex_in[self.index] == 2:
                    matrix_row.append( 1 )
                else:
                    matrix_row.append( 0 )
            matrix.append(matrix_row)
        return np.array(matrix)

    def operate(self, out):
        """Perform operation."""
        operand = self.args[0]
        # Set output layout
        layout = operand.layout
        out.set_layout(layout)
        np.copyto(out.data, operand.data[axindex(self.index,2)])


class S2AngularComponent(operators.AngularComponent):

    basis_type = SWSH

    def subproblem_matrix(self, subproblem):
        operand = self.args[0]
        basis = self.domain.get_basis(self.coordsys)
        S_in = basis.spin_weights(operand.tensorsig)
        S_out = basis.spin_weights(self.tensorsig)

        matrix = []
        for spinindex_out, spintotal_out in np.ndenumerate(S_out):
            matrix_row = []
            for spinindex_in, spintotal_in in np.ndenumerate(S_in):
                if spinindex_in == spinindex_out:
                    matrix_row.append( 1 )
                else:
                    matrix_row.append( 0 )
            matrix.append(matrix_row)
        return np.array(matrix)

    def operate(self, out):
        """Perform operation."""
        operand = self.args[0]
        # Set output layout
        layout = operand.layout
        out.set_layout(layout)
        np.copyto(out.data, operand.data[axslice(self.index,0,2)])


from . import transforms
