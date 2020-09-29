"""
Abstract and built-in classes for spectral bases.

"""

import math
import numpy as np
from scipy import sparse
from functools import reduce
import operator

from . import operators
from ..tools.array import kron
from ..tools.array import axslice
from ..tools.array import apply_matrix
from ..tools.array import permute_axis
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

    def __init__(self, coords):
        self.coords = coords
        self.dist = coords.dist
        self.axis = coords.axis
        self.domain = Domain(self.dist, bases=(self,))

    @CachedAttribute
    def constant(self):
        return tuple(False for i in range(self.dim))

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

    def local_elements(self):
        """
        Local element arrays used for output.
        Should correspond to all memory indices, NOT applying e.g. triangular truncation.
        """
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
        self.coordsystem = coord
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

    def local_elements(self):
        local_elements = self.dist.coeff_layout.local_elements(self.domain, scales=scale)[self.axis]
        return (local_elements,)

    def _native_grid(self, scale):
        """Native flat global grid."""
        # Subclasses must implement
        raise NotImplementedError

    def global_shape(self, layout, scales):
        grid_space = layout.grid_space[self.axis]
        if grid_space:
            return self.grid_shape(scales)
        else:
            return self.shape

    def chunk_shape(self, layout):
        grid_space = layout.grid_space[self.axis]
        if grid_space:
            return 1
        else:
            # Chunk groups together
            return self.group_shape[0]

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

    def __matmul__(self, other):
        if other is None:
            return self.__rmatmul__(other)
        else:
            return other.__rmatmul__(self)

    def __rmatmul__(self, other):
        return self.__mul__(other)

    # def include_mode(self, mode):
    #     return (0 <= mode < self.space.coeff_size)

    def Jacobi_matrix(self):
        return dedalus_sphere.jacobi.operator('Z')(self.size, self.a, self.b).square

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

    def multiplication_matrix(self, subproblem, arg_basis, coeffs, ncc_comp, arg_comp, out_comp, cutoff=1e-6):
        if arg_basis is None:
            return super().ncc_matrix(arg_basis, coeffs)
        # Jacobi parameters
        a_ncc = self.a
        b_ncc = self.b
        M = coeffs.size
        N = arg_basis.size
        J = arg_basis.Jacobi_matrix()
        A, B = clenshaw.jacobi_recursion(M, a_ncc, b_ncc, J)
        f0 = dedalus_sphere.jacobi.polynomials(1, a_ncc, b_ncc, 1)[0] * sparse.identity(N)
        return clenshaw.matrix_clenshaw(coeffs.ravel(), A, B, f0, cutoff=cutoff)

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
        interp_vector = jacobi.build_polynomials(N, a, b, x)
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

    def __matmul__(self, other):
        if other is None:
            return self.__rmatmul__(other)
        else:
            return other.__rmatmul__(self)

    def __rmatmul__(self, other):
        if other is None:
            return self
        else:
            return NotImplemented

    # def __pow__(self, other):
    #     return self.space.Fourier

    def __init__(self, coord, size, bounds, dealias=1, library=None):
        super().__init__(coord, size, bounds, dealias)
        if library is None:
            library = 'fftw'
        self.library = library
        self.kmax = kmax = (size - 1) // 2
        self.wavenumbers = np.concatenate((np.arange(0, kmax+2), np.arange(-kmax, 0)))  # Includes Nyquist mode
        # No permutations by default
        self.forward_coeff_permutation = None
        self.backward_coeff_permutation = None

    def _native_grid(self, scale):
        """Native flat global grid."""
        N, = self.grid_shape((scale,))
        return (2 * np.pi / N) * np.arange(N)

    @CachedMethod
    def transform_plan(self, grid_size):
        """Build transform plan."""
        return self.transforms[self.library](grid_size, self.size)

    def local_elements(self):
        local_elements = self.dist.coeff_layout.local_elements(self.domain, scales=scale)[self.axis]
        return (self.wavenumbers[local_elements],)

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
    dtype = np.complex128

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
    dtype = np.complex128

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


class RealFourier(IntervalBasis):
    """
    Fourier real sine/cosine basis.

    Modes: [cos 0, -sin 0, cos 1, -sin 1, ...]
    """

    group_shape = (2,)
    native_bounds = (0, 2*np.pi)
    transforms = {}
    default_library = 'fftw'

    def __init__(self, coord, size, bounds, dealias=1, library=None):
        if library is None:
            library = self.default_library
        super().__init__(coord, size, bounds, dealias)
        self.library = library
        self.kmax = kmax = (size - 1) // 2
        self.wavenumbers_no_repeats = np.arange(0, kmax+1)  # Excludes Nyquist mode
        self.wavenumbers = np.repeat(self.wavenumbers_no_repeats, 2)
        # No permutations by default
        self.forward_coeff_permutation = None
        self.backward_coeff_permutation = None


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

    def __matmul__(self, other):
        if other is None:
            return self.__rmatmul__(other)
        else:
            return other.__rmatmul__(self)

    def __rmatmul__(self, other):
        if other is None:
            return self
        else:
            return NotImplemented

    # def __pow__(self, other):
    #     return self.space.Fourier

    def _native_grid(self, scale):
        """Native flat global grid."""
        N, = self.grid_shape((scale,))
        return (2 * np.pi / N) * np.arange(N)

    @CachedMethod
    def transform_plan(self, grid_size):
        """Build transform plan."""
        return self.transforms[self.library](grid_size, self.size)

    def local_elements(self):
        local_elements = self.dist.coeff_layout.local_elements(self.domain, scales=scale)[self.axis]
        return (self.wavenumbers[local_elements],)

    def forward_transform(self, field, axis, gdata, cdata):
        super().forward_transform(field, axis, gdata, cdata)
        if self.forward_coeff_permutation is not None:
            permute_axis(cdata, axis+len(field.tensorsig), self.forward_coeff_permutation, out=cdata)

    def backward_transform(self, field, axis, cdata, gdata):
        if self.backward_coeff_permutation is not None:
            permute_axis(cdata, axis+len(field.tensorsig), self.backward_coeff_permutation, out=cdata)
        super().backward_transform(field, axis, cdata, gdata)

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
            permuted_wavenumbers = self.wavenumbers[self.forward_coeff_permutation][::2]
            local_groups = permuted_wavenumbers[local_chunks]
            local_index = list(local_groups).index(group)
            group_size = self.group_shape[0]
            return [slice(local_index*group_size, (local_index+1)*group_size)]

class ConvertConstantRealFourier(operators.Convert, operators.SpectralOperator1D):

    input_basis_type = type(None)
    output_basis_type = RealFourier
    subaxis_dependence = [True]
    subaxis_coupling = [False]

    @staticmethod
    def _subspace_matrix(input_basis, output_basis):
        basis = output_basis
        MMT = basis.transforms['matrix'](grid_size=1, coeff_size=output_basis.size)
        return MMT.forward_matrix


class DifferentiateRealFourier(operators.Differentiate):

    input_basis_type = RealFourier
    bands = [-1, 1]
    subaxis_dependence = [True]
    subaxis_coupling = [False]
    dtype = np.float64

    @staticmethod
    def _output_basis(input_basis):
        return input_basis

    @staticmethod
    def _subspace_entry(i, j, input_basis):
        # dx(cos(k*x)) = k*(-sin(k*x))
        # dx(-sin(k*x)) = -k*cos(k*x)
        n = j // 2
        k = n / input_basis.COV.stretch
        if n == 0:
            return 0
        elif (j % 2) == 0:
            # dx(cos(k*x)) = k*(-sin(k*x))
            if i == (j + 1):
                return k
            else:
                return 0
        else:
            # dx(-sin(k*x)) = -k*cos(k*x)
            if i == (j - 1):
                return -k
            else:
                return 0

class InterpolateRealFourier(operators.Interpolate, operators.SpectralOperator1D):

    input_basis_type = RealFourier
    subaxis_dependence = [True]
    subaxis_coupling = [True]
    dtype = np.float64

    @staticmethod
    def _output_basis(input_basis, position):
        return None

    @staticmethod
    def _subspace_matrix(input_basis, position):
        # Interleaved cos(k*x), -sin(k*x)
        x = input_basis.COV.native_coord(position)
        k = input_basis.wavenumbers_no_repeats
        interp_vector = np.exp(1j*k*x).view(dtype=np.float64)
        interp_vector[1::2] *= -1
        # Return as 1*N array
        return interp_vector[None,:]


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


class SpinRecombinationBasis:

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
            if cs == self.coordsystem or (type(cs) is SphericalCoordinates and cs.sub_cs(self.coordsystem)):
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

    def spin_recombination_matrix(self, tensorsig):
        U = self.spin_recombination_matrices(tensorsig)
        matrix = kron(*U)

        if self.dtype == np.float64:
            #matrix = np.array([[matrix.real,-matrix.imag],[matrix.imag,matrix.real]])
            matrix = (np.kron(matrix.real,np.array([[1,0],[0,1]]))
                      + np.kron(matrix.imag,np.array([[0,-1],[1,0]])))
        return matrix

    def forward_spin_recombination(self, tensorsig, gdata, temp):
        """Apply component-to-spin recombination."""
        # HACK: just copying the data so we can apply_matrix repeatedly
        np.copyto(temp, gdata)
        if tensorsig:
            U = self.spin_recombination_matrices(tensorsig)
            if self.dtype == np.complex128:
                for i, Ui in enumerate(U):
                    if Ui is not None:
                        # Directly apply U
                        apply_matrix(Ui, temp, axis=i, out=temp)
            elif self.dtype == np.float64:
                data_cos = temp[axslice(self.axis+len(tensorsig), 0, None, 2)]
                data_msin = temp[axslice(self.axis+len(tensorsig), 1, None, 2)]  # minus sine coefficient
                for i, Ui in enumerate(U):
                    if Ui is not None:
                        # Apply U split up into real and imaginary pieces
                        RC = apply_matrix(Ui.real, data_cos, axis=i)
                        RmS = apply_matrix(Ui.real, data_msin, axis=i)
                        IC = apply_matrix(Ui.imag, data_cos, axis=i)
                        ImS = apply_matrix(Ui.imag, data_msin, axis=i)
                        data_cos[:] = RC - ImS
                        data_msin[:] = RmS + IC

    def backward_spin_recombination(self, tensorsig, temp, gdata):
        """Apply spin-to-component recombination."""
        # HACK: just copying the data so we can apply_matrix repeatedly
        if tensorsig:
            U = self.spin_recombination_matrices(tensorsig)
            if temp.dtype == np.complex128:
                for i, Ui in enumerate(U):
                    if Ui is not None:
                        # Directly apply U
                        apply_matrix(Ui.T.conj(), temp, axis=i, out=temp)
            elif temp.dtype == np.float64:
                data_cos = temp[axslice(self.axis+len(tensorsig), 0, None, 2)]
                data_msin = temp[axslice(self.axis+len(tensorsig), 1, None, 2)]  # minus sine coefficient
                for i, Ui in enumerate(U):
                    if Ui is not None:
                        # Apply U split up into real and imaginary pieces
                        Ui_inv = Ui.T.conj()
                        RC = apply_matrix(Ui_inv.real, data_cos, axis=i)
                        RmS = apply_matrix(Ui_inv.real, data_msin, axis=i)
                        IC = apply_matrix(Ui_inv.imag, data_cos, axis=i)
                        ImS = apply_matrix(Ui_inv.imag, data_msin, axis=i)
                        data_cos[:] = RC - ImS
                        data_msin[:] = RmS + IC
        np.copyto(gdata, temp)


# These are common for S2 and D2
class SpinBasis(MultidimensionalBasis, SpinRecombinationBasis):

    def __init__(self, coordsystem, shape, dealias, dtype=np.complex128, azimuth_library=None):
        self.coordsystem = coordsystem
        self.shape = shape
        self.dtype = dtype
        if np.isscalar(dealias):
            self.dealias = (dealias,) * 2
        elif len(dealias) != 2:
            raise ValueError("dealias must either be a number or a tuple of two numbers")
        else:
            self.dealias = dealias
        self.azimuth_library = azimuth_library
        self.mmax = (shape[0] - 1) // 2
        if dtype == np.complex128:
            self.azimuth_basis = ComplexFourier(coordsystem.coords[0], shape[0], bounds=(0, 2*np.pi), library=azimuth_library)
        elif dtype == np.float64:
            self.azimuth_basis = RealFourier(coordsystem.coords[0], shape[0], bounds=(0, 2*np.pi), library=azimuth_library)
        else:
            raise NotImplementedError()
        self.global_grid_azimuth = self.azimuth_basis.global_grid
        self.local_grid_azimuth = self.azimuth_basis.local_grid
        super().__init__(coordsystem)

    @CachedAttribute
    def local_m(self):
        layout = self.dist.coeff_layout
        local_m_elements = layout.local_elements(self.domain, scales=1)[self.axis]
        return tuple(self.azimuth_basis.wavenumbers[local_m_elements])

    def local_elements(self):
        CL = self.dist.coeff_layout
        LE = CL.local_elements(self.domain, scales=1)[self.axis:self.axis+self.dim]
        LE[0] = self.local_m
        return tuple(LE)

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
    def spintotal(self, spinindex):
        spinorder = [-1, 1, 0]
        spin = lambda index: spinorder[index]
        return sum(spin(index) for index in spinindex)




class DiskBasis(SpinBasis):

    dim = 2
    dims = ['azimuth', 'radius']
    #group_shape = (1, 1)
    transforms = {}

    def __init__(self, coordsystem, shape, radius=1, k=0, alpha=0, dealias=(1,1), radius_library='matrix', **kw):
        super().__init__(coordsystem, shape, dealias, **kw)
        if radius <= 0:
            raise ValueError("Radius must be positive.")
        self.radius = radius
        self.k = k
        self.alpha = alpha
        self.radial_COV = AffineCOV((0, 1), (0, radius))
        self.radius_library = radius_library
        self.Nmax = shape[1] - 1
        if self.mmax > 2*self.Nmax:
            logger.warning("You are using more azimuthal modes than can be resolved with your current radial resolution")
            #raise ValueError("shape[0] cannot be more than twice shape[1].")
        if self.mmax == 0:
            self.forward_transforms = [self.forward_transform_azimuth_Mmax0,
                                       self.forward_transform_radius]
            self.backward_transforms = [self.backward_transform_azimuth_Mmax0,
                                        self.backward_transform_radius]
        else:
            self.forward_transforms = [self.forward_transform_azimuth,
                                       self.forward_transform_radius]
            self.backward_transforms = [self.backward_transform_azimuth,
                                        self.backward_transform_radius]

        self.grid_params = (coordsystem, radius, alpha, dealias)
        if self.mmax > 0 and self.Nmax > 0 and shape[0] % 2 != 0:
            raise ValueError("Don't use an odd phi resolution, please")
        if self.mmax > 0 and self.Nmax > 0 and self.dtype == np.float64 and shape[0] % 4 != 0:
            raise ValueError("Don't use a phi resolution that isn't divisible by 4, please")

        # ASSUMPTION: we assume we are dropping Nyquist mode, so shape=2 --> mmax = 0
        # m permutations for repacking triangular truncation
        if self.dtype == np.complex128:
            raise NotImplementedError("Complex values are not supported for Disk Bases.")
        elif self.dtype == np.float64:
            if self.mmax > 0:
                az_index = np.arange(shape[0])
                div2, mod2 = divmod(az_index, 2)
                div22 = div2 % 2
                self.forward_m_perm = (mod2 + div2) * (1 - div22) + (shape[0] - 1 + mod2 - div2) * div22
                self.backward_m_perm = np.argsort(self.forward_m_perm)
            else:
                self.forward_m_perm = None
                self.backward_m_perm = None

            self.group_shape = (2, 1)
        # this should probably be cleaned up later; needed for m permutation in disk
        self.azimuth_basis = self.S1_basis(radius=self.radius)

    @CachedAttribute
    def radial_basis(self):
        new_shape = (1, self.shape[1])
        dealias = self.dealias
        return DiskBasis(self.coordsystem, new_shape, radius=self.radius, k=self.k, alpha=self.alpha, dealias=dealias, radius_library=self.radius_library, dtype=self.dtype, azimuth_library=self.azimuth_library)

    @CachedMethod
    def S1_basis(self, radius=1):
        if self.dtype == np.complex128:
            S1_basis = ComplexFourier(self.coordsystem.coords[0], self.shape[0], bounds=(0, 2*np.pi), library=self.azimuth_library)
        elif self.dtype == np.float64:
            S1_basis = RealFourier(self.coordsystem.coords[0], self.shape[0], bounds=(0, 2*np.pi), library=self.azimuth_library)
        else:
            raise NotImplementedError()

        S1_basis.radius = radius
        S1_basis.forward_coeff_permutation  = self.forward_m_perm
        S1_basis.backward_coeff_permutation = self.backward_m_perm

        return S1_basis

    def global_shape(self, layout, scales):
        grid_space = layout.grid_space[self.first_axis:self.last_axis+1]
        grid_shape = self.grid_shape(scales)
        if grid_space[0]:
            # grid-grid space
            if self.mmax == 0:
                return (1, grid_shape[1])
            else:
                return grid_shape
        elif grid_space[1]:
            # coeff-grid space
            shape = list(grid_shape)
            shape[0] = self.shape[0]
            return tuple(shape)
        else:
            # coeff-coeff space
            Nphi = self.shape[0]
            if Nphi > 1:
                if self.dtype == np.complex128:
                    raise
                elif self.dtype == np.float64:
                    return self.shape
            else:
                if self.dtype == np.complex128:
                    raise
                elif self.dtype == np.float64:
                    return (2, self.shape[1])

            # DRAFT Repacked triangular truncation for DiskBasis
            # if Nphi > 1:
            #     if self.dtype == np.complex128:
            #         raise
            #     elif self.dtype == np.float64:
            #         return (Nphi//2, Nmax+1+max(0, Nmax+2-Nphi//4))
            # else:
            #     if self.dtype == np.complex128:
            #         raise
            #     elif self.dtype == np.float64:
            #         return (2, Nmax+1+max(0, Nmax+2-Nphi//4))

    def chunk_shape(self, layout):
        grid_space = layout.grid_space[self.first_axis:self.last_axis+1]
        Nmax = self.Nmax
        if grid_space[0]:
            # grid-grid space
            return (1, 1)
        elif grid_space[1]:
            # coeff-grid space
            if self.dtype == np.complex128:
                if Nmax > 0:
                    raise
                else:
                    raise
            elif self.dtype == np.float64:
                # pairs of m don't have to be distributed together
                # since folding is not implemented
                return (2, 1)
        else:
            # coeff-coeff space
            if self.dtype == np.complex128:
                raise
            elif self.dtype == np.float64:
                return (2, 1)

    def local_groups(self, basis_coupling):
        m_coupling, n_coupling = basis_coupling
        if (not m_coupling) and n_coupling:
            groups = []
            local_m = self.local_m

            for m in local_m:
                # Avoid writing repeats for real data
                if [m,None] not in groups:
                    groups.append([m, None])
            return groups
        else:
            raise NotImplementedError()

    def local_group_slices(self, basis_group):
        m_group, n_group = basis_group
        if (m_group is not None) and (n_group is None):
            local_m = self.local_m
            local_indices = np.where((local_m==m_group))
            m_index = local_indices[0][0]
            m_gs = self.group_shape[0]
            m_slice = slice(m_index, m_index+m_gs)
            n_slice = self.n_slice(m_group)
            return [m_slice, n_slice]
        else:
            raise NotImplementedError()

    def _n_limits(self, m):
        nmin = dedalus_sphere.zernike.min_degree(m)
        return (nmin, self.Nmax)

    def n_size(self, m):
        nmin, nmax = self._n_limits(m)
        return nmax - nmin + 1

    def n_slice(self, m):
        nmin, nmax = self._n_limits(m)
        return slice(nmin, nmax+1)

    def __eq__(self, other):
        if isinstance(other, DiskBasis):
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
        if isinstance(other, DiskBasis):
            if self.grid_params == other.grid_params:
                shape = tuple(np.maximum(self.shape, other.shape))
                k = max(self.k, other.k)
                return DiskBasis(self.coordsystem, shape, radius=self.radius, k=k, alpha=self.alpha, dealias=self.dealias, dtype=self.dtype)
        return NotImplemented

    def __mul__(self, other):
        if other is None:
            return self
        if other is self:
            return self
        if isinstance(other, DiskBasis):
            if self.grid_params == other.grid_params:
                shape = tuple(np.maximum(self.shape, other.shape))
                return DiskBasis(self.coordsystem, shape, radius=self.radius, k=0, alpha=self.alpha, dealias=self.dealias, dtype=self.dtype)
        return NotImplemented

    def __matmul__(self, other):
        """NCC is self.

        NB: This does not support NCCs with different number of modes than the fields.
        """
        if other is None:
            return self
        if isinstance(other, DiskBasis):
            return other
        return NotImplemented

    def __rmatmul__(self, other):
        if other is None:
            return self
        return NotImplemented

    @CachedAttribute
    def local_m(self):
        if self.shape[0] == 1:
            return tuple([0,])
        # Permute Fourier wavenumbers
        wavenumbers = self.azimuth_basis.wavenumbers[self.forward_m_perm]
        # Get layout before radius forward transform
        transform = self.dist.get_transform_object(axis=self.axis+1)
        layout = transform.layout1
        # Take local elements
        local_m_elements = layout.local_elements(self.domain, scales=1)[self.axis]
        local_wavenumbers = wavenumbers[local_m_elements]
        return tuple(local_wavenumbers)

    @CachedAttribute
    def local_n(self):
        layout = self.dist.coeff_layout
        local_j = layout.local_elements(self.domain, scales=1)[self.axis + 1][None, :]
        return local_j

    @CachedAttribute
    def m_maps(self):
        return self._compute_m_maps(self.local_m, Nmax=self.Nmax, Nphi=self.shape[0])

    def _compute_m_maps(self, local_m, Nmax, Nphi):
        """
        Tuple of (m, mg_slice, mc_slice, n_slice) for all local m's.
        """
        m_maps = []
        # Get continuous segments of unpacked m's
        segment = [local_m[0], 0, 0] # m, start, end
        segments = [segment]
        m = local_m[0]
        for i, m_i in enumerate(local_m):
            if (m_i == m):
                segment[2] = i + 1
            else:
                m = m_i
                segment = [m, i, i+1]
                segments.append(segment)
        # Build slices for each segment
        for dseg, (m, mg_start, mg_end) in enumerate(segments):
            mg_slice = slice(mg_start, mg_end)
            mc_slice = mg_slice
            m_maps.append((m, mg_slice, mc_slice, self.n_slice(m)))
        return tuple(m_maps)

    def global_grids(self, scales=None):
        if scales == None: scales = (1, 1)
        return (self.global_grid_azimuth(scales[0]),
                self.global_grid_radius(scales[1]))

    def global_grid_radius(self, scale):
        r = self.radial_COV.problem_coord(self._native_radius_grid(scale))
        return reshape_vector(r, dim=self.dist.dim, axis=self.axis+1)

    def local_grids(self, scales=None):
        if scales == None: scales = (1, 1)
        return (self.local_grid_azimuth(scales[0]),
                self.local_grid_radius(scales[1]))

    def local_grid_radius(self, scale):
        r = self.radial_COV.problem_coord(self._native_radius_grid(scale))
        local_elements = self.dist.grid_layout.local_elements(self.domain, scales=scale)[self.axis+1]
        return reshape_vector(r[local_elements], dim=self.dist.dim, axis=self.axis+1)

    def _native_radius_grid(self, scale):
        N = int(np.ceil(scale * self.shape[1]))
        z, weights = dedalus_sphere.zernike.quadrature(2,N,k=self.alpha)
        r = np.sqrt((z+1)/2).astype(np.float64)
        return r

    def global_radius_weights(self, scale=None):
        if scale == None: scale = 1
        N = int(np.ceil(scale * self.shape[1]))
        z, weights = dedalus_sphere.sphere.quadrature(2,N,k=self.alpha)
        return reshape_vector(weights.astype(np.float64), dim=self.dist.dim, axis=self.axis+1)

    def local_radius_weights(self, scale=None):
        if scale == None: scale = 1
        local_elements = self.dist.grid_layout.local_elements(self.domain, scales=scale)[self.axis+1]
        N = int(np.ceil(scale * self.shape[1]))
        z, weights = dedalus_sphere.sphere.quadrature(2,N,k=self.alpha)
        return reshape_vector(weights.astype(np.float64)[local_elements], dim=self.dist.dim, axis=self.axis+1)

    def _new_k(self, k):
        return DiskBasis(self.coordsystem, self.shape, radius = self.radius, k=k, alpha=self.alpha, dealias=self.dealias, dtype=self.dtype,
                         azimuth_library=self.azimuth_library,
                         radius_library=self.radius_library)

    @CachedMethod
    def transform_plan(self, grid_shape, axis, s):
        """Build transform plan."""
        return self.transforms[self.radius_library](grid_shape, self.shape, axis, self.m_maps, s, self.k, self.alpha)

    def forward_transform_azimuth_Mmax0(self, field, axis, gdata, cdata):
        # slice_axis = axis + len(field.tensorsig)
        # np.copyto(cdata[axslice(slice_axis, 0, 1)], gdata)
        np.copyto(cdata[axslice(self.axis+len(field.tensorsig), 0, 1)], gdata)

    def forward_transform_azimuth(self, field, axis, gdata, cdata):
        # Call Fourier transform
        self.azimuth_basis.forward_transform(field, axis, gdata, cdata)
        # Permute m for triangular truncation
        #permute_axis(cdata, axis+len(field.tensorsig), self.forward_m_perm, out=cdata)

    def backward_transform_azimuth_Mmax0(self, field, axis, cdata, gdata):
        # slice_axis = axis + len(field.tensorsig)
        # np.copyto(gdata, cdata[axslice(slice_axis, 0, 1)])
        np.copyto(gdata, cdata[axslice(self.axis+len(field.tensorsig), 0, 1)])

    def backward_transform_azimuth(self, field, axis, cdata, gdata):
        # Permute m back from triangular truncation
        #permute_axis(cdata, axis+len(field.tensorsig), self.backward_m_perm, out=cdata)
        # Call Fourier transform
        self.azimuth_basis.backward_transform(field, axis, cdata, gdata)

    def forward_transform_radius_Nmax0(self, field, axis, gdata, cdata):
        raise NotImplementedError("Not yet.")
        # # Create temporary
        # temp = np.zeros_like(gdata)
        # # Apply spin recombination from gdata to temp
        # self.forward_spin_recombination(field.tensorsig, gdata, out=temp)
        # np.copyto(cdata, temp)

    def forward_transform_radius(self, field, axis, gdata, cdata):
        # Create temporary
        temp = np.zeros_like(gdata)
        # Apply spin recombination from gdata to temp
        self.forward_spin_recombination(field.tensorsig, gdata, temp)
        cdata.fill(0)  # OPTIMIZE: shouldn't be necessary
        # Transform component-by-component from temp to cdata
        S = self.spin_weights(field.tensorsig)
        for i, s in np.ndenumerate(S):
            grid_shape = gdata[i].shape
            plan = self.transform_plan(grid_shape, axis, s)
            plan.forward(temp[i], cdata[i], axis)

    def backward_transform_radius_Nmax0(self, field, axis, cdata, gdata):
        raise NotImplementedError("Not yet.")
        # # Create temporary
        # temp = np.zeros_like(cdata)
        # # Apply spin recombination from cdata to temp
        # self.backward_spin_recombination(field.tensorsig, cdata, out=temp)
        # np.copyto(gdata, temp)

    def backward_transform_radius(self, field, axis, cdata, gdata):
        # Create temporary
        temp = np.zeros_like(gdata)
        # Transform component-by-component from cdata to temp
        S = self.spin_weights(field.tensorsig)
        for i, s in np.ndenumerate(S):
            grid_shape = gdata[i].shape
            plan = self.transform_plan(grid_shape, axis, s)
            plan.backward(cdata[i], temp[i], axis)

        # Apply spin recombination from temp to gdata
        gdata.fill(0)  # OPTIMIZE: shouldn't be necessary
        self.backward_spin_recombination(field.tensorsig, temp, gdata)

    @CachedMethod
    def conversion_matrix(self, m, spintotal, dk):
        E = dedalus_sphere.zernike.operator(2, 'E', radius=self.radius)
        operator = E(+1)**dk
        return operator(self.n_size(m), self.alpha + self.k, np.abs(m + spintotal)).square.astype(np.float64)

    @CachedMethod
    def operator_matrix(self,op,m,spin):

        if op[-1] in ['+', '-']:
            o = op[:-1]
            p = int(op[-1]+'1')
            if m+spin == 0:
                p = +1
            elif m+spin < 0:
                p = -p
            operator = dedalus_sphere.zernike.operator(2, o, radius=self.radius)(p)
        elif op == 'L':
            D = dedalus_sphere.zernike.operator(2, 'D', radius=self.radius)
            if m+spin < 0:
                operator = D(+1) @ D(-1)
            else:
                operator = D(-1) @ D(+1)

        else:
            operator = dedalus_sphere.zernike.operator(2, op, radius=self.radius)

        return operator(self.n_size(m), self.alpha + self.k, abs(m + spin)).square.astype(np.float64)

    @CachedMethod
    def interpolation(self, m, spintotal, position):
        native_position = self.radial_COV.native_coord(position)
        return dedalus_sphere.zernike.polynomials(2, self.n_size(m), self.alpha + self.k, np.abs(m + spintotal), native_position)

    @CachedMethod
    def radius_multiplication_matrix(self, m, spintotal, order, d):
        if order == 0:
            operator = dedalus_sphere.zernike.operator(2, 'Id', radius=self.radius)
        else:
            R = dedalus_sphere.zernike.operator(2, 'R', radius=self.radius)
            if order < 0:
                operator = R(-1)**abs(order)
            else: # order > 0
                operator = R(+1)**abs(order)
        if d > 0:
            R = dedalus_sphere.zernike.operator(2, 'R', radius=self.radius)
            R2 = R(-1) @ R(+1)
            operator = R2**(d//2) @ operator
        return operator(self.n_size(m), self.alpha + self.k, abs(m + spintotal)).square.astype(np.float64)

    def multiplication_matrix(self, subproblem, arg_basis, coeffs, ncc_comp, arg_comp, out_comp, cutoff=1e-6):
        m = subproblem.group[0]  # HACK
        spintotal_ncc = self.spintotal(ncc_comp)
        spintotal_arg = self.spintotal(arg_comp)
        spintotal_out = self.spintotal(out_comp)
        regtotal_ncc = abs(spintotal_ncc)
        regtotal_arg = abs(m + spintotal_arg)
        regtotal_out = abs(m + spintotal_out)
        diff_regtotal = regtotal_out - regtotal_arg
        # jacobi parameters
        a_ncc = self.alpha + self.k
        b_ncc = regtotal_ncc
        N = self.n_size(m)
        d = regtotal_ncc - abs(diff_regtotal)
        if (d >= 0) and (d % 2 == 0):
            J = arg_basis.operator_matrix('Z', m, spintotal_arg)
            A, B = clenshaw.jacobi_recursion(N, a_ncc, b_ncc, J)
            # assuming that we're doing ball for now...
            f0 = dedalus_sphere.zernike.polynomials(2, 1, a_ncc, b_ncc, 1)[0] * sparse.identity(N)
            prefactor = arg_basis.radius_multiplication_matrix(m, spintotal_arg, diff_regtotal, d)
            if self.dtype == np.float64:
                coeffs_filter = coeffs.ravel()[:2*N]
                matrix_cos = prefactor @ clenshaw.matrix_clenshaw(coeffs_filter[:N], A, B, f0, cutoff=cutoff)
                matrix_msin = prefactor @ clenshaw.matrix_clenshaw(coeffs_filter[N:], A, B, f0, cutoff=cutoff)
                matrix = sparse.bmat([[matrix_cos, -matrix_msin], [matrix_msin, matrix_cos]], format='csr')
            elif self.dtype == np.complex128:
                coeffs_filter = coeffs.ravel()[:N]
                matrix = prefactor @ clenshaw.matrix_clenshaw(coeffs_filter, A, B, f0, cutoff=cutoff)
        else:
            if self.dtype == np.float64:
                matrix = sparse.csr_matrix((2*N, 2*N))
            elif self.dtype == np.complex128:
                matrix = sparse.csr_matrix((N, N))
        return matrix


class ConvertPolar(operators.Convert, operators.PolarMOperator):

    input_basis_type = DiskBasis
    output_basis_type = DiskBasis

    def __init__(self, operand, output_basis, out=None):
        operators.Convert.__init__(self, operand, output_basis, out=out)
        self.radius_axis = self.last_axis

    def spinindex_out(self, spinindex_in):
        return (spinindex_in,)

    def radial_matrix(self, spinindex_in, spinindex_out, m):
        radial_basis = self.input_basis
        spintotal = radial_basis.spintotal(spinindex_in)
        dk = self.output_basis.k - radial_basis.k
        if spinindex_in == spinindex_out:
            return radial_basis.conversion_matrix(m, spintotal, dk)
        else:
            raise ValueError("This should never happen.")


class SpinWeightedSphericalHarmonics(SpinBasis):

    dim = 2
    dims = ['azimuth', 'colatitude']
    #group_shape = (1, 1)
    transforms = {}

    def __init__(self, coordsystem, shape, radius=1, dealias=(1,1), colatitude_library='matrix', **kw):
        super().__init__(coordsystem, shape, dealias, **kw)
        if radius <= 0:
            raise ValueError("Radius must be positive.")
        self.radius = radius
        self.colatitude_library = colatitude_library
        self.Lmax = shape[1] - 1
        if self.mmax > self.Lmax + 1:
            logger.warning("You are using more azimuthal modes than can be resolved with your current colatitude resolution")
            #raise ValueError("shape[0] cannot be more than twice shape[1].")
        self.forward_transforms = [self.forward_transform_azimuth,
                                   self.forward_transform_colatitude]
        self.backward_transforms = [self.backward_transform_azimuth,
                                    self.backward_transform_colatitude]
        self.grid_params = (coordsystem, radius, dealias)
        if self.Lmax > 0 and shape[0] % 2 != 0:
            raise ValueError("Don't use an odd phi resolution please")
        if self.Lmax > 0 and self.dtype == np.float64 and shape[0] % 4 != 0:
            raise ValueError("Don't use a phi resolution that isn't divisible by 4, please")
        # m permutations for repacking triangular truncation
        if self.dtype == np.complex128:
            az_index = np.arange(shape[0])
            az_div, az_mod = divmod(az_index, 2)
            self.forward_m_perm = az_div + shape[0] // 2 * az_mod
            self.backward_m_perm = np.argsort(self.forward_m_perm)
            self.group_shape = (1, 1)
        elif self.dtype == np.float64:
            az_index = np.arange(shape[0])
            div2, mod2 = divmod(az_index, 2)
            div22 = div2 % 2
            self.forward_m_perm = (mod2 + div2) * (1 - div22) + (shape[0] - 1 + mod2 - div2) * div22
            self.backward_m_perm = np.argsort(self.forward_m_perm)
            self.group_shape = (2, 1)

    def global_shape(self, layout, scales):
        grid_space = layout.grid_space[self.first_axis:self.last_axis+1]
        grid_shape = self.grid_shape(scales)
        if grid_space[0]:
            # grid-grid space
            return grid_shape
        elif grid_space[1]:
            # coeff-grid space
            shape = list(grid_shape)
            shape[0] = self.shape[0]
            return tuple(shape)
        else:
            # coeff-coeff space
            # Repacked triangular truncation
            Nphi = self.shape[0]
            Lmax = self.Lmax
            if Lmax > 0:
                if self.dtype == np.complex128:
                    return (Nphi//2, Lmax+1+max(0, Lmax+1-Nphi//2))
                elif self.dtype == np.float64:
                    return (Nphi//2, Lmax+1+max(0, Lmax+2-Nphi//2))
            else:
                if self.dtype == np.complex128:
                    return (1, 1)
                elif self.dtype == np.float64:
                    return (2, 1)

    def chunk_shape(self, layout):
        grid_space = layout.grid_space[self.first_axis:self.last_axis+1]
        Lmax = self.Lmax
        if grid_space[0]:
            # grid-grid space
            return (1, 1)
        elif grid_space[1]:
            # coeff-grid space
            if self.dtype == np.complex128:
                if Lmax > 0:
                    return (2, 1)
                else:
                    return (1, 1)
            elif self.dtype == np.float64:
                if Lmax > 0:
                    return (4, 1)
                else:
                    return (2, 1)
        else:
            # coeff-coeff space
            if self.dtype == np.complex128:
                return (1, 1)
            elif self.dtype == np.float64:
                return (2, 1)

    def local_groups(self, basis_coupling):
        m_coupling, ell_coupling = basis_coupling
        if (not m_coupling) and (not ell_coupling):
            groups = []
            local_m, local_ell = self.local_m_ell
            local_m = local_m.ravel()
            local_ell = local_ell.ravel()
            for (m, ell) in zip(local_m, local_ell):
                # Avoid writing repeats for real data
                if [m, ell] not in groups:
                    groups.append([m, ell])
            return groups
        else:
            raise NotImplementedError()

    def local_group_slices(self, basis_group):
        m_group, ell_group = basis_group
        if (m_group is not None) and (ell_group is not None):
            local_m, local_ell = self.local_m_ell
            local_indices = np.where((local_m==m_group)*(local_ell==ell_group))
            m_index = local_indices[0][0]
            m_gs = self.group_shape[0]
            m_slice = slice(m_index, m_index+m_gs)
            ell_index = local_indices[1][0]
            ell_gs = self.group_shape[1]
            ell_slice = slice(ell_index, ell_index+ell_gs)
            return [m_slice, ell_slice]
        else:
            raise NotImplementedError()

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
                return SpinWeightedSphericalHarmonics(self.coordsystem, shape, radius=self.radius, dealias=self.dealias, dtype=self.dtype)
        return NotImplemented

    def __mul__(self, other):
        if other is None:
            return self
        if other is self:
            return self
        if isinstance(other, SpinWeightedSphericalHarmonics):
            if self.radius == other.radius:
                shape = tuple(np.maximum(self.shape, other.shape))
                return SpinWeightedSphericalHarmonics(self.coordsystem, shape, radius=self.radius, dealias=self.dealias, dtype=self.dtype)
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
                                                   radius_library=other.library, dtype=self.dtype)
        return NotImplemented

    @CachedAttribute
    def local_unpacked_m(self):
        # Permute Fourier wavenumbers
        wavenumbers = self.azimuth_basis.wavenumbers[self.forward_m_perm]
        # Get layout before colatitude forward transform
        transform = self.dist.get_transform_object(axis=self.axis+1)
        layout = transform.layout1
        # Take local elements
        local_m_elements = layout.local_elements(self.domain, scales=1)[self.axis]
        local_wavenumbers = wavenumbers[local_m_elements]
        return tuple(local_wavenumbers)

    @CachedAttribute
    def local_m_ell(self):
        layout = self.dist.coeff_layout
        local_i = layout.local_elements(self.domain, scales=1)[self.axis][:, None]
        local_j = layout.local_elements(self.domain, scales=1)[self.axis + 1][None, :]
        local_i = local_i + 0*local_j
        local_j = local_j + 0*local_i
        Nphi = self.shape[0]
        Lmax = self.Lmax
        if self.dtype == np.complex128:
            # Valid for m > 0 except Nyquist
            shift = max(0, Lmax + 1 - Nphi//2)
            local_m = 1 * local_i
            local_ell = local_j - shift
            # Fix for m < 0
            neg_modes = (local_ell < local_m)
            local_m[neg_modes] = local_i[neg_modes] - (Nphi+1)//2
            local_ell[neg_modes] = Lmax - local_j[neg_modes]
            # Fix for m = 0
            m_zero = (local_i == 0)
            local_m[m_zero] = 0
            local_ell[m_zero] = local_j[m_zero]
            # Fix for Nyquist
            nyq_modes = (local_i == 0) * (local_j > Lmax)
            local_m[nyq_modes] = Nphi//2
            local_ell[nyq_modes] = local_j[nyq_modes] - shift
        elif self.dtype == np.float64:
            # Valid for 0 < m < Nphi//4
            shift = max(0, Lmax + 2 - Nphi//2)
            local_m = local_i // 2
            local_ell = local_j - shift
            # Fix for Nphi//4 <= m < Nphi//2-1
            neg_modes = (local_ell < local_m)
            local_m[neg_modes] = (Nphi//2 - 1) - local_m[neg_modes]
            local_ell[neg_modes] = Lmax - local_j[neg_modes]
            # Fix for m = 0
            m_zero = (local_i < 2)
            local_m[m_zero] = 0
            local_ell[m_zero] = local_j[m_zero]
            # Fix for m = Nphi//2 - 1
            m_max = (local_i < 2) * (local_j > Lmax)
            local_m[m_max] = Nphi//2 - 1
            local_ell[m_max] = local_j[m_max] - shift
        # Reshape as multidimensional vectors
        # HACK
        if self.first_axis != 0:
            raise ValueError("Need to reshape these")
        return local_m, local_ell

    @CachedAttribute
    def local_m(self):
        return self.local_m_ell[0]

    @CachedAttribute
    def local_ell(self):
        return self.local_m_ell[1]

    @CachedAttribute
    def m_maps(self):
        return self._compute_m_maps(self.local_unpacked_m, Lmax=self.Lmax, Nphi=self.shape[0])

    @CachedAttribute
    def ell_maps(self):
        return self._compute_ell_maps(self.local_ell)

    @staticmethod
    def _compute_m_maps(local_unpacked_m, Lmax, Nphi):
        """
        Tuple of (m, mg_slice, mc_slice, ell_slice) for all local m's.
        """
        m_maps = []
        # Get continuous segments of unpacked m's
        segment = [local_unpacked_m[0], 0, 0] # m, start, end
        segments = [segment]
        m = local_unpacked_m[0]
        for i, m_i in enumerate(local_unpacked_m):
            if (m_i == m):
                segment[2] = i + 1
            else:
                m = m_i
                segment = [m, i, i+1]
                segments.append(segment)
        # Build slices for each segment
        for dseg, (m, mg_start, mg_end) in enumerate(segments):
            mg_slice = slice(mg_start, mg_end)
            # Fold over every other segment
            gs = mg_end - mg_start
            shift = max(0, Lmax + gs - Nphi//2)  # Assuming gs=1 for complex and gs=2 for real
            mc_slice = slice(gs*(dseg//2), gs*(dseg//2) + gs)
            # Reverse ell's on folded segments for ell locality
            if m == 0:
                # Start m=0 at zero for easier parallelization
                ell_slice = slice(0, Lmax+1)
            elif (gs == 1) and (m == Nphi//2):  # Nyquist mode for complex folding
                ell_slice = slice(Lmax+1, None)
            elif (gs == 2) and (m == Nphi//2 - 1):  # One below Nyquist mode for real folding
                ell_slice = slice(Lmax+1, None)
            elif dseg % 2 == 0:
                # Shifted up
                ell_slice = slice(shift + np.abs(m), None)
            else:
                # Reversed for ell locality
                ell_slice = slice(Lmax - np.abs(m), None, -1)
            m_maps.append((m, mg_slice, mc_slice, ell_slice))
        return tuple(m_maps)

    @staticmethod
    def _compute_ell_maps(local_ell):
        """
        Tuple of (ell, m_slice, ell_slice) for all local ells.
        m_slice and ell_slice are local slices along the phi and theta axes.

        Data for each ell should be sliced as:

            for ell, m_slice, ell_slice in ell_maps:
                ell_data = data[m_slice, ell_slice]
        """
        ell_maps = []
        for dl, ell_row in enumerate(local_ell.T):
            if len(ell_row) == 0:
                continue
            # Get continuous segments of ells
            segment = [ell_row[0], 0, 0] # ell, start, end
            segments = [segment]
            ell = ell_row[0]
            for i, ell_i in enumerate(ell_row):
                if (ell_i == ell):
                    segment[2] = i + 1
                else:
                    ell = ell_i
                    segment = [ell, i, i+1]
                    segments.append(segment)
            # Build slices for each segment
            for ell, start, end in segments:
                ell_map = (ell, slice(start, end), slice(dl, dl+1))
                ell_maps.append(ell_map)
        return tuple(ell_maps)

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
        return self.transforms[self.colatitude_library](grid_shape, self.shape, axis, self.m_maps, s)

    def forward_transform_azimuth_Lmax0(self, field, axis, gdata, cdata):
        slice_axis = axis + len(field.tensorsig)
        np.copyto(cdata[axslice(slice_axis, 0, 1)], gdata)

    def forward_transform_azimuth(self, field, axis, gdata, cdata):
        # Call Fourier transform
        self.azimuth_basis.forward_transform(field, axis, gdata, cdata)
        # Permute m for triangular truncation
        permute_axis(cdata, axis+len(field.tensorsig), self.forward_m_perm, out=cdata)

    def backward_transform_azimuth_Lmax0(self, field, axis, cdata, gdata):
        slice_axis = axis + len(field.tensorsig)
        np.copyto(gdata, cdata[axslice(slice_axis, 0, 1)])

    def backward_transform_azimuth(self, field, axis, cdata, gdata):
        # Permute m back from triangular truncation
        permute_axis(cdata, axis+len(field.tensorsig), self.backward_m_perm, out=cdata)
        # Call Fourier transform
        self.azimuth_basis.backward_transform(field, axis, cdata, gdata)

    def forward_transform_colatitude_Lmax0(self, field, axis, gdata, cdata):
        # Create temporary
        temp = np.zeros_like(gdata)
        # Apply spin recombination from gdata to temp
        self.forward_spin_recombination(field.tensorsig, gdata, out=temp)
        np.copyto(cdata, temp)

    def forward_transform_colatitude(self, field, axis, gdata, cdata):
        # Create temporary
        temp = np.zeros_like(gdata)
        # Apply spin recombination from gdata to temp
        self.forward_spin_recombination(field.tensorsig, gdata, out=temp)
        cdata.fill(0)  # OPTIMIZE: shouldn't be necessary
        # Transform component-by-component from temp to cdata
        S = self.spin_weights(field.tensorsig)
        for i, s in np.ndenumerate(S):
            grid_shape = gdata[i].shape
            plan = self.transform_plan(grid_shape, axis, s)
            plan.forward(temp[i], cdata[i], axis)

    def backward_transform_colatitude_Lmax0(self, field, axis, cdata, gdata):
        # Create temporary
        temp = np.zeros_like(cdata)
        # Apply spin recombination from cdata to temp
        self.backward_spin_recombination(field.tensorsig, cdata, out=temp)
        np.copyto(gdata, temp)

    def backward_transform_colatitude(self, field, axis, cdata, gdata):
        # Create temporary
        temp = np.zeros_like(gdata)
        # Transform component-by-component from cdata to temp
        S = self.spin_weights(field.tensorsig)
        for i, s in np.ndenumerate(S):
            grid_shape = gdata[i].shape
            plan = self.transform_plan(grid_shape, axis, s)
            plan.backward(cdata[i], temp[i], axis)
        # Apply spin recombination from temp to gdata
        gdata.fill(0)  # OPTIMIZE: shouldn't be necessary
        self.backward_spin_recombination(field.tensorsig, temp, out=gdata)

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


# These are common for BallRadialBasis and SphericalShellRadialBasis
class RegularityBasis(SpinRecombinationBasis, MultidimensionalBasis):

    dim = 3
    dims = ['azimuth', 'colatitude', 'radius']
    group_shape = (1, 1, 1)
    transforms = {}
    subaxis_dependence = [False, False, True]

    def __init__(self, coordsystem, radial_size, k, dealias, dtype):
        self.coordsystem = coordsystem
        self.shape = (1, 1, radial_size)
        self.k = k
        self.dealias = (1, 1) + dealias
        self.Nmax = radial_size - 1
        self.dtype = dtype
        # Call at end because dealias is needed to build self.domain
        Basis.__init__(self, coordsystem)
        self.radial_axis = self.first_axis + 2

    @CachedAttribute
    def constant(self):
        return (True, True, False)

    # @CachedAttribute
    # def local_l(self):
    #     layout = self.dist.coeff_layout
    #     local_l_elements = layout.local_elements(self.domain, scales=1)[self.axis-1]
    #     if 0 in local_l_elements:
    #         return (0,)
    #     else:
    #         return ()

    # @CachedAttribute
    # def local_m(self):
    #     layout = self.dist.coeff_layout
    #     local_m_elements = layout.local_elements(self.domain, scales=1)[self.axis-2]
    #     if 0 in local_m_elements:
    #         return (0,)
    #     else:
    #         return ()

    @CachedAttribute
    def local_m_ell(self):
        layout = self.dist.coeff_layout
        local_i = layout.local_elements(self.domain, scales=1)[self.axis][:, None]
        local_j = layout.local_elements(self.domain, scales=1)[self.axis + 1][None, :]
        if self.dtype == np.complex128:
            local_m = local_i
            local_ell = local_j
        elif self.dtype == np.float64:
            local_m = local_i // 2
            local_ell = local_j
        # Reshape as multidimensional vectors
        # HACK
        if self.axis != 0:
            raise ValueError("Need to reshape these")
        return local_m, local_ell

    @CachedAttribute
    def local_m(self):
        return self.local_m_ell[0]

    @CachedAttribute
    def local_ell(self):
        return self.local_m_ell[1]

    def grid_shape(self, scales):
        grid_shape = list(super().grid_shape(scales))
        # Set constant directions back to size 1
        grid_shape[0] = 1
        grid_shape[1] = 1
        return tuple(grid_shape)

    def global_shape(self, layout, scales):
        grid_space = layout.grid_space[self.first_axis:self.last_axis+1]
        grid_shape = self.grid_shape(scales)
        if grid_space[0]:
            # grid-grid-grid space
            return grid_shape
        elif grid_space[1] or grid_space[2]:
            # coeff-grid-grid and coeff-coeff-grid space
            shape = list(grid_shape)
            if self.dtype == np.float64:
                shape[0] = 2
            return tuple(shape)
        else:
            # coeff-coeff-coeff space
            shape = list(grid_shape)
            if self.dtype == np.float64:
                shape[0] = 2
            shape[2] = self.shape[2]
            return tuple(shape)

    def chunk_shape(self, layout):
        grid_space = layout.grid_space[self.first_axis:self.last_axis+1]
        if grid_space[0]:
            # grid-grid-grid space
            return (1, 1, 1)
        else:
            if self.dtype == np.complex128:
                return (1, 1, 1)
            elif self.dtype == np.float64:
                return (2, 1, 1)

    @CachedAttribute
    def ell_maps(self):
        return SWSH._compute_ell_maps(self.local_ell)

    def get_radial_basis(self):
        return self

    def global_grid(self, scale):
        problem_grid = self._radius_grid(scale)
        return reshape_vector(problem_grid, dim=self.dist.dim, axis=self.radial_axis)

    def local_grid(self, scale):
        local_elements = self.dist.grid_layout.local_elements(self.domain, scales=scale)[self.radial_axis]
        problem_grid = self._radius_grid(scale)[local_elements]
        return reshape_vector(problem_grid, dim=self.dist.dim, axis=self.radial_axis)

    def global_weights(self, scale=None):
        if scale == None: scale = 1
        weights = self._radius_weights(scale)
        return reshape_vector(weights.astype(np.float64), dim=self.dist.dim, axis=self.radial_axis)

    def local_weights(self, scale=None):
        if scale == None: scale = 1
        local_elements = self.dist.grid_layout.local_elements(self.domain, scales=scale)[self.radial_axis]
        weights = self._radius_weights(scale)
        return reshape_vector(weights.astype(np.float64)[local_elements], dim=self.dist.dim, axis=self.radial_axis)

    @CachedMethod
    def regularity_allowed(self,l,regularity):
        Rb = np.array([-1, 1, 0], dtype=int)
        if regularity == (): return True
        intertwiner = dedalus_sphere.spin_operators.Intertwiner(l, indexing=(-1,+1,0))
        return not intertwiner.forbidden_regularity(Rb[np.array(regularity)])

    @CachedMethod
    def regtotal(self, regindex):
        regorder = [-1, 1, 0]
        reg = lambda index: regorder[index]
        return sum(reg(index) for index in regindex)

    @CachedMethod
    def xi(self,mu,l):
        return np.sqrt( (l + (mu+1)//2 )/(2*l + 1) )

    @CachedMethod
    def radial_recombinations(self, tensorsig, ell_list):
        # For now only implement recombinations for sphere-only tensors
        for cs in tensorsig:
            if self.coordsystem is not cs:
                raise ValueError("Only supports tensors over spherical coords.")
        order = len(tensorsig)
        Q_matrices = {}
        for ell in ell_list:
            if ell not in Q_matrices:
                Q = dedalus_sphere.spin_operators.Intertwiner(ell, indexing=(-1,+1,0))
                Q_matrices[ell] = Q(order)
        return Q_matrices

    @CachedMethod
    def regularity_classes(self, tensorsig):
        # Regularity-component ordering: [-, +, 0]
        Rb = np.array([-1, 1, 0], dtype=int)
        R = np.zeros([cs.dim for cs in tensorsig], dtype=int)
        for i, cs in enumerate(tensorsig):
            if self.coordsystem is cs: # kludge before we decide how compound coordinate systems work
                R[axslice(i, 0, cs.dim)] += reshape_vector(Rb, dim=len(tensorsig), axis=i)
            #elif self.space in vs.spaces:
            #    n = vs.get_index(self.space)
            #    R[axslice(i, n, n+self.dim)] += reshape_vector(Rb, dim=len(tensorsig), axis=i)
        return R

    def forward_regularity_recombination(self, tensorsig, axis, gdata, ell_maps=None):
        rank = len(tensorsig)
        if ell_maps is None:
            ell_maps = self.ell_maps
        ell_list = tuple(map[0] for map in ell_maps)
        # Apply radial recombinations
        if rank > 0:
            Q = self.radial_recombinations(tensorsig, ell_list)
            # Flatten tensor axes
            shape = gdata.shape
            temp = gdata.reshape((-1,)+shape[rank:])
            slices = [slice(None) for i in range(1+self.dist.dim)]
            # Apply Q transformations for each ell to flattened tensor data
            for ell, m_ind, ell_ind in ell_maps:
                slices[axis-2+1] = m_ind    # Add 1 for tensor axis
                slices[axis-1+1] = ell_ind  # Add 1 for tensor axis
                temp_ell = temp[tuple(slices)]
                apply_matrix(Q[ell].T, temp_ell, axis=0, out=temp_ell)

    def backward_regularity_recombination(self, tensorsig, axis, gdata, ell_maps=None):
        rank = len(tensorsig)
        if ell_maps is None:
            ell_maps = self.ell_maps
        ell_list = tuple(map[0] for map in ell_maps)
        # Apply radial recombinations
        if rank > 0:
            Q = self.radial_recombinations(tensorsig, ell_list)
            # Flatten tensor axes
            shape = gdata.shape
            temp = gdata.reshape((-1,)+shape[rank:])
            slices = [slice(None) for i in range(1+self.dist.dim)]
            # Apply Q transformations for each ell to flattened tensor data
            for ell, m_ind, ell_ind in ell_maps:
                slices[axis-2+1] = m_ind    # Add 1 for tensor axis
                slices[axis-1+1] = ell_ind  # Add 1 for tensor axis
                temp_ell = temp[tuple(slices)]
                apply_matrix(Q[ell], temp_ell, axis=0, out=temp_ell)

    def radial_vector_3(self, comp, m, ell, regindex, local_m=None, local_l=None):
        if local_m == None: local_m = self.local_m
        if local_l == None: local_l = self.local_l
        slices = self.radial_vector_slices(m, ell, regindex, local_m, local_l)
        if slices is None:
            return None
        comp5 = reduced_view(comp, axis=self.axis, dim=3)
        return comp5[(slice(None),) + slices + (slice(None),)]

    @CachedMethod
    def radial_vector_slices(self, m, ell, regindex, local_m, local_l):
        if m > ell:
            return None
        if not self.regularity_allowed(ell, regindex):
            return None
        mi = local_m.index(m)
        li = local_l.index(ell)
        return (mi, li, self.n_slice(ell))

    def local_groups(self, basis_coupling):
        m_coupling, ell_coupling, r_coupling = basis_coupling
        if (not m_coupling) and (not ell_coupling) and r_coupling:
            groups = []
            local_m, local_ell = self.local_m_ell
            if len(local_m) and len(local_ell):
                groups.append([0, 0, None])
            return groups
        else:
            raise NotImplementedError()

    def local_group_slices(self, basis_group):
        raise NotImplementedError("Hmm so what called this??")
        m_group, ell_group, r_group = basis_group
        if (m_group is not None) and (ell_group is not None) and (r_group is None):
            if m_group == 0 and ell_group == 0:
                if self.dtype == np.float64:
                    m_slice = slice(0, 2)
                else:
                    m_slice = slice(0, 1)
                ell_slice = slice(0, 1)
            else:
                m_slice = slice(0, 0)
                ell_slice = slice(0, 0)
            n_slice = slice(None)
            return [m_slice, ell_slice, n_slice]
        else:
            raise NotImplementedError()

    def forward_transform_azimuth(self, field, axis, gdata, cdata):
        # Copy over real part of m = 0
        data_axis = len(field.tensorsig) + axis
        np.copyto(cdata[axslice(data_axis, 0, 1)], gdata)
        cdata[axslice(data_axis, 1, None)] = 0

    def forward_transform_colatitude(self, field, axis, gdata, cdata):
        # Spin recombination
        self.forward_spin_recombination(field.tensorsig, gdata, out=cdata)

    def backward_transform_colatitude(self, field, axis, cdata, gdata):
        # Spin recombination
        self.backward_spin_recombination(field.tensorsig, cdata, out=gdata)

    def backward_transform_azimuth(self, field, axis, cdata, gdata):
        # Copy over real part of m = 0
        np.copyto(gdata, cdata[axslice(len(field.tensorsig)+axis, 0, 1)])


class SphericalShellRadialBasis(RegularityBasis):

    transforms = {}

    def __init__(self, coordsystem, radial_size, radii=(1,2), alpha=(-0.5,-0.5), dealias=(1,), k=0, radius_library='matrix', dtype=np.complex128):
        super().__init__(coordsystem, radial_size, k=k, dealias=dealias, dtype=dtype)
        if radii[0] <= 0:
            raise ValueError("Inner radius must be positive.")
        self.radii = radii
        self.dR = self.radii[1] - self.radii[0]
        self.rho = (self.radii[1] + self.radii[0])/self.dR
        self.alpha = alpha
        self.radius_library = radius_library
        self.grid_params = (coordsystem, radii, alpha, dealias)
        self.forward_transforms = [self.forward_transform_azimuth,
                                   self.forward_transform_colatitude,
                                   self.forward_transform_radius]
        self.backward_transforms = [self.backward_transform_azimuth,
                                    self.backward_transform_colatitude,
                                    self.backward_transform_radius]

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
                radial_size = max(self.shape[2], other.shape[2])
                k = max(self.k, other.k)
                return SphericalShellRadialBasis(self.coordsystem, radial_size, radii=self.radii, alpha=self.alpha, dealias=self.dealias[2:], k=k, radius_library=self.radius_library, dtype=self.dtype)
        return NotImplemented

    def __mul__(self, other):
        if other is None:
            return self
        if isinstance(other, SphericalShellRadialBasis):
            if self.grid_params == other.grid_params:
                radial_size = max(self.shape[2], other.shape[2])
                k = 0
                return SphericalShellRadialBasis(self.coordsystem, radial_size, radii=self.radii, alpha=self.alpha, dealias=self.dealias[2:], k=k, radius_library=self.radius_library, dtype=self.dtype)
        return NotImplemented

    def __matmul__(self, other):
        return other.__rmatmul__(self)

    def __rmatmul__(self, other):
        if other is None:
            return self
        if isinstance(other, SphericalShellRadialBasis):
            if self.grid_params == other.grid_params:
                radial_size = max(self.shape[2], other.shape[2])
                k = self.k + other.k
                return SphericalShellRadialBasis(self.coordsystem, radial_size, radii=self.radii, k=k, alpha=self.alpha, dealias=self.dealias[2:], radius_library=self.radius_library, dtype=self.dtype)
        return NotImplemented

    def _new_k(self, k):
        return SphericalShellRadialBasis(self.coordsystem, self.shape[2], radii=self.radii, alpha=self.alpha, dealias=self.dealias[2:], k=k, radius_library=self.radius_library, dtype=self.dtype)

    @CachedMethod
    def _radius_grid(self, scale):
        N = int(np.ceil(scale * self.shape[2]))
        z, weights = dedalus_sphere.jacobi.quadrature(N, self.alpha[0], self.alpha[1])
        r = self.dR/2*(z + self.rho)
        return r.astype(np.float64)

    @CachedMethod
    def _radius_weights(self, scale):
        N = int(np.ceil(scale * self.shape[2]))
        #z_proj, weights_proj = dedalus_sphere.annulus.quadrature(N, alpha=self.alpha)
        z_proj, weights_proj = dedalus_sphere.jacobi.quadrature(N, self.alpha[0], self.alpha[1])
        z0, weights0 = dedalus_sphere.jacobi.quadrature(N, 0, 0)
        Q0 = dedalus_sphere.jacobi.polynomials(N, self.alpha[0], self.alpha[1], z0)
        Q_proj = dedalus_sphere.jacobi.polynomials(N, self.alpha[0], self.alpha[1], z_proj)
        normalization = self.dR/2
        return normalization * ( (Q0 @ weights0).T ) @ (weights_proj*Q_proj)

    @CachedMethod
    def radial_transform_factor(self, scale, data_axis, dk):
        r = reshape_vector(self._radius_grid(scale), dim=data_axis, axis=data_axis-1)
        return (self.dR/r)**dk

    @CachedMethod
    def interpolation(self, position):
        native_position = position*2/self.dR - self.rho
        a = self.alpha[0] + self.k
        b = self.alpha[1] + self.k
        radial_factor = (self.dR/position)**(self.k)
        return radial_factor*dedalus_sphere.jacobi.polynomials(self.n_size(0), a, b, native_position)

    @CachedMethod
    def transform_plan(self, grid_size, k):
        """Build transform plan."""
        a = self.alpha[0] + k
        b = self.alpha[1] + k
        a0 = self.alpha[0]
        b0 = self.alpha[1]
        return self.transforms[self.radius_library](grid_size, self.Nmax+1, a, b, a0, b0)

    def forward_transform_radius(self, field, axis, gdata, cdata):
        data_axis = len(field.tensorsig) + axis
        grid_size = gdata.shape[data_axis]
        # Multiply by radial factor
        if self.k > 0:
            gdata *= self.radial_transform_factor(field.scales[axis], data_axis, -self.k)
        # Regularity recombination
        self.forward_regularity_recombination(field.tensorsig, axis, gdata)
        # Perform radial transforms component-by-component
        R = self.regularity_classes(field.tensorsig)
        # HACK -- don't want to make a new array every transform
        temp = np.copy(cdata)
        for regindex, regtotal in np.ndenumerate(R):
           plan = self.transform_plan(grid_size, self.k)
           plan.forward(gdata[regindex], temp[regindex], axis)
        np.copyto(cdata, temp)

    def backward_transform_radius(self, field, axis, cdata, gdata):
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
        # Regularity recombination
        self.backward_regularity_recombination(field.tensorsig, axis, gdata)
        # Multiply by radial factor
        if self.k > 0:
            gdata *= self.radial_transform_factor(field.scales[axis], data_axis, self.k)

    @CachedMethod
    def operator_matrix(self, op, l, regtotal):

        l = l + regtotal

        if op in ['D+', 'D-']:
            p = int(op[-1]+'1')
            operator = dedalus_sphere.shell.operator(3, self.radii, 'D', self.alpha)(p, l)
        elif op == 'L':
            D = dedalus_sphere.shell.operator(3, self.radii, 'D', self.alpha)
            operator = D(-1, l+1) @ D(+1, l)
        else:
            operator = dedalus_sphere.shell.operator(3, self.radii, op, self.alpha)
        return operator(self.n_size(l), self.k).square.astype(np.float64)

    def jacobi_conversion(self, l, dk):
        AB = dedalus_sphere.shell.operator(3, self.radii, 'AB', self.alpha)
        operator = AB**dk
        return operator(self.n_size(l), self.k).square.astype(np.float64)

    @CachedMethod
    def conversion_matrix(self, l, regtotal, dk):
        E = dedalus_sphere.shell.operator(3, self.radii, 'E', self.alpha)
        operator = E**dk
        return operator(self.n_size(l), self.k).square.astype(np.float64)

    def n_size(self, ell, Nmax=None):
        if Nmax == None: Nmax = self.Nmax
        return Nmax + 1

    def n_slice(self, ell):
        return slice(0, self.Nmax + 1)

    def start(self, groups):
        return 0

    def multiplication_matrix(self, subproblem, arg_basis, coeffs, ncc_comp, arg_comp, out_comp, cutoff=1e-6):
        ell = subproblem.group[1]  # HACK
        arg_radial_basis = arg_basis.radial_basis
        regtotal_arg = self.regtotal(arg_comp)
        # Jacobi parameters
        a_ncc = self.k + self.alpha[0]
        b_ncc = self.k + self.alpha[1]
        N = self.n_size(ell)
        J = arg_radial_basis.operator_matrix('Z', ell, regtotal_arg)
        A, B = clenshaw.jacobi_recursion(N, a_ncc, b_ncc, J)
        f0 = dedalus_sphere.jacobi.polynomials(1, a_ncc, b_ncc, 1)[0] * sparse.identity(N)
        # Conversions to account for radial prefactors
        prefactor = arg_radial_basis.jacobi_conversion(ell, dk=self.k)
        if self.dtype == np.float64:
            coeffs_filter = coeffs.ravel()[:2*N]
            matrix_cos = prefactor @ clenshaw.matrix_clenshaw(coeffs_filter[:N], A, B, f0, cutoff=cutoff)
            matrix_msin = prefactor @ clenshaw.matrix_clenshaw(coeffs_filter[N:], A, B, f0, cutoff=cutoff)
            matrix = sparse.bmat([[matrix_cos, -matrix_msin], [matrix_msin, matrix_cos]], format='csr')
        elif self.dtype == np.complex128:
            coeffs_filter = coeffs.ravel()[:N]
            matrix = prefactor @ clenshaw.matrix_clenshaw(coeffs_filter, A, B, f0, cutoff=cutoff)
        return matrix


class BallRadialBasis(RegularityBasis):

    transforms = {}

    def __init__(self, coordsystem, radial_size, radius=1, k=0, alpha=0, dealias=(1,), radius_library='matrix', dtype=np.complex128):
        super().__init__(coordsystem, radial_size, k=k, dealias=dealias, dtype=dtype)
        if radius <= 0:
            raise ValueError("Radius must be positive.")
        self.radius = radius
        self.alpha = alpha
        self.radial_COV = AffineCOV((0, 1), (0, radius))
        self.radius_library = radius_library
        self.grid_params = (coordsystem, radius, alpha, dealias)
        self.forward_transforms = [self.forward_transform_azimuth,
                                   self.forward_transform_colatitude,
                                   self.forward_transform_radius]
        self.backward_transforms = [self.backward_transform_azimuth,
                                    self.backward_transform_colatitude,
                                    self.backward_transform_radius]

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
                radial_size = max(self.shape[2], other.shape[2])
                k = max(self.k, other.k)
                return BallRadialBasis(self.coordsystem, radial_size, radius=self.radius, k=k, alpha=self.alpha, dealias=self.dealias[2:], radius_library=self.radius_library, dtype=self.dtype)
        return NotImplemented

    def __mul__(self, other):
        if other is None:
            return self
        if isinstance(other, BallRadialBasis):
            if self.grid_params == other.grid_params:
                radial_size = max(self.shape[2], other.shape[2])
                k = 0
                return BallRadialBasis(self.coordsystem, radial_size, radius=self.radius, k=k, alpha=self.alpha, dealias=self.dealias[2:], radius_library=self.radius_library, dtype=self.dtype)
        return NotImplemented

    def __matmul__(self, other):
        return other.__rmatmul__(self)

    def __rmatmul__(self, other):
        if other is None:
            return self
        if isinstance(other, BallRadialBasis):
            if self.grid_params == other.grid_params:
                radial_size = max(self.shape[2], other.shape[2])
                k = self.k
                return BallRadialBasis(self.coordsystem, radial_size, radius=self.radius, k=k, alpha=self.alpha, dealias=self.dealias[2:], radius_library=self.radius_library, dtype=self.dtype)
        return NotImplemented

    def _new_k(self, k):
        return BallRadialBasis(self.coordsystem, self.shape[2], radius=self.radius, k=k, alpha=self.alpha, dealias=self.dealias[2:], radius_library=self.radius_library, dtype=self.dtype)

    @CachedMethod
    def _radius_grid(self, scale):
        return self.radial_COV.problem_coord(self._native_radius_grid(scale))

    def _native_radius_grid(self, scale):
        N = int(np.ceil(scale * self.shape[2]))
        z, weights = dedalus_sphere.zernike.quadrature(3, N, k=self.alpha)
        r = np.sqrt((z + 1) / 2)
        return r.astype(np.float64)

    @CachedMethod
    def _radius_weights(self, scale):
        N = int(np.ceil(scale * self.shape[2]))
        z, weights = dedalus_sphere.zernike.quadrature(3, N, k=self.alpha)
        return weights

    @CachedMethod
    def interpolation(self, ell, regtotal, position):
        native_position = self.radial_COV.native_coord(position)
        return dedalus_sphere.zernike.polynomials(3, self.n_size(ell), self.alpha + self.k, ell + regtotal, native_position)

    @CachedMethod
    def transform_plan(self, grid_shape, regindex, axis, regtotal, k, alpha):
        """Build transform plan."""
        return self.transforms[self.radius_library](grid_shape, self.Nmax+1, axis, self.ell_maps, regindex, regtotal, k, alpha)

    def forward_transform_radius(self, field, axis, gdata, cdata):
        # Regularity recombination
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

    def backward_transform_radius(self, field, axis, cdata, gdata):
        # Perform radial transforms component-by-component
        R = self.regularity_classes(field.tensorsig)
        # HACK -- don't want to make a new array every transform
        temp = np.copy(gdata)
        for regindex, regtotal in np.ndenumerate(R):
           grid_shape = gdata[regindex].shape
           plan = self.transform_plan(grid_shape, regindex, axis, regtotal, self.k, self.alpha)
           plan.backward(cdata[regindex], temp[regindex], axis)
        np.copyto(gdata, temp)
        # Regularity recombination
        self.backward_regularity_recombination(field.tensorsig, axis, gdata)

    @CachedMethod
    def operator_matrix(self,op,l,deg):

        if op[-1] in ['+', '-']:
            o = op[:-1]
            p = int(op[-1]+'1')
            operator = dedalus_sphere.zernike.operator(3, o, radius=self.radius)(p)
        elif op == 'L':
            D = dedalus_sphere.zernike.operator(3, 'D', radius=self.radius)
            operator = D(-1) @ D(+1)
        else:
            operator = dedalus_sphere.zernike.operator(3, op, radius=self.radius)

        return operator(self.n_size(l), self.alpha + self.k, l + deg).square.astype(np.float64)

    @CachedMethod
    def conversion_matrix(self, ell, regtotal, dk):
        E = dedalus_sphere.zernike.operator(3, 'E', radius=self.radius)
        operator = E(+1)**dk
        return operator(self.n_size(ell), self.alpha + self.k, ell + regtotal).square.astype(np.float64)

    @CachedMethod
    def radius_multiplication_matrix(self, ell, regtotal, order, d):
        if order == 0:
            operator = dedalus_sphere.zernike.operator(3, 'Id', radius=self.radius)
        else:
            R = dedalus_sphere.zernike.operator(3, 'R', radius=self.radius)

            if order < 0:
                operator = R(-1)**abs(order)
            else: # order > 0
                operator = R(+1)**abs(order)

        if d > 0:
            R = dedalus_sphere.zernike.operator(3, 'R', radius=self.radius)
            R2 = R(-1) @ R(+1)
            operator = R2**(d//2) @ operator

        return operator(self.n_size(ell), self.alpha + self.k, ell + regtotal).square.astype(np.float64)

    def _n_limits(self, ell):
        nmin = dedalus_sphere.zernike.min_degree(ell)
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

    def multiplication_matrix(self, subproblem, arg_basis, coeffs, ncc_comp, arg_comp, out_comp, cutoff=1e-6):
        ell = subproblem.group[1]  # HACK
        arg_radial_basis = arg_basis.radial_basis
        regtotal_ncc = self.regtotal(ncc_comp)
        regtotal_arg = self.regtotal(arg_comp)
        regtotal_out = self.regtotal(out_comp)
        diff_regtotal = regtotal_out - regtotal_arg
        # jacobi parameters
        a_ncc = self.alpha + self.k
        b_ncc = regtotal_ncc + 1/2
        N = self.n_size(ell)
        d = regtotal_ncc - abs(diff_regtotal)
        if (d >= 0) and (d % 2 == 0):
            J = arg_radial_basis.operator_matrix('Z', ell, regtotal_arg)
            A, B = clenshaw.jacobi_recursion(N, a_ncc, b_ncc, J)
            # assuming that we're doing ball for now...
            f0 = dedalus_sphere.zernike.polynomials(3, 1, a_ncc, regtotal_ncc, 1)[0] * sparse.identity(N)
            prefactor = arg_radial_basis.radius_multiplication_matrix(ell, regtotal_arg, diff_regtotal, d)
            if self.dtype == np.float64:
                coeffs_filter = coeffs.ravel()[:2*N]
                matrix_cos = prefactor @ clenshaw.matrix_clenshaw(coeffs_filter[:N], A, B, f0, cutoff=cutoff)
                matrix_msin = prefactor @ clenshaw.matrix_clenshaw(coeffs_filter[N:], A, B, f0, cutoff=cutoff)
                matrix = sparse.bmat([[matrix_cos, -matrix_msin], [matrix_msin, matrix_cos]], format='csr')
            elif self.dtype == np.complex128:
                coeffs_filter = coeffs.ravel()[:N]
                matrix = prefactor @ clenshaw.matrix_clenshaw(coeffs_filter, A, B, f0, cutoff=cutoff)
        else:
            if self.dtype == np.float64:
                matrix = sparse.csr_matrix((2*N, 2*N))
            elif self.dtype == np.complex128:
                matrix = sparse.csr_matrix((N, N))
        return matrix


class Spherical3DBasis(MultidimensionalBasis):

    dim = 3
    dims = ['azimuth', 'colatitude', 'radius']
    group_shape = (1, 1, 1)
    transforms = {}
    subaxis_dependence = [False, True, True]

    def __init__(self, coordsystem, shape_angular, dealias_angular, radial_basis, dtype, azimuth_library=None, colatitude_library='matrix'):
        self.coordsystem = coordsystem
        self.shape = tuple( (*shape_angular, radial_basis.shape[2] ) )
        self.dtype = dtype
        if np.isscalar(dealias_angular):
            self.dealias = ( (dealias_angular, dealias_angular, radial_basis.dealias[2]) )
        elif len(dealias_angular) != 2:
            raise ValueError("dealias_angular must either be a number or a tuple of two numbers")
        else:
            self.dealias = tuple( (*dealias_angular, radial_basis.dealias[2]) )
        self.radial_basis = radial_basis
        self.k = radial_basis.k
        self.azimuth_library = azimuth_library
        self.colatitude_library = colatitude_library
        self.sphere_basis = self.S2_basis()
        self.mmax = self.sphere_basis.mmax
        self.Lmax = self.sphere_basis.Lmax
        self.local_m = self.sphere_basis.local_m
        self.local_ell = self.sphere_basis.local_ell
        self.ell_maps = self.sphere_basis.ell_maps
        self.global_grid_azimuth = self.sphere_basis.global_grid_azimuth
        self.global_grid_colatitude = self.sphere_basis.global_grid_colatitude
        self.global_grid_radius = self.radial_basis.global_grid
        self.local_grid_azimuth = self.sphere_basis.local_grid_azimuth
        self.local_grid_colatitude = self.sphere_basis.local_grid_colatitude
        self.local_grid_radius = self.radial_basis.local_grid
        self.global_colatitude_weights = self.sphere_basis.global_colatitude_weights
        self.global_radial_weights = self.radial_basis.global_weights
        self.local_colatitude_weights = self.sphere_basis.local_colatitude_weights
        self.local_radial_weights = self.radial_basis.local_weights
        if self.Lmax > 0:
            self.forward_transform_azimuth = self.sphere_basis.forward_transform_azimuth
            self.forward_transform_colatitude = self.sphere_basis.forward_transform_colatitude
            self.backward_transform_azimuth = self.sphere_basis.backward_transform_azimuth
            self.backward_transform_colatitude = self.sphere_basis.backward_transform_colatitude
        else:
            self.forward_transform_azimuth = self.sphere_basis.forward_transform_azimuth_Lmax0
            self.forward_transform_colatitude = self.sphere_basis.forward_transform_colatitude_Lmax0
            self.backward_transform_azimuth = self.sphere_basis.backward_transform_azimuth_Lmax0
            self.backward_transform_colatitude = self.sphere_basis.backward_transform_colatitude_Lmax0
        Basis.__init__(self, coordsystem)

    @CachedAttribute
    def constant(self):
        return (self.Lmax==0, self.Lmax==0, False)

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

    def local_elements(self):
        CL = self.dist.coeff_layout
        LE = CL.local_elements(self.domain, scales=1)[self.axis:self.axis+self.dim]
        LE[0] = np.array(self.local_m)
        return tuple(LE)

    def get_radial_basis(self):
        return self.radial_basis

    def S2_basis(self,radius=1):
        return SWSH(self.coordsystem.S2coordsys, self.shape[:2], radius=radius, dealias=self.dealias[:2], dtype=self.dtype,
                    azimuth_library=self.azimuth_library, colatitude_library=self.colatitude_library)

    @CachedMethod
    def operator_matrix(self, op, l, regtotal, dk=0):
        return self.radial_basis.operator_matrix(op, l, regtotal)

    @CachedMethod
    def conversion_matrix(self, l, regtotal, dk):
        return self.radial_basis.conversion_matrix(l, regtotal, dk)

    def n_size(self, ell):
        return self.radial_basis.n_size(ell)

    def n_slice(self, ell):
        return self.radial_basis.n_slice(ell)

    def start(self, groups):
        return self.radial_basis.start(groups)

    def global_shape(self, layout, scales):
        grid_space = layout.grid_space[self.first_axis:self.last_axis+1]
        grid_shape = self.grid_shape(scales)
        if grid_space[2]:
            s2_shape = self.sphere_basis.global_shape(layout, scales)
            return s2_shape + (grid_shape[2],)
        else:
            # coeff-coeff-coeff space
            s2_shape = self.sphere_basis.global_shape(layout, scales)
            return s2_shape + (self.shape[2],)

    def chunk_shape(self, layout):
        s2_chunk = self.sphere_basis.chunk_shape(layout)
        return s2_chunk + (1,)

    def local_groups(self, basis_coupling):
        m_coupling, ell_coupling, n_coupling = basis_coupling
        if (not m_coupling) and (not ell_coupling) and (n_coupling):
            groups = self.sphere_basis.local_groups((m_coupling, ell_coupling))
            for group in groups:
                group.append(None)
            return groups
        else:
            raise NotImplementedError()

    def local_group_slices(self, basis_group):
        m_group, ell_group, n_group = basis_group
        if (m_group is not None) and (ell_group is not None) and (n_group is None):
            slices = self.sphere_basis.local_group_slices((m_group, ell_group))
            n_slice = self.radial_basis.n_slice(ell=ell_group)
            slices.append(n_slice)
            return slices
        else:
            raise NotImplementedError()


class SphericalShellBasis(Spherical3DBasis):

    def __init__(self, coordsystem, shape, radii=(1,2), alpha=(-0.5,-0.5), dealias=(1,1,1), k=0, dtype=np.complex128, azimuth_library=None, colatitude_library='matrix', radius_library='matrix'):
        self.radial_basis = SphericalShellRadialBasis(coordsystem, shape[2], radii=radii, alpha=alpha, dealias=(dealias[2],), k=k, dtype=dtype, radius_library=radius_library)
        Spherical3DBasis.__init__(self, coordsystem, shape[:2], dealias[:2], self.radial_basis, dtype=dtype, azimuth_library=azimuth_library, colatitude_library=colatitude_library)
        self.grid_params = (coordsystem, radii, alpha, dealias)
#        self.forward_transform_radius = self.radial_basis.forward_transform
#        self.backward_transform_radius = self.radial_basis.backward_transform
        self.forward_transforms = [self.forward_transform_azimuth,
                                   self.forward_transform_colatitude,
                                   self.forward_transform_radius]
        self.backward_transforms = [self.backward_transform_azimuth,
                                    self.backward_transform_colatitude,
                                    self.backward_transform_radius]

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
                                           dtype=self.dtype, azimuth_library=self.azimuth_library, colatitude_library=self.colatitude_library,
                                           radius_library=self.radial_basis.radius_library)
        return NotImplemented

    def __mul__(self, other):
        if other is None:
            return self
        if isinstance(other, SphericalShellBasis):
            if self.grid_params == other.grid_params:
                shape = tuple(np.maximum(self.shape, other.shape))
                k = 0
                return SphericalShellBasis(self.coordsystem, shape, radii=self.radial_basis.radii, alpha=self.radial_basis.alpha, dealias=self.dealias, k=k,
                                           dtype=self.dtype, azimuth_library=self.azimuth_library, colatitude_library=self.colatitude_library,
                                           radius_library=self.radial_basis.radius_library)
        if isinstance(other, SphericalShellRadialBasis):
            radial_basis = other * self.radial_basis
            return self._new_k(radial_basis.k)
        return NotImplemented

    def __rmatmul__(self, other):
        if other is None:
            return self
        if isinstance(other, SphericalShellRadialBasis):
            radial_basis = other @ self.radial_basis
            return self._new_k(radial_basis.k)
        return NotImplemented

    def _new_k(self, k):
        return SphericalShellBasis(self.coordsystem, self.shape, radii=self.radial_basis.radii, alpha=self.radial_basis.alpha, dealias=self.dealias, k=k,
                                   dtype=self.dtype, azimuth_library=self.azimuth_library, colatitude_library=self.colatitude_library,
                                   radius_library=self.radial_basis.radius_library)

    def forward_transform_radius(self, field, axis, gdata, cdata):
        radial_basis = self.radial_basis
        data_axis = len(field.tensorsig) + axis
        grid_size = gdata.shape[data_axis]
        # Multiply by radial factor
        if self.k > 0:
            gdata *= radial_basis.radial_transform_factor(field.scales[axis], data_axis, -self.k)
        # Apply regularity recombination using 3D ell map
        radial_basis.forward_regularity_recombination(field.tensorsig, axis, gdata, ell_maps=self.ell_maps)
        # Perform radial transforms component-by-component
        R = radial_basis.regularity_classes(field.tensorsig)
        # HACK -- don't want to make a new array every transform
        temp = np.copy(cdata)
        for regindex, regtotal in np.ndenumerate(R):
           plan = radial_basis.transform_plan(grid_size, self.k)
           plan.forward(gdata[regindex], temp[regindex], axis)
        np.copyto(cdata, temp)

    def backward_transform_radius(self, field, axis, cdata, gdata):
        radial_basis = self.radial_basis
        data_axis = len(field.tensorsig) + axis
        grid_size = gdata.shape[data_axis]
        # Perform radial transforms component-by-component
        R = radial_basis.regularity_classes(field.tensorsig)
        # HACK -- don't want to make a new array every transform
        temp = np.copy(gdata)
        for i, r in np.ndenumerate(R):
           plan = radial_basis.transform_plan(grid_size, self.k)
           plan.backward(cdata[i], temp[i], axis)
        np.copyto(gdata, temp)
        # Apply regularity recombinations using 3D ell map
        radial_basis.backward_regularity_recombination(field.tensorsig, axis, gdata, ell_maps=self.ell_maps)
        # Multiply by radial factor
        if self.k > 0:
            gdata *= radial_basis.radial_transform_factor(field.scales[axis], data_axis, self.k)

class BallBasis(Spherical3DBasis):

    def __init__(self, coordsystem, shape, radius=1, k=0, alpha=0, dealias=(1,1,1), dtype=np.complex128, azimuth_library=None, colatitude_library='matrix', radius_library='matrix'):
        self.radial_basis = BallRadialBasis(coordsystem, shape[2], radius=radius, k=k, alpha=alpha, dealias=(dealias[2],), dtype=dtype, radius_library=radius_library)
        Spherical3DBasis.__init__(self, coordsystem, shape[:2], dealias[:2], self.radial_basis, dtype=dtype, azimuth_library=azimuth_library, colatitude_library=colatitude_library)
        self.grid_params = (coordsystem, radius, alpha, dealias)
        self.forward_transforms = [self.forward_transform_azimuth,
                                   self.forward_transform_colatitude,
                                   self.forward_transform_radius]
        self.backward_transforms = [self.backward_transform_azimuth,
                                    self.backward_transform_colatitude,
                                    self.backward_transform_radius]

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
                                 dtype=self.dtype, azimuth_library=self.azimuth_library, colatitude_library=self.colatitude_library,
                                 radius_library=self.radial_basis.radius_library)
        return NotImplemented

    def __mul__(self, other):
        if other is None:
            return self
        if isinstance(other, BallBasis):
            if self.grid_params == other.grid_params:
                shape = tuple(np.maximum(self.shape, other.shape))
                k = 0
                return BallBasis(self.coordsystem, shape, radius=self.radial_basis.radius, k=k, alpha=self.radial_basis.alpha, dealias=self.dealias, dtype=self.dtype,
                                 azimuth_library=self.azimuth_library, colatitude_library=self.colatitude_library,
                                 radius_library=self.radial_basis.radius_library)
        if isinstance(other, BallRadialBasis):
            radial_basis = other * self.radial_basis
            return self._new_k(radial_basis.k)
        return NotImplemented

    def __rmatmul__(self, other):
        if other is None:
            return self
        if isinstance(other, BallRadialBasis):
            radial_basis = other @ self.radial_basis
            return self._new_k(radial_basis.k)
        return NotImplemented

    def _new_k(self, k):
        return BallBasis(self.coordsystem, self.shape, radius = self.radial_basis.radius, k=k, alpha=self.radial_basis.alpha, dealias=self.dealias, dtype=self.dtype,
                         azimuth_library=self.azimuth_library, colatitude_library=self.colatitude_library,
                         radius_library=self.radial_basis.radius_library)

    @CachedMethod
    def transform_plan(self, grid_shape, regindex, axis, regtotal, k, alpha):
        """Build transform plan."""
        radius_library = self.radial_basis.radius_library
        Nmax = self.radial_basis.Nmax
        return self.transforms[radius_library](grid_shape, Nmax+1, axis, self.ell_maps, regindex, regtotal, k, alpha)

    def forward_transform_radius(self, field, axis, gdata, cdata):
        # apply transforms based off the 3D basis' local_l
        radial_basis = self.radial_basis
        # Apply regularity recombination
        radial_basis.forward_regularity_recombination(field.tensorsig, axis, gdata, ell_maps=self.ell_maps)
        # Perform radial transforms component-by-component
        R = radial_basis.regularity_classes(field.tensorsig)
        # HACK -- don't want to make a new array every transform
        # transforms don't seem to work properly if we don't zero these
        temp = np.zeros_like(cdata)
        for regindex, regtotal in np.ndenumerate(R):
           grid_shape = gdata[regindex].shape
           plan = self.transform_plan(grid_shape, regindex, axis, regtotal, radial_basis.k, radial_basis.alpha)
           plan.forward(gdata[regindex], temp[regindex], axis)
        np.copyto(cdata, temp)

    def backward_transform_radius(self, field, axis, cdata, gdata):
        # apply transforms based off the 3D basis' local_l
        radial_basis = self.radial_basis
        # Perform radial transforms component-by-component
        R = radial_basis.regularity_classes(field.tensorsig)
        # HACK -- don't want to make a new array every transform
        # transforms don't seem to work properly if we don't zero these
        temp = np.zeros_like(gdata)
        for regindex, regtotal in np.ndenumerate(R):
           grid_shape = gdata[regindex].shape
           plan = self.transform_plan(grid_shape, regindex, axis, regtotal, radial_basis.k, radial_basis.alpha)
           plan.backward(cdata[regindex], temp[regindex], axis)
        np.copyto(gdata, temp)
        # Apply regularity recombinations
        radial_basis.backward_regularity_recombination(field.tensorsig, axis, gdata, ell_maps=self.ell_maps)

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
        operators.Convert.__init__(self, operand, output_basis, out=out)
        self.radial_basis = self.input_basis.get_radial_basis()

    def regindex_out(self, regindex_in):
        return (regindex_in,)

    def radial_matrix(self, regindex_in, regindex_out, ell):
        radial_basis = self.radial_basis
        regtotal = radial_basis.regtotal(regindex_in)
        dk = self.output_basis.k - radial_basis.k
        if regindex_in == regindex_out:
            return radial_basis.conversion_matrix(ell, regtotal, dk)
        else:
            raise ValueError("This should never happen.")


class DiskInterpolate(operators.Interpolate, operators.PolarMOperator):

    basis_type = DiskBasis
    basis_subaxis = 1

    @classmethod
    def _check_args(cls, operand, coord, position, out=None):
        if isinstance(operand, Operand):
            if isinstance(operand.domain.get_basis(coord), cls.basis_type):
                if operand.domain.get_basis_subaxis(coord) == cls.basis_subaxis:
                    return True
        return False

    @staticmethod
    def _output_basis(input_basis, position):
        return input_basis.S1_basis(radius=position)

    def __init__(self, operand, coord, position, out=None):
        operators.Interpolate.__init__(self, operand, coord, position, out=None)

    def subproblem_matrix(self, subproblem):
        m = subproblem.group[self.last_axis - 1]
        matrix = super().subproblem_matrix(subproblem)
        radial_basis = self.input_basis
        if self.tensorsig != ():
            U = radial_basis.spin_recombination_matrix(self.tensorsig)
            matrix = U @ matrix

        return matrix

    def operate(self, out):
        """Perform operation."""
        operand = self.args[0]
        input_basis = self.input_basis
        output_basis = self.output_basis
        radial_basis = self.input_basis
        axis = self.last_axis
        # Set output layout
        out.set_layout(operand.layout)
        # Apply operator
        S = radial_basis.spin_weights(operand.tensorsig)
        slices_in  = [slice(None) for i in range(input_basis.dist.dim)]
        slices_out = [slice(None) for i in range(input_basis.dist.dim)]
        for spinindex, spintotal in np.ndenumerate(S):
           comp_in = operand.data[spinindex]
           comp_out = out.data[spinindex]
           for m, mg_slice, mc_slice, n_slice in input_basis.m_maps:
               slices_in[axis-1] = slices_out[axis-1] = mc_slice
               slices_in[axis] = n_slice
               vec_in  = comp_in[tuple(slices_in)]
               vec_out = comp_out[tuple(slices_out)]
               A = self.radial_matrix(spinindex, spinindex, m)
               apply_matrix(A, vec_in, axis=axis, out=vec_out)
        radial_basis.backward_spin_recombination(operand.tensorsig, out.data)

    def radial_matrix(self, spinindex_in, spinindex_out, m):
        position = self.position
        basis = self.input_basis
        if spinindex_in == spinindex_out:
            return self._radial_matrix(basis, m, basis.spintotal(spinindex_in), position)
        else:
            return np.zeros((1,basis.n_size(m)))

    def spinindex_out(self, spinindex_in):
        return (spinindex_in,)

    @staticmethod
    @CachedMethod
    def _radial_matrix(basis, m, spintotal, position):
        return reshape_vector(basis.interpolation(m, spintotal, position), dim=2, axis=1)


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

    def __init__(self, operand, coord, position, out=None):
        operators.Interpolate.__init__(self, operand, coord, position, out=None)
        self.radial_basis = self.input_basis.get_radial_basis()

    def subproblem_matrix(self, subproblem):
        ell = subproblem.group[self.last_axis - 1]
        matrix = super().subproblem_matrix(subproblem)
        radial_basis = self.radial_basis
        if self.tensorsig != ():
            Q = radial_basis.radial_recombinations(self.tensorsig, ell_list=(ell,))[ell]
            if self.dtype == np.float64:
                # Block-diag for sin/cos parts for real dtype
                Q = np.kron(Q, np.eye(2))
            matrix = Q @ matrix
        return matrix

    def operate(self, out):
        """Perform operation."""
        operand = self.args[0]
        input_basis = self.input_basis
        output_basis = self.output_basis
        radial_basis = self.radial_basis
        axis = radial_basis.radial_axis
        # Set output layout
        out.set_layout(operand.layout)
        # Apply operator
        R = radial_basis.regularity_classes(operand.tensorsig)
        slices_in  = [slice(None) for i in range(input_basis.dist.dim)]
        slices_out = [slice(None) for i in range(input_basis.dist.dim)]
        for regindex, regtotal in np.ndenumerate(R):
           comp_in = operand.data[regindex]
           comp_out = out.data[regindex]
           for ell, m_ind, ell_ind in input_basis.ell_maps:
               allowed  = radial_basis.regularity_allowed(ell, regindex)
               if allowed:
                   slices_in[axis-2] = slices_out[axis-2] = m_ind
                   slices_in[axis-1] = slices_out[axis-1] = ell_ind
                   slices_in[axis] = radial_basis.n_slice(ell)
                   vec_in  = comp_in[slices_in]
                   vec_out = comp_out[slices_out]
                   A = self.radial_matrix(regindex, regindex, ell)
                   apply_matrix(A, vec_in, axis=axis, out=vec_out)
        radial_basis.backward_regularity_recombination(operand.tensorsig, self.basis_subaxis, out.data, ell_maps=input_basis.ell_maps)

    def radial_matrix(self, regindex_in, regindex_out, ell):
        position = self.position
        basis = self.radial_basis
        if regindex_in == regindex_out:
            return self._radial_matrix(basis, ell, basis.regtotal(regindex_in), position)
        else:
            return np.zeros((1,basis.n_size(ell)))

    def regindex_out(self, regindex_in):
        return (regindex_in,)

    @staticmethod
    @CachedMethod
    def _radial_matrix(basis, ell, regtotal, position):
        return reshape_vector(basis.interpolation(ell, regtotal, position), dim=2, axis=1)


class SphericalShellRadialInterpolate(operators.Interpolate, operators.SphericalEllOperator):

    basis_type = SphericalShellBasis
    basis_subaxis = 2

    def __init__(self, operand, coord, position, out=None):
        operators.Interpolate.__init__(self, operand, coord, position, out=None)
        self.radial_basis = self.input_basis.get_radial_basis()

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
        basis_in = self.radial_basis
        matrix = super().subproblem_matrix(subproblem)
        if self.tensorsig != ():
            Q = basis_in.radial_recombinations(self.tensorsig, ell_list=(ell,))[ell]
            if self.dtype == np.float64:
                # Block-diag for sin/cos parts for real dtype
                Q = np.kron(Q, np.eye(2))
            matrix = Q @ matrix
        # Radial rescaling
        return matrix

    def operate(self, out):
        """Perform operation."""
        operators.SphericalEllOperator.operate(self, out)
        operand = self.args[0]
        radial_basis = self.radial_basis
        input_basis = self.input_basis
        # Q matrix
        radial_basis.backward_regularity_recombination(operand.tensorsig, self.basis_subaxis, out.data, ell_maps=input_basis.ell_maps)

    def radial_matrix(self, regindex_in, regindex_out, ell):
        position = self.position
        basis = self.radial_basis
        if regindex_in == regindex_out:
            return self._radial_matrix(basis, position)
        else:
            raise ValueError("This should never happen.")

    def regindex_out(self, regindex_in):
        return (regindex_in,)

    @staticmethod
    @CachedMethod
    def _radial_matrix(basis, position):
        return reshape_vector(basis.interpolation(position), dim=2, axis=1)


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
        matrix = np.array(matrix)
        if self.dtype == np.float64:
            # Block-diag for sin/cos parts for real dtype
            matrix = np.kron(matrix, np.eye(2))
        return matrix

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
        matrix = np.array(matrix)
        if self.dtype == np.float64:
            # Block-diag for sin/cos parts for real dtype
            matrix = np.kron(matrix, np.eye(2))
        return matrix

    def operate(self, out):
        """Perform operation."""
        operand = self.args[0]
        # Set output layout
        layout = operand.layout
        out.set_layout(layout)
        np.copyto(out.data, operand.data[axslice(self.index,0,2)])


from . import transforms
