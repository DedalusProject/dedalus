"""Abstract and built-in classes for spectral bases."""

import numpy as np
from scipy import sparse
from functools import reduce
import inspect
from math import prod

from . import operators
from ..libraries import spin_recombination
from ..tools.array import kron, axslice, apply_matrix, permute_axis
from ..tools.cache import CachedAttribute, CachedMethod, CachedClass
from ..tools import jacobi
from ..tools import clenshaw
from ..tools.array import reshape_vector, axindex, axslice, interleave_matrices
from ..tools.dispatch import MultiClass, SkipDispatchException
from ..tools.general import unify, DeferredTuple
from .coords import Coordinate, CartesianCoordinates, S2Coordinates, SphericalCoordinates, PolarCoordinates, AzimuthalCoordinate, DirectProduct
from .domain import Domain
from .field  import Operand, LockedField
from .future import FutureLockedField
from ..libraries import dedalus_sphere

import logging
logger = logging.getLogger(__name__.split('.')[-1])


# Public interface
__all__ = ['Jacobi',
           'Legendre',
           'Ultraspherical',
           'Chebyshev',
           'ChebyshevT',
           'ChebyshevU',
           'ChebyshevV',
           'RealFourier',
           'ComplexFourier',
           'Fourier',
           'DiskBasis',
           'AnnulusBasis',
           'SphereBasis',
           'BallBasis',
           'ShellBasis']


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

    @CachedMethod
    def domain(self, dist):
        return Domain(dist, (self,))

    def clone_with(self, **new_kw):
        (_, *argnames), _, _, _, _, _, _ = inspect.getfullargspec(type(self).__init__)
        kw = {name: getattr(self, name) for name in argnames}
        kw.update(new_kw)
        return type(self)(**kw)

    @CachedAttribute
    def constant(self):
        return tuple(False for i in range(self.dim))

    def __matmul__(self, other):
        # NCC (self) * operand (other)
        if other is None:
            # All multiplications by constants return same basis
            return self
        else:
            # Call __rmatmul__ codepath for operand basis
            return other.__rmatmul__(self)

    # def __repr__(self):
    #     return '<%s %i>' %(self.__class__.__name__, id(self))

    # def __str__(self):
    #     return '%s.%s' %(self.space.name, self.__class__.__name__)

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def grid_shape(self, scales):
        shape = np.array([int(np.ceil(s*n)) for s, n in zip(scales, self.shape)])
        shape[np.array(self.shape) == 1] = 1
        return tuple(shape)

    def global_shape(self, grid_space, scales):
        # Subclasses must implement
        raise NotImplementedError

    def chunk_shape(self, grid_space):
        # Subclasses must implement
        raise NotImplementedError

    def elements_to_groups(self, grid_space, elements):
        # Subclasses must implement
        raise NotImplementedError

    def global_grids(self, dist, scales):
        """Global grids."""
        # Subclasses must implement
        # Returns tuple of global grids along each subaxis
        raise NotImplementedError(f"{type(self)} has not implement global_grids.")

    def local_grids(self, dist, scales):
        """Local grids."""
        # Subclasses must implement
        # Returns tuple of local grids along each subaxis
        raise NotImplementedError(f"{type(self)} has not implement local_grids.")

    def local_modes(self, dist, scales):
        """Local modes."""
        # Subclasses must implement
        # Returns tuple of local modes along each subaxis
        raise NotImplementedError(f"{type(self)} has not implement local_modes.")

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

    @classmethod
    def _last_axis_component_ncc_matrix(cls, subproblem, ncc_basis, arg_basis, out_basis, coeffs, ncc_comp, arg_comp, out_comp, ncc_tensorsig, arg_tensorsig, out_tensorsig, cutoff):
        # Simple matrix method by default
        return cls.ncc_matrix(ncc_basis, arg_basis, out_basis, coeffs.ravel(), cutoff=cutoff)

    @classmethod
    def ncc_matrix(cls, ncc_basis, arg_basis, out_basis, coeffs, cutoff=1e-6):
        """Build NCC matrix via direct summation."""
        N = len(coeffs)
        for i in range(N):
            coeff = coeffs[i]
            # Build initial matrix
            if i == 0:
                matrix = ncc_basis.product_matrix(arg_basis, out_basis, i)
                total = 0 * sparse.kron(matrix, coeff)
                total.eliminate_zeros()
            if len(coeff.shape) or (abs(coeff) > cutoff):
                matrix = ncc_basis.product_matrix(arg_basis, out_basis, i)
                total = total + sparse.kron(matrix, coeff)
        return total

    def product_matrix(self, arg_basis, out_basis, i):
        if arg_basis is None:
            N = self.size
            return sparse.coo_matrix(([1],([i],[0])), shape=(N,1)).tocsr()
        else:
            raise NotImplementedError()

    def build_ncc_matrix(self, product, subproblem, ncc_cutoff, max_ncc_terms):
        # Default to last axis only
        dist = product.dist
        if any(product.ncc.domain.nonconstant[dist.first_axis(self):dist.last_axis(self)]):
            raise NotImplementedError("Only last-axis NCCs implemented for this basis.")
        axis = dist.last_axis(self)
        ncc_basis = product.ncc.domain.get_basis(axis)
        arg_basis = product.operand.domain.get_basis(axis)
        out_basis = product.domain.get_basis(axis)
        coeffs = product._ncc_data
        return self._last_axis_field_ncc_matrix(product, subproblem, axis, ncc_basis, arg_basis, out_basis, coeffs, ncc_cutoff, max_ncc_terms)

    @classmethod
    def _last_axis_field_ncc_matrix(cls, product, subproblem, axis, ncc_basis, arg_basis, out_basis, coeffs, ncc_cutoff, max_ncc_terms):
        def enum_indices(tensorsig):
            shape = tuple(cs.dim for cs in tensorsig)
            return enumerate(np.ndindex(shape))
        operand = product.operand
        ncc = product.ncc
        group = subproblem.group
        ncc_first = (ncc is product.args[0])
        ncc_group = tuple(0*g if g is not None else None for g in group)
        if ncc_first:
            Gamma = product.Gamma(ncc.tensorsig, operand.tensorsig, product.tensorsig, ncc_group, group, group, axis)
        else:
            Gamma = product.Gamma(operand.tensorsig, ncc.tensorsig, product.tensorsig, group, ncc_group, group, axis)
            Gamma = Gamma.transpose((1,0,2))
        # Loop over input and output components to build matrix blocks
        M = prod(subproblem.coeff_shape(product.domain)[:axis+1])
        N = prod(subproblem.coeff_shape(operand.domain)[:axis+1])
        blocks = []
        for ic, out_comp in enum_indices(product.tensorsig):
            block_row = []
            for ib, arg_comp in enum_indices(operand.tensorsig):
                block = sparse.csr_matrix((M, N))
                # Loop over ncc components
                for ia, ncc_comp in enum_indices(ncc.tensorsig):
                    G = Gamma[ia, ib, ic]
                    if abs(G) > ncc_cutoff:
                        if ncc_basis is None:
                            if coeffs[ncc_comp].size != 1:
                                raise NotImplementedError()
                            matrix = coeffs[ncc_comp].ravel()[0] * sparse.eye(M, N)
                        else:
                            matrix = cls._last_axis_component_ncc_matrix(subproblem, ncc_basis, arg_basis, out_basis, coeffs[ncc_comp].squeeze(), ncc_comp, arg_comp, out_comp, ncc.tensorsig, operand.tensorsig, product.tensorsig, cutoff=ncc_cutoff)
                            # Domains with real Fourier bases require kroneckering the Jacobi NCC matrix up to match the subsystem shape including the sin and cos parts of RealFourier data
                            # This fix assumes the Jacobi basis is on the last axis
                            if matrix.shape != (M,N):
                                m, n = matrix.shape
                                matrix = sparse.kron(sparse.eye(M//m, N//n), matrix)
                        if product.dtype == np.float64:
                            if G.imag != 0:
                                raise NotImplementedError()
                            else:
                                block += G.real * matrix
                        else:
                            block += G * matrix
                block_row.append(block)
            blocks.append(block_row)
        matrix = sparse.bmat(blocks, format='csr')
        return matrix
        #return getattr(ncc_basis, self.ncc_method)(arg_basis, coeffs, ncc_ts, arg_ts, out_ts, subproblem, ncc_first, *gamma_args, cutoff=1e-6)
        # tshape = [cs.dim for cs in ncc.tensorsig]
        # self._ncc_matrices = [self._ncc_matrix_recursion(ncc.data[ind], ncc.domain.full_bases, operand.domain.full_bases, separability, **kw) for ind in np.ndindex(*tshape)]


class IntervalBasis(Basis):

    dim = 1
    subaxis_dependence = [True]

    def __init__(self, coord, size, bounds, dealias):
        self.coord = coord
        coord.check_bounds(bounds)
        self.coordsys = coord
        self.size = size
        self.shape = (size,)
        self.bounds = bounds
        self.volume = bounds[1] - bounds[0]
        if isinstance(dealias, tuple):
            self.dealias = dealias
        else:
            self.dealias = (dealias,)
        self.COV = AffineCOV(self.native_bounds, bounds)
        super().__init__(coord)

    def matrix_dependence(self, matrix_coupling):
        return matrix_coupling

    def global_grids(self, dist, scales):
        """Global grids."""
        return (self.global_grid(dist, scales[0]),)

    def global_grid(self, dist, scale):
        """Global grid."""
        native_grid = self._native_grid(scale)
        problem_grid = self.COV.problem_coord(native_grid)
        return reshape_vector(problem_grid, dim=dist.dim, axis=dist.get_basis_axis(self))

    def local_grids(self, dist, scales):
        """Local grids."""
        return (self.local_grid(dist, scales[0]),)

    def local_grid(self, dist, scale):
        """Local grid."""
        local_elements = dist.grid_layout.local_elements(self.domain(dist), scales=scale)
        native_grid = self._native_grid(scale)[local_elements[dist.get_basis_axis(self)]]
        problem_grid = self.COV.problem_coord(native_grid)
        return reshape_vector(problem_grid, dim=dist.dim, axis=dist.get_basis_axis(self))

    def global_grid_spacing(self, dist, scale):
        """Global grid spacings."""
        grid = self.global_grid(dist, scale=scale)
        return np.gradient(grid, axis=dist.first_axis(self), edge_order=2)

    def local_grid_spacing(self, dist, scale):
        """Local grids spacings."""
        global_spacing = self.global_grid_spacing(dist, scale=scale)
        local_elements = dist.grid_layout.local_elements(self.domain(dist), scales=scale)
        local_spacing_flat = np.ravel(global_spacing)[local_elements[dist.get_basis_axis(self)]]
        out = reshape_vector(local_spacing_flat, dim=dist.dim, axis=dist.get_basis_axis(self))
        return out

    def local_modes(self, dist):
        """Local grid."""
        local_elements = dist.coeff_layout.local_elements(self.domain(dist), scales=1)
        return reshape_vector(local_elements[dist.get_basis_axis(self)], dim=dist.dim, axis=dist.get_basis_axis(self))

    def _native_grid(self, scale):
        """Native flat global grid."""
        # Subclasses must implement
        raise NotImplementedError

    def global_shape(self, grid_space, scales):
        if grid_space[0]:
            return self.grid_shape(scales)
        else:
            return self.shape

    def chunk_shape(self, grid_space):
        if grid_space[0]:
            return (1,)
        else:
            return self.group_shape

    def forward_transform(self, field, axis, gdata, cdata):
        """Forward transform field data."""
        data_axis = len(field.tensorsig) + axis
        grid_size = gdata.shape[data_axis]
        plan = self.transform_plan(field.dist, grid_size)
        plan.forward(gdata, cdata, data_axis)

    def backward_transform(self, field, axis, cdata, gdata):
        """Backward transform field data."""
        data_axis = len(field.tensorsig) + axis
        grid_size = gdata.shape[data_axis]
        plan = self.transform_plan(field.dist, grid_size)
        plan.backward(cdata, gdata, data_axis)

    def transform_plan(self, dist, grid_size):
        # Subclasses must implement
        raise NotImplementedError


class Jacobi(IntervalBasis, metaclass=CachedClass):
    """Jacobi polynomial basis."""

    group_shape = (1,)
    native_bounds = (-1, 1)
    transforms = {}
    default_dct = "fftw_dct"
    default_library = "matrix"

    @classmethod
    def _preprocess_cache_args(cls, coord, size, bounds, a, b, a0, b0, dealias, library):
        """Preprocess arguments into canonical form for caching. Must accept and return __init__ arguments."""
        # coord: Coordinate
        if not isinstance(coord, Coordinate):
            raise ValueError("Jacobi coord must be Coordinate object.")
        # size: positive int
        size = int(size)
        if size <= 0:
            raise ValueError("Jacobi size must be positive.")
        # bounds: length-2 tuple
        bounds = tuple(bounds)
        if len(bounds) != 2:
            raise ValueError("Jacobi bounds must have length 2.")
        # a: float
        a = float(a)
        # b: float
        b = float(b)
        # a0: float, pick default
        if a0 is None:
            a0 = a
        a0 = float(a0)
        # b0: float, pick default
        if b0 is None:
            b0 = b
        b0 = float(b0)
        # dealias: length-1 tuple
        if isinstance(dealias, (int, float)):
            dealias = (dealias,)
        else:
            dealias = tuple(dealias)
        if len(dealias) != 1:
            raise ValueError("Jacobi dealias must have length 1.")
        # library: pick default based on (a0, b0)
        if library is None:
            if a0 == b0 == -1/2:
                library = cls.default_dct
            else:
                library = cls.default_library
        return (coord, size, bounds, a, b, a0, b0, dealias, library)

    def __init__(self, coord, size, bounds, a, b, a0=None, b0=None, dealias=(1,), library=None):
        super().__init__(coord, size, bounds, dealias)
        # Save arguments without modification for caching
        self.coord = coord
        self.size = size
        self.bounds = bounds
        self.a = a
        self.b = b
        self.a0 = a0
        self.b0 = b0
        self.dealias = dealias
        self.library = library
        # Other attributes
        self.grid_params = (coord, bounds, a0, b0, dealias, library)
        self.constant_mode_value = 1 / np.sqrt(jacobi.mass(self.a, self.b))

    def _native_grid(self, scale):
        """Native flat global grid."""
        N, = self.grid_shape((scale,))
        return jacobi.build_grid(N, a=self.a0, b=self.b0)

    @CachedMethod
    def transform_plan(self, dist, grid_size):
        """Build transform plan."""
        return self.transforms[self.library](grid_size, self.size, self.a, self.b, self.a0, self.b0)

    # def weights(self, scales):
    #     """Gauss-Jacobi weights."""
    #     N = self.grid_shape(scales)[0]
    #     return jacobi.build_weights(N, a=self.a, b=self.b)

    def __str__(self):
        return f"Jacobi({self.coord.name}, {self.size}, a0={self.a0}, b0={self.b0}, a={self.a}, b={self.b}, dealias={self.dealias[0]})"

    def __add__(self, other):
        if other is None or other is self:
            return self
        if isinstance(other, Jacobi):
            if self.grid_params == other.grid_params:
                # Everything matches except size, a, b
                size = max(self.size, other.size)
                a = max(self.a, other.a)
                b = max(self.b, other.b)
                return self.clone_with(size=size, a=a, b=b)
        return NotImplemented

    def __mul__(self, other):
        if other is None or other is self:
            return self
        if isinstance(other, Jacobi):
            if self.grid_params == other.grid_params:
                # Take maximum size
                size = max(self.size, other.size)
                # Take grid (a0, b0) for minimal conversions
                a = self.a0
                b = self.b0
                return self.clone_with(size=size, a=a, b=b)
        if isinstance(other, SphereBasis):
            return other.__mul__(self)
        return NotImplemented

    def __rmatmul__(self, other):
        # NCC (other) * operand (self)
        if other is None or other is self:
            return self
        if isinstance(other, Jacobi):
            if self.grid_params == other.grid_params:
                # Take maximum size
                size = max(self.size, other.size)
                # Take operand (a, b) for minimal conversions
                a = self.a
                b = self.b
                return self.clone_with(size=size, a=a, b=b)
        if isinstance(other, SphereBasis):
            return other.__mul__(self)
        return NotImplemented

    # def include_mode(self, mode):
    #     return (0 <= mode < self.space.coeff_size)

    def elements_to_groups(self, grid_space, elements):
        # No permutations
        return elements

    def valid_elements(self, tensorsig, grid_space, elements):
        # No invalid modes
        vshape = tuple(cs.dim for cs in tensorsig) + elements[0].shape
        return np.ones(shape=vshape, dtype=bool)

    def Jacobi_matrix(self, size):
        if size is None:
            size = self.size
        return dedalus_sphere.jacobi.operator('Z')(size, self.a, self.b).square

    # def ncc_matrix(self, arg_basis, coeffs, cutoff=1e-6):
    #     """Build NCC matrix via Clenshaw algorithm."""
    #     if arg_basis is None:
    #         return super().ncc_matrix(arg_basis, coeffs)
    #     # Kronecker Clenshaw on argument Jacobi matrix
    #     elif isinstance(arg_basis, Jacobi):
    #         N = self.size
    #         J = jacobi.jacobi_matrix(N, arg_basis.a, arg_basis.b)
    #         A, B = clenshaw.jacobi_recursion(N, self.a, self.b, J)
    #         f0 = self.const * sparse.identity(N)
    #         total = clenshaw.kronecker_clenshaw(coeffs, A, B, f0, cutoff=cutoff)
    #         # Conversion matrix
    #         input_basis = arg_basis
    #         output_basis = (self * arg_basis)
    #         conversion = ConvertJacobi._subspace_matrix(input_basis, output_basis)
    #         # Kronecker with identity for matrix coefficients
    #         coeff_size = total.shape[0] // conversion.shape[0]
    #         if coeff_size > 1:
    #             conversion = sparse.kron(conversion, sparse.identity(coeff_size))
    #         return (conversion @ total)
    #     else:
    #         raise ValueError("Jacobi ncc_matrix not implemented for basis type: %s" %type(arg_basis))

    @CachedMethod
    def product_matrix(self, arg_basis, out_basis, i):
        if arg_basis is None:
            return super().product_matrix(arg_basis, out_basis, i)
        coeffs = np.zeros(i+1)
        coeffs[i] = 1
        return self._last_axis_component_ncc_matrix(None, self, arg_basis, out_basis, coeffs, None, None, None, None, None, None, 0)

    @classmethod
    def _last_axis_component_ncc_matrix(cls, subproblem, ncc_basis, arg_basis, out_basis, coeffs, ncc_comp, arg_comp, out_comp, ncc_tensorsig, arg_tensorsig, out_tensorsig, cutoff):
        if arg_basis is None:
            return super().ncc_matrix(ncc_basis, arg_basis, out_basis, coeffs.ravel(), cutoff=cutoff)
        # Jacobi parameters
        a_ncc = ncc_basis.a
        b_ncc = ncc_basis.b
        N = arg_basis.size
        da = int(np.round(out_basis.a - arg_basis.a))
        db = int(np.round(out_basis.b - arg_basis.b))
        # Pad for dealiasing with conversion
        Nmat = 3*((N+1)//2) + min((N+1)//2, (da+db+1)//2)
        J = arg_basis.Jacobi_matrix(size=Nmat)
        A, B = clenshaw.jacobi_recursion(Nmat, a_ncc, b_ncc, J)
        f0 = dedalus_sphere.jacobi.polynomials(1, a_ncc, b_ncc, 1)[0] * sparse.identity(Nmat)
        matrix = clenshaw.matrix_clenshaw(coeffs.ravel(), A, B, f0, cutoff=cutoff)
        convert = jacobi.conversion_matrix(Nmat, arg_basis.a, arg_basis.b, out_basis.a, out_basis.b)
        matrix = convert @ matrix
        return matrix[:N, :N]

    def derivative_basis(self, order=1):
        a = self.a + order
        b = self.b + order
        return self.clone_with(a=a, b=b)


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


Chebyshev = ChebyshevT


class ConvertJacobi(operators.Convert, operators.SpectralOperator1D):
    """Jacobi polynomial conversion."""

    input_basis_type = Jacobi
    output_basis_type = Jacobi
    subaxis_dependence = [True]
    subaxis_coupling = [True]

    @staticmethod
    @CachedMethod
    def _full_matrix(input_basis, output_basis):
        N = input_basis.size
        a0, b0 = input_basis.a, input_basis.b
        a1, b1 = output_basis.a, output_basis.b
        matrix = jacobi.conversion_matrix(N, a0, b0, a1, b1)
        return matrix.tocsr()


class ConvertConstantJacobi(operators.ConvertConstant, operators.SpectralOperator1D):
    """Upcast constants to Jacobi."""

    output_basis_type = Jacobi
    subaxis_dependence = [True]
    subaxis_coupling = [False]

    @staticmethod
    @CachedMethod
    def _group_matrix(group, input_basis, output_basis):
        n = group
        if n == 0:
            unit_amplitude = 1 / output_basis.constant_mode_value
            return np.array([[unit_amplitude]])
        else:
            # Constructor should only loop over group 0.
            raise ValueError("This should never happen.")


class DifferentiateJacobi(operators.Differentiate, operators.SpectralOperator1D):
    """Jacobi polynomial differentiation."""

    input_basis_type = Jacobi
    subaxis_dependence = [True]
    subaxis_coupling = [True]

    @staticmethod
    def _output_basis(input_basis):
        return input_basis.derivative_basis(order=1)

    @staticmethod
    @CachedMethod
    def _full_matrix(input_basis, output_basis):
        N = input_basis.size
        a, b = input_basis.a, input_basis.b
        matrix = jacobi.differentiation_matrix(N, a, b) / input_basis.COV.stretch
        return matrix.tocsr()


class InterpolateJacobi(operators.Interpolate, operators.SpectralOperator1D):
    """Jacobi polynomial interpolation."""

    input_basis_type = Jacobi
    basis_subaxis = 0
    subaxis_dependence = [True]
    subaxis_coupling = [True]

    @staticmethod
    def _output_basis(input_basis, position):
        return None

    @staticmethod
    @CachedMethod
    def _full_matrix(input_basis, output_basis, position):
        # Build native interpolation vector
        N = input_basis.size
        a, b = input_basis.a, input_basis.b
        x = input_basis.COV.native_coord(position)
        interp_vector = jacobi.build_polynomials(N, a, b, x)
        # Return with shape (1, N)
        return interp_vector[None, :]


class IntegrateJacobi(operators.Integrate, operators.SpectralOperator1D):
    """Jacobi polynomial integration."""

    input_coord_type = Coordinate
    input_basis_type = Jacobi
    subaxis_dependence = [True]
    subaxis_coupling = [True]

    @staticmethod
    def _output_basis(input_basis):
        return None

    @staticmethod
    def _full_matrix(input_basis, output_basis):
        # Build native integration vector
        N = input_basis.size
        a, b = input_basis.a, input_basis.b
        integ_vector = jacobi.integration_vector(N, a, b)
        # Rescale and return with shape (1, N)
        return integ_vector[None, :] * input_basis.COV.stretch


class AverageJacobi(operators.Average, operators.SpectralOperator1D):
    """Jacobi polynomial averaging."""

    input_coord_type = Coordinate
    input_basis_type = Jacobi
    subaxis_dependence = [True]
    subaxis_coupling = [True]

    @staticmethod
    def _output_basis(input_basis):
        return None

    @staticmethod
    def _full_matrix(input_basis, output_basis):
        # Build native integration vector
        N = input_basis.size
        a, b = input_basis.a, input_basis.b
        integ_vector = jacobi.integration_vector(N, a, b)
        ave_vector = integ_vector / 2
        # Rescale and return with shape (1, N)
        return ave_vector[None, :]


class LiftJacobi(operators.Lift, operators.Copy):
    """Jacobi polynomial lift."""

    input_basis_type = type(None)
    output_basis_type = Jacobi
    subaxis_dependence = [True]
    subaxis_coupling = [True]

    @classmethod
    def _check_args(cls, operand, output_basis, n, out=None):
        if super()._check_args(operand, output_basis, n):
            P = cls.build_polynomial(operand.dist, output_basis, n)
            raise SkipDispatchException(output=P*operand)
        else:
            return False

    @staticmethod
    @CachedMethod
    def build_polynomial(dist, basis, n):
        if n < 0:
            n += basis.size
        P = dist.Field(bases=basis)
        axis = dist.get_basis_axis(basis)
        P['c'][axslice(axis, n, n+1)] = 1
        return P


class FourierBase(IntervalBasis):
    """Base class for RealFourier and ComplexFourier."""

    native_bounds = (0, 2*np.pi)
    default_library = "fftw"

    @classmethod
    def _preprocess_cache_args(cls, coord, size, bounds, dealias, library):
        """Preprocess arguments into canonical form for caching. Must accept and return __init__ arguments."""
        # coord: Coordinate
        if not isinstance(coord, Coordinate):
            raise ValueError("Fourier coord must be Coordinate object.")
        # size: positive int
        size = int(size)
        if size <= 0:
            raise ValueError("Fourier size must be positive.")
        # bounds: length-2 tuple
        bounds = tuple(bounds)
        if len(bounds) != 2:
            raise ValueError("Fourier bounds must have length 2.")
        # dealias: length-1 tuple
        if isinstance(dealias, (int, float)):
            dealias = (dealias,)
        else:
            dealias = tuple(dealias)
        if len(dealias) != 1:
            raise ValueError("Fourier dealias must have length 1.")
        # library: pick default based on (a0, b0)
        if library is None:
            library = cls.default_library
        return (coord, size, bounds, dealias, library)

    def __init__(self, coord, size, bounds, dealias=(1,), library=None):
        super().__init__(coord, size, bounds, dealias)
        # Save arguments without modification for caching
        self.coord = coord
        self.size = size
        self.bounds = bounds
        self.dealias = dealias
        self.library = library
        # Other attributes
        self.constant_mode_value = 1
        # No permutations by default
        self.forward_coeff_permutation = None
        self.backward_coeff_permutation = None

    def __add__(self, other):
        if other is None or other is self:
            return self
        # TODO: support different sizes
        return NotImplemented

    def __mul__(self, other):
        if other is None or other is self:
            return self
        # TODO: support different sizes
        return NotImplemented

    def __rmatmul__(self, other):
        if other is None or other is self:
            return self
        # TODO: support different sizes
        return NotImplemented

    def __pow__(self, other):
        return self

    def elements_to_groups(self, grid_space, elements):
        if grid_space[0]:
            groups = elements
        else:
            # Use native wavenumbers for readability
            groups = self.native_wavenumbers[elements]
        return groups

    @CachedAttribute
    def _wavenumbers(self):
        return self._native_wavenumbers / self.COV.stretch

    @property
    def native_wavenumbers(self):
        if self.forward_coeff_permutation is None:
            return self._native_wavenumbers
        else:
            return self._native_wavenumbers[self.forward_coeff_permutation]

    @property
    def wavenumbers(self):
        if self.forward_coeff_permutation is None:
            return self._wavenumbers
        else:
            return self._wavenumbers[self.forward_coeff_permutation]

    def _native_grid(self, scale):
        """Native flat global grid."""
        N, = self.grid_shape((scale,))
        return (2 * np.pi / N) * np.arange(N)

    @CachedMethod
    def transform_plan(self, dist, grid_size):
        """Build transform plan."""
        # Shortcut trivial transforms
        if grid_size == 1 or self.size == 1:
            return self.transforms['matrix'](grid_size, self.size)
        else:
            return self.transforms[self.library](grid_size, self.size)

    def forward_transform(self, field, axis, gdata, cdata):
        # Transform
        super().forward_transform(field, axis, gdata, cdata)
        # Permute coefficients
        if self.forward_coeff_permutation is not None:
            permute_axis(cdata, axis+len(field.tensorsig), self.forward_coeff_permutation, out=cdata)

    def backward_transform(self, field, axis, cdata, gdata):
        # Permute coefficients
        if self.backward_coeff_permutation is not None:
            permute_axis(cdata, axis+len(field.tensorsig), self.backward_coeff_permutation, out=cdata)
        # Transform
        super().backward_transform(field, axis, cdata, gdata)


def Fourier(*args, dtype=None, **kw):
    """Factory function dispatching to RealFourier and ComplexFourier based on provided dtype."""
    if dtype is None:
        raise ValueError("dtype must be specified")
    elif dtype == np.float64:
        return RealFourier(*args, **kw)
    elif dtype == np.complex128:
        return ComplexFourier(*args, **kw)
    else:
        raise ValueError(f"Unrecognized dtype: {dtype}")


class ComplexFourier(FourierBase, metaclass=CachedClass):
    """
    Fourier complex exponential basis.
    Modes: [exp(0j*x), exp(1j*x), exp(2j*x), ...]
    """

    group_shape = (1,)
    transforms = {}

    @CachedAttribute
    def _native_wavenumbers(self):
        kmax = self.size // 2
        if self.size % 2:
            # Odd size
            # [0, 1, ..., kmax, -kmax, ..., -1]
            return np.concatenate((np.arange(0, kmax+1), np.arange(-kmax, 0)))
        else:
            # Even size, include Nyquist mode
            # [0, 1, ..., kmax, 1-kmax, ..., -1]
            return np.concatenate((np.arange(0, kmax+1), np.arange(1-kmax, 0)))

    def valid_elements(self, tensorsig, grid_space, elements):
        vshape = tuple(cs.dim for cs in tensorsig) + elements[0].shape
        valid = np.ones(shape=vshape, dtype=bool)
        # # Drop Nyquist mode
        # if not grid_space[0]:
        #     groups = self.elements_to_groups(grid_space, elements[0])
        #     valid_groups = np.abs(groups) <= self.kmax
        #     valid *= valid_groups
        return valid

    @CachedMethod
    def product_matrix(self, arg_basis, out_basis, i):
        # Directly compare wavenumber arrays to handle any permutations
        # First turn them into integer mode numbers to avoid floating point issues
        k0 = self.wavenumbers[1]
        k_ncc = np.round(self.wavenumbers[i]/k0).astype(int)
        k_out = np.round(out_basis.wavenumbers/k0).astype(int)
        if arg_basis is None:
            k_arg = np.array([0])
        else:
            k_arg = np.round(arg_basis.wavenumbers/k0).astype(int)
        k_prod = k_arg + k_ncc
        _, rows, cols = np.intersect1d(k_out, k_prod, assume_unique=True, return_indices=True)
        data = np.ones_like(rows)
        return sparse.coo_matrix((data, (rows, cols)), shape=(k_out.size, k_arg.size)).tocsr()


class ConvertConstantComplexFourier(operators.ConvertConstant, operators.SpectralOperator1D):
    """Upcast constants to ComplexFourier."""

    output_basis_type = ComplexFourier
    subaxis_dependence = [True]
    subaxis_coupling = [False]

    @staticmethod
    def _group_matrix(group, input_basis, output_basis):
        # Rescale group (native wavenumber) to get physical wavenumber
        k = group / output_basis.COV.stretch
        # 1 = exp(1j*0*x)
        if k == 0:
            unit_amplitude = 1 / output_basis.constant_mode_value
            return np.array([[unit_amplitude]])
        else:
            return np.zeros(shape=(1, 0))


class DifferentiateComplexFourier(operators.Differentiate, operators.SpectralOperator1D):
    """ComplexFourier differentiation."""

    input_basis_type = ComplexFourier
    subaxis_dependence = [True]
    subaxis_coupling = [False]

    @staticmethod
    def _output_basis(input_basis):
        return input_basis

    @staticmethod
    def _group_matrix(group, input_basis, output_basis):
        # Rescale group (native wavenumber) to get physical wavenumber
        k = group / input_basis.COV.stretch
        # dx exp(1j*k*x) = 1j * k * exp(1j*k*x)
        return np.array([[1j*k]])


class InterpolateComplexFourier(operators.Interpolate, operators.SpectralOperator1D):
    """ComplexFourier interpolation."""

    input_basis_type = ComplexFourier
    basis_subaxis = 0
    subaxis_dependence = [True]
    subaxis_coupling = [True]

    @staticmethod
    def _output_basis(input_basis, position):
        return None

    @staticmethod
    def _full_matrix(input_basis, output_basis, position):
        # Build native interpolation vector
        x = input_basis.COV.native_coord(position)
        k = input_basis.native_wavenumbers
        interp_vector = np.exp(1j * k * x)
        # Return with shape (1, N)
        return interp_vector[None, :]


class IntegrateComplexFourier(operators.Integrate, operators.SpectralOperator1D):
    """ComplexFourier integration."""

    input_coord_type = Coordinate
    input_basis_type = ComplexFourier
    subaxis_dependence = [True]
    subaxis_coupling = [False]

    @staticmethod
    def _output_basis(input_basis):
        return None

    @staticmethod
    def _group_matrix(group, input_basis, output_basis):
        # Rescale group (native wavenumber) to get physical wavenumber
        k = group / input_basis.COV.stretch
        # integ exp(1j*k*x) = L * δ(k, 0)
        if k == 0:
            L = input_basis.COV.problem_length
            return np.array([[L]])
        else:
            # Constructor should only loop over group 0.
            raise ValueError("This should never happen.")


class AverageComplexFourier(operators.Average, operators.SpectralOperator1D):
    """ComplexFourier averaging."""

    input_coord_type = Coordinate
    input_basis_type = ComplexFourier
    subaxis_dependence = [True]
    subaxis_coupling = [False]

    @staticmethod
    def _output_basis(input_basis):
        return None

    @staticmethod
    def _group_matrix(group, input_basis, output_basis):
        # Rescale group (native wavenumber) to get physical wavenumber
        k = group / input_basis.COV.stretch
        # integ exp(1j*k*x) / L = δ(k, 0)
        if k == 0:
            return np.array([[1]])
        else:
            # Constructor should only loop over group 0.
            raise ValueError("This should never happen.")


class RealFourier(FourierBase, metaclass=CachedClass):
    """
    Fourier real sine/cosine basis.
    Modes: [cos(0*x), -sin(0*x), cos(1*x), -sin(1*x), ...]
    """

    group_shape = (2,)
    transforms = {}

    @CachedAttribute
    def _native_wavenumbers(self):
        # Excludes Nyquist mode
        kmax = (self.size - 1) // 2
        return np.repeat(np.arange(0, kmax+1), 2)

    def valid_elements(self, tensorsig, grid_space, elements):
        vshape = tuple(cs.dim for cs in tensorsig) + elements[0].shape
        valid = np.ones(shape=vshape, dtype=bool)
        if not grid_space[0]:
            # Drop msin part of k=0 for all cartesian components and spin scalars
            if not isinstance(self.coord, AzimuthalCoordinate) or not tensorsig:
                # Drop msin part of k=0
                groups = self.elements_to_groups(grid_space, elements)
                allcomps = tuple(slice(None) for cs in tensorsig)
                selection = (groups[0] == 0) * (elements[0] % 2 == 1)
                valid[allcomps + (selection,)] = False
        return valid

    @CachedMethod
    def product_matrix(self, arg_basis, out_basis, i):
        # Directly compare wavenumber arrays to handle any permutations
        # First turn them into integer mode numbers to avoid floating point issues
        k0 = self.wavenumbers[2]
        k_ncc = np.round(self.wavenumbers[i]/k0).astype(int)
        k_out = np.round(out_basis.wavenumbers/k0).astype(int)
        if arg_basis is None:
            k_arg = np.array([0])
        else:
            k_arg = np.round(arg_basis.wavenumbers/k0).astype(int)
        # Skip multiplication by constant
        if k_ncc == 0:
            if i % 2 == 0:
                return sparse.eye(k_out.size, k_arg.size, format='csr')
            else:
                return sparse.csr_matrix((k_out.size, k_arg.size))
        # Find wavenumber couplings (this is crazy)
        _, rows_p, cols_p = np.intersect1d(k_out[::2], k_ncc + k_arg[::2], assume_unique=True, return_indices=True)
        _, rows_m, cols_m = np.intersect1d(k_out[::2], k_ncc - k_arg[::2], assume_unique=True, return_indices=True)
        _, rows_mn, cols_mn = np.intersect1d(k_out[2::2], k_arg[::2] - k_ncc, assume_unique=True, return_indices=True)
        rows_mn += 1
        ones_p = np.ones_like(rows_p)
        ones_m = np.ones_like(rows_m)
        ones_mn = np.ones_like(rows_mn)
        if i % 2 == 0:
            # 2 cos(mx) cos(nx) = cos((m+n)x) + cos((m-n)x)
            rows = [2*rows_p, 2*rows_m, 2*rows_mn]
            cols = [2*cols_p, 2*cols_m, 2*cols_mn]
            data = [ones_p/2, ones_m/2, ones_mn/2]
            # 2 cos(mx) msin(nx) = msin((m+n)x) - msin((m-n)x)
            rows += [2*rows_p+1, 2*rows_m+1, 2*rows_mn+1]
            cols += [2*cols_p+1, 2*cols_m+1, 2*cols_mn+1]
            data += [ones_p/2, -ones_m/2, ones_mn/2]
        else:
            # 2 msin(mx) cos(nx) = msin((m+n)x) + msin((m-n)x)
            rows = [2*rows_p+1, 2*rows_m+1, 2*rows_mn+1]
            cols = [2*cols_p, 2*cols_m, 2*cols_mn]
            data = [ones_p/2, ones_m/2, -ones_mn/2]
            # 2 msin(mx) msin(nx) = - cos((m+n)x) + cos((m-n)x)
            rows += [2*rows_p, 2*rows_m, 2*rows_mn]
            cols += [2*cols_p+1, 2*cols_m+1, 2*cols_mn+1]
            data += [-ones_p/2, ones_m/2, ones_mn/2]
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        data = np.concatenate(data)
        matrix = sparse.coo_matrix((data, (rows, cols)), shape=(k_out.size, k_arg.size)).tocsr()
        return matrix


class ConvertConstantRealFourier(operators.ConvertConstant, operators.SpectralOperator1D):
    """Upcast constants to RealFourier."""

    output_basis_type = RealFourier
    subaxis_dependence = [True]
    subaxis_coupling = [False]

    @staticmethod
    def _group_matrix(group, input_basis, output_basis):
        # Rescale group (native wavenumber) to get physical wavenumber
        k = group / output_basis.COV.stretch
        # 1 = cos(0*x)
        if k == 0:
            unit_amplitude = 1 / output_basis.constant_mode_value
            return np.array([[unit_amplitude],
                             [0]])
        else:
            return np.zeros(shape=(2, 0))


class DifferentiateRealFourier(operators.Differentiate, operators.SpectralOperator1D):
    """RealFourier differentiation."""

    input_basis_type = RealFourier
    subaxis_dependence = [True]
    subaxis_coupling = [False]

    @staticmethod
    def _output_basis(input_basis):
        return input_basis

    @staticmethod
    def _group_matrix(group, input_basis, output_basis):
        # Rescale group (native wavenumber) to get physical wavenumber
        k = group / input_basis.COV.stretch
        # dx  cos(k*x) = k * -sin(k*x)
        # dx -sin(k*x) = -k * cos(k*x)
        return np.array([[0, -k],
                         [k,  0]])


class InterpolateRealFourier(operators.Interpolate, operators.SpectralOperator1D):
    """RealFourier interpolation."""

    input_basis_type = RealFourier
    basis_subaxis = 0
    subaxis_dependence = [True]
    subaxis_coupling = [True]

    @staticmethod
    def _output_basis(input_basis, position):
        return None

    @staticmethod
    def _full_matrix(input_basis, output_basis, position):
        # Build native interpolation vector
        # Interleaved cos(k*x), -sin(k*x)
        x = input_basis.COV.native_coord(position)
        k = input_basis.native_wavenumbers
        interp_vector = np.zeros(k.size)
        interp_vector[0::2] = np.cos(k[0::2] * x)
        interp_vector[1::2] = -np.sin(k[1::2] * x)
        # Return with shape (1, N)
        return interp_vector[None, :]


class IntegrateRealFourier(operators.Integrate, operators.SpectralOperator1D):
    """RealFourier integration."""

    input_coord_type = Coordinate
    input_basis_type = RealFourier
    subaxis_dependence = [True]
    subaxis_coupling = [False]

    @staticmethod
    def _output_basis(input_basis):
        return None

    @staticmethod
    def _group_matrix(group, input_basis, output_basis):
        # Rescale group (native wavenumber) to get physical wavenumber
        k = group / input_basis.COV.stretch
        # integ  cos(k*x) = L * δ(k, 0)
        # integ -sin(k*x) = 0
        if k == 0:
            L = input_basis.COV.problem_length
            return np.array([[L, 0]])
        else:
            # Constructor should only loop over group 0.
            raise ValueError("This should never happen.")


class AverageRealFourier(operators.Average, operators.SpectralOperator1D):
    """RealFourier averaging."""

    input_coord_type = Coordinate
    input_basis_type = RealFourier
    subaxis_dependence = [True]
    subaxis_coupling = [False]

    @staticmethod
    def _output_basis(input_basis):
        return None

    @staticmethod
    def _group_matrix(group, input_basis, output_basis):
        # Rescale group (native wavenumber) to get physical wavenumber
        k = group / input_basis.COV.stretch
        # integ  cos(k*x) / L = δ(k, 0)
        # integ -sin(k*x) / L = 0
        if k == 0:
            return np.array([[1, 0]])
        else:
            # Constructor should only loop over group 0.
            raise ValueError("This should never happen.")


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
        subaxis = axis - field.dist.get_basis_axis(self)
        return self.forward_transforms[subaxis](field, axis, gdata, cdata)

    def backward_transform(self, field, axis, cdata, gdata):
        subaxis = axis - field.dist.get_basis_axis(self)
        return self.backward_transforms[subaxis](field, axis, cdata, gdata)


def reduced_view_5(data, axis1, axis2):
    shape = data.shape
    N0 = prod(shape[:axis1])
    N1 = shape[axis1]
    N2 = prod(shape[axis1+1:axis2])
    N3 = shape[axis2]
    N4 = prod(shape[axis2+1:])
    return data.reshape((N0, N1, N2, N3, N4))


class SpinRecombinationBasis:

    @CachedMethod
    def spin_recombination_factors(self, tensorsig):
        """Build matrices for applying spin recombination to each tensor rank."""
        # Spin recombinations are separable over tensor index
        factors = []
        coord1 = self.coordsys.coords[1]
        for i, cs in enumerate(tensorsig):
            if coord1 in cs.coords:
                subaxis = cs.coords.index(coord1)
                factors.append(cs.forward_intertwiner(subaxis, order=1, group=None))  # no group dependence
            else:
                factors.append(None)
        return factors

    def spin_recombination_matrix(self, tensorsig, n_upstream):
        # Combine factors, replacing None placeholders
        factors = self.spin_recombination_factors(tensorsig)
        for i, cs in enumerate(tensorsig):
            if factors[i] is None:
                factors[i] = np.identity(cs.dim)
        # Add space for upstream groups
        if n_upstream > 1:
            # Create new list to avoid modifying cached list
            factors = factors + [np.identity(n_upstream)]
        matrix = kron(*factors)
        # Expand for cos and msin parts
        if self.dtype == np.float64:
            #matrix = np.array([[matrix.real,-matrix.imag],[matrix.imag,matrix.real]])
            matrix = (np.kron(matrix.real,np.array([[1,0],[0,1]]))
                      + np.kron(matrix.imag,np.array([[0,-1],[1,0]])))
        return matrix

    def forward_spin_recombination(self, tensorsig, colat_axis, gdata, out):
        """Apply component-to-spin recombination."""
        # We assume gdata and out are different data buffers
        # and that we can safely overwrite gdata
        if not tensorsig:
            np.copyto(out, gdata)
        else:
            azimuth_axis = colat_axis - 1
            factors = self.spin_recombination_factors(tensorsig)
            if gdata.dtype == np.complex128:
                # HACK: just copying the data so we can apply_matrix repeatedly
                np.copyto(out, gdata)
                for i, Ui in enumerate(factors):
                    if Ui is not None:
                        # Directly apply U
                        apply_matrix(Ui, out, axis=i, out=out)
            elif gdata.dtype == np.float64:
                # Recombinations alternate between using gdata/out as input/output
                # For an even number of transforms, we need a final copy
                num_recombinations = 0
                coord0 = self.coordsys.coords[0]
                for i, Ui in enumerate(factors):
                    if Ui is not None:
                        subaxis = tensorsig[i].coords.index(coord0)
                        if num_recombinations % 2 == 0:
                            input_view = reduced_view_5(gdata, i, azimuth_axis+len(tensorsig))
                            output_view = reduced_view_5(out, i, azimuth_axis+len(tensorsig))
                        else:
                            input_view = reduced_view_5(out, i, azimuth_axis+len(tensorsig))
                            output_view = reduced_view_5(gdata, i, azimuth_axis+len(tensorsig))
                        spin_recombination.recombine_forward(subaxis, input_view, output_view)
                        num_recombinations += 1
                if num_recombinations % 2 == 0:
                    np.copyto(out, gdata)

    def backward_spin_recombination(self, tensorsig, colat_axis, gdata, out):
        """Apply spin-to-component recombination."""
        # We assume gdata and out are different data buffers
        # and that we can safely overwrite gdata
        if not tensorsig:
            np.copyto(out, gdata)
        else:
            azimuth_axis = colat_axis - 1
            factors = self.spin_recombination_factors(tensorsig)
            if gdata.dtype == np.complex128:
                # HACK: just copying the data so we can apply_matrix repeatedly
                np.copyto(out, gdata)
                for i, Ui in enumerate(factors):
                    if Ui is not None:
                        # Directly apply U
                        apply_matrix(Ui.T.conj(), out, axis=i, out=out)
            elif gdata.dtype == np.float64:
                # Recombinations alternate between using gdata/out as input/output
                # For an even number of transforms, we need a final copy
                num_recombinations = 0
                coord0 = self.coordsys.coords[0]
                for i, Ui in enumerate(factors):
                    if Ui is not None:
                        subaxis = tensorsig[i].coords.index(coord0)
                        if num_recombinations % 2 == 0:
                            input_view = reduced_view_5(gdata, i, azimuth_axis+len(tensorsig))
                            output_view = reduced_view_5(out, i, azimuth_axis+len(tensorsig))
                        else:
                            input_view = reduced_view_5(out, i, azimuth_axis+len(tensorsig))
                            output_view = reduced_view_5(gdata, i, azimuth_axis+len(tensorsig))
                        spin_recombination.recombine_backward(subaxis, input_view, output_view)
                        num_recombinations += 1
                if num_recombinations % 2 == 0:
                    np.copyto(out, gdata)


# These are common for S2 and D2
class SpinBasis(MultidimensionalBasis, SpinRecombinationBasis):

    def __init__(self, coordsys, shape, dtype, dealias, azimuth_library=None):
        self.coordsys = coordsys
        self.shape = shape
        self.dtype = dtype
        if np.isscalar(dealias):
            self.dealias = (dealias, dealias)
        elif len(dealias) != 2:
            raise ValueError("dealias must either be a number or a tuple of two numbers")
        else:
            self.dealias = dealias
        self.azimuth_library = azimuth_library
        self.mmax = (shape[0] - 1) // 2
        if dtype == np.complex128:
            self.azimuth_basis = ComplexFourier(coordsys.coords[0], shape[0], bounds=(0, 2*np.pi), library=azimuth_library, dealias=self.dealias[0])
        elif dtype == np.float64:
            self.azimuth_basis = RealFourier(coordsys.coords[0], shape[0], bounds=(0, 2*np.pi), library=azimuth_library, dealias=self.dealias[0])
        else:
            raise NotImplementedError()
        self.global_grid_azimuth = self.azimuth_basis.global_grid
        self.local_grid_azimuth = self.azimuth_basis.local_grid
        super().__init__(coordsys)

    @CachedAttribute
    def constant(self):
        return (self.mmax==0, False)

    def local_elements(self):
        raise NotImplementedError()
        # CL = self.dist.coeff_layout
        # LE = CL.local_elements(self.domain, scales=1)[self.axis:self.axis+self.dim]
        # LE[0] = self.local_m
        # return tuple(LE)

    @CachedMethod
    def spin_weights(self, tensorsig):
        S = np.zeros([cs.dim for cs in tensorsig], dtype=int)
        coord1 = self.coordsys.coords[1]
        spin_order = np.array(self.coordsys.spin_ordering)
        for i, cs in enumerate(tensorsig):
            if coord1 in cs.coords:
                start = cs.coords.index(coord1) - 1
                S[axslice(i, start, start+self.coordsys.dim)] += reshape_vector(spin_order[:cs.dim], dim=len(tensorsig), axis=i)
                # The slice in spin_order is a hack so that 2-vectors on S2 with 3D spherical coordinates work
                # Really it is more like cs.spin_ordering, but that fails for DirectProducts
        return S

    @CachedMethod
    def spintotal(self, tensorsig, spinindex):
        return int(self.spin_weights(tensorsig)[spinindex])


class PolarBasis(SpinBasis):

    dim = 2
    dims = ['azimuth', 'radius']

    def __init__(self, coordsys, shape, dtype, k=0, dealias=(1,1), azimuth_library=None):
        super().__init__(coordsys, shape, dtype, dealias, azimuth_library=azimuth_library)
        self.k = k
        self.Nmax = shape[1] - 1
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

        if self.mmax > 0 and self.Nmax > 0 and shape[0] % 2 != 0:
            raise ValueError("Don't use an odd phi resolution, please")
        if self.mmax > 0 and self.Nmax > 0 and self.dtype == np.float64 and shape[0] % 4 != 0:
            # TODO: probably we can get away with pairs rather than factors of 4...
            raise ValueError("Don't use a phi resolution that isn't divisible by 4, please")

        # ASSUMPTION: we assume we are dropping Nyquist mode, so shape=2 --> mmax = 0
        # m permutations for repacking triangular truncation
        if self.dtype == np.complex128:
            if self.mmax > 0:
                az_index = np.arange(shape[0])
                az_div, az_mod = divmod(az_index, 2)
                self.forward_m_perm = az_div + shape[0] // 2 * az_mod
                self.backward_m_perm = np.argsort(self.forward_m_perm)
            else:
                self.forward_m_perm = None
                self.backward_m_perm = None

            self.group_shape = (1, 1)
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
        self.azimuth_basis = self.S1_basis(radius=None)

    def matrix_dependence(self, matrix_coupling):
        matrix_dependence = matrix_coupling.copy()
        if matrix_coupling[1]:
            matrix_dependence[0] = True
        return matrix_dependence

    def valid_elements(self, tensorsig, grid_space, elements):
        if grid_space[1]:
            # grid-grid and coeff-grid
            # Same as Fourier validity before spin recombination
            return self.azimuth_basis.valid_elements(tensorsig, grid_space, elements)
        else:
            # coeff-coeff
            m, n = self.elements_to_groups(grid_space, elements)
            if tensorsig:
                rank = len(tensorsig)
                m = m[(None,) * rank]
                n = n[(None,) * rank]
            valid = n >= self._nmin(m)
            if not tensorsig:
                # Drop msin part of m = 0
                valid[(m == 0) * (elements[0] % 2 == 1)] = False
            return valid

    @CachedMethod
    def S1_basis(self, radius=1):
        if self.dtype == np.complex128:
            S1_basis = ComplexFourier(self.coordsys.coords[0], self.shape[0], bounds=(0, 2*np.pi), dealias=self.dealias[0], library=self.azimuth_library)
        elif self.dtype == np.float64:
            S1_basis = RealFourier(self.coordsys.coords[0], self.shape[0], bounds=(0, 2*np.pi), dealias=self.dealias[0], library=self.azimuth_library)
        else:
            raise NotImplementedError()
        S1_basis.radius = radius
        S1_basis.forward_coeff_permutation  = self.forward_m_perm
        S1_basis.backward_coeff_permutation = self.backward_m_perm
        return S1_basis

    def global_shape(self, grid_space, scales):
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
            if self.dtype == np.complex128:
                return self.shape
            elif self.dtype == np.float64:
                if Nphi > 1:
                    return self.shape
                else:
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

    def chunk_shape(self, grid_space):
        if grid_space[0]:
            # grid-grid space
            return (1, 1)
        elif grid_space[1]:
            # coeff-grid space
            # pairs of m don't have to be distributed together
            # since folding is not implemented
            if self.dtype == np.complex128:
                return (1, 1)
            elif self.dtype == np.float64:
                # for mmax == 0, the additional sin mode is added *after* the transpose
                # in radial transform, not here.
                if self.mmax == 0:
                    return (1, 1)
                else:
                    return (2, 1)
        else:
            # coeff-coeff space
            if self.dtype == np.complex128:
                return (1, 1)
            elif self.dtype == np.float64:
                return (2, 1)

    def elements_to_groups(self, grid_space, elements):
        s1_groups = self.azimuth_basis.elements_to_groups(grid_space, elements[:1])
        radial_groups = elements[1]
        groups = np.array([*s1_groups, radial_groups])
        if not grid_space[1]:
            # coeff-coeff space
            m, n = groups
            nmin = self._nmin(m)
            groups = np.ma.masked_array(groups)
            groups[:, n < nmin] = np.ma.masked
        return groups

    @CachedMethod
    def n_size(self, m):
        nmin = self._nmin(m)
        nmax = self.Nmax
        return nmax - nmin + 1

    @CachedMethod
    def n_slice(self, m):
        nmin = self._nmin(m)
        nmax = self.Nmax
        return slice(nmin, nmax+1)

    def __eq__(self, other):
        if isinstance(other, type(self)):
            if self.dtype == other.dtype:
                if self.coordsys == other.coordsys:
                    if self.grid_params == other.grid_params:
                        if self.k == other.k:
                            return True
        return False

    def __hash__(self):
        return id(self)

    @CachedMethod
    def m_maps(self, dist):
        """
        Tuple of (m, mg_slice, mc_slice, n_slice) for all local m's.
        """
        # Get colatitude transform object
        colat_transform = dist.get_transform_object(dist.first_axis(self)+1)
        colat_coeff_layout = colat_transform.layout0
        colat_grid_layout = colat_transform.layout1
        # Get groupsets
        domain = self.domain(dist)
        group_coupling = [True] * domain.dist.dim
        group_coupling[dist.first_axis(self)] = False
        group_coupling = tuple(group_coupling)
        groupsets = colat_grid_layout.local_groupsets(group_coupling, domain, scales=domain.dealias, broadcast=True)
        # Build m_maps from groupset slices
        m_maps = []
        for groupset in groupsets:
            m = groupset[dist.first_axis(self)]
            coeff_slices = colat_coeff_layout.local_groupset_slices(groupset, domain, scales=domain.dealias, broadcast=True)
            grid_slices = colat_grid_layout.local_groupset_slices(groupset, domain, scales=domain.dealias, broadcast=True)
            if len(coeff_slices) != 1 or len(grid_slices) != 1:
                raise ValueError("This should never happpen. Ask for help.")
            mg_slice = grid_slices[0][dist.first_axis(self)]
            mc_slice = coeff_slices[0][dist.first_axis(self)]
            n_slice = coeff_slices[0][dist.first_axis(self)+1]
            m_maps.append((m, mg_slice, mc_slice, n_slice))
        return tuple(m_maps)

    @CachedMethod
    def ell_reversed(self, dist):
        ell_reversed = {}
        for m, mg_slice, mc_slice, ell_slice in self.m_maps(dist):
            ell_reversed[m] = False
        return ell_reversed

    def global_grids(self, dist, scales):
        return (self.global_grid_azimuth(dist, scales[0]),
                self.global_grid_radius(dist, scales[1]))

    def global_grid_radius(self, dist, scale):
        r = self.radial_COV.problem_coord(self._native_radius_grid(scale))
        return reshape_vector(r, dim=dist.dim, axis=dist.get_basis_axis(self)+1)

    @CachedMethod
    def global_grid_spacing(self, dist, subaxis, scales):
        """Global grids spacings."""
        axis = dist.first_axis(self) + subaxis
        return np.gradient(self.global_grids(dist, scales=scales)[subaxis], axis=axis, edge_order=2)

    @CachedMethod
    def local_grid_spacing(self, dist, subaxis, scales):
        """Local grids spacings."""
        axis = dist.first_axis(self) + subaxis
        global_spacing = self.global_grid_spacing(dist, subaxis, scales=scales)
        local_elements = dist.grid_layout.local_elements(self.domain(dist), scales=scales[subaxis])[axis]
        return reshape_vector(np.ravel(global_spacing)[local_elements], dim=dist.dim, axis=axis)

    def local_grids(self, dist, scales):
        return (self.local_grid_azimuth(dist, scales[0]),
                self.local_grid_radius(dist, scales[1]))

    def forward_transform_azimuth_Mmax0(self, field, axis, gdata, cdata):
        # slice_axis = axis + len(field.tensorsig)
        # np.copyto(cdata[axslice(slice_axis, 0, 1)], gdata)
        np.copyto(cdata[axslice(axis+len(field.tensorsig), 0, 1)], gdata)

    def forward_transform_azimuth(self, field, axis, gdata, cdata):
        # Call Fourier transform
        self.azimuth_basis.forward_transform(field, axis, gdata, cdata)
        # Permute m for triangular truncation
        #permute_axis(cdata, axis+len(field.tensorsig), self.forward_m_perm, out=cdata)

    def backward_transform_azimuth_Mmax0(self, field, axis, cdata, gdata):
        # slice_axis = axis + len(field.tensorsig)
        # np.copyto(gdata, cdata[axslice(slice_axis, 0, 1)])
        np.copyto(gdata, cdata[axslice(axis+len(field.tensorsig), 0, 1)])

    def backward_transform_azimuth(self, field, axis, cdata, gdata):
        # Permute m back from triangular truncation
        #permute_axis(cdata, axis+len(field.tensorsig), self.backward_m_perm, out=cdata)
        # Call Fourier transform
        self.azimuth_basis.backward_transform(field, axis, cdata, gdata)

    @CachedMethod
    def radius_multiplication_matrix(self, m, spintotal, order, d):
        if order == 0:
            operator = dedalus_sphere.zernike.operator(2, 'Id', radius=self.radius)
        else:
            R = dedalus_sphere.zernike.operator(2, 'R', radius=1)
            if order < 0:
                operator = R(-1)**abs(order)
            else: # order > 0
                operator = R(+1)**abs(order)
        if d > 0:
            R = dedalus_sphere.zernike.operator(2, 'R', radius=1)
            R2 = R(-1) @ R(+1)
            operator = R2**(d//2) @ operator
        return operator(self.n_size(m), self.alpha + self.k, abs(m + spintotal)).square.astype(np.float64)

    def derivative_basis(self, order=1):
        k = self.k + order
        return self.clone_with(k=k)


class AnnulusBasis(PolarBasis, metaclass=CachedClass):

    transforms = {}
    subaxis_dependence = (False, True)

    @classmethod
    def _preprocess_cache_args(cls, coordsys, shape, dtype, radii, k, alpha, dealias, azimuth_library, radius_library):
        """Preprocess arguments into canonical form for caching. Must accept and return __init__ arguments."""
        # coordsys: PolarCoordinates
        if not isinstance(coordsys, PolarCoordinates):
            raise ValueError("Annulus coordsys must be PolarCoordinates.")
        # shape: length-2 tuple
        shape = tuple(shape)
        if len(shape) != 2:
            raise ValueError("Annulus shape must have length 2.")
        # radii: length-2 tuple of increasing positive numbers
        radii = tuple(radii)
        if len(radii) != 2:
            raise ValueError("Annulus radii must have length 2.")
        if min(radii) <= 0:
            raise ValueError("Annulus radii must be positive.")
        if radii[0] >= radii[1]:
            raise ValueError("Annulus radii must be in increasing order.")
        # k: int
        k = int(k)
        # alpha: length-2 tuple
        if isinstance(alpha, (int, float)):
            alpha = (alpha,) * 2
        else:
            alpha = tuple(alpha)
        if len(alpha) != 2:
            raise ValueError("Annulus alpha must have length 2.")
        # dealias: length-2 tuple
        if isinstance(dealias, (int, float)):
            dealias = (dealias,) * 2
        else:
            dealias = tuple(dealias)
        if len(dealias) != 2:
            raise ValueError("Annulus dealias must have length 2.")
        # azimuth_library: pick default
        if azimuth_library is None:
            azimuth_library = RealFourier.default_library
        # radius_library: pick default based on alpha
        if radius_library is None:
            if alpha[0] == alpha[1] == -1/2:
                radius_library = Jacobi.default_dct
            else:
                radius_library = Jacobi.default_library
        return (coordsys, shape, dtype, radii, k, alpha, dealias, azimuth_library, radius_library)

    def __init__(self, coordsys, shape, dtype, radii=(1,2), k=0, alpha=(-0.5,-0.5), dealias=(1,1), azimuth_library=None, radius_library=None):
        super().__init__(coordsys, shape, dtype, k=k, dealias=dealias, azimuth_library=azimuth_library)
        # Save arguments without modification for caching
        self.coordsys = coordsys
        self.shape = shape
        self.dtype = dtype
        self.radii = radii
        self.k = k
        self.alpha = alpha
        self.dealias = dealias
        self.azimuth_library = azimuth_library
        self.radius_library = radius_library
        # Other attributes
        self.volume = np.pi * (radii[1]**2 - radii[0]**2)
        self.dR = radii[1] - radii[0]
        self.rho = (radii[1] + radii[0])/self.dR
        self.grid_params = (coordsys, dtype, radii, alpha, dealias, azimuth_library, radius_library)
        self.inner_edge = self.S1_basis(radii[0])
        self.outer_edge = self.S1_basis(radii[1])

    @CachedAttribute
    def radial_basis(self):
        shape = (1, self.shape[1])
        return self.clone_with(shape=shape)

    @staticmethod
    def _nmin(m):
        return 0 * m  # To have same array shape as m

    def __add__(self, other):
        if other is None or other is self:
            return self
        if isinstance(other, AnnulusBasis):
            if self.grid_params == other.grid_params:
                # Everything matches except shape and k
                shape = tuple(np.maximum(self.shape, other.shape))
                k = max(self.k, other.k)  # minimizes conversions
                return self.clone_with(shape=shape, k=k)
        return NotImplemented

    def __mul__(self, other):
        if other is None:
            return self
        if isinstance(other, AnnulusBasis):
            if self.grid_params == other.grid_params:
                # Everything matches except shape and k
                shape = tuple(np.maximum(self.shape, other.shape))
                k = self.k + other.k
                return self.clone_with(shape=shape, k=k)
        return NotImplemented

    def __rmatmul__(self, other):
        # NCC (other) * operand (self)
        # Same as __mul__ since conversion only needs to be upwards in k
        return self.__mul__(other)

    def global_grid_radius(self, dist, scale):
        r = self._radius_grid(scale)
        return reshape_vector(r, dim=dist.dim, axis=dist.first_axis(self)+1)

    def local_grid_radius(self, dist, scale):
        r = self._radius_grid(scale)
        local_elements = dist.grid_layout.local_elements(self.domain(dist), scales=scale)[dist.first_axis(self)+1]
        return reshape_vector(r[local_elements], dim=dist.dim, axis=dist.get_basis_axis(self)+1)

    @CachedMethod
    def _radius_grid(self, scale):
        N = int(np.ceil(scale * self.shape[1]))
        z, weights = dedalus_sphere.jacobi.quadrature(N, self.alpha[0], self.alpha[1])
        r = self.dR/2*(z + self.rho)
        return r.astype(np.float64)

    @CachedMethod
    def _radius_weights(self, scale):
        N = int(np.ceil(scale * self.shape[1]))
        z_proj, weights_proj = dedalus_sphere.jacobi.quadrature(N, self.alpha[0], self.alpha[1])
        z0, weights0 = dedalus_sphere.jacobi.quadrature(N, 0, 0)
        Q0 = dedalus_sphere.jacobi.polynomials(N, self.alpha[0], self.alpha[1], z0)
        Q_proj = dedalus_sphere.jacobi.polynomials(N, self.alpha[0], self.alpha[1], z_proj)
        normalization = self.dR/2
        return normalization * ( (Q0 @ weights0).T ) @ (weights_proj*Q_proj)

    def global_radius_weights(self, dist, scale=None):
        if scale == None: scale = 1
        N = int(np.ceil(scale * self.shape[1]))
        z, weights = dedalus_sphere.sphere.quadrature(2,N,k=self.alpha)
        return reshape_vector(weights.astype(np.float64), dim=dist.dim, axis=dist.get_basis_axis(self)+1)

    def local_radius_weights(self, dist, scale=None):
        if scale == None: scale = 1
        local_elements = dist.grid_layout.local_elements(self.domain(dist), scales=scale)[dist.get_basis_axis(self)+1]
        N = int(np.ceil(scale * self.shape[1]))
        z, weights = dedalus_sphere.sphere.quadrature(2,N,k=self.alpha)
        return reshape_vector(weights.astype(np.float64)[local_elements], dim=dist.dim, axis=dist.get_basis_axis(self)+1)

    @CachedAttribute
    def constant_mode_value(self):
        # Note the zeroth mode is constant only for k=0
        Q0 = dedalus_sphere.jacobi.polynomials(1, self.alpha[0], self.alpha[1], np.array([0.0]))
        return Q0[0,0]

    def _new_k(self, k):
        # TODO: replace this in all operators
        return self.clone_with(k=k)

    @CachedMethod
    def transform_plan(self, dist, grid_size, k):
        """Build transform plan."""
        a = self.alpha[0] + k
        b = self.alpha[1] + k
        a0 = self.alpha[0]
        b0 = self.alpha[1]
        return Jacobi.transforms[self.radius_library](grid_size, self.Nmax+1, a, b, a0, b0)

    @CachedMethod
    def radial_transform_factor(self, scale, data_axis, dk):
        r = reshape_vector(self._radius_grid(scale), dim=data_axis, axis=data_axis-1)
        return (self.dR/r)**dk

    def forward_transform_radius(self, field, axis, gdata, cdata):
        data_axis = len(field.tensorsig) + axis
        grid_size = gdata.shape[data_axis]
        # Multiply by radial factor
        if self.k > 0:
            gdata *= self.radial_transform_factor(field.scales[axis], data_axis, -self.k)
        # Expand gdata if mmax=0 and dtype=float for spin recombination
        if self.mmax == 0 and self.dtype == np.float64:
            m_axis = len(field.tensorsig) + axis - 1
            gdata = np.concatenate((gdata, np.zeros_like(gdata)), axis=m_axis)
        # Apply spin recombination from gdata to temp
        temp = np.zeros_like(gdata)
        self.forward_spin_recombination(field.tensorsig, axis, gdata, temp)
        cdata.fill(0)  # OPTIMIZE: shouldn't be necessary
        # Transform component-by-component from temp to cdata
        S = self.spin_weights(field.tensorsig)
        for i, s in np.ndenumerate(S):
           plan = self.transform_plan(field.dist, grid_size, self.k)
           plan.forward(temp[i], cdata[i], axis)

    def backward_transform_radius(self, field, axis, cdata, gdata):
        data_axis = len(field.tensorsig) + axis
        grid_size = gdata.shape[data_axis]
        # Create temporary
        if self.mmax == 0 and self.dtype == np.float64:
            m_axis = len(field.tensorsig) + axis - 1
            shape = list(gdata.shape)
            shape[m_axis] = 2
            temp = np.zeros(shape, dtype=gdata.dtype)
            # Expand gdata for spin recombination
            gdata_orig = gdata
            gdata = np.zeros(shape, dtype=gdata.dtype)
        else:
            temp = np.zeros_like(gdata)
        # Transform component-by-component from cdata to temp
        S = self.spin_weights(field.tensorsig)
        for i, s in np.ndenumerate(S):
           plan = self.transform_plan(field.dist, grid_size, self.k)
           plan.backward(cdata[i], temp[i], axis)
        # Apply spin recombination from temp to gdata
        gdata.fill(0)  # OPTIMIZE: shouldn't be necessary
        self.backward_spin_recombination(field.tensorsig, axis, temp, gdata)
        # Multiply by radial factor
        if self.k > 0:
            gdata *= self.radial_transform_factor(field.scales[axis], data_axis, self.k)
        if self.mmax == 0 and self.dtype == np.float64:
            gdata_orig[:] = gdata[axslice(m_axis, 0, 1)]

    def interpolation(self, m, spintotal, position):
        return self._interpolation(position)

    @CachedMethod
    def _interpolation(self, position):
        native_position = position*2/self.dR - self.rho
        a = self.alpha[0] + self.k
        b = self.alpha[1] + self.k
        radial_factor = (self.dR/position)**(self.k)
        return radial_factor*dedalus_sphere.jacobi.polynomials(self.n_size(0), a, b, native_position)

    @CachedMethod
    def operator_matrix(self,op,m,spintotal, size=None):
        ms = m + spintotal
        if op[-1] in ['+', '-']:
            o = op[:-1]
            p = int(op[-1]+'1')
            if ms == 0:
                p = +1
            elif ms < 0:
                p = -p
                ms = -ms
            operator = dedalus_sphere.shell.operator(2, self.radii, o, self.alpha)(p,ms)
        elif op == 'L':
            D = dedalus_sphere.shell.operator(2, self.radii, 'D', self.alpha)
            if ms < 0:
                operator = D(+1, ms-1) @ D(-1, ms)
            else:
                operator = D(-1, ms+1) @ D(+1, ms)
        else:
            operator = dedalus_sphere.shell.operator(2, self.radii, op, self.alpha)
        if size is None:
            size = self.n_size(m)
        return operator(size, self.k).square.astype(np.float64)

    def jacobi_conversion(self, m, dk, size=None):
        AB = dedalus_sphere.shell.operator(2, self.radii, 'AB', self.alpha)
        operator = AB**dk
        if size is None:
            size = self.n_size(m)
        return operator(size, self.k).square.astype(np.float64)

    @CachedMethod
    def conversion_matrix(self, m, spintotal, dk):
        E = dedalus_sphere.shell.operator(2, self.radii, 'E', self.alpha)
        operator = E**dk
        return operator(self.n_size(m), self.k).square.astype(np.float64)

    @classmethod
    def _last_axis_component_ncc_matrix(cls, subproblem, ncc_basis, arg_basis, out_basis, coeffs, ncc_comp, arg_comp, out_comp, ncc_tensorsig, arg_tensorsig, out_tensorsig, cutoff=1e-6):
        first_axis = subproblem.dist.first_axis(out_basis)
        m = subproblem.group[first_axis]
        spintotal_arg = out_basis.spintotal(arg_tensorsig, arg_comp)
        # Jacobi parameters
        a_ncc = ncc_basis.k + ncc_basis.alpha[0]
        b_ncc = ncc_basis.k + ncc_basis.alpha[1]
        N = ncc_basis.n_size(m)
        N0 = ncc_basis.n_size(0)
        # Pad for dealiasing with conversion
        Nmat = 3*((N0+1)//2) + ncc_basis.k
        J = arg_basis.operator_matrix('Z', m, spintotal_arg, size=Nmat)
        A, B = clenshaw.jacobi_recursion(Nmat, a_ncc, b_ncc, J)
        f0 = dedalus_sphere.jacobi.polynomials(1, a_ncc, b_ncc, 1)[0] * sparse.identity(Nmat)
        # Conversions to account for radial prefactors
        prefactor = arg_basis.jacobi_conversion(m, dk=ncc_basis.k, size=Nmat)
        if ncc_basis.dtype == np.float64:
            coeffs_cos_filter = coeffs[0].ravel()[:N0]
            coeffs_msin_filter = coeffs[1].ravel()[:N0]
            matrix_cos = (prefactor @ clenshaw.matrix_clenshaw(coeffs_cos_filter, A, B, f0, cutoff=cutoff))[:N,:N]
            matrix_msin = (prefactor @ clenshaw.matrix_clenshaw(coeffs_msin_filter, A, B, f0, cutoff=cutoff))[:N,:N]
            matrix = sparse.bmat([[matrix_cos, -matrix_msin], [matrix_msin, matrix_cos]], format='csr')
        elif ncc_basis.dtype == np.complex128:
            coeffs_filter = coeffs.ravel()[:N0]
            matrix = (prefactor @ clenshaw.matrix_clenshaw(coeffs_filter, A, B, f0, cutoff=cutoff))[:N,:N]
        return matrix


class DiskBasis(PolarBasis, metaclass=CachedClass):

    transforms = {}
    subaxis_dependence = (True, True)
    default_library = "matrix"

    @classmethod
    def _preprocess_cache_args(cls, coordsys, shape, dtype, radius, k, alpha, dealias, azimuth_library, radius_library):
        """Preprocess arguments into canonical form for caching. Must accept and return __init__ arguments."""
        # coordsys: PolarCoordinates
        if not isinstance(coordsys, PolarCoordinates):
            raise ValueError("Disk coordsys must be PolarCoordinates.")
        # shape: length-2 tuple
        shape = tuple(shape)
        if len(shape) != 2:
            raise ValueError("Disk shape must have length 2.")
        # radius: positive number
        if radius <= 0:
            raise ValueError("Disk radius must be positive.")
        # k: int
        k = int(k)
        # alpha
        # dealias: length-2 tuple
        if isinstance(dealias, (int, float)):
            dealias = (dealias,) * 2
        else:
            dealias = tuple(dealias)
        if len(dealias) != 2:
            raise ValueError("Disk dealias must have length 2.")
        # azimuth_library: pick default
        if azimuth_library is None:
            azimuth_library = RealFourier.default_library
        # radius_library: pick default
        if radius_library is None:
            radius_library = cls.default_library
        return (coordsys, shape, dtype, radius, k, alpha, dealias, azimuth_library, radius_library)

    def __init__(self, coordsys, shape, dtype, radius=1, k=0, alpha=0, dealias=(1,1), azimuth_library=None, radius_library=None):
        super().__init__(coordsys, shape, dtype, k=k, dealias=dealias, azimuth_library=azimuth_library)
        # Save arguments without modification for caching
        self.coordsys = coordsys
        self.shape = shape
        self.dtype = dtype
        self.radius = radius
        self.k = k
        self.alpha = alpha
        self.dealias = dealias
        self.azimuth_library = azimuth_library
        self.radius_library = radius_library
        # Other attributes
        self.volume = np.pi * radius**2
        self.radial_COV = AffineCOV((0, 1), (0, radius))
        if self.mmax > 2*self.Nmax:
            logger.warning("You are using more azimuthal modes than can be resolved with your current radial resolution")
        self.grid_params = (coordsys, dtype, radius, alpha, dealias, azimuth_library, radius_library)
        self.edge = self.S1_basis(radius)

    @CachedAttribute
    def radial_basis(self):
        shape = (1, self.shape[1])
        return self.clone_with(shape=shape)

    @staticmethod
    def _nmin(m):
        return abs(m) // 2

    def __add__(self, other):
        if other is None or other is self:
            return self
        if isinstance(other, DiskBasis):
            if self.grid_params == other.grid_params:
                # Everything matches except shape and k
                shape = tuple(np.maximum(self.shape, other.shape))
                k = max(self.k, other.k)  # minimizes conversions
                return self.clone_with(shape=shape, k=k)
        return NotImplemented

    def __mul__(self, other):
        if other is None:
            return self
        if isinstance(other, DiskBasis):
            if self.grid_params == other.grid_params:
                # Everything matches except shape and k
                shape = tuple(np.maximum(self.shape, other.shape))
                k = 0  # kind of arbitrary, but minimizes k in RHS expressions
                return self.clone_with(shape=shape, k=k)
        return NotImplemented

    def __rmatmul__(self, other):
        # NCC (other) * operand (self)
        if other is None:
            return self
        if isinstance(other, DiskBasis):
            if self.grid_params == other.grid_params:
                # Everything matches except shape and k
                shape = tuple(np.maximum(self.shape, other.shape))
                k = self.k  # use operand's k value to minimize conversions
                return self.clone_with(shape=shape, k=k)
        return NotImplemented

    def global_grid_radius(self, dist, scale):
        r = self.radial_COV.problem_coord(self._native_radius_grid(scale))
        return reshape_vector(r, dim=dist.dim, axis=dist.get_basis_axis(self)+1)

    def local_grid_radius(self, dist, scale):
        r = self.radial_COV.problem_coord(self._native_radius_grid(scale))
        local_elements = dist.grid_layout.local_elements(self.domain(dist), scales=scale)[dist.get_basis_axis(self)+1]
        return reshape_vector(r[local_elements], dim=dist.dim, axis=dist.get_basis_axis(self)+1)

    def _native_radius_grid(self, scale):
        N = int(np.ceil(scale * self.shape[1]))
        z, weights = dedalus_sphere.zernike.quadrature(2,N,k=self.alpha)
        r = np.sqrt((z+1)/2).astype(np.float64)
        return r

    def global_radius_weights(self, dist, scale=None):
        if scale == None: scale = 1
        N = int(np.ceil(scale * self.shape[1]))
        z, weights = dedalus_sphere.sphere.quadrature(2,N,k=self.alpha)
        return reshape_vector(weights.astype(np.float64), dim=dist.dim, axis=dist.get_basis_axis(self)+1)

    def local_radius_weights(self, dist, scale=None):
        if scale == None: scale = 1
        local_elements = dist.grid_layout.local_elements(self.domain(dist), scales=scale)[dist.get_basis_axis(self)+1]
        N = int(np.ceil(scale * self.shape[1]))
        z, weights = dedalus_sphere.sphere.quadrature(2,N,k=self.alpha)
        return reshape_vector(weights.astype(np.float64)[local_elements], dim=dist.dim, axis=dist.get_basis_axis(self)+1)

    @CachedAttribute
    def constant_mode_value(self):
        Qk = dedalus_sphere.zernike.polynomials(2, 1, self.alpha+self.k, 0, np.array([0]))
        return Qk[0,0]

    def _new_k(self, k):
        # TODO: replace this in all operators
        return self.clone_with(k=k)

    @CachedMethod
    def transform_plan(self, dist, grid_shape, axis, s):
        """Build transform plan."""
        return self.transforms[self.radius_library](grid_shape, self.shape, axis, self.m_maps(dist), s, self.k, self.alpha)

    def forward_transform_radius_Nmax0(self, field, axis, gdata, cdata):
        raise NotImplementedError("Not yet.")
        # # Create temporary
        # temp = np.zeros_like(gdata)
        # # Apply spin recombination from gdata to temp
        # self.forward_spin_recombination(field.tensorsig, gdata, out=temp)
        # np.copyto(cdata, temp)

    def forward_transform_radius(self, field, axis, gdata, cdata):
        # Expand gdata if mmax=0 and dtype=float for spin recombination
        if self.mmax == 0 and self.dtype == np.float64:
            m_axis = len(field.tensorsig) + axis - 1
            gdata = np.concatenate((gdata, np.zeros_like(gdata)), axis=m_axis)
        # Apply spin recombination from gdata to temp
        temp = np.zeros_like(gdata)
        self.forward_spin_recombination(field.tensorsig, axis, gdata, temp)
        cdata.fill(0)  # OPTIMIZE: shouldn't be necessary
        # Transform component-by-component from temp to cdata
        S = self.spin_weights(field.tensorsig)
        for i, s in np.ndenumerate(S):
            grid_shape = gdata[i].shape
            plan = self.transform_plan(field.dist, grid_shape, axis, s)
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
        if self.mmax == 0 and self.dtype == np.float64:
            m_axis = len(field.tensorsig) + axis - 1
            shape = list(gdata.shape)
            shape[m_axis] = 2
            temp = np.zeros(shape, dtype=gdata.dtype)
            # Expand gdata for spin recombination
            gdata_orig = gdata
            gdata = np.zeros(shape, dtype=gdata.dtype)
        else:
            temp = np.zeros_like(gdata)
        # Transform component-by-component from cdata to temp
        S = self.spin_weights(field.tensorsig)
        for i, s in np.ndenumerate(S):
            grid_shape = gdata[i].shape
            plan = self.transform_plan(field.dist, grid_shape, axis, s)
            plan.backward(cdata[i], temp[i], axis)
        # Apply spin recombination from temp to gdata
        gdata.fill(0)  # OPTIMIZE: shouldn't be necessary
        self.backward_spin_recombination(field.tensorsig, axis, temp, gdata)
        if self.mmax == 0 and self.dtype == np.float64:
            gdata_orig[:] = gdata[axslice(m_axis, 0, 1)]

    @CachedMethod
    def conversion_matrix(self, m, spintotal, dk):
        E = dedalus_sphere.zernike.operator(2, 'E', radius=self.radius)
        operator = E(+1)**dk
        return operator(self.n_size(m), self.alpha + self.k, np.abs(m + spintotal)).square.astype(np.float64)

    @CachedMethod
    def operator_matrix(self, op, m, spin, size=None):
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
        if size is None:
            size = self.n_size(m)
        return operator(size, self.alpha + self.k, abs(m + spin)).square.astype(np.float64)

    @CachedMethod
    def interpolation(self, m, spintotal, position):
        native_position = self.radial_COV.native_coord(position)
        native_z = 2*native_position**2 - 1
        return dedalus_sphere.zernike.polynomials(2, self.n_size(m), self.alpha + self.k, np.abs(m + spintotal), native_z)

    @CachedMethod
    def radius_multiplication_matrix(self, m, spintotal, order, d):
        if order == 0:
            operator = dedalus_sphere.zernike.operator(2, 'Id', radius=self.radius)
        else:
            R = dedalus_sphere.zernike.operator(2, 'R', radius=1)
            if order < 0:
                operator = R(-1)**abs(order)
            else: # order > 0
                operator = R(+1)**abs(order)
        if d > 0:
            R = dedalus_sphere.zernike.operator(2, 'R', radius=1)
            R2 = R(-1) @ R(+1)
            operator = R2**(d//2) @ operator
        return operator(self.n_size(m), self.alpha + self.k, abs(m + spintotal)).square.astype(np.float64)

    @classmethod
    def _last_axis_component_ncc_matrix(cls, subproblem, ncc_basis, arg_basis, out_basis, coeffs, ncc_comp, arg_comp, out_comp, ncc_tensorsig, arg_tensorsig, out_tensorsig, cutoff=1e-6):
        first_axis = subproblem.dist.first_axis(out_basis)
        m = subproblem.group[first_axis]
        spintotal_ncc = out_basis.spintotal(ncc_tensorsig, ncc_comp)
        spintotal_arg = out_basis.spintotal(arg_tensorsig, arg_comp)
        spintotal_out = out_basis.spintotal(out_tensorsig, out_comp)
        regtotal_ncc = abs(spintotal_ncc)
        regtotal_arg = abs(m + spintotal_arg)
        regtotal_out = abs(m + spintotal_out)
        diff_regtotal = regtotal_out - regtotal_arg
        # jacobi parameters
        a_ncc = ncc_basis.alpha + ncc_basis.k
        b_ncc = regtotal_ncc
        N = ncc_basis.n_size(m)
        d = regtotal_ncc - abs(diff_regtotal)
        if (d >= 0) and (d % 2 == 0):
            J = arg_basis.operator_matrix('Z', m, spintotal_arg)
            A, B = clenshaw.jacobi_recursion(N, a_ncc, b_ncc, J)
            # assuming that we're doing ball for now...
            f0 = dedalus_sphere.zernike.polynomials(2, 1, a_ncc, b_ncc, 1)[0] * sparse.identity(N)
            prefactor = arg_basis.radius_multiplication_matrix(m, spintotal_arg, diff_regtotal, d)
            if ncc_basis.dtype == np.float64:
                coeffs_filter = coeffs.ravel()[:2*N]
                matrix_cos = prefactor @ clenshaw.matrix_clenshaw(coeffs_filter[:N], A, B, f0, cutoff=cutoff)
                matrix_msin = prefactor @ clenshaw.matrix_clenshaw(coeffs_filter[N:], A, B, f0, cutoff=cutoff)
                matrix = sparse.bmat([[matrix_cos, -matrix_msin], [matrix_msin, matrix_cos]], format='csr')
            elif ncc_basis.dtype == np.complex128:
                coeffs_filter = coeffs.ravel()[:N]
                matrix = prefactor @ clenshaw.matrix_clenshaw(coeffs_filter, A, B, f0, cutoff=cutoff)
        else:
            if ncc_basis.dtype == np.float64:
                matrix = sparse.csr_matrix((2*N, 2*N))
            elif ncc_basis.dtype == np.complex128:
                matrix = sparse.csr_matrix((N, N))
        return matrix


class ConvertPolar(operators.Convert, operators.PolarMOperator):

    input_basis_type = PolarBasis
    output_basis_type = PolarBasis

    def __init__(self, operand, output_basis, out=None):
        operators.Convert.__init__(self, operand, output_basis, out=out)
        self.radius_axis = self.dist.last_axis(output_basis)

    def spinindex_out(self, spinindex_in):
        return (spinindex_in,)

    def radial_matrix(self, spinindex_in, spinindex_out, m):
        radial_basis = self.input_basis
        spintotal = radial_basis.spintotal(self.operand.tensorsig, spinindex_in)
        dk = self.output_basis.k - radial_basis.k
        if spinindex_in == spinindex_out:
            return radial_basis.conversion_matrix(m, spintotal, dk)
        else:
            raise ValueError("This should never happen.")


class ConvertConstantDisk(operators.ConvertConstant, operators.PolarMOperator):

    output_basis_type = DiskBasis
    subaxis_dependence = [True, True]
    subaxis_coupling = [False, False]

    def __init__(self, operand, output_basis, out=None):
        super().__init__(operand, output_basis, out=out)
        for coordsys in operand.tensorsig:
            if self.coords == coordsys:
                raise ValueError("Tensors not yet supported.")
            if isinstance(coordsys, DirectProduct) and self.coords in coordsys.coordsystems:
                raise ValueError("DirectProduct tensors not yet supported.")

    def spinindex_out(self, spinindex_in):
        return (spinindex_in,)

    def radial_matrix(self, spinindex_in, spinindex_out, m):
        radial_basis = self.output_basis
        coeff_size = radial_basis.shape[-1]
        if m == 0 and spinindex_in == spinindex_out:
            unit_amplitude = 1 / self.output_basis.constant_mode_value
            matrix = np.zeros((coeff_size, 1))
            matrix[0, 0] = unit_amplitude
            return matrix
        else:
            raise ValueError("This should never happen.")


class ConvertConstantAnnulus(operators.ConvertConstant, operators.PolarMOperator):

    output_basis_type = AnnulusBasis
    subaxis_dependence = [True, True]
    subaxis_coupling = [False, True]

    def __init__(self, operand, output_basis, out=None):
        super().__init__(operand, output_basis, out=out)
        if self.coords in operand.tensorsig:
            raise ValueError("Tensors not yet supported.")

    def spinindex_out(self, spinindex_in):
        return (spinindex_in,)

    def radial_matrix(self, spinindex_in, spinindex_out, m):
        radial_basis = self.output_basis
        spintotal = radial_basis.spintotal(self.operand.tensorsig, spinindex_in)
        coeff_size = radial_basis.shape[-1]
        if m == 0 and spinindex_in == spinindex_out:
            # Convert to k=0
            const_to_k0 = np.zeros((coeff_size, 1))
            const_to_k0[0, 0] = 1 / self.output_basis.constant_mode_value  # This is really for k=0
            # Convert up in k
            k0_to_k = radial_basis._new_k(0).conversion_matrix(m, spintotal, radial_basis.k)
            matrix = k0_to_k @ const_to_k0
            return matrix
        else:
            raise ValueError("This should never happen.")


class SphereBasis(SpinBasis, metaclass=CachedClass):

    dim = 2
    dims = ['azimuth', 'colatitude']
    subaxis_dependence = [True, True]
    transforms = {}
    constant_mode_value = 1 / np.sqrt(2)
    default_library = "matrix"

    @classmethod
    def _preprocess_cache_args(cls, coordsys, shape, dtype, radius, dealias, azimuth_library, colatitude_library):
        """Preprocess arguments into canonical form for caching. Must accept and return __init__ arguments."""
        # coordsys: S2Coordinates or SphericalCoordinates
        if not isinstance(coordsys, (S2Coordinates, SphericalCoordinates)):
            raise ValueError("Sphere coordsys must be S2Coordinates.")
        # shape: length-2 tuple
        shape = tuple(shape)
        if len(shape) != 2:
            raise ValueError("Sphere shape must have length 2.")
        # radius: non-negative number
        # can be zero for interpolation at origin in SphericalCoordinates
        if radius < 0:
            raise ValueError("Sphere radius must be non-negative.")
        # dealias: length-2 tuple
        if isinstance(dealias, (int, float)):
            dealias = (dealias,) * 2
        else:
            dealias = tuple(dealias)
        if len(dealias) != 2:
            raise ValueError("Sphere dealias must have length 2.")
        # azimuth_library: pick default
        if azimuth_library is None:
            azimuth_library = RealFourier.default_library
        # colatitude_library: pick default
        if colatitude_library is None:
            colatitude_library = cls.default_library
        return (coordsys, shape, dtype, radius, dealias, azimuth_library, colatitude_library)

    def __init__(self, coordsys, shape, dtype, radius=1, dealias=(1,1), azimuth_library=None, colatitude_library=None):
        super().__init__(coordsys, shape, dtype, dealias, azimuth_library=azimuth_library)
        # Save arguments without modification for caching
        self.coordsys = coordsys
        self.shape = shape
        self.dtype = dtype
        self.radius = radius
        self.dealias = dealias
        self.azimuth_library = azimuth_library
        self.colatitude_library = colatitude_library
        # Other attributes
        self.volume = 4 * np.pi * radius**2
        # Set Lmax for optimal load balancing
        if self.dtype == np.float64:
            self.Lmax = max(0, shape[1] - 2)
        elif self.dtype == np.complex128:
            self.Lmax = shape[1] - 1
        if self.Lmax > 0 and (self.mmax > self.Lmax + 1):
            logger.warning("You are using more azimuthal modes than can be resolved with your current colatitude resolution")
        # TODO: make this less hacky
        if self.mmax == 0:
            self.forward_transform_azimuth = self.forward_transform_azimuth_Mmax0
            self.backward_transform_azimuth = self.backward_transform_azimuth_Mmax0
        # if self.Lmax == 0:
        #     self.forward_transform_colatitude = self.forward_transform_colatitude_Lmax0
        #     self.backward_transform_colatitude = self.backward_transform_colatitude_Lmax0
        self.forward_transforms = [self.forward_transform_azimuth,
                                   self.forward_transform_colatitude,
                                   self.forward_transform_radius]
        self.backward_transforms = [self.backward_transform_azimuth,
                                    self.backward_transform_colatitude,
                                    self.backward_transform_radius]
        self.grid_params = (coordsys, dtype, radius, dealias, azimuth_library, colatitude_library)
        if self.shape[0] > 1 and shape[0] % 2 != 0:
            raise ValueError("Don't use an odd phi resolution please")
        if self.shape[0] > 1 and self.dtype == np.float64 and shape[0] % 4 != 0:
            raise ValueError("Don't use a phi resolution that isn't divisible by 4, please")
        # m permutations for repacking triangular truncation
        if self.dtype == np.complex128:
            az_index = np.arange(shape[0])
            az_div, az_mod = divmod(az_index, 2)
            self.forward_m_perm = az_div + shape[0] // 2 * az_mod
            self.backward_m_perm = np.argsort(self.forward_m_perm)
            self.group_shape = (1, 1)
        elif self.dtype == np.float64:
            if shape[0] == 1:
                # Include sine and cosine parts of m=0
                az_index = np.arange(2)
            else:
                az_index = np.arange(shape[0])
            div2, mod2 = divmod(az_index, 2)
            div22 = div2 % 2
            self.forward_m_perm = (mod2 + div2) * (1 - div22) + (shape[0] - 1 + mod2 - div2) * div22
            self.backward_m_perm = np.argsort(self.forward_m_perm)
            self.group_shape = (2, 1)
        # Set permutations on azimuth basis
        if self.mmax > 0:
            self.azimuth_basis.forward_coeff_permutation = self.forward_m_perm
            self.azimuth_basis.backward_coeff_permutation = self.backward_m_perm

    @CachedAttribute
    def latitude_basis(self):
        shape = (1, self.shape[1])
        return self.clone_with(shape=shape)

    def matrix_dependence(self, matrix_coupling):
        matrix_dependence = matrix_coupling.copy()
        if matrix_coupling[1]:
            matrix_dependence[0] = True
        return matrix_dependence

    def global_shape(self, grid_space, scales):
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
            if self.mmax > 0:
                shape[0] = self.shape[0]
            else:
                if self.dtype == np.complex128:
                    shape[0] = 1
                elif self.dtype == np.float64:
                    shape[0] = 2
            return tuple(shape)
        else:
            # coeff-coeff space
            # Repacked triangular truncation
            Nphi = self.shape[0]
            Lmax = self.Lmax
            if self.mmax > 0:
                if self.dtype == np.complex128:
                    return (Nphi//2, Lmax+1+max(0, Lmax+1-Nphi//2))
                elif self.dtype == np.float64:
                    return (Nphi//2, Lmax+1+max(0, Lmax+2-Nphi//2))
            else:
                if self.dtype == np.complex128:
                    return (1, Lmax+1)
                elif self.dtype == np.float64:
                    return (2, Lmax+1)

    def chunk_shape(self, grid_space):
        if grid_space[0]:
            # grid-grid space
            return (1, 1)
        elif grid_space[1]:
            # coeff-grid space
            if self.dtype == np.complex128:
                if self.mmax > 0:
                    return (2, 1)
                else:
                    return (1, 1)
            elif self.dtype == np.float64:
                if self.mmax > 0:
                    return (4, 1)
                else:
                    return (2, 1)
        else:
            # coeff-coeff space
            if self.dtype == np.complex128:
                return (1, 1)
            elif self.dtype == np.float64:
                return (2, 1)

    def elements_to_groups(self, grid_space, elements):
        if grid_space[0]:
            # grid-grid space
            groups = elements
        elif grid_space[1]:
            # coeff-grid space
            # Unpacked m
            permuted_native_wavenumbers = self.azimuth_basis.native_wavenumbers
            groups = elements.copy()
            groups[0] = permuted_native_wavenumbers[elements[0]]
        else:
            # coeff-coeff space
            # Repacked triangular truncation
            i, j = elements
            Nphi = self.shape[0]
            Lmax = self.Lmax
            if self.dtype == np.complex128:
                # Valid for m > 0 except Nyquist
                shift = max(0, Lmax + 1 - Nphi//2)
                m = 1 * i
                ell = j - shift
                # Fix for m < 0
                neg_modes = (ell < m)
                m[neg_modes] = i[neg_modes] - (Nphi+1)//2
                ell[neg_modes] = Lmax - j[neg_modes]
                # Fix for m = 0
                m_zero = (i == 0)
                m[m_zero] = 0
                ell[m_zero] = j[m_zero]
                # Fix for Nyquist
                nyq_modes = (i == 0) * (j > Lmax)
                m[nyq_modes] = Nphi//2
                ell[nyq_modes] = j[nyq_modes] - shift
            elif self.dtype == np.float64:
                # Valid for 0 < m < Nphi//4
                shift = max(0, Lmax + 2 - Nphi//2)
                m = i // 2
                ell = j - shift
                # Fix for Nphi//4 <= m < Nphi//2-1
                neg_modes = (ell < m)
                m[neg_modes] = (Nphi//2 - 1) - m[neg_modes]
                ell[neg_modes] = Lmax - j[neg_modes]
                # Fix for m = 0
                m_zero = (i < 2)
                m[m_zero] = 0
                ell[m_zero] = j[m_zero]
                # Fix for m = Nphi//2 - 1
                m_max = (i < 2) * (j > Lmax)
                m[m_max] = Nphi//2 - 1
                ell[m_max] = j[m_max] - shift
            groups = np.array([m, ell])
        return groups

    def __eq__(self, other):
        if isinstance(other, SphereBasis):
            if self.grid_params == other.grid_params:
                if self.shape == other.shape:
                    return True
        return False

    def __hash__(self):
        return id(self)

    def __add__(self, other):
        if other is None or other is self:
            return self
        if isinstance(other, SphereBasis):
            # Better error message for common mistake
            if self.radius != other.radius:
                raise NotImplementedError(f"Cannot add two sphere bases with different radii ({self.radius} and {other.radius}).")
            if self.grid_params == other.grid_params:
                # Everything matches except shape
                shape = tuple(np.maximum(self.shape, other.shape))
                return self.clone_with(shape=shape)
        return NotImplemented

    def __mul__(self, other):
        if other is None or other is self:
            return self
        if isinstance(other, SphereBasis):
            # Better error message for common mistake
            if self.radius != other.radius:
                raise NotImplementedError(f"Cannot multiply two sphere bases with different radii ({self.radius} and {other.radius}).")
            if self.grid_params == other.grid_params:
                # Everything matches except shape
                shape = tuple(np.maximum(self.shape, other.shape))
                return self.clone_with(shape=shape)
        return NotImplemented

    def __rmatmul__(self, other):
        # NCC (other) * operand (self)
        return self.__mul__(other)

    # @staticmethod
    # @CachedAttribute
    # def constant_mode_value():
    #     # Adjust for SWSH normalization
    #     # TODO: check this is right for regtotal != 0?
    #     return 1 / np.sqrt(2)

    @CachedMethod
    def m_maps(self, dist):
        """
        Tuple of (m, mg_slice, mc_slice, ell_slice) for all local m's when
        the colatitude axis is local.
        """
        # Get colatitude transform object
        colat_transform = dist.get_transform_object(dist.first_axis(self)+1)
        colat_coeff_layout = colat_transform.layout0
        colat_grid_layout = colat_transform.layout1
        # Get groupsets
        domain = self.domain(dist)
        group_coupling = [True] * domain.dist.dim
        group_coupling[dist.first_axis(self)] = False
        group_coupling = tuple(group_coupling)
        groupsets = colat_coeff_layout.local_groupsets(group_coupling, domain, scales=domain.dealias, broadcast=True)
        # Build m_maps from groupset slices
        m_maps = []
        for groupset in groupsets:
            m = groupset[dist.first_axis(self)]
            coeff_slices = colat_coeff_layout.local_groupset_slices(groupset, domain, scales=domain.dealias, broadcast=True)
            grid_slices = colat_grid_layout.local_groupset_slices(groupset, domain, scales=domain.dealias, broadcast=True)
            if len(coeff_slices) != 1 or len(grid_slices) != 1:
                raise ValueError("This should never happpen. Ask for help.")
            mg_slice = grid_slices[0][dist.first_axis(self)]
            mc_slice = coeff_slices[0][dist.first_axis(self)]
            ell_slice = coeff_slices[0][dist.first_axis(self)+1]
            # Reverse n_slice for folded modes so that ells are well-ordered
            if ell_slice.start == 0 and m != 0:
                ell_slice = slice(ell_slice.stop-1, None, -1)
            m_maps.append((m, mg_slice, mc_slice, ell_slice))
        return tuple(m_maps)

    @CachedMethod
    def ell_reversed(self, dist):
        ell_reversed = {}
        for m, mg_slice, mc_slice, ell_slice in self.m_maps(dist):
            ell_reversed[m] = False
            if ell_slice.step is not None:
                if ell_slice.step < 0:
                    ell_reversed[m] = True
        return ell_reversed

    @CachedMethod
    def ell_maps(self, dist):
        """
        Tuple of (ell, m_slice, ell_slice) for all local ells in coeff space.
        m_slice and ell_slice are local slices along the phi and theta axes.

        Data for each ell should be sliced as:

            for ell, m_slice, ell_slice in ell_maps:
                ell_data = data[m_slice, ell_slice]
        """
        coeff_layout = dist.coeff_layout
        azimuth_axis = dist.first_axis(self)
        colatitude_axis = dist.first_axis(self) + 1
        # Get groupsets
        domain = self.domain(dist)
        group_coupling = [True] * domain.dist.dim
        group_coupling[colatitude_axis] = False
        group_coupling = tuple(group_coupling)
        groupsets = coeff_layout.local_groupsets(group_coupling, domain, scales=domain.dealias, broadcast=True)
        # Build ell_maps from groupset slices
        ell_maps = []
        for groupset in groupsets:
            ell = groupset[colatitude_axis]
            groupset_slices = coeff_layout.local_groupset_slices(groupset, domain, scales=domain.dealias, broadcast=True)
            for groupset_slice in groupset_slices:
                m_slice = groupset_slice[azimuth_axis]
                ell_slice = groupset_slice[colatitude_axis]
                ell_maps.append((ell, m_slice, ell_slice))
        return tuple(ell_maps)

    def global_grids(self, dist, scales):
        return (self.global_grid_azimuth(dist, scales[0]),
                self.global_grid_colatitude(dist, scales[1]))

    def global_grid_colatitude(self, dist, scale):
        theta = self._native_colatitude_grid(scale)
        return reshape_vector(theta, dim=dist.dim, axis=dist.get_basis_axis(self)+1)

    @CachedMethod
    def global_grid_spacing(self, dist, subaxis, scales):
        """Global grids spacings."""
        axis = dist.first_axis(self) + subaxis
        return np.gradient(self.global_grids(dist, scales=scales)[subaxis], axis=axis, edge_order=2)

    @CachedMethod
    def local_grid_spacing(self, dist, subaxis, scales):
        """Local grids spacings."""
        axis = dist.first_axis(self) + subaxis
        global_spacing = self.global_grid_spacing(dist, subaxis, scales=scales)
        local_elements = dist.grid_layout.local_elements(self.domain(dist), scales=scales[subaxis])[axis]
        return reshape_vector(np.ravel(global_spacing)[local_elements], dim=dist.dim, axis=axis)

    def local_grids(self, dist, scales):
        return (self.local_grid_azimuth(dist, scales[0]),
                self.local_grid_colatitude(dist, scales[1]))

    def local_grid_colatitude(self, dist, scale):
        local_elements = dist.grid_layout.local_elements(self.domain(dist), scales=scale)[dist.get_basis_axis(self)+1]
        theta = self._native_colatitude_grid(scale)[local_elements]
        return reshape_vector(theta, dim=dist.dim, axis=dist.get_basis_axis(self)+1)

    def _native_colatitude_grid(self, scale):
        N = int(np.ceil(scale * self.shape[1]))
        cos_theta, weights = dedalus_sphere.sphere.quadrature(Lmax=N-1)
        theta = np.arccos(cos_theta).astype(np.float64)
        return theta

    def global_colatitude_weights(self, dist, scale=None):
        if scale == None: scale = 1
        N = int(np.ceil(scale * self.shape[1]))
        cos_theta, weights = dedalus_sphere.sphere.quadrature(Lmax=N-1)
        return reshape_vector(weights.astype(np.float64), dim=dist.dim, axis=dist.get_basis_axis(self)+1)

    def local_colatitude_weights(self, dist, scale=None):
        if scale == None: scale = 1
        local_elements = dist.grid_layout.local_elements(self.domain(dist), scales=scale)[dist.get_basis_axis(self)+1]
        N = int(np.ceil(scale * self.shape[1]))
        cos_theta, weights = dedalus_sphere.sphere.quadrature(Lmax=N-1)
        return reshape_vector(weights.astype(np.float64)[local_elements], dim=dist.dim, axis=dist.get_basis_axis(self)+1)

    @CachedMethod
    def transform_plan(self, dist, Ntheta, s):
        """Build transform plan."""
        return self.transforms[self.colatitude_library](Ntheta, self.Lmax, self.m_maps(dist), s)

    def forward_transform_azimuth_Mmax0(self, field, axis, gdata, cdata):
        slice_axis = axis + len(field.tensorsig)
        # do we need temp here? I'm worried that cdata and gdata are different views of the same buffer
        temp = np.zeros(cdata.shape, dtype=cdata.dtype)
        np.copyto(temp[axslice(slice_axis, 0, 1)], gdata)
        np.copyto(cdata, temp)

    def forward_transform_azimuth(self, field, axis, gdata, cdata):
        # Call Fourier transform
        self.azimuth_basis.forward_transform(field, axis, gdata, cdata)
        # Permute m for triangular truncation
        #permute_axis(cdata, axis+len(field.tensorsig), self.forward_m_perm, out=cdata)

    def forward_transform_radius(self, field, axis, gdata, cdata):
        # Placeholder for when self.coordsys is SphericalCoordinates
        pass

    def backward_transform_azimuth_Mmax0(self, field, axis, cdata, gdata):
        slice_axis = axis + len(field.tensorsig)
        np.copyto(gdata, cdata[axslice(slice_axis, 0, 1)])

    def backward_transform_azimuth(self, field, axis, cdata, gdata):
        # Permute m back from triangular truncation
        #permute_axis(cdata, axis+len(field.tensorsig), self.backward_m_perm, out=cdata)
        # Call Fourier transform
        self.azimuth_basis.backward_transform(field, axis, cdata, gdata)

    def backward_transform_radius(self, field, axis, cdata, gdata):
        # Placeholder for when self.coordsys is SphericalCoordinates
        pass

    def forward_transform_colatitude_Lmax0(self, field, axis, gdata, cdata):
        # Apply spin recombination from gdata to temp
        temp = np.zeros_like(gdata)
        self.forward_spin_recombination(field.tensorsig, axis, gdata, temp)
        # Copy from temp to cdata
        np.copyto(cdata, temp)
        # Scale to account for SWSH normalization? Is this right for all spins?
        cdata *= np.sqrt(2)

    def forward_transform_colatitude(self, field, axis, gdata, cdata):
        # Expand gdata if mmax=0 and dtype=float for spin recombination
        if self.mmax == 0 and self.dtype == np.float64:
            m_axis = len(field.tensorsig) + axis - 1
            gdata = np.concatenate((gdata, np.zeros_like(gdata)), axis=m_axis)
        # Apply spin recombination from gdata to temp
        temp = np.zeros_like(gdata)
        self.forward_spin_recombination(field.tensorsig, axis, gdata, temp)
        cdata.fill(0)  # OPTIMIZE: shouldn't be necessary
        # Transform component-by-component from temp to cdata
        S = self.spin_weights(field.tensorsig)
        for i, s in np.ndenumerate(S):
            Ntheta = gdata[i].shape[axis]
            plan = self.transform_plan(field.dist, Ntheta, s)
            plan.forward(temp[i], cdata[i], axis)

    def backward_transform_colatitude_Lmax0(self, field, axis, cdata, gdata):
        # Copy from cdata to temp
        temp = np.copy(cdata)
        # Apply spin recombination from temp to gdata
        self.backward_spin_recombination(field.tensorsig, axis, temp, gdata)
        # Scale to account for SWSH normalization? Is this right for all spins?
        gdata /= np.sqrt(2)

    def backward_transform_colatitude(self, field, axis, cdata, gdata):
        # Create temporary
        if self.mmax == 0 and self.dtype == np.float64:
            m_axis = len(field.tensorsig) + axis - 1
            shape = list(gdata.shape)
            shape[m_axis] = 2
            temp = np.zeros(shape, dtype=gdata.dtype)
            # Expand gdata for spin recombination
            gdata_orig = gdata
            gdata = np.zeros(shape, dtype=gdata.dtype)
        else:
            temp = np.zeros_like(gdata)
        S = self.spin_weights(field.tensorsig)
        for i, s in np.ndenumerate(S):
            Ntheta = gdata[i].shape[axis]
            plan = self.transform_plan(field.dist, Ntheta, s)
            plan.backward(cdata[i], temp[i], axis)
        # Apply spin recombination from temp to gdata
        gdata.fill(0)  # OPTIMIZE: shouldn't be necessary
        self.backward_spin_recombination(field.tensorsig, axis, temp, gdata)
        if self.mmax == 0 and self.dtype == np.float64:
            gdata_orig[:] = gdata[axslice(m_axis, 0, 1)]

    @staticmethod
    def k(l, s, mu):
        return -mu*np.sqrt(np.maximum(0, (l-mu*s)*(l+mu*s+1)/2))

    @CachedMethod
    def k_vector(self,mu,m,s,local_l):
        vector = np.zeros(len(local_l))
        Lmin = max(abs(m),abs(s),abs(s+mu))
        for i,l in enumerate(local_l):
            if l < Lmin: vector[i] = 0
            else: vector[i] = dedalus_sphere.sphere.k_element(mu,l,s,self.radius)
        return vector

    # @CachedMethod
    # def vector_slice(self, m, ell):
    #     if m > ell:
    #         return None
    #     mi = self.local_m.index(m)
    #     li = self.local_l.index(ell)
    #     return (mi, li)

    # def vector_3(self, comp, m, ell):
    #     slices = self.vector_slice(m, ell)
    #     if slices is None:
    #         return None
    #     comp5 = reduced_view(comp, axis=self.axis, dim=self.dist.dim)
    #     return comp5[(slice(None),) + slices + (slice(None),)]

    def valid_elements(self, tensorsig, grid_space, elements):
        if grid_space[1]:
            # grid-grid and coeff-grid
            # Same as Fourier validity before spin recombination
            return self.azimuth_basis.valid_elements(tensorsig, grid_space, elements)
        else:
            # coeff-coeff
            m, ell = self.elements_to_groups(grid_space, elements)
            tshape = tuple(cs.dim for cs in tensorsig)
            valid = np.ones(tshape + m.shape, dtype=bool)
            # Drop disallowed spin components
            if tensorsig:
                rank = len(tensorsig)
                dim = len(m.shape)
                full_m = m[(None,) * rank]
                full_ell = ell[(None,) * rank]
                s = self.spin_weights(tensorsig)
                full_s = s.T[(None,) * dim].T
                valid[np.maximum(np.abs(full_m), np.abs(full_s)) > full_ell] = False
            else:
                valid[np.abs(m) > ell] = False
            # Drop msin part of ell == 0 for real scalars and vectors
            # Does not impose m == 0 symmetry for ell > 0 or tensors
            if self.dtype == np.float64:
                if len(tensorsig) == 0:
                    valid[(ell == 0) * (elements[0] % 2 == 1)] = False
                elif len(tensorsig) == 1:
                    valid[:, (ell == 0) * (elements[0] % 2 == 1)] = False
            return valid

    def ell_size(self, m):
        return self.Lmax + 1 - abs(m)

    @CachedMethod
    def operator_matrix(self, op, m, spintotal, size=None):
        if op in ['Cos']:
            operator = dedalus_sphere.sphere.operator(op)
        return operator(size-1+max(abs(m), abs(spintotal)), m, spintotal).square.astype(np.float64)

    @CachedMethod
    def sine_multiplication_matrix(self, m, spintotal, order, size=None):
        if size is None:
            size = self.ell_size(m)
        if order == 0:
            return sparse.identity(size)
        else:
            S = dedalus_sphere.sphere.operator('Sin')
            if order < 0:
                operator = S(-1)**abs(order)
            else: # order > 0
                operator = S(+1)**abs(order)
        return (-1)**(max(0,-order))*operator(size - 1 + max(abs(m), abs(spintotal)), m, spintotal).square.astype(np.float64)

    @classmethod
    def _last_axis_component_ncc_matrix(cls, subproblem, ncc_basis, arg_basis, out_basis, coeffs, ncc_comp, arg_comp, out_comp, ncc_tensorsig, arg_tensorsig, out_tensorsig, cutoff):
        m = subproblem.group[0]  # HACK
        spintotal_arg = out_basis.spintotal(arg_tensorsig, arg_comp)
        spintotal_ncc = out_basis.spintotal(ncc_tensorsig, ncc_comp)
        spintotal_out = out_basis.spintotal(out_tensorsig, out_comp)
        # Jacobi parameters
        a_ncc = abs(spintotal_ncc)
        b_ncc = abs(spintotal_ncc)
        N = ncc_basis.ell_size(m)
        N0 = ncc_basis.ell_size(0)
        # Pad for dealiasing with conversion
        Nmat = 3*((N0+1)//2)
        J = arg_basis.operator_matrix('Cos', m, spintotal_arg, size=Nmat)
        A, B = clenshaw.jacobi_recursion(Nmat, a_ncc, b_ncc, J)
        f0 = dedalus_sphere.jacobi.polynomials(1, a_ncc, b_ncc, 1)[0] * sparse.identity(Nmat)
        # Sine and sign prefactors
        prefactor = ncc_basis.sine_multiplication_matrix(m, spintotal_arg, spintotal_ncc, size=Nmat)
        if ncc_basis.dtype == np.float64:
            coeffs_cos_filter = coeffs[0].ravel()[abs(spintotal_ncc):N0]
            coeffs_msin_filter = coeffs[1].ravel()[abs(spintotal_ncc):N0]
            # ell_min of data is based off |m|, but ell_min of matrix takes into account |s| and |m|
            lmin_out = max(abs(spintotal_out) - abs(m), 0)
            lmin_arg = max(abs(spintotal_arg) - abs(m), 0 )
            matrix_cos = sparse.lil_matrix((N, N))
            matrix_cos[lmin_out:,lmin_arg:] = (prefactor @ clenshaw.matrix_clenshaw(coeffs_cos_filter, A, B, f0, cutoff=cutoff))[:N-lmin_out,:N-lmin_arg]
            matrix_msin = sparse.lil_matrix((N, N))
            matrix_msin[lmin_out:,lmin_arg:] = (prefactor @ clenshaw.matrix_clenshaw(coeffs_msin_filter, A, B, f0, cutoff=cutoff))[:N-lmin_out,:N-lmin_arg]
            matrix = sparse.bmat([[matrix_cos, -matrix_msin], [matrix_msin, matrix_cos]], format='csr')
            if m >= arg_basis.mmax//4:
                matrix = matrix[::-1, ::-1] #for second half of m's, ell's go in opposite direction
        elif ncc_basis.dtype == np.complex128:
            coeffs_filter = coeffs.ravel()[abs(spintotal_ncc):N0]
            # ell_min of data is based off |m|, but ell_min of matrix takes into account |s| and |m|
            lmin_out = max(abs(spintotal_out) - abs(m), 0)
            lmin_arg = max(abs(spintotal_arg) - abs(m), 0 )
            matrix = sparse.lil_matrix((N, N), dtype=np.complex128)
            matrix[lmin_out:,lmin_arg:] = (prefactor @ clenshaw.matrix_clenshaw(coeffs_filter, A, B, f0, cutoff=cutoff))[:N-lmin_out,:N-lmin_arg]
            if m < 0:
                matrix = matrix[::-1, ::-1] #for negative m, ell's go in opposite direction
        return matrix.tocsr()


class ConvertConstantSphere(operators.ConvertConstant, operators.SeparableSphereOperator):

    output_basis_type = SphereBasis
    subaxis_dependence = [False, True]
    complex_operator = False

    def __init__(self, operand, output_basis, out=None):
        super().__init__(operand, output_basis, out=out)
        if self.coords in operand.tensorsig:
            raise ValueError("Tensors not yet supported.")

    def spinindex_out(self, spinindex_in):
        return (spinindex_in,)

    @staticmethod
    def symbol(spinindex_in, spinindex_out, spintotal_in, spintotal_out, ell, radius):
        unit_amplitude = 1 / SphereBasis.constant_mode_value
        return unit_amplitude * (ell == 0) * (spinindex_in == spinindex_out)


class SphereDivergence(operators.Divergence, operators.SeparableSphereOperator):
    """Divergence on S2."""

    cs_type = S2Coordinates
    input_basis_type = SphereBasis
    subaxis_dependence = [False, True]  # Depends on ell
    complex_operator = False

    def __init__(self, operand, index=0, out=None):
        operators.Divergence.__init__(self, operand, out=out)  # Gradient has no __init__
        if index != 0:
            raise ValueError("Divergence only implemented along index 0.")
        self.index = index
        coordsys = operand.tensorsig[index]
        self.coordsys = coordsys
        self.operand = operand
        self.input_basis = operand.domain.get_basis(coordsys)
        self.output_basis = self.input_basis
        self.first_axis = self.dist.first_axis(self.input_basis)
        self.last_axis = self.dist.last_axis(self.input_basis)
        # FutureField requirements
        self.domain  = operand.domain#.substitute_basis(self.input_basis, self.output_basis)
        self.tensorsig = operand.tensorsig[:index] + operand.tensorsig[index+1:]
        self.dtype = operand.dtype

    @staticmethod
    def _output_basis(input_basis):
        return input_basis

    def spinindex_out(self, spinindex_in):
        # Spinorder: -, +
        # Divergence feels - and +
        if spinindex_in[0] in (0, 1):
            return (spinindex_in[1:],)
        else:
            return tuple()

    @staticmethod
    def symbol(spinindex_in, spinindex_out, spintotal_in, spintotal_out, ell, radius):
        return SphereGradient.symbol(spinindex_in, spinindex_out, spintotal_in, spintotal_out, ell, radius)


class SphereGradient(operators.Gradient, operators.SeparableSphereOperator):
    """Gradient on S2."""

    cs_type = S2Coordinates
    input_basis_type = SphereBasis
    subaxis_dependence = [False, True]  # Depends on ell
    complex_operator = False

    def __init__(self, operand, coordsys, out=None):
        operators.Gradient.__init__(self, operand, out=out)  # Gradient has no __init__
        self.coordsys = coordsys
        self.operand = operand
        self.input_basis = operand.domain.get_basis(coordsys)
        self.output_basis = self.input_basis
        self.first_axis = self.dist.first_axis(self.input_basis)
        self.last_axis = self.dist.last_axis(self.input_basis)
        # FutureField requirements
        self.domain  = operand.domain#.substitute_basis(self.input_basis, self.output_basis)
        self.tensorsig = (coordsys,) + operand.tensorsig
        self.dtype = operand.dtype

    @staticmethod
    def _output_basis(input_basis):
        return input_basis

    def spinindex_out(self, spinindex_in):
        # Spinorder: -, +
        # Gradients hits - and +
        return ((0,) + spinindex_in, (1,) + spinindex_in)

    @staticmethod
    def symbol(spinindex_in, spinindex_out, spintotal_in, spintotal_out, ell, radius):
        mu = spintotal_out - spintotal_in
        k = SphereBasis.k(ell, spintotal_in, mu)
        k[np.abs(spintotal_in) > ell] = 0
        k[np.abs(spintotal_out) > ell] = 0
        return k / radius


class SphereLaplacian(operators.Laplacian, operators.SeparableSphereOperator):
    """Laplacian on S2."""

    cs_type = S2Coordinates
    input_basis_type = SphereBasis
    subaxis_dependence = [False, True]  # Depends on ell
    complex_operator = False

    def __init__(self, operand, coordsys, out=None):
        operators.Laplacian.__init__(self, operand, out=out)
        self.coordsys = coordsys
        self.operand = operand
        self.input_basis = operand.domain.get_basis(coordsys)
        self.output_basis = self.input_basis
        self.first_axis = self.dist.first_axis(self.input_basis)
        self.last_axis = self.dist.last_axis(self.input_basis)
        # FutureField requirements
        self.domain  = operand.domain#.substitute_basis(self.input_basis, self.output_basis)
        self.tensorsig = operand.tensorsig
        self.dtype = operand.dtype

    @staticmethod
    def _output_basis(input_basis):
        return input_basis

    def spinindex_out(self, spinindex_in):
        return (spinindex_in,)

    @staticmethod
    def symbol(spinindex_in, spinindex_out, spintotal_in, spintotal_out, ell, radius):
        k = SphereBasis.k
        kp = k(ell, spintotal_in, +1)
        km = k(ell, spintotal_in, -1)
        kp_1 = k(ell, spintotal_in-1, +1)
        km_1 = k(ell, spintotal_in+1, -1)
        k_lap = km_1*kp + kp_1*km
        k_lap[np.abs(spintotal_in) > ell] = 0
        k_lap[np.abs(spintotal_out) > ell] = 0
        return k_lap / radius**2


# These are common for BallRadialBasis and ShellRadialBasis
class RegularityBasis(SpinRecombinationBasis, MultidimensionalBasis):

    dim = 3
    dims = ['azimuth', 'colatitude', 'radius']
    subaxis_dependence = [False, False, True]

    def __init__(self, coordsys, radial_size, k, dealias, dtype):
        self.coordsys = coordsys
        self.radial_size = radial_size
        self.shape = (1, 1, radial_size)
        self.k = k
        if len(dealias) == 1:
            self.dealias = (1, 1) + dealias
        else:
            self.dealias = dealias
        self.Nmax = radial_size - 1
        self.dtype = dtype
        # Call at end because dealias is needed to build self.domain
        Basis.__init__(self, coordsys)
        if dtype == np.float64:
            self.group_shape = (2, 1, 1)
        elif dtype == np.complex128:
            self.group_shape = (1, 1, 1)

    @CachedAttribute
    def constant(self):
        return (True, True, False)

    def grid_shape(self, scales):
        grid_shape = list(super().grid_shape(scales))
        # Set constant directions back to size 1
        grid_shape[0] = 1
        grid_shape[1] = 1
        return tuple(grid_shape)

    def global_shape(self, grid_space, scales):
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

    def chunk_shape(self, grid_space):
        if grid_space[0]:
            # grid-grid-grid space
            return (1, 1, 1)
        else:
            if self.dtype == np.complex128:
                return (1, 1, 1)
            elif self.dtype == np.float64:
                return (2, 1, 1)

    def elements_to_groups(self, grid_space, elements):
        groups = elements.copy()
        if not grid_space[0]:
            # coeff-*-* space
            groups[0] = 0
        if not grid_space[2]:
            # coeff-coeff-coeff space
            m, ell, n = groups
            nmin = self._nmin(ell)
            groups = np.ma.masked_array(groups)
            groups[:, n < nmin] = np.ma.masked
        return groups

    @CachedMethod
    def ell_maps(self, dist):
        return SphereBasis.ell_maps(self, dist)

    def get_radial_basis(self):
        return self

    def global_grid(self, dist, scale):
        problem_grid = self._radius_grid(scale)
        radial_axis = dist.get_basis_axis(self) + 2
        return reshape_vector(problem_grid, dim=dist.dim, axis=radial_axis)

    def local_grid(self, dist, scale):
        radial_axis = dist.get_basis_axis(self) + 2
        local_elements = dist.grid_layout.local_elements(self.domain(dist), scales=scale)[radial_axis]
        problem_grid = self._radius_grid(scale)[local_elements]
        return reshape_vector(problem_grid, dim=dist.dim, axis=radial_axis)

    def global_weights(self, dist, scale=None):
        if scale == None: scale = 1
        radial_axis = dist.get_basis_axis(self) + 2
        weights = self._radius_weights(scale)
        return reshape_vector(weights.astype(np.float64), dim=dist.dim, axis=radial_axis)

    def local_weights(self, dist, scale=None):
        if scale == None: scale = 1
        radial_axis = dist.get_basis_axis(self) + 2
        local_elements = dist.grid_layout.local_elements(self.domain(dist), scales=scale)[radial_axis]
        weights = self._radius_weights(scale)
        return reshape_vector(weights.astype(np.float64)[local_elements], dim=dist.dim, axis=radial_axis)

    @CachedMethod
    def regularity_allowed(self,l,regularity):
        Rb = np.array([-1, 1, 0], dtype=int)
        if regularity == (): return True
        intertwiner = dedalus_sphere.spin_operators.Intertwiner(l, indexing=(-1,+1,0))
        return not intertwiner.forbidden_regularity(Rb[np.array(regularity)])

    @staticmethod
    @CachedMethod
    def regtotal(regindex):
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
            if self.coordsys is not cs:
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
            if self.coordsys is cs: # kludge before we decide how compound coordinate systems work
                R[axslice(i, 0, cs.dim)] += reshape_vector(Rb, dim=len(tensorsig), axis=i)
            #elif self.space in vs.spaces:
            #    n = vs.get_index(self.space)
            #    R[axslice(i, n, n+self.dim)] += reshape_vector(Rb, dim=len(tensorsig), axis=i)
        return R

    def regularity_indices(self, tensorsig):
        """Return array of regularity multiindeces."""
        indices = []
        tshape = [cs.dim for cs in tensorsig]
        for i, cs in enumerate(tensorsig):
            if self.coordsys is cs:
                index = np.zeros(tshape) + reshape_vector(np.array([-1, 1, 0]), dim=len(tshape), axis=i)
                indices.append(index)
        return np.stack(indices, axis=-1)

    def regularity_allowed_vectorized(self, l, regindex):
        valid = np.ones(l.shape, dtype=bool)
        walk = l
        for dr in regindex[::-1]:
            walk = walk + dr
            valid[walk < 0] = False
            if dr == 0:
                valid[walk == 0] = False
        return valid

    def forward_regularity_recombination(self, tensorsig, axis, gdata, ell_maps):
        rank = len(tensorsig)
        ell_list = tuple(map[0] for map in ell_maps)
        # Apply radial recombinations
        if rank > 0:
            Q = self.radial_recombinations(tensorsig, ell_list)
            # Flatten tensor axes
            shape = gdata.shape
            temp = gdata.reshape((np.prod(shape[:rank]),)+shape[rank:])
            slices = [slice(None) for i in range(temp.ndim)]
            # Apply Q transformations for each ell to flattened tensor data
            for ell, m_ind, ell_ind in ell_maps:
                slices[axis-2+1] = m_ind    # Add 1 for tensor axis
                slices[axis-1+1] = ell_ind  # Add 1 for tensor axis
                temp_ell = temp[tuple(slices)]
                apply_matrix(Q[ell].T, temp_ell, axis=0, out=temp_ell)

    def backward_regularity_recombination(self, tensorsig, axis, gdata, ell_maps):
        rank = len(tensorsig)
        ell_list = tuple(map[0] for map in ell_maps)
        # Apply radial recombinations
        if rank > 0:
            Q = self.radial_recombinations(tensorsig, ell_list)
            # Flatten tensor axes
            shape = gdata.shape
            temp = gdata.reshape((np.prod(shape[:rank]),)+shape[rank:])
            slices = [slice(None) for i in range(temp.ndim)]
            # Apply Q transformations for each ell to flattened tensor data
            for ell, m_ind, ell_ind in ell_maps:
                slices[axis-2+1] = m_ind    # Add 1 for tensor axis
                slices[axis-1+1] = ell_ind  # Add 1 for tensor axis
                temp_ell = temp[tuple(slices)]
                apply_matrix(Q[ell], temp_ell, axis=0, out=temp_ell)

    # def radial_vector_3(self, comp, m, ell, regindex, local_m=None, local_l=None):
    #     if local_m == None: local_m = self.local_m
    #     if local_l == None: local_l = self.local_l
    #     slices = self.radial_vector_slices(m, ell, regindex, local_m, local_l)
    #     if slices is None:
    #         return None
    #     comp5 = reduced_view(comp, axis=self.axis, dim=3)
    #     return comp5[(slice(None),) + slices + (slice(None),)]

    # @CachedMethod
    # def radial_vector_slices(self, m, ell, regindex, local_m, local_l):
    #     if m > ell:
    #         return None
    #     if not self.regularity_allowed(ell, regindex):
    #         return None
    #     mi = local_m.index(m)
    #     li = local_l.index(ell)
    #     return (mi, li, self.n_slice(ell))

    def forward_transform_azimuth(self, field, axis, gdata, cdata):
        # Copy over real part of m = 0
        data_axis = len(field.tensorsig) + axis
        np.copyto(cdata[axslice(data_axis, 0, 1)], gdata)
        cdata[axslice(data_axis, 1, None)] = 0

    def forward_transform_colatitude(self, field, axis, gdata, cdata):
        # Spin recombination
        temp = np.zeros_like(gdata)
        self.forward_spin_recombination(field.tensorsig, axis, gdata, temp)
        np.copyto(cdata, temp)

    def backward_transform_colatitude(self, field, axis, cdata, gdata):
        # Spin recombination
        temp = np.copy(cdata)
        self.backward_spin_recombination(field.tensorsig, axis, temp, gdata)

    def backward_transform_azimuth(self, field, axis, cdata, gdata):
        # Copy over real part of m = 0
        np.copyto(gdata, cdata[axslice(len(field.tensorsig)+axis, 0, 1)])

    @CachedMethod
    def n_size(self, ell):
        nmin = self._nmin(ell)
        nmax = self.Nmax
        return nmax - nmin + 1

    @CachedMethod
    def n_slice(self, ell):
        nmin = self._nmin(ell)
        nmax = self.Nmax
        return slice(nmin, nmax+1)


class ShellRadialBasis(RegularityBasis, metaclass=CachedClass):

    def __init__(self, coordsys, radial_size, dtype, radii=(1,2), alpha=(-0.5,-0.5), dealias=(1,), k=0, radius_library=None):
        super().__init__(coordsys, radial_size, k=k, dealias=dealias, dtype=dtype)
        if radii[0] <= 0:
            raise ValueError("Inner radius must be positive.")
        if radius_library is None:
            if alpha[0] == alpha[1] == -1/2:
                radius_library = Jacobi.default_dct
            else:
                radius_library = Jacobi.default_library
        self.radii = radii
        self.volume = 4 / 3 * np.pi * (radii[1]**3 - radii[0]**3)
        self.dR = self.radii[1] - self.radii[0]
        self.rho = (self.radii[1] + self.radii[0])/self.dR
        self.alpha = alpha
        self.radius_library = radius_library
        self.grid_params = (coordsys, radii, alpha, self.dealias)
        self.forward_transforms = [self.forward_transform_azimuth,
                                   self.forward_transform_colatitude,
                                   self.forward_transform_radius]
        self.backward_transforms = [self.backward_transform_azimuth,
                                    self.backward_transform_colatitude,
                                    self.backward_transform_radius]

    def __eq__(self, other):
        if isinstance(other, ShellRadialBasis):
            if self.coordsys == other.coordsys:
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
        if isinstance(other, ShellRadialBasis):
            if self.grid_params == other.grid_params:
                radial_size = max(self.shape[2], other.shape[2])
                k = max(self.k, other.k)
                return self.clone_with(radial_size=radial_size, k=k)
        return NotImplemented

    def __mul__(self, other):
        if other is None:
            return self
        if isinstance(other, ShellRadialBasis):
            if self.grid_params == other.grid_params:
                radial_size = max(self.shape[2], other.shape[2])
                k = self.k + other.k
                return self.clone_with(radial_size=radial_size, k=k)
        if isinstance(other, SphereBasis):
            unify((self.coordsys, other.coordsys))
            args = {}
            args['coordsys'] = self.coordsys
            args['shape'] = other.shape + self.shape[-1:] # Because ShellRadialBasis shape is padded up to 3d
            args['radii'] = self.radii
            args['alpha'] = self.alpha
            args['dealias'] = other.dealias + (self.dealias[-1],)
            args['k'] = self.k
            args['dtype'] = unify((self.dtype, other.dtype))
            args['azimuth_library'] = other.azimuth_library
            args['colatitude_library'] = other.colatitude_library
            args['radius_library'] = self.radius_library
            return ShellBasis(**args)
        return NotImplemented

    def __rmatmul__(self, other):
        # NCC (other) * operand (self)
        if other is None:
            return self
        if isinstance(other, ShellRadialBasis):
            if self.grid_params == other.grid_params:
                radial_size = max(self.shape[2], other.shape[2])
                k = self.k + other.k
                return self.clone_with(radial_size=radial_size, k=k)
        return NotImplemented

    def _new_k(self, k):
        return self.clone_with(k=k)

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

    @CachedAttribute
    def constant_mode_value(self):
        # Note the zeroth mode is constant only for k=0
        Q0 = dedalus_sphere.jacobi.polynomials(1, self.alpha[0], self.alpha[1], np.array([0.0]))
        return Q0[0,0]

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
    def transform_plan(self, dist, grid_size, k):
        """Build transform plan."""
        a = self.alpha[0] + k
        b = self.alpha[1] + k
        a0 = self.alpha[0]
        b0 = self.alpha[1]
        return Jacobi.transforms[self.radius_library](grid_size, self.Nmax+1, a, b, a0, b0)

    def forward_transform_radius(self, field, axis, gdata, cdata):
        data_axis = len(field.tensorsig) + axis
        grid_size = gdata.shape[data_axis]
        # Multiply by radial factor
        if self.k > 0:
            gdata *= self.radial_transform_factor(field.scales[axis], data_axis, -self.k)
        # Apply recombinations
        self.forward_regularity_recombination(field.tensorsig, axis, gdata, ell_maps=self.ell_maps(field.dist))
        temp = np.copy(gdata)
        # Perform radial transforms component-by-component
        R = self.regularity_classes(field.tensorsig)
        for regindex, regtotal in np.ndenumerate(R):
           plan = self.transform_plan(field.dist, grid_size, self.k)
           plan.forward(temp[regindex], cdata[regindex], axis)

    def backward_transform_radius(self, field, axis, cdata, gdata):
        data_axis = len(field.tensorsig) + axis
        grid_size = gdata.shape[data_axis]
        # Perform radial transforms component-by-component
        R = self.regularity_classes(field.tensorsig)
        # HACK -- don't want to make a new array every transform
        temp = np.zeros_like(gdata)
        for i, r in np.ndenumerate(R):
           plan = self.transform_plan(field.dist, grid_size, self.k)
           plan.backward(cdata[i], temp[i], axis)
        np.copyto(gdata, temp)
        # Regularity recombination
        self.backward_regularity_recombination(field.tensorsig, axis, gdata, self.ell_maps(field.dist))
        # Multiply by radial factor
        if self.k > 0:
            gdata *= self.radial_transform_factor(field.scales[axis], data_axis, self.k)

    @CachedMethod
    def operator_matrix(self, op, l, regtotal, size=None):
        l = l + regtotal
        if op in ['D+', 'D-']:
            p = int(op[-1]+'1')
            operator = dedalus_sphere.shell.operator(3, self.radii, 'D', self.alpha)(p, l)
        elif op == 'L':
            D = dedalus_sphere.shell.operator(3, self.radii, 'D', self.alpha)
            operator = D(-1, l+1) @ D(+1, l)
        else:
            operator = dedalus_sphere.shell.operator(3, self.radii, op, self.alpha)
        if size is None:
            size = self.n_size(l)
        return operator(size, self.k).square.astype(np.float64)

    def jacobi_conversion(self, l, dk, size=None):
        AB = dedalus_sphere.shell.operator(3, self.radii, 'AB', self.alpha)
        operator = AB**dk
        if size is None:
            size = self.n_size(l)
        return operator(size, self.k).square.astype(np.float64)

    @CachedMethod
    def conversion_matrix(self, l, regtotal, dk):
        E = dedalus_sphere.shell.operator(3, self.radii, 'E', self.alpha)
        operator = E**dk
        return operator(self.n_size(l), self.k).square.astype(np.float64)

    @staticmethod
    def _nmin(ell):
        return 0 * ell  # To have same array shape as ell

    @classmethod
    def _last_axis_component_ncc_matrix(cls, subproblem, ncc_basis, arg_basis, out_basis, coeffs, ncc_comp, arg_comp, out_comp, ncc_tensorsig, arg_tensorsig, out_tensorsig, cutoff=1e-6):
        ell = 0  # HACK, independent of ell for shell
        arg_radial_basis = arg_basis.radial_basis
        regtotal_arg = cls.regtotal(arg_comp)
        # Jacobi parameters
        a_ncc = ncc_basis.k + ncc_basis.alpha[0]
        b_ncc = ncc_basis.k + ncc_basis.alpha[1]
        N = ncc_basis.n_size(ell)
        N0 = ncc_basis.n_size(0)
        # Pad for dealiasing with conversion
        Nmat = 3*((N0+1)//2) + ncc_basis.k
        J = arg_radial_basis.operator_matrix('Z', ell, regtotal_arg, size=Nmat)
        A, B = clenshaw.jacobi_recursion(Nmat, a_ncc, b_ncc, J)
        f0 = dedalus_sphere.jacobi.polynomials(1, a_ncc, b_ncc, 1)[0] * sparse.identity(Nmat)
        # Conversions to account for radial prefactors
        prefactor = arg_radial_basis.jacobi_conversion(ell, dk=ncc_basis.k, size=Nmat)
        if ncc_basis.dtype == np.float64:
            coeffs_cos, coeffs_msin = coeffs
            if not np.isscalar(coeffs_cos[0]):
                raise NotImplementedError("Recursive Clenshaw not finished for float64.")
            matrix_cos = (prefactor @ clenshaw.kronecker_clenshaw(coeffs_cos, None, A, B, f0, cutoff=cutoff))[:N,:N]
            matrix_msin = (prefactor @ clenshaw.kronecker_clenshaw(coeffs_msin, None, A, B, f0, cutoff=cutoff))[:N,:N]
            matrix = sparse.bmat([[matrix_cos, -matrix_msin], [matrix_msin, matrix_cos]], format='csr')
        elif ncc_basis.dtype == np.complex128:
            if np.isscalar(coeffs[0]):
                matrix = (prefactor @ clenshaw.matrix_clenshaw(coeffs, A, B, f0, cutoff=cutoff))[:N,:N]
            else:
                coeff_vals, coeff_norms = coeffs
                i0, i1 = coeff_vals[0].shape
                I0 = sparse.identity(i0)
                I1 = sparse.identity(i1)
                matrix = sparse.kron(I0, prefactor) @ clenshaw.kronecker_clenshaw(coeff_vals, coeff_norms, A, B, f0, cutoff=cutoff)
                dealias0 = sparse.kron(I0, sparse.eye(N, Nmat))
                dealias1 = sparse.kron(I1, sparse.eye(N, Nmat))
                matrix = dealias0 @ matrix @ dealias1.T
        return matrix


class BallRadialBasis(RegularityBasis, metaclass=CachedClass):

    transforms = {}

    def __init__(self, coordsys, radial_size, dtype, radius=1, k=0, alpha=0, dealias=(1,), radius_library=None):
        super().__init__(coordsys, radial_size, k=k, dealias=dealias, dtype=dtype)
        if radius <= 0:
            raise ValueError("Radius must be positive.")
        if radius_library is None:
            radius_library = "matrix"
        self.radius = radius
        self.volume = 4 / 3 * np.pi * radius**3
        self.alpha = alpha
        self.radial_COV = AffineCOV((0, 1), (0, radius))
        self.radius_library = radius_library
        self.grid_params = (coordsys, radius, alpha, self.dealias)
        self.forward_transforms = [self.forward_transform_azimuth,
                                   self.forward_transform_colatitude,
                                   self.forward_transform_radius]
        self.backward_transforms = [self.backward_transform_azimuth,
                                    self.backward_transform_colatitude,
                                    self.backward_transform_radius]

    def __eq__(self, other):
        if isinstance(other, BallRadialBasis):
            if self.coordsys == other.coordsys:
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
                return self.clone_with(radial_size=radial_size, k=k)
        return NotImplemented

    def __mul__(self, other):
        if other is None:
            return self
        if isinstance(other, BallRadialBasis):
            if self.grid_params == other.grid_params:
                radial_size = max(self.shape[2], other.shape[2])
                k = max(self.k, other.k)
                return self.clone_with(radial_size=radial_size, k=k)
        return NotImplemented

    def __rmatmul__(self, other):
        # NCC (other) * operand (self)
        if other is None:
            return self
        if isinstance(other, BallRadialBasis):
            if self.grid_params == other.grid_params:
                radial_size = max(self.shape[2], other.shape[2])
                k = max(self.k, other.k)
                return self.clone_with(radial_size=radial_size, k=k)
        return NotImplemented

    def _new_k(self, k):
        return self.clone_with(k=k)

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

    @CachedAttribute
    def constant_mode_value(self):
        Qk = dedalus_sphere.zernike.polynomials(3, 1, self.alpha+self.k, 0, np.array([0]))
        return Qk[0,0]

    @CachedMethod
    def interpolation(self, ell, regtotal, position):
        native_position = self.radial_COV.native_coord(position)
        native_z = 2*native_position**2 - 1
        return dedalus_sphere.zernike.polynomials(3, self.n_size(ell), self.alpha + self.k, ell + regtotal, native_z)

    @CachedMethod
    def transform_plan(self, dist, grid_shape, regindex, axis, regtotal, k, alpha):
        """Build transform plan."""
        return self.transforms[self.radius_library](grid_shape, self.Nmax+1, axis, self.ell_maps(dist), regindex, regtotal, k, alpha)

    def forward_transform_radius(self, field, axis, gdata, cdata):
        # Apply recombination
        self.forward_regularity_recombination(field.tensorsig, axis, gdata, ell_maps=self.ell_maps(field.dist))
        # Perform radial transforms component-by-component
        R = self.regularity_classes(field.tensorsig)
        temp = np.copy(gdata)
        for regindex, regtotal in np.ndenumerate(R):
           grid_shape = gdata[regindex].shape
           plan = self.transform_plan(field.dist, grid_shape, regindex, axis, regtotal, self.k, self.alpha)
           plan.forward(temp[regindex], cdata[regindex], axis)

    def backward_transform_radius(self, field, axis, cdata, gdata):
        # Perform radial transforms component-by-component
        R = self.regularity_classes(field.tensorsig)
        # HACK -- don't want to make a new array every transform
        temp = np.zeros_like(gdata)
        for regindex, regtotal in np.ndenumerate(R):
           grid_shape = gdata[regindex].shape
           plan = self.transform_plan(field.dist, grid_shape, regindex, axis, regtotal, self.k, self.alpha)
           plan.backward(cdata[regindex], temp[regindex], axis)
        np.copyto(gdata, temp)
        # Apply recombinations
        self.backward_regularity_recombination(field.tensorsig, axis, gdata, self.ell_maps(field.dist))

    @CachedMethod
    def operator_matrix(self, op, l, deg, size=None):
        if op[-1] in ['+', '-']:
            o = op[:-1]
            p = int(op[-1]+'1')
            operator = dedalus_sphere.zernike.operator(3, o, radius=self.radius)(p)
        elif op == 'L':
            D = dedalus_sphere.zernike.operator(3, 'D', radius=self.radius)
            operator = D(-1) @ D(+1)
        else:
            operator = dedalus_sphere.zernike.operator(3, op, radius=self.radius)
        if size is None:
            size = self.n_size(l)
        return operator(size, self.alpha + self.k, l + deg).square.astype(np.float64)

    @CachedMethod
    def conversion_matrix(self, ell, regtotal, dk, size=None):
        E = dedalus_sphere.zernike.operator(3, 'E', radius=self.radius)
        operator = E(+1)**dk
        if size is None:
            size = self.n_size(ell)
        return operator(size, self.alpha + self.k, ell + regtotal).square.astype(np.float64)

    @CachedMethod
    def radius_multiplication_matrix(self, ell, regtotal, order, d, size=None):
        if order == 0:
            operator = dedalus_sphere.zernike.operator(3, 'Id', radius=self.radius)
        else:
            R = dedalus_sphere.zernike.operator(3, 'R', radius=1)
            if order < 0:
                operator = R(-1)**abs(order)
            else: # order > 0
                operator = R(+1)**abs(order)
        if d > 0:
            R = dedalus_sphere.zernike.operator(3, 'R', radius=1)
            R2 = R(-1) @ R(+1)
            operator = R2**(d//2) @ operator
        if size is None:
            size = self.n_size(ell)
        return operator(size, self.alpha + self.k, ell + regtotal).square.astype(np.float64)

    @staticmethod
    def _nmin(ell):
        return ell // 2

    @classmethod
    def _last_axis_component_ncc_matrix(cls, subproblem, ncc_basis, arg_basis, out_basis, coeffs, ncc_comp, arg_comp, out_comp, ncc_tensorsig, arg_tensorsig, out_tensorsig, cutoff=1e-6):
        ell = subproblem.group[1]  # HACK
        if isinstance(arg_basis, BallBasis):
            arg_radial_basis = arg_basis.radial_basis
        elif isinstance(arg_basis, BallRadialBasis):
            arg_radial_basis = arg_basis
        elif arg_basis is None:
            # Reshape coeffs as column vector
            matrix = coeffs.ravel()[:, None]
            return sparse.csr_matrix(matrix)
        else:
            raise NotImplementedError()
        regtotal_ncc = cls.regtotal(ncc_comp)
        regtotal_arg = cls.regtotal(arg_comp)
        regtotal_out = cls.regtotal(out_comp)
        diff_regtotal = regtotal_out - regtotal_arg
        # jacobi parameters
        a_ncc = ncc_basis.alpha + ncc_basis.k
        b_ncc = regtotal_ncc + 1/2
        N = ncc_basis.n_size(ell)
        N0 = ncc_basis.n_size(0)
        d = regtotal_ncc - abs(diff_regtotal)
        dk = max(ncc_basis.k, arg_radial_basis.k) - arg_radial_basis.k
        # Pad for dealiasing with conversion
        Nmat = 3*((N0+1)//2) + (dk+1)//2
        if (d >= 0) and (d % 2 == 0):
            J = arg_radial_basis.operator_matrix('Z', ell, regtotal_arg, size=Nmat)
            A, B = clenshaw.jacobi_recursion(N0, a_ncc, b_ncc, J)
            f0 = dedalus_sphere.zernike.polynomials(3, 1, a_ncc, regtotal_ncc, 1)[0] * sparse.identity(Nmat)
            radial_factor = arg_radial_basis.radius_multiplication_matrix(ell, regtotal_arg, diff_regtotal, d, size=Nmat)
            conversion = arg_radial_basis.conversion_matrix(ell, regtotal_out, dk, size=Nmat)
            prefactor = conversion @ radial_factor
            if ncc_basis.dtype == np.float64:
                coeffs_cos_filter = coeffs[0].ravel()[:N0]
                coeffs_msin_filter = coeffs[1].ravel()[:N0]
                matrix_cos = (prefactor @ clenshaw.matrix_clenshaw(coeffs_cos_filter, A, B, f0, cutoff=cutoff))[:N, :N]
                matrix_msin = (prefactor @ clenshaw.matrix_clenshaw(coeffs_msin_filter, A, B, f0, cutoff=cutoff))[:N, :N]
                matrix = sparse.bmat([[matrix_cos, -matrix_msin], [matrix_msin, matrix_cos]], format='csr')
            elif ncc_basis.dtype == np.complex128:
                coeffs_filter = coeffs.ravel()[:N0]
                matrix = (prefactor @ clenshaw.matrix_clenshaw(coeffs_filter, A, B, f0, cutoff=cutoff))[:N, :N]
        else:
            if ncc_basis.dtype == np.float64:
                matrix = sparse.csr_matrix((2*N, 2*N), dtype=np.float64)
            elif ncc_basis.dtype == np.complex128:
                matrix = sparse.csr_matrix((N, N), dtype=np.complex128)
        return matrix


class Spherical3DBasis(MultidimensionalBasis):

    dim = 3
    dims = ['azimuth', 'colatitude', 'radius']
    subaxis_dependence = [False, True, True]

    def __init__(self, coordsys, shape_angular, dealias_angular, radial_basis, dtype, azimuth_library=None, colatitude_library=None):
        self.coordsys = coordsys
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
        self.azimuth_basis = self.sphere_basis.azimuth_basis
        self.mmax = self.sphere_basis.mmax
        self.Lmax = self.sphere_basis.Lmax
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
        self.forward_transform_azimuth = self.sphere_basis.forward_transform_azimuth
        self.backward_transform_azimuth = self.sphere_basis.backward_transform_azimuth
        self.forward_transform_colatitude = self.sphere_basis.forward_transform_colatitude
        self.backward_transform_colatitude = self.sphere_basis.backward_transform_colatitude
        if dtype == np.float64:
            self.group_shape = (2, 1, 1)
        elif dtype == np.complex128:
            self.group_shape = (1, 1, 1)
        Basis.__init__(self, coordsys)

    @CachedMethod
    def ell_reversed(self, dist):
        return self.S2_basis().ell_reversed(dist)

    def matrix_dependence(self, matrix_coupling):
        matrix_dependence = matrix_coupling.copy()
        if matrix_coupling[1]:
            matrix_dependence[0] = True
        return matrix_dependence

    @CachedAttribute
    def constant(self):
        return (self.shape[0]==1, self.shape[1]==1, False)

    def derivative_basis(self, order=1):
        k = self.k + order
        return self.clone_with(k=k)

    @CachedAttribute
    def constant_mode_value(self):
        # Adjust for SWSH normalization
        # TODO: check this is right for regtotal != 0?
        return self.radial_basis.constant_mode_value / np.sqrt(2)

    def global_grids(self, dist, scales):
        return (self.global_grid_azimuth(dist, scales[0]),
                self.global_grid_colatitude(dist, scales[1]),
                self.global_grid_radius(dist, scales[2]))

    def local_grids(self, dist, scales):
        return (self.local_grid_azimuth(dist, scales[0]),
                self.local_grid_colatitude(dist, scales[1]),
                self.local_grid_radius(dist, scales[2]))

    @CachedMethod
    def global_grid_spacing(self, dist, subaxis, scales):
        """Global grids spacings."""
        grid = self.global_grids(dist, scales=scales)[subaxis]
        if grid.size == 1:
            return np.array([np.inf,], dtype=grid.dtype)
        else:
            axis = dist.first_axis(self) + subaxis
            return np.gradient(grid, axis=axis, edge_order=2)

    @CachedMethod
    def local_grid_spacing(self, dist, subaxis, scales):
        """Local grids spacings."""
        axis = dist.first_axis(self) + subaxis
        global_spacing = self.global_grid_spacing(dist, subaxis, scales=scales)
        local_elements = dist.grid_layout.local_elements(self.domain(dist), scales=scales[subaxis])[axis]
        return reshape_vector(np.ravel(global_spacing)[local_elements], dim=dist.dim, axis=axis)

    def local_elements(self):
        raise NotImplementedError()
        # CL = self.dist.coeff_layout
        # LE = CL.local_elements(self.domain, scales=1)[self.axis:self.axis+self.dim]
        # LE[0] = np.array(self.local_m)
        # return tuple(LE)

    def get_radial_basis(self):
        return self.radial_basis

    def S2_basis(self, radius=None):
        if radius is None:
            if hasattr(self, "radius"):
                radius = self.radius
            else:
                radius = max(self.radii)
        return SphereBasis(self.coordsys, self.shape[:2], self.dtype, radius=radius, dealias=self.dealias[:2],
                    azimuth_library=self.azimuth_library, colatitude_library=self.colatitude_library)

    @CachedMethod
    def operator_matrix(self, op, l, regtotal, dk=0, size=None):
        return self.radial_basis.operator_matrix(op, l, regtotal, size=size)

    @CachedMethod
    def conversion_matrix(self, l, regtotal, dk):
        return self.radial_basis.conversion_matrix(l, regtotal, dk)

    def n_size(self, ell):
        return self.radial_basis.n_size(ell)

    def n_slice(self, ell):
        return self.radial_basis.n_slice(ell)

    def global_shape(self, grid_space, scales):
        grid_shape = self.grid_shape(scales)
        s2_shape = self.sphere_basis.global_shape(grid_space, scales)
        if grid_space[2]:
            # *-*-grid space
            radial_shape = (grid_shape[2],)
        else:
            # coeff-coeff-coeff space
            radial_shape = (self.shape[2],)
        return s2_shape + radial_shape

    def chunk_shape(self, grid_space):
        s2_chunk = self.sphere_basis.chunk_shape(grid_space)
        return s2_chunk + (1,)

    def valid_elements(self, tensorsig, grid_space, elements):
        if grid_space[2]:
            # ggg, cgg, ccg
            # Same as Sphere validity before regularity recombination
            return self.S2_basis().valid_elements(tensorsig, grid_space, elements)
        else:
            # coeff-coeff-coeff
            m, l, n = self.elements_to_groups(grid_space, elements)
            tshape = tuple(cs.dim for cs in tensorsig)
            valid = np.ones(tshape + m.shape, dtype=bool)
            # Drop disallowed regularity components
            if tensorsig:
                regindices = self.radial_basis.regularity_indices(tensorsig)
                for regcomp in np.ndindex(tshape):
                    regindex = regindices[regcomp]
                    valid[regcomp] = self.radial_basis.regularity_allowed_vectorized(l, regindex)
            # Drop msin part of ell == 0 for real scalars and vectors
            # Does not impose m == 0 symmetry for ell > 0 or tensors
            if self.dtype == np.float64:
                if len(tensorsig) == 0:
                    valid[(l == 0) * (elements[0] % 2 == 1)] = False
                elif len(tensorsig) == 1:
                    valid[:, (l == 0) * (elements[0] % 2 == 1)] = False
            return valid

    def elements_to_groups(self, grid_space, elements):
        s2_groups = self.sphere_basis.elements_to_groups(grid_space, elements[:2])
        radial_groups = elements[2]
        groups = np.array([*s2_groups, radial_groups])
        if not grid_space[2]:
            # coeff-coeff-coeff space
            m, ell, n = groups
            nmin = self.radial_basis._nmin(ell)
            groups = np.ma.masked_array(groups)
            groups[:, n < nmin] = np.ma.masked
        return groups

    # def valid_elements(self, *args):
    #     # Implemented in RegularityBasis
    #     return self.radial_basis.valid_elements(*args)

    @classmethod
    def _last_axis_component_ncc_matrix(cls, subproblem, ncc_basis, arg_basis, out_basis, coeffs, *args, **kw):
        if isinstance(ncc_basis, RegularityBasis):
            return ncc_basis._last_axis_component_ncc_matrix(subproblem, ncc_basis, arg_basis, out_basis, coeffs, *args, **kw)
        elif ncc_basis.shape[0:2] == (1, 1):
            # Scale to account for SWSH normalization? Is this right for all spins?
            coeffs = coeffs / np.sqrt(2) # Need to add this back in once nccs can be non-radial bases?
            return ncc_basis.radial_basis._last_axis_component_ncc_matrix(subproblem, ncc_basis, arg_basis, out_basis, coeffs, *args, **kw)
        else:
            raise ValueError("Cannot build NCCs of non-radial fields.")


class ShellBasis(Spherical3DBasis, metaclass=CachedClass):

    @classmethod
    def _preprocess_cache_args(cls, coordsys, shape, dtype, radii, k, alpha, dealias, azimuth_library, colatitude_library, radius_library):
        """Preprocess arguments into canonical form for caching. Must accept and return __init__ arguments."""
        # coordsys: SphericalCoordinates
        if not isinstance(coordsys, SphericalCoordinates):
            raise ValueError("Shell coordsys must be SphericalCoordinates.")
        # shape: length-3 tuple
        shape = tuple(shape)
        if len(shape) != 3:
            raise ValueError("Shell shape must have length 3.")
        # radii: length-2 tuple of increasing positive numbers
        radii = tuple(radii)
        if len(radii) != 2:
            raise ValueError("Shell radii must have length 2")
        if min(radii) <= 0:
            raise ValueError("Shell radii must be positive.")
        if radii[0] >= radii[1]:
            raise ValueError("Shell radii must be in increasing order.")
        # k: int
        k = int(k)
        # alpha: length-2 tuple
        if isinstance(alpha, (int, float)):
            alpha = (alpha,) * 2
        else:
            alpha = tuple(alpha)
        if len(alpha) != 2:
            raise ValueError("Shell alpha must have length 2.")
        # dealias: length-3 tuple
        if isinstance(dealias, (int, float)):
            dealias = (dealias,) * 3
        else:
            dealias = tuple(dealias)
        if len(dealias) != 3:
            raise ValueError("Shell dealias must have length 3.")
        # azimuth_library: pick default
        if azimuth_library is None:
            azimuth_library = RealFourier.default_library
        # colatitude_library: pick default
        if colatitude_library is None:
            colatitude_library = SphereBasis.default_library
        # radius_library: pick default based on alpha
        if radius_library is None:
            if alpha[0] == alpha[1] == -1/2:
                radius_library = Jacobi.default_dct
            else:
                radius_library = Jacobi.default_library
        return (coordsys, shape, dtype, radii, k, alpha, dealias, azimuth_library, colatitude_library, radius_library)

    def __init__(self, coordsys, shape, dtype, radii=(1,2), k=0, alpha=(-0.5,-0.5), dealias=(1,1,1), azimuth_library=None, colatitude_library=None, radius_library=None):
        # Save arguments without modification for caching
        self.coordsys = coordsys
        self.shape = shape
        self.dtype = dtype
        self.radii = radii
        self.k = k
        self.alpha = alpha
        self.dealias = dealias
        self.azimuth_library = azimuth_library
        self.colatitude_library = colatitude_library
        self.radius_library = radius_library
        # Other attributes
        self.volume = 4 / 3 * np.pi * (radii[1]**3 - radii[0]**3)
        self.radius_library = radius_library
        self.radial_basis = ShellRadialBasis(coordsys, shape[2], radii=radii, alpha=alpha, dealias=(dealias[2],), k=k, dtype=dtype, radius_library=radius_library)
        Spherical3DBasis.__init__(self, coordsys, shape[:2], dealias[:2], self.radial_basis, dtype=dtype, azimuth_library=azimuth_library, colatitude_library=colatitude_library)
        self.grid_params = (coordsys, dtype, radii, alpha, dealias, azimuth_library, colatitude_library, radius_library)
#        self.forward_transform_radius = self.radial_basis.forward_transform
#        self.backward_transform_radius = self.radial_basis.backward_transform
        self.forward_transforms = [self.forward_transform_azimuth,
                                   self.forward_transform_colatitude,
                                   self.forward_transform_radius]
        self.backward_transforms = [self.backward_transform_azimuth,
                                    self.backward_transform_colatitude,
                                    self.backward_transform_radius]
        self.inner_surface = self.S2_basis(radius=radii[0])
        self.outer_surface = self.S2_basis(radius=radii[1])

    def __eq__(self, other):
        if isinstance(other, ShellBasis):
            if self.coordsys == other.coordsys:
                if self.grid_params == other.grid_params:
                    if self.k == other.k:
                        return True
        return False

    def __hash__(self):
        return id(self)

    def __add__(self, other):
        if other is None or other is self:
            return self
        if isinstance(other, ShellBasis):
            if self.grid_params == other.grid_params:
                # Everything matches except shape and k
                shape = tuple(np.maximum(self.shape, other.shape))
                k = max(self.k, other.k)  # minimizes conversions
                return self.clone_with(shape=shape, k=k)
        return NotImplemented

    def __mul__(self, other):
        if other is None:
            return self
        if isinstance(other, ShellBasis):
            if self.grid_params == other.grid_params:
                # Everything matches except shape and k
                shape = tuple(np.maximum(self.shape, other.shape))
                k = (self.radial_basis * other.radial_basis).k  # TODO: inline this
                return self.clone_with(shape=shape, k=k)
        if isinstance(other, ShellRadialBasis):
            k = (other * self.radial_basis).k  # TODO: inline this
            return self.clone_with(k=k)
        return NotImplemented

    def __rmatmul__(self, other):
        # NCC (other) * operand (self)
        if other is None:
            return self
        if isinstance(other, ShellRadialBasis):
            radial_basis = other @ self.radial_basis
            return self._new_k(radial_basis.k)
        if isinstance(other, ShellBasis):
            if other.shape[0] != 1:
                raise ValueError("Azimuthal NCCs not yet supported.")
            radial_basis = other.radial_basis @ self.radial_basis
            return self._new_k(radial_basis.k)
        return NotImplemented

    def _new_k(self, k):
        # TODO: replace this in all operators
        return self.clone_with(k=k)

    @property
    def meridional_basis(self):
        meridional_shape = (1,) + self.shape[1:]
        return self.clone_with(shape=meridional_shape)

    def forward_transform_radius(self, field, axis, gdata, cdata):
        radial_basis = self.radial_basis
        data_axis = len(field.tensorsig) + axis
        grid_size = gdata.shape[data_axis]
        # Multiply by radial factor
        if self.k > 0:
            gdata *= radial_basis.radial_transform_factor(field.scales[axis], data_axis, -self.k)
        # Apply regularity recombination using 3D ell map
        radial_basis.forward_regularity_recombination(field.tensorsig, axis, gdata, ell_maps=self.ell_maps(field.dist))
        # Perform radial transforms component-by-component
        R = radial_basis.regularity_classes(field.tensorsig)
        # HACK -- don't want to make a new array every transform
        temp = np.copy(cdata)
        for regindex, regtotal in np.ndenumerate(R):
           plan = radial_basis.transform_plan(field.dist, grid_size, self.k)
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
           plan = radial_basis.transform_plan(field.dist, grid_size, self.k)
           plan.backward(cdata[i], temp[i], axis)
        np.copyto(gdata, temp)
        # Apply regularity recombinations using 3D ell map
        radial_basis.backward_regularity_recombination(field.tensorsig, axis, gdata, ell_maps=self.ell_maps(field.dist))
        # Multiply by radial factor
        if self.k > 0:
            gdata *= radial_basis.radial_transform_factor(field.scales[axis], data_axis, self.k)

    def build_ncc_matrix(self, product, subproblem, ncc_cutoff, max_ncc_terms):
        axis = product.dist.last_axis(self)
        ncc_basis = product.ncc.domain.get_basis(axis)
        if ncc_basis is None:
            # NCC is constant
            return super().build_ncc_matrix(product, subproblem, ncc_cutoff, max_ncc_terms)
        elif ncc_basis.shape[0] == 1:
            if ncc_basis.shape[1] == 1:
                # NCC is radial
                return super().build_ncc_matrix(product, subproblem, ncc_cutoff, max_ncc_terms)
            else:
                # NCC is meridional
                return self.build_meridional_ncc_matrix(product, subproblem, ncc_cutoff, max_ncc_terms)
        else:
            raise NotImplementedError("Azimuthal NCCs not yet supported.")

    def build_meridional_ncc_matrix(self, product, subproblem, ncc_cutoff, max_ncc_terms):
        axis = product.dist.last_axis(self)
        ncc_basis = product.ncc.domain.get_basis(axis)
        arg_basis = product.operand.domain.get_basis(axis)
        out_basis = product.domain.get_basis(axis)
        # Build coeff_norm function that returns same result for all components
        coeffs = product._ncc_data
        # Take Frobenius norm of tensor components
        tensor_axes = tuple(range(len(product.ncc.tensorsig)))
        subcoeff_norms = np.sum(np.abs(coeffs)**2, axis=tensor_axes)**0.5
        # Sum over sin and cos for real
        subcoeff_norms = np.sum(subcoeff_norms**2, axis=0)**0.5
        # Take max over ell
        subcoeff_norms = np.max(subcoeff_norms, axis=0)
        # Convert NCC coeffs to spin components
        spin_coeffs = coeffs.copy()
        self.radial_basis.backward_regularity_recombination(product.ncc.tensorsig, axis, spin_coeffs, ell_maps=ncc_basis.ell_maps(product.dist))
        # Build deferred regcomp S2 NCC
        S2_basis = self.S2_basis()
        def reg_NCC_matrix(radial_index):
            # Select radial spin data
            subcoeffs = spin_coeffs[..., radial_index]
            # Call S2 NCC
            submatrix = S2_basis._last_axis_field_ncc_matrix(product, subproblem, axis-1, ncc_basis.S2_basis(), arg_basis.S2_basis(), out_basis.S2_basis(), subcoeffs, ncc_cutoff, max_ncc_terms)
            # Apply forward Q transformations
            m = subproblem.group[axis-2]
            ells = np.arange(abs(m), self.Lmax+1)
            if S2_basis.ell_reversed(product.dist)[m]:
                ells = ells[::-1]
            ells = tuple(ells)
            Qout = self.radial_basis.radial_recombinations(product.tensorsig, ells)
            Qout = interleave_matrices([Qout[ell].T for ell in ells])
            Qarg = self.radial_basis.radial_recombinations(product.operand.tensorsig, ells)
            Qarg = interleave_matrices([Qarg[ell].T for ell in ells])
            return Qout @ submatrix @ Qarg.T
        subcoeff_vals = DeferredTuple(reg_NCC_matrix, size=len(subcoeff_norms))
        # Call last axis Clenshaw via ShellRadialBasis
        subcoeffs = (subcoeff_vals, subcoeff_norms)
        ncc_comp = arg_comp = out_comp = ncc_tensorsig = arg_tensorsig = out_tensorsig = tuple()
        return self.radial_basis._last_axis_component_ncc_matrix(subproblem, ncc_basis, arg_basis, out_basis, subcoeffs, ncc_comp, arg_comp, out_comp, ncc_tensorsig, arg_tensorsig, out_tensorsig, cutoff=ncc_cutoff)


class BallBasis(Spherical3DBasis, metaclass=CachedClass):

    transforms = {}
    default_library = "matrix"

    @classmethod
    def _preprocess_cache_args(cls, coordsys, shape, dtype, radius, k, alpha, dealias, azimuth_library, colatitude_library, radius_library):
        """Preprocess arguments into canonical form for caching. Must accept and return __init__ arguments."""
        # coordsys: SphericalCoordinates
        if not isinstance(coordsys, SphericalCoordinates):
            raise ValueError("Ball coordsys must be SphericalCoordinates.")
        # shape: length-3 tuple
        shape = tuple(shape)
        if len(shape) != 3:
            raise ValueError("Ball shape must have length 3.")
        # radius: positive number
        if radius <= 0:
            raise ValueError("Ball radius must be positive.")
        # k: int
        k = int(k)
        # alpha
        # dealias: length-3 tuple
        if isinstance(dealias, (int, float)):
            dealias = (dealias,) * 3
        else:
            dealias = tuple(dealias)
        if len(dealias) != 3:
            raise ValueError("Ball dealias must have length 3.")
        # azimuth_library: pick default
        if azimuth_library is None:
            azimuth_library = RealFourier.default_library
        # colatitude_library: pick default
        if colatitude_library is None:
            colatitude_library = SphereBasis.default_library
        # radius_library: pick default
        if radius_library is None:
            radius_library = cls.default_library
        return (coordsys, shape, dtype, radius, k, alpha, dealias, azimuth_library, colatitude_library, radius_library)

    def __init__(self, coordsys, shape, dtype, radius=1, k=0, alpha=0, dealias=(1,1,1), azimuth_library=None, colatitude_library=None, radius_library=None):
        # Save arguments without modification for caching
        self.coordsys = coordsys
        self.shape = shape
        self.dtype = dtype
        self.radius = radius
        self.k = k
        self.alpha = alpha
        self.dealias = dealias
        self.azimuth_library = azimuth_library
        self.colatitude_library = colatitude_library
        self.radius_library = radius_library
        # Other attributes
        self.volume = 4 / 3 * np.pi * radius**3
        self.radial_basis = BallRadialBasis(coordsys, shape[2], radius=radius, k=k, alpha=alpha, dealias=(dealias[2],), dtype=dtype, radius_library=radius_library)
        Spherical3DBasis.__init__(self, coordsys, shape[:2], dealias[:2], self.radial_basis, dtype=dtype, azimuth_library=azimuth_library, colatitude_library=colatitude_library)
        self.grid_params = (coordsys, dtype, radius, alpha, dealias, azimuth_library, colatitude_library, radius_library)
        self.forward_transforms = [self.forward_transform_azimuth,
                                   self.forward_transform_colatitude,
                                   self.forward_transform_radius]
        self.backward_transforms = [self.backward_transform_azimuth,
                                    self.backward_transform_colatitude,
                                    self.backward_transform_radius]
        self.surface = self.S2_basis(radius=radius)

    def __eq__(self, other):
        if isinstance(other, BallBasis):
            if self.coordsys == other.coordsys:
                if self.grid_params == other.grid_params:
                    if self.k == other.k:
                        return True
        return False

    def __hash__(self):
        return id(self)

    def __add__(self, other):
        if other is None or other is self:
            return self
        if isinstance(other, BallBasis):
            if self.grid_params == other.grid_params:
                # Everything matches except shape and k
                shape = tuple(np.maximum(self.shape, other.shape))
                k = max(self.k, other.k)  # minimizes conversions
                return self.clone_with(shape=shape, k=k)
        return NotImplemented

    def __mul__(self, other):
        if other is None:
            return self
        if isinstance(other, BallBasis):
            if self.grid_params == other.grid_params:
                # Everything matches except shape and k
                shape = tuple(np.maximum(self.shape, other.shape))
                k = 0
                return self.clone_with(shape=shape, k=k)
        if isinstance(other, BallRadialBasis):
            k = (other * self.radial_basis).k  # TODO: inline this
            return self.clone_with(k=k)
        return NotImplemented

    def __rmatmul__(self, other):
        # NCC (other) * operand (self)
        if other is None:
            return self
        if isinstance(other, BallRadialBasis):
            radial_basis = other @ self.radial_basis
            return self._new_k(radial_basis.k)
        if isinstance(other, BallBasis):
            radial_basis = other.radial_basis @ self.radial_basis
            return self._new_k(radial_basis.k)
        return NotImplemented

    def _new_k(self, k):
        # TODO: replace this in all operators
        return self.clone_with(k=k)

    @CachedMethod
    def transform_plan(self, dist, grid_shape, regindex, axis, regtotal, k, alpha):
        """Build transform plan."""
        radius_library = self.radial_basis.radius_library
        Nmax = self.radial_basis.Nmax
        return self.transforms[radius_library](grid_shape, Nmax+1, axis, self.ell_maps(dist), regindex, regtotal, k, alpha)

    def forward_transform_radius(self, field, axis, gdata, cdata):
        # apply transforms based off the 3D basis' local_l
        radial_basis = self.radial_basis
        # Apply regularity recombination
        radial_basis.forward_regularity_recombination(field.tensorsig, axis, gdata, ell_maps=self.ell_maps(field.dist))
        # Perform radial transforms component-by-component
        R = radial_basis.regularity_classes(field.tensorsig)
        # HACK -- don't want to make a new array every transform
        # transforms don't seem to work properly if we don't zero these
        temp = np.zeros_like(cdata)
        for regindex, regtotal in np.ndenumerate(R):
           grid_shape = gdata[regindex].shape
           plan = self.transform_plan(field.dist, grid_shape, regindex, axis, regtotal, radial_basis.k, radial_basis.alpha)
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
           plan = self.transform_plan(field.dist, grid_shape, regindex, axis, regtotal, radial_basis.k, radial_basis.alpha)
           plan.backward(cdata[regindex], temp[regindex], axis)
        np.copyto(gdata, temp)
        # Apply regularity recombinations
        radial_basis.backward_regularity_recombination(field.tensorsig, axis, gdata, ell_maps=self.ell_maps(field.dist))


def reduced_view(data, axis, dim):
    shape = data.shape
    Na = prod(shape[:axis])
    Nb = shape[axis:axis+dim]
    Nc = prod(shape[axis+dim:])
    return data.reshape((Na,)+Nb+(Nc,))


class ConvertRegularity(operators.Convert, operators.SphericalEllOperator):

    input_basis_type = RegularityBasis
    output_basis_type = RegularityBasis

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


class ConvertConstantBall(operators.ConvertConstant, operators.SphericalEllOperator):

    output_basis_type = (BallBasis, BallRadialBasis)
    subaxis_dependence = [False, True, True]
    subaxis_coupling = [False, False, False]

    def __init__(self, operand, output_basis, out=None):
        super().__init__(operand, output_basis, out=out)
        self.radial_basis = self.output_basis.get_radial_basis()
        if self.coords in operand.tensorsig:
            raise ValueError("Tensors not yet supported.")

    def regindex_out(self, regindex_in):
        return (regindex_in,)

    def radial_matrix(self, regindex_in, regindex_out, ell):
        radial_basis = self.radial_basis
        regtotal = radial_basis.regtotal(regindex_in)
        coeff_size = radial_basis.shape[-1]
        if ell == 0 and regindex_in == regindex_out:
            unit_amplitude = 1 / self.output_basis.constant_mode_value
            matrix = np.zeros((coeff_size, 1))
            matrix[0, 0] = unit_amplitude
            return matrix
        else:
            raise ValueError("This should never happen.")


class ConvertConstantShell(operators.ConvertConstant, operators.SphericalEllOperator):

    output_basis_type = (ShellBasis, ShellRadialBasis)
    subaxis_dependence = [False, True, True]
    subaxis_coupling = [False, False, True]

    def __init__(self, operand, output_basis, out=None):
        super().__init__(operand, output_basis, out=out)
        self.radial_basis = self.output_basis.get_radial_basis()
        if self.coords in operand.tensorsig:
            raise ValueError("Tensors not yet supported.")

    @CachedAttribute
    def radial_basis(self):
        return self.output_basis.radial_basis

    @CachedAttribute
    def S2_basis(self):
        return self.output_basis.S2_basis()

    def regindex_out(self, regindex_in):
        return (regindex_in,)

    def radial_matrix(self, regindex_in, regindex_out, ell):
        radial_basis = self.radial_basis
        regtotal = radial_basis.regtotal(regindex_in)
        coeff_size = radial_basis.shape[-1]
        if ell == 0 and regindex_in == regindex_out:
            # Convert to k=0
            const_to_k0 = np.zeros((coeff_size, 1))
            const_to_k0[0, 0] = 1 / self.output_basis.constant_mode_value  # This is really for k=0
            # Convert up in k
            k0_to_k = radial_basis._new_k(0).conversion_matrix(ell, regtotal, radial_basis.k)
            matrix = k0_to_k @ const_to_k0
            return matrix
        else:
            return sparse.csr_matrix((self.output_basis.n_size(ell), 0))
            #raise ValueError("This should never happen.")


class ConvertSpherical3D(operators.Convert, operators.SphericalEllOperator):

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


# class ConvertConstantRegularity(operators.Convert, operators.SphericalEllOperator):

#     input_basis_type = type(None)
#     output_basis_type = Spherical3DBasis
#     subaxis_dependence = [False, True, True]
#     subaxis_coupling = [False, False, False]

#     def __init__(self, operand, output_basis, out=None):
#         operators.Convert.__init__(self, operand, output_basis, out=out)
#         self.radial_basis = self.output_basis.get_radial_basis()

#     def subproblem_matrix(self, subproblem):
#         operand = self.args[0]
#         radial_basis = self.radial_basis
#         ell = subproblem.group[dist.last_axis(self) - 1]
#         # Build identity matrices for each axis
#         subshape_in = subproblem.coeff_shape(self.operand.domain)
#         subshape_out = subproblem.coeff_shape(self.domain)
#         # Substitute factor for radial axis
#         factors = [sparse.eye(m, n, format='csr') for m, n in zip(subshape_out, subshape_in)]
#         factors[dist.last_axis(self)][:] = 0
#         if ell == 0:
#             factors[dist.last_axis(self)][0, 0] = 1
#         return reduce(sparse.kron, factors, 1).tocsr()


class PolarRadialInterpolate(operators.Interpolate, operators.PolarMOperator):

    basis_type = PolarBasis
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
            n_upstream = prod(subproblem.coeff_shape(self.domain)[:self.last_axis-1])
            U = radial_basis.spin_recombination_matrix(self.tensorsig, n_upstream=n_upstream)
            matrix = U.T.conj() @ matrix
        return matrix

    def operate(self, out):
        """Perform operation."""
        operand = self.args[0]
        input_basis = self.input_basis
        output_basis = self.output_basis
        radial_basis = self.input_basis
        axis = self.last_axis
        # Set output layout
        out.preset_layout(operand.layout)
        # Apply operator
        S = radial_basis.spin_weights(operand.tensorsig)
        slices_in  = [slice(None) for i in range(self.dist.dim)]
        slices_out = [slice(None) for i in range(self.dist.dim)]
        for spinindex, spintotal in np.ndenumerate(S):
           comp_in = operand.data[spinindex]
           comp_out = out.data[spinindex]
           for m, mg_slice, mc_slice, n_slice in input_basis.m_maps(self.dist):
               slices_in[axis-1] = slices_out[axis-1] = mc_slice
               slices_in[axis] = n_slice
               vec_in  = comp_in[tuple(slices_in)]
               vec_out = comp_out[tuple(slices_out)]
               A = self.radial_matrix(spinindex, spinindex, m)
               apply_matrix(A, vec_in, axis=axis, out=vec_out)
        temp = np.copy(out.data)
        radial_basis.backward_spin_recombination(operand.tensorsig, axis, temp, out.data)

    def radial_matrix(self, spinindex_in, spinindex_out, m):
        position = self.position
        basis = self.input_basis
        if spinindex_in == spinindex_out:
            return self._radial_matrix(basis, m, basis.spintotal(self.operand.tensorsig, spinindex_in), position)
        else:
            return np.zeros((1,basis.n_size(m)))

    def spinindex_out(self, spinindex_in):
        return (spinindex_in,)

    @staticmethod
    @CachedMethod
    def _radial_matrix(basis, m, spintotal, position):
        return reshape_vector(basis.interpolation(m, spintotal, position), dim=2, axis=1)


class LiftDisk(operators.Lift, operators.PolarMOperator):

    input_basis_type = (RealFourier, ComplexFourier)
    output_basis_type = DiskBasis

    def spinindex_out(self, spinindex_in):
        return (spinindex_in,)

    def subproblem_matrix(self, subproblem):
        operand = self.args[0]
        radial_basis = self.output_basis  ## CHANGED RELATIVE TO POLARMOPERATOR
        S_in = radial_basis.spin_weights(operand.tensorsig)
        S_out = radial_basis.spin_weights(self.tensorsig)  # Should this use output_basis?
        radial_axis = self.dist.last_axis(self.output_basis)
        m = subproblem.group[radial_axis - 1]
        subshape_in = subproblem.coeff_shape(self.operand.domain)
        subshape_out = subproblem.coeff_shape(self.domain)
        # Loop over components
        submatrices = []
        for spinindex_out, spintotal_out in np.ndenumerate(S_out):
            submatrix_row = []
            for spinindex_in, spintotal_in in np.ndenumerate(S_in):
                # Build identity matrices for each axis
                if spinindex_out in self.spinindex_out(spinindex_in):
                    # Substitute factor for radial axis
                    factors = [sparse.eye(i, j, format='csr') for i, j in zip(subshape_out, subshape_in)]
                    factors[radial_axis] = self.radial_matrix(spinindex_in, spinindex_out, m)
                    comp_matrix = reduce(sparse.kron, factors, 1).tocsr()
                else:
                    # Build zero matrix
                    comp_matrix = sparse.csr_matrix((prod(subshape_out), prod(subshape_in)))
                submatrix_row.append(comp_matrix)
            submatrices.append(submatrix_row)
        matrix = sparse.bmat(submatrices)
        matrix.tocsr()
        # Convert tau to spin first
        if self.tensorsig:
            n_upstream = prod(subshape_out[:radial_axis-1])
            U = radial_basis.spin_recombination_matrix(self.tensorsig, n_upstream=n_upstream)
            matrix = (matrix @ sparse.csr_matrix(U)).tocsr()
        return matrix

    def radial_matrix(self, spinindex_in, spinindex_out, m):
        if spinindex_in == spinindex_out:
            n_size = self.output_basis.n_size(m)
            matrix = np.zeros((n_size, 1))
            matrix[self.n, 0] = 1
            return matrix
        else:
            raise ValueError("This should never happen.")


class LiftAnnulus(operators.Lift, operators.PolarMOperator):

    input_basis_type = (RealFourier, ComplexFourier)
    output_basis_type = AnnulusBasis

    def spinindex_out(self, spinindex_in):
        return (spinindex_in,)

    def subproblem_matrix(self, subproblem):
        operand = self.args[0]
        radial_basis = self.output_basis  ## CHANGED RELATIVE TO POLARMOPERATOR
        S_in = radial_basis.spin_weights(operand.tensorsig)
        S_out = radial_basis.spin_weights(self.tensorsig)  # Should this use output_basis?
        m = subproblem.group[self.last_axis - 1]
        subshape_in = subproblem.coeff_shape(self.operand.domain)
        subshape_out = subproblem.coeff_shape(self.domain)
        # Loop over components
        submatrices = []
        for spinindex_out, spintotal_out in np.ndenumerate(S_out):
            submatrix_row = []
            for spinindex_in, spintotal_in in np.ndenumerate(S_in):
                # Build identity matrices for each axis
                if spinindex_out in self.spinindex_out(spinindex_in):
                    # Substitute factor for radial axis
                    factors = [sparse.eye(i, j, format='csr') for i, j in zip(subshape_out, subshape_in)]
                    factors[self.last_axis] = self.radial_matrix(spinindex_in, spinindex_out, m)
                    comp_matrix = reduce(sparse.kron, factors, 1).tocsr()
                else:
                    # Build zero matrix
                    comp_matrix = sparse.csr_matrix((prod(subshape_out), prod(subshape_in)))
                submatrix_row.append(comp_matrix)
            submatrices.append(submatrix_row)
        matrix = sparse.bmat(submatrices)
        matrix.tocsr()
        # Convert tau to spin first
        if self.tensorsig:
            n_upstream = prod(subshape_out[:self.last_axis-1])
            U = radial_basis.spin_recombination_matrix(self.tensorsig, n_upstream=n_upstream)
            matrix = (matrix @ sparse.csr_matrix(U)).tocsr()
        return matrix

    def radial_matrix(self, spinindex_in, spinindex_out, m):
        if spinindex_in == spinindex_out:
            n_size = self.output_basis.n_size(m)
            matrix = np.zeros((n_size, 1))
            matrix[self.n, 0] = 1
            return matrix
        else:
            raise ValueError("This should never happen.")


class LiftBall(operators.Lift, operators.SphericalEllOperator):

    input_basis_type = SphereBasis
    output_basis_type = BallBasis

    def __init__(self, operand, output_basis, n, out=None):
        super().__init__(operand, output_basis, n)
        self.radial_basis = self.output_basis.radial_basis

    def regindex_out(self, regindex_in):
        return (regindex_in,)

    def subproblem_matrix(self, subproblem):
        operand = self.args[0]
        radial_basis = self.output_basis.radial_basis  ## CHANGED RELATIVE TO POLARMOPERATOR
        radial_axis = self.dist.last_axis(radial_basis)
        R_in = radial_basis.regularity_classes(operand.tensorsig)
        R_out = radial_basis.regularity_classes(self.tensorsig)  # Should this use output_basis?
        ell = subproblem.group[radial_axis - 1]
        # Loop over components
        submatrices = []
        for regindex_out, regtotal_out in np.ndenumerate(R_out):
            submatrix_row = []
            for regindex_in, regtotal_in in np.ndenumerate(R_in):
                # Build identity matrices for each axis
                subshape_in = subproblem.coeff_shape(self.operand.domain)
                subshape_out = subproblem.coeff_shape(self.domain)
                # Check if regularity component exists for this ell
                if (regindex_out in self.regindex_out(regindex_in)) and radial_basis.regularity_allowed(ell, regindex_in) and radial_basis.regularity_allowed(ell, regindex_out):
                    # Substitute factor for radial axis
                    factors = [sparse.eye(m, n, format='csr') for m, n in zip(subshape_out, subshape_in)]
                    factors[radial_axis] = self.radial_matrix(regindex_in, regindex_out, ell)
                    comp_matrix = reduce(sparse.kron, factors, 1).tocsr()
                else:
                    # Build zero matrix
                    comp_matrix = sparse.csr_matrix((prod(subshape_out), prod(subshape_in)))
                submatrix_row.append(comp_matrix)
            submatrices.append(submatrix_row)
        matrix = sparse.bmat(submatrices)
        matrix.tocsr()
        # Convert tau from spin to regularity first
        Q = dedalus_sphere.spin_operators.Intertwiner(ell, indexing=(-1,+1,0))(len(self.tensorsig))  # Fix for product domains
        matrix = matrix @ sparse.kron(Q.T, sparse.identity(prod(subshape_in), format='csr'))
        return matrix.tocsr()

    def radial_matrix(self, regindex_in, regindex_out, m):
        if regindex_in == regindex_out:
            n_size = self.output_basis.n_size(m)
            matrix = np.zeros((n_size, 1))
            matrix[self.n, 0] = 1
            return matrix
        else:
            raise ValueError("This should never happen.")


class LiftBallRadius(operators.Lift, operators.SphericalEllOperator):

    input_basis_type = type(None)
    output_basis_type = BallRadialBasis

    def __init__(self, operand, output_basis, n, out=None):
        super().__init__(operand, output_basis, n)
        self.radial_basis = self.output_basis

    def regindex_out(self, regindex_in):
        return (regindex_in,)

    def subproblem_matrix(self, subproblem):
        operand = self.args[0]
        radial_basis = self.output_basis
        R_in = radial_basis.regularity_classes(operand.tensorsig)
        R_out = radial_basis.regularity_classes(self.tensorsig)  # Should this use output_basis?
        ell = subproblem.group[self.last_axis - 1]
        # Loop over components
        submatrices = []
        for regindex_out, regtotal_out in np.ndenumerate(R_out):
            submatrix_row = []
            for regindex_in, regtotal_in in np.ndenumerate(R_in):
                # Build identity matrices for each axis
                subshape_in = subproblem.coeff_shape(self.operand.domain)
                subshape_out = subproblem.coeff_shape(self.domain)
                # Check if regularity component exists for this ell
                if (regindex_out in self.regindex_out(regindex_in)) and radial_basis.regularity_allowed(ell, regindex_in) and radial_basis.regularity_allowed(ell, regindex_out):
                    # Substitute factor for radial axis
                    factors = [sparse.eye(m, n, format='csr') for m, n in zip(subshape_out, subshape_in)]
                    factors[self.last_axis] = self.radial_matrix(regindex_in, regindex_out, ell)
                    comp_matrix = reduce(sparse.kron, factors, 1).tocsr()
                else:
                    # Build zero matrix
                    comp_matrix = sparse.csr_matrix((prod(subshape_out), prod(subshape_in)))
                submatrix_row.append(comp_matrix)
            submatrices.append(submatrix_row)
        matrix = sparse.bmat(submatrices)
        matrix.tocsr()
        # Convert tau from spin to regularity first
        Q = dedalus_sphere.spin_operators.Intertwiner(ell, indexing=(-1,+1,0))(len(self.tensorsig))  # Fix for product domains
        matrix = matrix @ sparse.kron(Q.T, sparse.identity(prod(subshape_in), format='csr'))
        return matrix.tocsr()

    def radial_matrix(self, regindex_in, regindex_out, m):
        if regindex_in == regindex_out:
            n_size = self.output_basis.n_size(m)
            matrix = np.zeros((n_size, 1))
            matrix[self.n, 0] = 1
            return matrix
        else:
            raise ValueError("This should never happen.")


class LiftShell(operators.Lift, operators.SphericalEllOperator):

    input_basis_type = SphereBasis
    output_basis_type = ShellBasis

    @CachedAttribute
    def radial_basis(self):
        return self.output_basis.radial_basis

    @CachedAttribute
    def S2_basis(self):
        return self.output_basis.S2_basis()

    def regindex_out(self, regindex_in):
        return (regindex_in,)

    def subproblem_matrix(self, subproblem):
        matrix = super().subproblem_matrix(subproblem)
        # Get relevant Qs
        m = subproblem.group[self.last_axis - 2]
        ell = subproblem.group[self.last_axis - 1]
        if ell is None:
            ell_list = np.arange(abs(m), self.input_basis.Lmax + 1)
            if self.input_basis.ell_reversed(self.dist)[m]:
                ell_list = ell_list[::-1]
        else:
            ell_list = [ell]
        Q_dict = self.output_basis.radial_basis.radial_recombinations(self.tensorsig, ell_list=tuple(ell_list))
        # Block-diag for sin/cos parts for real dtype
        if self.dtype == np.float64:
            I2 = sparse.eye(2)
            Q_dict = {ell: sparse.kron(Q_dict[ell], I2) for ell in ell_list}
        Q_forward = interleave_matrices([Q_dict[ell].T for ell in ell_list])
        # Convert tau from spin to regularity before lifting
        matrix = matrix @ Q_forward
        return sparse.csr_matrix(matrix)

    def radial_matrix(self, regindex_in, regindex_out, m):
        if regindex_in == regindex_out:
            n_size = self.output_basis.n_size(m)
            matrix = np.zeros((n_size, 1))
            matrix[self.n, 0] = 1
            return matrix
        else:
            raise ValueError("This should never happen.")


class AzimuthalAverage(metaclass=MultiClass):

    input_coord_type = AzimuthalCoordinate

    @staticmethod
    def _output_basis(input_basis):
        # Clone input basis with N_azimuth = 1
        shape = list(input_basis.shape)
        shape[0] = 1
        return input_basis.clone_with(shape=tuple(shape))

    def new_operand(self, operand, **kw):
        return AzimuthalAverage(operand, self.coord, **kw)


class PolarAzimuthalAverage(AzimuthalAverage, operators.Average, operators.PolarMOperator):

    input_basis_type = (DiskBasis, AnnulusBasis)

    @CachedAttribute
    def radial_basis(self):
        return self.input_basis.radial_basis

    def spinindex_out(self, spinindex_in):
        return (spinindex_in,)

    def radial_matrix(self, spinindex_in, spinindex_out, m):
        if spinindex_out != spinindex_in:
            raise ValueError("This should never happen.")
        n_size = self.input_basis.n_size(m)
        if m == 0:
            return sparse.identity(n_size).tocsr()
        else:
            return sparse.csr_matrix((0, n_size), dtype=self.dtype).tocsr()


class SphereAzimuthalAverage(AzimuthalAverage, operators.Average, operators.SpectralOperator):

    input_basis_type = (SphereBasis,)
    subaxis_dependence = [True, False]  # Depends on m only
    subaxis_coupling = [False, False]  # No coupling

    def operate(self, out):
        """Perform operation."""
        operand = self.args[0]
        # Set output layout
        layout = operand.layout
        out.preset_layout(layout)
        out.data[:] = 0
        # Apply operator
        azimuth_axis = self.dist.first_axis(self.input_basis)
        domain_in = self.operand.domain
        domain_out = self.domain
        groups_in = layout.local_group_arrays(domain_in, scales=domain_in.dealias)
        groups_out = layout.local_group_arrays(domain_out, scales=domain_out.dealias)
        m0_in = (groups_in[azimuth_axis] == 0)
        m0_out = (groups_out[azimuth_axis] == 0)
        spincomps = self.input_basis.spin_weights(operand.tensorsig)
        # Copy m = 0 for every component
        for spinindex, spintotal in np.ndenumerate(spincomps):
            comp_in = operand.data[spinindex]
            comp_out = out.data[spinindex]
            comp_out[m0_out] = comp_in[m0_in]


class SphericalAzimuthalAverage(AzimuthalAverage, operators.Average, operators.SpectralOperator):

    input_basis_type = (BallBasis, ShellBasis)
    subaxis_dependence = [True, False, False]  # Depends on m only
    subaxis_coupling = [False, False, False]  # No coupling

    def operate(self, out):
        """Perform operation."""
        operand = self.args[0]
        # Set output layout
        layout = operand.layout
        out.preset_layout(layout)
        out.data[:] = 0
        # Apply operator
        azimuth_axis = self.dist.first_axis(self.input_basis)
        domain_in = self.operand.domain
        domain_out = self.domain
        groups_in = layout.local_group_arrays(domain_in, scales=domain_in.dealias)
        groups_out = layout.local_group_arrays(domain_out, scales=domain_out.dealias)
        m0_in = (groups_in[azimuth_axis] == 0)
        m0_out = (groups_out[azimuth_axis] == 0)
        regcomps = self.input_basis.radial_basis.regularity_classes(operand.tensorsig)
        # Copy m = 0 for every component
        for regindex, regtotal in np.ndenumerate(regcomps):
            comp_in = operand.data[regindex]
            comp_out = out.data[regindex]
            comp_out[m0_out] = comp_in[m0_in]


class SphereAverage(operators.Average, operators.SeparableSphereOperator):
    """Todo: skip when Nphi = Ntheta = 1."""

    input_coord_type = S2Coordinates
    input_basis_type = SphereBasis
    subaxis_dependence = [False, True]
    complex_operator = False

    def _output_basis(self, input_basis):
        # Copy with Nphi = Ntheta = 1
        shape = list(input_basis.shape)
        shape[0] = shape[1] = 1
        return input_basis.clone_with(shape=tuple(shape))

    def new_operand(self, operand, **kw):
        return operators.Average(operand, self.coord, **kw)

    def spinindex_out(self, spinindex_in):
        return (spinindex_in,)

    @staticmethod
    def symbol(spinindex_in, spinindex_out, spintotal_in, spintotal_out, ell, radius):
        return 1.0 * (ell == 0) * (spinindex_in == spinindex_out)


class SphericalAverage(operators.Average, operators.SphericalEllOperator):
    """Todo: skip when Nphi = Ntheta = 1."""

    input_coord_type = S2Coordinates
    input_basis_type = (BallBasis, ShellBasis)

    @CachedAttribute
    def radial_basis(self):
        return self.input_basis.radial_basis

    def _output_basis(self, input_basis):
        # Copy with Nphi = Ntheta = 1
        shape = list(input_basis.shape)
        shape[0] = shape[1] = 1
        return input_basis.clone_with(shape=tuple(shape))

    def new_operand(self, operand, **kw):
        return SphericalAverage(operand, self.coord, **kw)

    def regindex_out(self, regindex_in):
        return (regindex_in,)

    def radial_matrix(self, regindex_in, regindex_out, ell):
        if regindex_out != regindex_in:
            raise ValueError("This should never happen.")
        n_size = self.input_basis.n_size(ell)
        if ell == 0:
            return sparse.identity(n_size).tocsr()
        else:
            return sparse.csr_matrix((0, n_size), dtype=self.dtype).tocsr()


class IntegrateSpinBasis(operators.PolarMOperator):
    """Integrate SpinBasis scalar fields."""

    input_coord_type = PolarCoordinates

    @CachedAttribute
    def radial_basis(self):
        return self.input_basis.radial_basis

    @classmethod
    def _check_coords(cls, basis, coords):
        return coords is basis.coordsys

    def _output_basis(self, input_basis):
        return None

    def new_operand(self, operand, **kw):
        return type(self)(operand, self.coord, **kw)

    def spinindex_out(self, spinindex_in):
        return (spinindex_in,)

    def radial_matrix(self, spinindex_in, spinindex_out, m):
        # Wrap cached method
        return self._radial_matrix(self.radial_basis, m)

    def operate(self, out):
        """Perform operation."""
        operand = self.args[0]
        basis = self.input_basis
        axis = self.dist.last_axis(self.radial_basis)
        # Set output layout
        out.preset_layout(operand.layout)
        out.data[:] = 0
        # Apply operator
        for m, mg_slice, mc_slice, n_slice in basis.m_maps(self.dist):
            if m == 0:
                # Modify mc_slice to ignore sin component
                slices = [slice(None) for i in range(self.dist.dim)]
                slices[axis-1] = slice(mc_slice.start, mc_slice.start+1)
                slices[axis] = n_slice
                slices = tuple(slices)
                vec_in  = operand.data[slices]
                vec_out = out.data[slices]
                A = self._radial_matrix(self.radial_basis, m)
                vec_out += apply_matrix(A, vec_in, axis=axis)


class IntegrateDisk(operators.Integrate, IntegrateSpinBasis):
    """Integrate DiskBasis scalar fields."""

    input_basis_type = DiskBasis

    @staticmethod
    @CachedMethod
    def _radial_matrix(basis, m):
        n_size = basis.n_size(m)
        if m == 0:
            N = basis.shape[1]
            z0, w0 = dedalus_sphere.zernike.quadrature(2, N, k=0)
            Qk = dedalus_sphere.zernike.polynomials(2, n_size, basis.alpha+basis.k, abs(m), z0)
            matrix = (w0[None, :] @ Qk.T).astype(basis.dtype)
            matrix *= basis.radius**2
            matrix *= 2 * np.pi # Fourier contribution
        else:
            matrix= sparse.csr_matrix((0, n_size), dtype=basis.dtype)
        return matrix


class AverageDisk(operators.Average, IntegrateSpinBasis):
    """Average DiskBasis scalar fields."""

    input_basis_type = DiskBasis

    @staticmethod
    @CachedMethod
    def _radial_matrix(basis, m):
        n_size = basis.n_size(m)
        if m == 0:
            N = basis.shape[1]
            z0, w0 = dedalus_sphere.zernike.quadrature(2, N, k=0)
            Qk = dedalus_sphere.zernike.polynomials(2, n_size, basis.alpha+basis.k, abs(m), z0)
            matrix = (w0[None, :] @ Qk.T).astype(basis.dtype)
            matrix *= 2 # Fourier contribution
        else:
            matrix= sparse.csr_matrix((0, n_size), dtype=basis.dtype)
        return matrix


class IntegrateAnnulus(operators.Integrate, IntegrateSpinBasis):
    """Integrate AnnulusBasis scalar fields."""

    input_basis_type = AnnulusBasis

    @staticmethod
    @CachedMethod
    def _radial_matrix(basis, m):
        n_size = basis.n_size(m)
        if m == 0:
            N = 2 * basis.shape[1]  # Add some dealiasing to help with large k
            z0, w0 = dedalus_sphere.jacobi.quadrature(N, a=0, b=0)
            r0 = basis.dR / 2 * (z0 + basis.rho)
            Qk = dedalus_sphere.jacobi.polynomials(n_size, basis.alpha[0]+basis.k, basis.alpha[1]+basis.k, z0)
            w0_geom = r0 * w0 * (r0 / basis.dR)**(-basis.k)
            matrix = (w0_geom[None, :] @ Qk.T).astype(basis.dtype)
            matrix *= basis.dR / 2
            matrix *= 2 * np.pi # Fourier contribution
        else:
            matrix= sparse.csr_matrix((0, n_size), dtype=basis.dtype)
        return matrix


class AverageAnnulus(operators.Average, IntegrateSpinBasis):
    """Average AnnulusBasis scalar fields."""

    input_basis_type = AnnulusBasis

    @staticmethod
    @CachedMethod
    def _radial_matrix(basis, m):
        n_size = basis.n_size(m)
        if m == 0:
            N = 2 * basis.shape[1]  # Add some dealiasing to help with large k
            z0, w0 = dedalus_sphere.jacobi.quadrature(N, a=0, b=0)
            r0 = basis.dR / 2 * (z0 + basis.rho)
            Qk = dedalus_sphere.jacobi.polynomials(n_size, basis.alpha[0]+basis.k, basis.alpha[1]+basis.k, z0)
            w0_geom = r0 * w0 * (r0 / basis.dR)**(-basis.k)
            matrix = (w0_geom[None, :] @ Qk.T).astype(basis.dtype)
            matrix *= basis.dR / 2
            matrix *= 2 * np.pi # Fourier contribution
            matrix /= np.pi * (basis.radii[1]**2 - basis.radii[0]**2)
        else:
            matrix= sparse.csr_matrix((0, n_size), dtype=basis.dtype)
        return matrix


class IntegrateSpherical(operators.SphericalEllOperator):
    """Integrate spherical scalar fields."""

    input_coord_type = SphericalCoordinates

    @CachedAttribute
    def radial_basis(self):
        return self.input_basis.radial_basis

    def _output_basis(self, input_basis):
        return None

    def new_operand(self, operand, **kw):
        return type(self)(operand, self.coord, **kw)

    def regindex_out(self, regindex_in):
        return (regindex_in,)

    def radial_matrix(self, regindex_in, regindex_out, ell):
        # Wrap cached method
        return self._radial_matrix(self.radial_basis, ell)

    def operate(self, out):
        """Perform operation."""
        operand = self.args[0]
        basis = self.input_basis
        axis = self.dist.last_axis(basis.radial_basis)
        # Set output layout
        out.preset_layout(operand.layout)
        out.data[:] = 0
        # Apply operator
        for ell, m_ind, ell_ind in basis.ell_maps(self.dist):
            if ell == 0:
                slices = [slice(None) for i in range(self.dist.dim)]
                # Modify m slice to ignore sin component
                slices[axis-2] = slice(m_ind.start, m_ind.start+1)
                slices[axis-1] = ell_ind
                slices[axis] = basis.n_slice(ell)
                slices = tuple(slices)
                vec_in  = operand.data[slices]
                vec_out = out.data[slices]
                A = self.radial_matrix(None, None, ell)
                vec_out += apply_matrix(A, vec_in, axis=axis)


class IntegrateBall(operators.Integrate, IntegrateSpherical):
    """Integrate BallBasis scalar fields."""

    input_basis_type = BallBasis

    @staticmethod
    @CachedMethod
    def _radial_matrix(basis, ell):
        n_size = basis.n_size(ell)
        if ell == 0:
            N = basis.shape[2]
            z0, w0 = dedalus_sphere.zernike.quadrature(3, N, k=0)
            Qk = dedalus_sphere.zernike.polynomials(3, n_size, basis.alpha+basis.k, ell, z0)
            matrix = (w0[None, :] @ Qk.T).astype(basis.dtype)
            matrix *= basis.radius**3
            matrix *= 4 * np.pi / np.sqrt(2) # SWSH contribution
        else:
            matrix= sparse.csr_matrix((0, n_size), dtype=basis.dtype)
        return matrix


class IntegrateShell(operators.Integrate, IntegrateSpherical):
    """Integrate ShellBasis scalar fields."""

    input_basis_type = ShellBasis

    @staticmethod
    @CachedMethod
    def _radial_matrix(basis, ell):
        n_size = basis.n_size(ell)
        if ell == 0:
            N = 2 * basis.shape[2]  # Add some dealiasing to help with large k
            z0, w0 = dedalus_sphere.jacobi.quadrature(N, a=0, b=0)
            r0 = basis.dR / 2 * (z0 + basis.rho)
            Qk = dedalus_sphere.jacobi.polynomials(n_size, basis.alpha[0]+basis.k, basis.alpha[1]+basis.k, z0)
            w0_geom = r0**2 * w0 * (r0 / basis.dR)**(-basis.k)
            matrix = (w0_geom[None, :] @ Qk.T).astype(basis.dtype)
            matrix *= basis.dR / 2
            matrix *= 4 * np.pi / np.sqrt(2) # SWSH contribution
        else:
            matrix= sparse.csr_matrix((0, n_size), dtype=basis.dtype)
        return matrix


class InterpolateAzimuth(FutureLockedField, operators.Interpolate):

    input_basis_type = (SphereBasis, BallBasis, ShellBasis, DiskBasis, AnnulusBasis)
    basis_subaxis = 0

    @staticmethod
    def _output_basis(input_basis, position):
        # Clone input basis with N_azimuth = 1
        shape = list(input_basis.shape)
        shape[0] = 1
        return input_basis.clone_with(shape=tuple(shape))

    def check_conditions(self):
        """Check that arguments are in a proper layout."""
        arg0 = self.args[0]
        azimuth_axis = self.first_axis
        # Require grid space and locality along azimuthal axis
        is_grid = arg0.layout.grid_space[azimuth_axis]
        is_local = arg0.layout.local[azimuth_axis]
        return is_grid and is_local

    def enforce_conditions(self):
        """Require arguments to be in a proper layout."""
        arg0 = self.args[0]
        azimuth_axis = self.first_axis
        # Require grid space and locality along azimuthal axis
        arg0.require_grid_space(azimuth_axis)
        arg0.require_local(azimuth_axis)

    def interpolation_vector(self):
        # Wrap class-based caching
        return self._interpolation_vector(self.input_basis, self.position)

    @staticmethod
    @CachedMethod
    def _interpolation_vector(input_basis, position):
        # Construct collocation interpolation using forward transform matrix and spectral interpolation
        azimuth_basis = input_basis.azimuth_basis
        grid_size = azimuth_basis.grid_shape(scales=azimuth_basis.dealias)[0]
        forward = azimuth_basis.transforms['matrix'](grid_size, azimuth_basis.size).forward_matrix[azimuth_basis.forward_coeff_permutation]
        if input_basis.dtype is np.float64:
            interp = InterpolateRealFourier._full_matrix(azimuth_basis, None, position)
        elif input_basis.dtype is np.complex128:
            interp = InterpolateComplexFourier._full_matrix(azimuth_basis, None, position)
        return interp @ forward

    def operate(self, out):
        """Perform operation."""
        arg = self.args[0]
        layout = arg.layout
        # Set output layout
        out.preset_layout(layout)
        # Set output lock
        out.lock_axis_to_grid(self.first_axis)
        # Apply matrix
        data_axis = self.first_axis + len(arg.tensorsig)
        apply_matrix(self.interpolation_vector(), arg.data, data_axis, out=out.data)


class InterpolateColatitude(FutureLockedField, operators.Interpolate):

    future_type = LockedField
    input_basis_type = (SphereBasis, BallBasis, ShellBasis)
    basis_subaxis = 1

    @CachedAttribute
    def sphere_basis(self):
        if isinstance(self.input_basis, SphereBasis):
            return self.input_basis
        else:
            return self.input_basis.sphere_basis

    @staticmethod
    def _output_basis(input_basis, position):
        # Clone input basis with N_colatitude = 1
        # Todo: just a function of radius if interpolation is at poles?
        shape = list(input_basis.shape)
        shape[1] = 1
        return input_basis.clone_with(shape=tuple(shape))

    def check_conditions(self):
        """Check that arguments are in a proper layout."""
        arg0 = self.args[0]
        azimuth_axis = self.first_axis
        colat_axis = azimuth_axis + 1
        # Require azimuth coeff, colat grid, colat, local
        az_coeff = not arg0.layout.grid_space[azimuth_axis]
        colat_grid = arg0.layout.grid_space[colat_axis]
        colat_local = arg0.layout.local[colat_axis]
        return az_coeff and colat_grid and colat_local

    def enforce_conditions(self):
        """Require arguments to be in a proper layout."""
        arg0 = self.args[0]
        azimuth_axis = self.first_axis
        colat_axis = azimuth_axis + 1
        # Require azimuth coeff, colat grid, colat, local
        arg0.require_coeff_space(azimuth_axis)
        arg0.require_grid_space(colat_axis)
        arg0.require_local(colat_axis)

    def interpolation_vectors(self, Ntheta, s):
        # Wrap class-based caching
        return self._interpolation_vectors(self.dist, self.sphere_basis, Ntheta, s, self.position)

    @staticmethod
    @CachedMethod
    def _interpolation_vectors(dist, sphere_basis, Ntheta, s, theta):
        interp_vectors = {}
        z = np.cos(theta)
        colat_transform = dist.get_transform_object(dist.first_axis(sphere_basis)+1)
        layout = colat_transform.layout1
        coupling = [True] * dist.dim
        coupling[dist.first_axis(sphere_basis)] = False
        coupling = tuple(coupling)
        domain = sphere_basis.domain(dist)
        m_groupsets = layout.local_groupsets(coupling, domain, scales=domain.dealias, broadcast=True)
        forward = sphere_basis.transform_plan(dist, Ntheta, s)
        for group in m_groupsets:
            m = group[dist.first_axis(sphere_basis)]
            if m <= sphere_basis.Lmax:
                Lmin = max(abs(m), abs(s))
                interp_m = dedalus_sphere.sphere.harmonics(sphere_basis.Lmax, m, s, z)[None, :]
                forward_m = forward._forward_SWSH_matrices[m][Lmin-abs(m):]
                interp_vectors[m] = interp_m @ forward_m
            else:
                interp_vectors[m] = np.zeros((1, Ntheta))
        return interp_vectors

    def operate(self, out):
        """Perform operation."""
        arg = self.args[0]
        basis = self.sphere_basis
        layout = arg.layout
        azimuth_axis = self.first_axis
        colat_axis = azimuth_axis + 1
        Ntheta = arg.data.shape[len(arg.tensorsig) + colat_axis]
        # Set output layout
        out.preset_layout(layout)
        # Set output lock
        out.lock_axis_to_grid(colat_axis)
        # Forward spin recombination
        arg_temp = np.zeros_like(arg.data)
        out_temp = np.zeros_like(out.data)
        basis.forward_spin_recombination(arg.tensorsig, colat_axis, arg.data, arg_temp)
        # Loop over spin components
        S = basis.spin_weights(arg.tensorsig)
        for i, s in np.ndenumerate(S):
            arg_s = arg_temp[i]
            out_s = out_temp[i]
            interp_vectors = self.interpolation_vectors(Ntheta, s)
            # Loop over m
            for m, mg_slice, _, _ in basis.m_maps(self.dist):
                mg_slice = axindex(azimuth_axis, mg_slice)
                arg_sm = arg_s[mg_slice]
                out_sm = out_s[mg_slice]
                apply_matrix(interp_vectors[m], arg_sm, axis=colat_axis, out=out_sm)
        # Backward spin recombination
        basis.backward_spin_recombination(out.tensorsig, colat_axis, out_temp, out.data)


class BallRadialInterpolate(operators.Interpolate, operators.SphericalEllOperator):

    basis_type = (BallBasis, BallRadialBasis)
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
        if isinstance(input_basis, BallBasis):
            return input_basis.S2_basis(radius=position)
        elif isinstance(input_basis, BallRadialBasis):
            return None

    def __init__(self, operand, coord, position, out=None):
        operators.Interpolate.__init__(self, operand, coord, position, out=None)
        if isinstance(self.input_basis, BallBasis):
            self.radial_basis = self.input_basis.get_radial_basis()
        elif isinstance(self.input_basis, BallRadialBasis):
            self.radial_basis = self.input_basis

    def subproblem_matrix(self, subproblem):
        ell = subproblem.group[self.dist.last_axis(self.radial_basis) - 1]
        matrix = super().subproblem_matrix(subproblem)
        radial_basis = self.radial_basis
        if self.tensorsig != ():
            Q = radial_basis.radial_recombinations(self.tensorsig, ell_list=(ell,))[ell]
            if self.dtype == np.float64:
                # Block-diag for sin/cos parts for real dtype
                Q = sparse.kron(Q, sparse.eye(2))
            matrix = Q @ matrix
        return matrix

    def operate(self, out):
        """Perform operation."""
        operand = self.args[0]
        input_basis = self.input_basis
        output_basis = self.output_basis
        radial_basis = self.radial_basis
        axis = self.dist.last_axis(radial_basis)
        # Set output layout
        out.preset_layout(operand.layout)
        # Apply operator
        R = radial_basis.regularity_classes(operand.tensorsig)
        slices_in  = [slice(None) for i in range(self.dist.dim)]
        slices_out = [slice(None) for i in range(self.dist.dim)]
        for regindex, regtotal in np.ndenumerate(R):
           comp_in = operand.data[regindex]
           comp_out = out.data[regindex]
           for ell, m_ind, ell_ind in input_basis.ell_maps(self.dist):
               allowed  = radial_basis.regularity_allowed(ell, regindex)
               if allowed:
                   slices_in[axis-2] = slices_out[axis-2] = m_ind
                   slices_in[axis-1] = slices_out[axis-1] = ell_ind
                   slices_in[axis] = radial_basis.n_slice(ell)
                   vec_in  = comp_in[tuple(slices_in)]
                   vec_out = comp_out[tuple(slices_out)]
                   A = self.radial_matrix(regindex, regindex, ell)
                   apply_matrix(A, vec_in, axis=axis, out=vec_out)
        radial_basis.backward_regularity_recombination(operand.tensorsig, self.basis_subaxis, out.data, ell_maps=input_basis.ell_maps(self.dist))

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


class ShellRadialInterpolate(operators.Interpolate, operators.SphericalEllOperator):

    basis_type = ShellBasis
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
        matrix = super().subproblem_matrix(subproblem)
        # Get relevant Qs
        m = subproblem.group[self.last_axis - 2]
        ell = subproblem.group[self.last_axis - 1]
        if ell is None:
            ell_list = np.arange(abs(m), self.input_basis.Lmax + 1)
            if self.input_basis.ell_reversed(self.dist)[m]:
                ell_list = ell_list[::-1]
        else:
            ell_list = [ell]
        Q_dict = self.input_basis.radial_basis.radial_recombinations(self.tensorsig, ell_list=tuple(ell_list))
        # Block-diag for sin/cos parts for real dtype
        if self.dtype == np.float64:
            I2 = sparse.eye(2)
            Q_dict = {ell: sparse.kron(Q_dict[ell], I2) for ell in ell_list}
        Q_backward = interleave_matrices([Q_dict[ell] for ell in ell_list])
        # Convert tau from spin to regularity before lifting
        matrix = Q_backward @ matrix
        return sparse.csr_matrix(matrix)

    def operate(self, out):
        """Perform operation."""
        operators.SphericalEllOperator.operate(self, out)
        operand = self.args[0]
        radial_basis = self.radial_basis
        input_basis = self.input_basis
        # Q matrix
        radial_basis.backward_regularity_recombination(operand.tensorsig, self.basis_subaxis, out.data, ell_maps=input_basis.ell_maps(self.dist))

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

    basis_type = SphereBasis

    def subproblem_matrix(self, subproblem):
        operand = self.args[0]
        basis = self.domain.get_basis(self.coordsys)
        S_in = basis.spin_weights(operand.tensorsig)
        S_out = basis.spin_weights(self.tensorsig)
        matrix = []
        for spinindex_out, spintotal_out in np.ndenumerate(S_out):
            matrix_row = []
            for spinindex_in, spintotal_in in np.ndenumerate(S_in):
                matrix_row.append(0)
                if spinindex_in[self.index] == 2:
                    spinindex_in = list(spinindex_in)
                    spinindex_in.pop(self.index)
                    if tuple(spinindex_in) == spinindex_out:
                        matrix_row[-1] = 1
            matrix.append(matrix_row)
        matrix = np.array(matrix)
        # Block-diag for sin/cos parts for real dtype
        if self.dtype == np.float64:
            matrix = sparse.kron(matrix, sparse.eye(2))
        # Block over ell
        m = subproblem.group[self.dist.last_axis(self.input_basis) - 1]
        ell = subproblem.group[self.dist.last_axis(self.input_basis)]
        if ell is None:
            n_ell = self.input_basis.Lmax + 1 - np.abs(m)
            matrix = sparse.kron(matrix, sparse.eye(n_ell))
        return matrix

    def operate(self, out):
        """Perform operation."""
        operand = self.args[0]
        # Set output layout
        layout = operand.layout
        out.preset_layout(layout)
        np.copyto(out.data, operand.data[axindex(self.index,2)])


class S2AngularComponent(operators.AngularComponent):

    basis_type = SphereBasis

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
        # Block-diag for sin/cos parts for real dtype
        if self.dtype == np.float64:
            matrix = sparse.kron(matrix, sparse.eye(2))
        # Block over ell
        m = subproblem.group[self.dist.last_axis(self.input_basis) - 1]
        ell = subproblem.group[self.dist.last_axis(self.input_basis)]
        if ell is None:
            n_ell = self.input_basis.Lmax + 1 - np.abs(m)
            matrix = sparse.kron(matrix, sparse.eye(n_ell))
        return matrix

    def operate(self, out):
        """Perform operation."""
        operand = self.args[0]
        # Set output layout
        layout = operand.layout
        out.preset_layout(layout)
        np.copyto(out.data, operand.data[axslice(self.index,0,2)])


class PolarAzimuthalComponent(operators.AzimuthalComponent):

    basis_type = IntervalBasis

    def subproblem_matrix(self, subproblem):
        operand = self.args[0]
        input_dim = len(operand.tensorsig)
        output_dim = len(self.tensorsig)
        matrix = []
        for output in range(2**output_dim):
            index_out = np.unravel_index(output, [2]*output_dim)
            matrix_row = []
            for input in range(2**input_dim):
                index_in = np.unravel_index(input, [2]*input_dim)
                if tuple(index_in[:self.index] + index_in[self.index+1:]) == index_out and index_in[self.index] == 0:
                    matrix_row.append(1)
                else:
                    matrix_row.append(0)
            matrix.append(matrix_row)
        matrix = np.array(matrix)
        if self.dtype == np.float64:
            # Block-diag for sin/cos parts for real dtype
            matrix = sparse.kron(matrix, sparse.eye(2))
        return matrix

    def operate(self, out):
        """Perform operation."""
        operand = self.args[0]
        # Set output layout
        layout = operand.layout
        out.preset_layout(layout)
        np.copyto(out.data, operand.data[axindex(self.index,0)])


class PolarRadialComponent(operators.RadialComponent):

    basis_type = IntervalBasis

    def subproblem_matrix(self, subproblem):
        operand = self.args[0]
        input_dim = len(operand.tensorsig)
        output_dim = len(self.tensorsig)
        matrix = []
        for output in range(2**output_dim):
            index_out = np.unravel_index(output, [2]*output_dim)
            matrix_row = []
            for input in range(2**input_dim):
                index_in = np.unravel_index(input, [2]*input_dim)
                if tuple(index_in[:self.index] + index_in[self.index+1:]) == index_out and index_in[self.index] == 1:
                    matrix_row.append(1)
                else:
                    matrix_row.append(0)
            matrix.append(matrix_row)
        matrix = np.array(matrix)
        if self.dtype == np.float64:
            # Block-diag for sin/cos parts for real dtype
            matrix = sparse.kron(matrix, sparse.eye(2))
        return matrix

    def operate(self, out):
        """Perform operation."""
        operand = self.args[0]
        # Set output layout
        layout = operand.layout
        out.preset_layout(layout)
        np.copyto(out.data, operand.data[axindex(self.index,1)])


class PolarAngularComponent(operators.AngularComponent):

    basis_type = IntervalBasis

    def subproblem_matrix(self, subproblem):
        # I'm not sure how to generalize this to higher order tensors, since we do
        # not have spin_weights for the S1 basis.
        matrix = np.array([[1,0]])
        if self.dtype == np.float64:
            # Block-diag for sin/cos parts for real dtype
            matrix = sparse.kron(matrix, sparse.eye(2))

#        operand = self.args[0]
#        basis = self.domain.get_basis(self.coordsys)
#        S_in = basis.spin_weights(operand.tensorsig)
#        S_out = basis.spin_weights(self.tensorsig)
#
#        matrix = []
#        for spinindex_out, spintotal_out in np.ndenumerate(S_out):
#            matrix_row = []
#            for spinindex_in, spintotal_in in np.ndenumerate(S_in):
#                if tuple(spinindex_in[:self.index] + spinindex_in[self.index+1:]) == spinindex_out and spinindex_in[self.index] == 2:
#                    matrix_row.append( 1 )
#                else:
#                    matrix_row.append( 0 )
#            matrix.append(matrix_row)
#        matrix = np.array(matrix)
        return matrix

    def operate(self, out):
        """Perform operation."""
        operand = self.args[0]
        # Set output layout
        layout = operand.layout
        out.preset_layout(layout)
        np.copyto(out.data, operand.data[axindex(self.index,0)])


class CartesianAdvectiveCFL(operators.AdvectiveCFL):

    input_coord_type = (CartesianCoordinates, Coordinate)

    @CachedMethod
    def cfl_spacing(self):
        velocity = self.operand
        coordsys = velocity.tensorsig[0]
        spacing = []
        for i, c in enumerate(coordsys.coords):
            basis = velocity.domain.get_basis(c)
            if basis:
                dealias = basis.dealias[0]
                axis_spacing = basis.local_grid_spacing(self.dist, dealias) * dealias
                N = basis.grid_shape((dealias,))[0]
            if isinstance(basis, Jacobi) and basis.a == -1/2 and basis.b == -1/2:
                #Special case for ChebyshevT (a=b=-1/2)
                local_elements = self.dist.grid_layout.local_elements(basis.domain(self.dist), scales=dealias)[i]
                i = np.arange(N)[local_elements].reshape(axis_spacing.shape)
                theta = np.pi * (i + 1/2) / N
                axis_spacing[:] = dealias * basis.COV.stretch * np.sin(theta) * np.pi / N
            elif isinstance(basis, (ComplexFourier, RealFourier)):
                #Special case for Fourier
                native_spacing = 2 * np.pi / N
                axis_spacing[:] = dealias * native_spacing * basis.COV.stretch
            elif basis is None:
                axis_spacing = np.inf
            spacing.append(axis_spacing)
        return spacing

    def compute_cfl_frequency(self, velocity, out):
        u_mag = np.abs(velocity)
        for i, dx in enumerate(self.cfl_spacing()):
            out += u_mag[i] / dx


class DirectProductAdvectiveCFL(operators.AdvectiveCFL):

    input_coord_type = DirectProduct

    def __init__(self, operand, coords):
        super().__init__(operand, coords)
        self.comps = [operators.DirectProductComponent(operand, index=0, comp=cs) for cs in coords.coordsystems]
        self.comp_cfls = [operators.AdvectiveCFL(comp, comp.comp) for comp in self.comps]

    def compute_cfl_frequency(self, velocity, out):
        # Add contributions from each component
        for comp, comp_cfl in zip(self.comps, self.comp_cfls):
            comp_velocity = velocity[comp.comp_slices]
            comp_cfl.compute_cfl_frequency(comp_velocity, out)


class PolarAdvectiveCFL(operators.AdvectiveCFL):

    input_coord_type = PolarCoordinates

    @CachedMethod
    def cfl_spacing(self):
        #Assumes velocity is a 2-length vector over polar coordinates
        basis   = self.input_basis
        dealias = basis.dealias
        azimuth_spacing = basis.local_grid_spacing(self.dist, 0, scales=dealias)
        if basis.mmax == 0:
            azimuth_spacing[:] = np.inf
        elif isinstance(basis, DiskBasis):
            azimuth_spacing[:] = basis.radius / basis.mmax
        elif isinstance(basis, AnnulusBasis):
            azimuth_spacing = basis.local_grid_radius(self.dist, dealias[1]) / basis.mmax
        radial_spacing = dealias[1] * basis.local_grid_spacing(self.dist, 1, scales=dealias)
        return [azimuth_spacing, radial_spacing]

    def compute_cfl_frequency(self, velocity, out):
        #Assumes velocity is a 2-length vector over polar coordinates
        u_theta, u_r = np.abs(velocity)
        out += u_theta / self.cfl_spacing()[0]
        out += u_r / self.cfl_spacing()[1]


class S2AdvectiveCFL(operators.AdvectiveCFL):

    input_coord_type = S2Coordinates

    @CachedMethod
    def cfl_spacing(self, r=None):
        #Assumes velocity is a 2-length vector over spherical coordinates
        basis   = self.input_basis
        dealias = basis.dealias
        if r is None:
            r = basis.radius
        s2_spacing = basis.local_grid_spacing(self.dist, 0, scales=dealias)
        if basis.Lmax == 0:
            s2_spacing[:] = np.inf
        else:
            s2_spacing[:] = r / np.sqrt(basis.Lmax*(basis.Lmax + 1))
        return [s2_spacing,]

    def compute_cfl_frequency(self, velocity, out):
        #compute u_mag * sqrt(ell*(ell+1)) / r
        #Assumes velocity is a 2-length vector over spherical coordinates
        u_phi, u_theta = velocity[0], velocity[1]
        u_mag = np.sqrt(u_phi**2 + u_theta**2)
        #Again assumes that this field only has an S2 basis
        out += u_mag / self.cfl_spacing()[0]


class Spherical3DAdvectiveCFL(operators.AdvectiveCFL):

    input_coord_type = SphericalCoordinates

    @CachedMethod
    def cfl_spacing(self):
        #Assumes velocity is a 3-length vector over spherical coordinates.
        basis   = self.input_basis
        dealias = basis.dealias
        if isinstance(basis, BallBasis):
            spacings = S2AdvectiveCFL.cfl_spacing(self, r=basis.radial_basis.radius)
        elif isinstance(basis, ShellBasis):
            spacings = S2AdvectiveCFL.cfl_spacing(self, r=1)
            spacings[0] = spacings[0] * basis.local_grid_radius(self.dist, dealias[2]) #get proper radial scaling for shell
        spacings.append(basis.local_grid_spacing(self.dist, 2, scales=dealias) * dealias[2])
        return spacings

    def compute_cfl_frequency(self, velocity, out):
        #Assumes velocity is a 3-length vector in spherical coordinates
        S2AdvectiveCFL.compute_cfl_frequency(self, velocity, out)
        u_r = np.abs(velocity[2])
        out += u_r / self.cfl_spacing()[1]


from . import transforms
