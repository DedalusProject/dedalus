"""
Abstract and built-in classes for spectral bases.

"""

import logging
import math
import numpy as np
from numpy import pi
from scipy import sparse
from scipy import fftpack
from scipy import special

from . import operators
from . import polynomials
from ..libraries.fftw import fftw_wrappers as fftw
from ..tools.config import config
from ..tools.cache import CachedAttribute
from ..tools.cache import CachedMethod
from ..tools.cache import CachedFunction
from ..tools.array import interleaved_view
from ..tools.array import reshape_vector
from ..tools.array import axslice
from ..tools.array import apply_matrix
from ..tools.exceptions import UndefinedParityError

logger = logging.getLogger(__name__.split('.')[-1])
DEFAULT_LIBRARY = config['transforms'].get('DEFAULT_LIBRARY')
FFTW_RIGOR = config['transforms-fftw'].get('PLANNING_RIGOR')
DIRICHLET_PRECONDITIONING = lambda: config['matrix construction'].getboolean('DIRICHLET_PRECONDITIONING')


class Basis:
    """
    Base class for spectral bases.

    These classes define methods for transforming, differentiating, and
    integrating corresponding series represented by their spectral coefficients.

    Parameters
    ----------
    base_grid_size : int
        Number of grid points
    interval : tuple of floats
        Spatial domain of basis
    dealias : float, optional
        Fraction of modes to keep after dealiasing (default: 1.)

    Attributes
    ----------
    grid_dtype : dtype
        Grid data type
    coeff_size : int
        Number of spectral coefficients
    coeff_embed : int
        Padded number of spectral coefficients for transform
    coeff_dtype : dtype
        Coefficient data type

    """

    def __repr__(self):
        return '<%s %i>' %(self.__class__.__name__, id(self))

    def __str__(self):
        if self.name:
            return self.name
        else:
            return self.__repr__()

    def set_dtype(self, grid_dtype):
        """Set transforms based on grid data type."""
        raise NotImplementedError()
        return coeff_dtype

    def forward(self, gdata, cdata, axis, meta, scale):
        """Grid-to-coefficient transform."""
        raise NotImplementedError()
        return cdata

    def backward(self, cdata, gdata, axis, meta, scale):
        """Coefficient-to-grid transform."""
        raise NotImplementedError()
        return gdata

    def differentiate(self, cdata, cderiv, axis):
        """Differentiate using coefficients."""
        raise NotImplementedError()

    def integrate(self, cdata, cint, axis):
        """Integrate over interval using coefficients."""
        raise NotImplementedError()

    def interpolate(self, cdata, cint, position, axis):
        """Interpolate in interval using coefficients."""
        raise NotImplementedError()

    @property
    def library(self):
        return self._library

    @library.setter
    def library(self, value):
        self.forward = getattr(self, '_forward_%s' %value.lower())
        self.backward = getattr(self, '_backward_%s' %value.lower())
        self._library = value.lower()

    def grid_size(self, scale):
        """Compute scaled grid size."""
        grid_size = float(scale) * self.base_grid_size
        if not grid_size.is_integer():
            raise ValueError("Scaled grid size is not an integer: %f" %grid_size)
        return int(grid_size)

    @CachedMethod
    def grid_spacing(self, scale=1.):
        """Compute grid spacings."""
        grid = self.grid(scale)
        return np.gradient(grid)

    @CachedMethod
    def grid_array_object(self, domain, axis):
        """Grid array object."""
        from .field import Array
        grid = Array(domain, name=self.name)
        grid.from_global_vector(self.grid(self.dealias), axis)
        return grid

    @CachedMethod
    def grid_spacing_object(self, domain, axis):
        from .field import Array
        spacing = Array(domain, name='s'+self.name)
        spacing.from_global_vector(self.grid_spacing(self.dealias), axis)
        return spacing

    def check_arrays(self, cdata, gdata, axis, scale):
        """
        Verify provided arrays sizes and dtypes are correct.
        Build compliant arrays if not provided.

        """

        if cdata is None:
            # Build cdata
            cshape = list(gdata.shape)
            cshape[axis] = self.coeff_size
            cdata = fftw.create_array(cshape, self.coeff_dtype)
        else:
            # Check cdata
            if cdata.shape[axis] != self.coeff_size:
                raise ValueError("cdata does not match coeff_size")
            if cdata.dtype != self.coeff_dtype:
                raise ValueError("cdata does not match coeff_dtype")

        grid_size = self.grid_size(scale)
        if gdata is None:
            # Build gdata
            gshape = list(cdata.shape)
            gshape[axis] = grid_size
            gdata = fftw.create_array(gshape, self.grid_dtype)
        else:
            # Check gdata
            if gdata.shape[axis] != grid_size:
                raise ValueError("gdata does not match scaled grid_size")
            if gdata.dtype != self.grid_dtype:
                raise ValueError("gdata does not match grid_dtype: gdata = {}; grid_dtype = {}".format(gdata.dtype, self.grid_dtype))

        return cdata, gdata


class TransverseBasis(Basis):
    """Base class for bases supporting transverse differentiation."""
    pass


class ImplicitBasis(Basis):
    """
    Base class for bases supporting implicit methods.

    These bases define the following matrices encoding the respective linear
    functions acting on a series represented by its spectral coefficients:

    Linear operators (square matrices):
        Pre     : preconditioning (default: identity)
        Diff    : differentiation
        Mult(p) : multiplication by p-th basis element

    Linear functionals (vectors):
        left_vector   : left-endpoint evaluation
        right_vector  : right-endpoint evaluation
        integ_vector  : integration over interval
        interp_vector : interpolation in interval

    """

    @staticmethod
    def _resize_coeffs(cdata_in, cdata_out, axis):
        """Resize coefficient data by padding/truncation."""
        size_in = cdata_in.shape[axis]
        size_out = cdata_out.shape[axis]
        if size_in < size_out:
            # Pad with higher order polynomials at end of data
            np.copyto(cdata_out[axslice(axis, 0, size_in)], cdata_in)
            np.copyto(cdata_out[axslice(axis, size_in, None)], 0)
        elif size_in > size_out:
            # Truncate higher order polynomials at end of data
            np.copyto(cdata_out, cdata_in[axslice(axis, 0, size_out)])
        else:
            np.copyto(cdata_out, cdata_in)

    def Multiply(self, p, ncc_basis_meta, arg_basis_meta):
        """p-element multiplication matrix."""
        raise NotImplementedError()

    @CachedAttribute
    def Precondition(self):
        """Preconditioning matrix."""
        raise NotImplementedError()

    @CachedMethod
    def DropTau(self, n_tau):
        """Matrix dropping tau+match rows."""
        N = self.coeff_size
        DLR = sparse.eye(N-n_tau, N, dtype=self.coeff_dtype, format='csr')
        return DLR

    @CachedAttribute
    def DropNonfirst(self):
        """Matrix dropping non-first rows."""
        N = self.coeff_size
        DNR = sparse.eye(1, N, dtype=self.coeff_dtype, format='csr')
        return DNR

    @CachedAttribute
    def DropNonconstant(self):
        """Matrix dropping non-constant rows."""
        return self.DropNonfirst

    @CachedAttribute
    def DropMatch(self):
        """Matrix dropping match rows."""
        N = self.coeff_size
        return sparse.eye(N, N, dtype=self.coeff_dtype, format='csr')

    @CachedMethod
    def PreconditionDropTau(self, n_tau):
        """Preconditioning with tau+match filtering."""
        if self.tau_after_pre:
            return self.DropTau(n_tau) @ self.Precondition
        else:
            return self.Precondition[:-n_tau,:-n_tau] @ self.DropTau(n_tau)

    @CachedAttribute
    def PreconditionDropMatch(self):
        """Preconditioning with match filtering."""
        return self.Precondition

    def NCC(self, ncc_basis_meta, arg_basis_meta, coeffs, cutoff, max_terms):
        """Build NCC multiplication matrix."""
        if max_terms is None:
            max_terms = self.coeff_size
        n_terms = max_term = matrix = 0
        for p in range(max_terms):
            if abs(coeffs[p]) >= cutoff:
                matrix = matrix + coeffs[p]*self.Multiply(p, ncc_basis_meta, arg_basis_meta)
                n_terms += 1
                max_term = p
        return n_terms, max_term, matrix


class Chebyshev(ImplicitBasis):
    """Chebyshev polynomial basis on the roots grid."""

    element_label = 'T'
    separable = False
    coupled = True

    def __init__(self, name, base_grid_size, interval=(-1,1), dealias=1, tau_after_pre=True):

        # Coordinate transformation
        # Native interval: (-1, 1)
        radius = (interval[1] - interval[0]) / 2
        center = (interval[1] + interval[0]) / 2
        self._grid_stretch = radius / 1
        self._native_coord = lambda xp: (xp - center) / radius
        self._problem_coord = lambda xn: center + (xn * radius)

        # Attributes
        self.subbases = [self]
        self.name = name
        self.element_name = 'T' + name
        self.base_grid_size = base_grid_size
        self.interval = tuple(interval)
        self.dealias = dealias
        self.tau_after_pre = tau_after_pre
        self.library = DEFAULT_LIBRARY
        self.operators = (self.Integrate,
                          self.Interpolate,
                          self.Differentiate)

    def default_meta(self):
        return {'constant': False,
                'dirichlet': DIRICHLET_PRECONDITIONING()}

    @CachedMethod
    def grid(self, scale=1.):
        """Build Chebyshev roots grid."""
        N = self.grid_size(scale)
        i = np.arange(N)
        theta = pi * (i + 1/2) / N
        native_grid = -np.cos(theta)
        return self._problem_coord(native_grid)

    @CachedMethod
    def grid_spacing(self, scale=1.):
        """Build Chebyshev roots grid spacing."""
        N = self.grid_size(scale)
        i = np.arange(N)
        theta = pi * (i + 1/2) / N
        native_spacing = np.sin(theta) * pi / N
        return native_spacing * self._grid_stretch

    def set_dtype(self, grid_dtype):
        """Determine coefficient properties from grid dtype."""
        # Transform retains data type
        self.grid_dtype = np.dtype(grid_dtype)
        self.coeff_dtype = self.grid_dtype
        # Same number of modes and grid points
        self.coeff_size = self.base_grid_size
        self.elements = np.arange(self.coeff_size)
        return self.coeff_dtype

    @staticmethod
    def _forward_scaling(pdata, axis):
        """Scale DCT output to Chebyshev coefficients."""
        # Scale as Chebyshev amplitudes
        pdata *= 1 / pdata.shape[axis]
        pdata[axslice(axis, 0, 1)] *= 0.5
        # Negate odd modes for natural grid ordering
        pdata[axslice(axis, 1, None, 2)] *= -1.

    @staticmethod
    def _backward_scaling(pdata, axis):
        """Scale Chebyshev coefficients to IDCT input."""
        # Negate odd modes for natural grid ordering
        pdata[axslice(axis, 1, None, 2)] *= -1.
        # Scale from Chebyshev amplitudes
        pdata[axslice(axis, 1, None)] *= 0.5

    def _forward_scipy(self, gdata, cdata, axis, meta, scale):
        """Forward transform using scipy DCT."""
        cdata, gdata = self.check_arrays(cdata, gdata, axis, scale)
        complex = (gdata.dtype == np.complex128)
        # View complex data as interleaved real data
        if complex:
            cdata_complex = cdata
            gdata = interleaved_view(gdata)
            cdata = interleaved_view(cdata)
        # Scipy DCT
        temp = fftpack.dct(gdata, type=2, axis=axis)
        # Scale DCT output to Chebyshev coefficients
        self._forward_scaling(temp, axis)
        # Pad / truncate coefficients
        self._resize_coeffs(temp, cdata, axis)
        if complex:
            cdata = cdata_complex
        return cdata

    def _backward_scipy(self, cdata, gdata, axis, meta, scale):
        """Backward transform using scipy IDCT."""
        cdata, gdata = self.check_arrays(cdata, gdata, axis, scale)
        complex = (gdata.dtype == np.complex128)
        # Pad / truncate coefficients
        # Store in gdata for memory efficiency (transform preserves shape/dtype)
        self._resize_coeffs(cdata, gdata, axis)
        # Scale Chebyshev coefficients to IDCT input
        self._backward_scaling(gdata, axis)
        # View complex data as interleaved real data
        if complex:
            gdata_complex = gdata
            gdata = interleaved_view(gdata)
        # Scipy IDCT
        temp = fftpack.dct(gdata, type=3, axis=axis)
        np.copyto(gdata, temp)
        if complex:
            gdata = gdata_complex
        return gdata

    @CachedMethod
    def _fftw_setup(self, dtype, gshape, axis):
        """Build FFTW plans and temporary arrays."""
        # Note: regular method used to cache through basis instance
        logger.debug("Building FFTW DCT plan for (dtype, gshape, axis) = (%s, %s, %s)" %(dtype, gshape, axis))
        flags = ['FFTW_'+FFTW_RIGOR.upper()]
        plan = fftw.DiscreteCosineTransform(dtype, gshape, axis, flags=flags)
        temp = fftw.create_array(gshape, dtype)

        return plan, temp

    def _forward_fftw(self, gdata, cdata, axis, meta, scale):
        """Forward transform using FFTW DCT."""
        cdata, gdata = self.check_arrays(cdata, gdata, axis, scale)
        plan, temp = self._fftw_setup(gdata.dtype, gdata.shape, axis)
        # Execute FFTW plan
        plan.forward(gdata, temp)
        # Scale DCT output to Chebyshev coefficients
        self._forward_scaling(temp, axis)
        # Pad / truncate coefficients
        self._resize_coeffs(temp, cdata, axis)
        return cdata

    def _backward_fftw(self, cdata, gdata, axis, meta, scale):
        """Backward transform using FFTW IDCT."""
        cdata, gdata = self.check_arrays(cdata, gdata, axis, scale)
        plan, temp = self._fftw_setup(gdata.dtype, gdata.shape, axis)
        # Pad / truncate coefficients
        self._resize_coeffs(cdata, temp, axis)
        # Scale Chebyshev coefficients to IDCT input
        self._backward_scaling(temp, axis)
        # Execute FFTW plan
        plan.backward(temp, gdata)
        return gdata

    @CachedAttribute
    def Integrate(self):
        """Build integration class."""

        class IntegrateChebyshev(operators.Integrate, operators.Coupled):
            name = 'integ_{}'.format(self.name)
            basis = self

            def matrix_form(self):
                # Call class method with arg metadata
                return self._cls_matrix_form(self.args[0].meta[self.axis])

            @classmethod
            def _cls_matrix_form(cls, arg_basis_meta):
                # Call cached constructor
                return cls._integ_matrix()

            @classmethod
            def _cls_vector_form(cls, arg_basis_meta):
                # Call cached constructor
                return cls._integ_vector()

            @classmethod
            @CachedMethod
            def _integ_matrix(cls):
                """Chebyshev integration: int(T_n) = (1 + (-1)^n) / (1 - n^2)"""
                size = cls.basis.coeff_size
                matrix = sparse.lil_matrix((size, size), dtype=cls.basis.coeff_dtype)
                matrix[0,:] = cls._integ_vector()
                return matrix.tocsr()

            @classmethod
            @CachedMethod
            def _integ_vector(cls):
                """Chebyshev integration: int(T_n) = (1 + (-1)^n) / (1 - n^2)"""
                vector = np.zeros(cls.basis.coeff_size, dtype=cls.basis.coeff_dtype)
                for n in range(0, cls.basis.coeff_size, 2):
                    vector[n] = 2. / (1. - n*n)
                vector *= cls.basis._grid_stretch
                return vector

        return IntegrateChebyshev

    @CachedAttribute
    def Interpolate(self):
        """Buld interpolation class."""

        class InterpolateChebyshev(operators.Interpolate, operators.Coupled):
            name = 'interp_{}'.format(self.name)
            basis = self

            def matrix_form(self):
                # Call class method with arg metadata
                return self._cls_matrix_form(self.args[0].meta[self.axis], self.position)

            @classmethod
            def _cls_matrix_form(cls, arg_basis_meta, position):
                # Call cached constructor
                return cls._interp_matrix(position)

            @classmethod
            def _cls_vector_form(cls, arg_basis_meta, position):
                # Call cached constructor
                return cls._interp_vector(position)

            @classmethod
            @CachedMethod
            def _interp_matrix(cls, position):
                size = cls.basis.coeff_size
                matrix = sparse.lil_matrix((size, size), dtype=cls.basis.coeff_dtype)
                matrix[0,:] = cls._interp_vector(position)
                return matrix.tocsr()

            @classmethod
            @CachedMethod
            def _interp_vector(cls, position):
                """Chebyshev interpolation: Tn(xn) = cos(n * acos(xn))"""
                if position == 'left':
                    theta = np.pi
                elif position == 'right':
                    theta = 0
                elif position == 'center':
                    theta = np.pi / 2
                else:
                    xn = cls.basis._native_coord(position)
                    theta = np.arccos(xn)
                return np.cos(cls.basis.elements * theta)

        return InterpolateChebyshev

    @CachedAttribute
    def Differentiate(self):
        """Build differentiation class."""

        class DifferentiateChebyshev(operators.Differentiate, operators.Coupled):
            name = 'd' + self.name
            basis = self

            def matrix_form(self):
                # Call class method with arg metadata
                return self._cls_matrix_form(self.args[0].meta[self.axis])

            @classmethod
            def _cls_matrix_form(cls, arg_basis_meta):
                # Call cached constructor
                return cls._diff_matrix()

            @classmethod
            @CachedMethod
            def _diff_matrix(cls):
                """Chebyshev differentiation: d_x(T_n) / n = 2 T_(n-1) + d_x(T_(n-2)) / (n-2)"""
                size = cls.basis.coeff_size
                dtype = cls.basis.coeff_dtype
                stretch = cls.basis._grid_stretch
                matrix = sparse.lil_matrix((size, size), dtype=dtype)
                for i in range(size-1):
                    for j in range(i+1, size, 2):
                        if i == 0:
                            matrix[i, j] = j / stretch
                        else:
                            matrix[i, j] = 2. * j / stretch
                return matrix.tocsr()

            def explicit_form(self, input, output, axis):
                """Differentiation by recursion on coefficients."""
                shape = input.shape
                # Currently setup just for last axis
                if axis != -1:
                    if axis != (len(shape) - 1):
                        raise NotImplementedError("Chebyshev derivative only implemented for last axis.")
                # Create 2D views of arrays
                reduced_shape = (int(np.prod(shape[:-1])), shape[-1])
                input_view = input.reshape(reduced_shape)
                output_view = output.reshape(reduced_shape)
                # Call cythonized derivative
                polynomials.chebyshev_derivative_2d(input_view, output_view)
                # Scale for interval
                output /= self.basis._grid_stretch

        return DifferentiateChebyshev

    @CachedAttribute
    def Precondition(self):
        """
        Preconditioning matrix.

        T_n = (U_n - U_(n-2)) / 2
        U_(-n) = -U_(n-2)

        """
        size = self.coeff_size
        # Construct sparse matrix
        Pre = sparse.lil_matrix((size, size), dtype=self.coeff_dtype)
        Pre[0, 0] = 1.
        Pre[1, 1] = 0.5
        for n in range(2, size):
            Pre[n, n] = 0.5
            Pre[n-2, n] = -0.5
        return Pre.tocsr()

    @CachedAttribute
    def Dirichlet(self):
        """
        Dirichlet recombination matrix.

        D[0] = T[0]
        D[1] = T[1]
        D[n] = T[n] - T[n-2]

        <T[i]|D[j]> = <T[i]|T[j]> - <T[i]|T[j-2]>
                    = δ(i,j) - δ(i,j-2)
        """
        size = self.coeff_size
        # Construct sparse matrix
        Dir = sparse.lil_matrix((size, size), dtype=self.coeff_dtype)
        for n in range(size):
            Dir[n, n] = 1
            if n > 1:
                Dir[n-2, n] = -1
        return Dir.tocsr()

    def Multiply(self, p, ncc_basis_meta, arg_basis_meta):
        # No meta dependency -- just cache on p
        return self._Multiply_T_T(p)

    @CachedMethod
    def _Multiply_T_T(self, p):
        """
        Multiplication matrix:
            T_p * T_n = (T_(n+p) + T_(n-p)) / 2
            T_(-n) = T_n
        """
        size = self.coeff_size
        # Construct sparse matrix
        Mult = sparse.lil_matrix((size, size), dtype=self.coeff_dtype)
        for n in range(size):
            upper = n + p
            if upper < size:
                Mult[upper, n] += 0.5
            lower = abs(n - p)
            if lower < size:
                Mult[lower, n] += 0.5
        return Mult.tocsr()


class Legendre(ImplicitBasis):
    """Legendre polynomial basis on the roots grid."""

    element_label = 'P'
    separable = False
    coupled = True

    def __init__(self, name, base_grid_size, interval=(-1,1), dealias=1, tau_after_pre=True):

        # Coordinate transformation
        # Native interval: (-1, 1)
        radius = (interval[1] - interval[0]) / 2
        center = (interval[1] + interval[0]) / 2
        self._grid_stretch = radius / 1
        self._native_coord = lambda xp: (xp - center) / radius
        self._problem_coord = lambda xn: center + (xn * radius)

        # Attributes
        self.subbases = [self]
        self.name = name
        self.element_name = 'P' + name
        self.base_grid_size = base_grid_size
        self.interval = tuple(interval)
        self.dealias = dealias
        self.tau_after_pre = tau_after_pre
        self.library = 'MMT'
        self.operators = (self.Integrate,
                          self.Interpolate,
                          self.Differentiate)

    def default_meta(self):
        return {'constant': False,
                'dirichlet': DIRICHLET_PRECONDITIONING()}

    @CachedMethod
    def grid(self, scale=1.):
        """Build Legendre polynomial roots grid."""
        N = self.grid_size(scale)
        native_grid, _ = special.roots_legendre(N)
        return self._problem_coord(native_grid)

    def set_dtype(self, grid_dtype):
        """Determine coefficient properties from grid dtype."""
        # Transform retains data type
        self.grid_dtype = np.dtype(grid_dtype)
        self.coeff_dtype = self.grid_dtype
        # Same number of modes and grid points
        self.coeff_size = self.base_grid_size
        self.elements = np.arange(self.coeff_size)
        return self.coeff_dtype

    @staticmethod
    def _recursion(N, X):
        # Preallocate list for outputs
        Y = [None] * N
        # Use powers to get identity element of X type and copy X
        Y[0] = X**0
        if N == 1:
            return Y
        Y[1] = X**1
        # Run recursion
        for n in range(1, N-1):
            Y[n+1] = (2*n + 1) / (n + 1) * X * Y[n] - n / (n + 1) * Y[n-1]
        return Y

    @classmethod
    @CachedMethod
    def _Jacobi_matrix(cls, N):
        # X * Y[n] = (n + 1) / (2*n + 1) * Y[n+1] + n / (2*n + 1) * Y[n-1]
        J = sparse.lil_matrix((N, N))
        J[1, 0] = 1
        for n in range(1, N):
            J[n-1, n] = n / (2*n + 1)
            if n < N - 1:
                J[n+1, n] = (n + 1) / (2*n + 1)
        return J.tocsr()

    @classmethod
    def _evaluate_basis_functions(cls, N, xn):
        """Evaluate basis functions on a specified native grid."""
        # Run recursion in higher precision
        g = cls._recursion(N, xn.astype(np.longdouble))
        return np.array(g).astype(xn.dtype)

    @CachedMethod
    def _mmt_setup(self, gsize, csize):
        """Build transform matrices."""
        logger.debug("Building Legendre MMT matrices for (gsize, csize) = (%s, %s)" %(gsize, csize))
        # Get grid and weights
        native_grid, weights = special.roots_legendre(gsize)
        # Evaluate polynomials on grid
        functions = self._evaluate_basis_functions(csize, native_grid)
        backward_mat = functions.T.copy()
        # Forward transform:
        # sum w P[n] P[m] = int P[n] P[m] dx
        #                 = δ[m,n] N[n]^2
        forward_mat = weights * functions
        forward_mat[gsize:] = 0
        n = np.arange(csize)[:, None]
        N2 = 2 / (2*n + 1)
        forward_mat /= N2
        return forward_mat, backward_mat

    def _forward_mmt(self, gdata, cdata, axis, meta, scale):
        """Forward transform using matrix transform."""
        cdata, gdata = self.check_arrays(cdata, gdata, axis, scale)
        # Apply transform matrices
        forward_mat, _ = self._mmt_setup(gdata.shape[axis], cdata.shape[axis])
        apply_matrix(forward_mat, gdata, axis, out=cdata)
        return cdata

    def _backward_mmt(self, cdata, gdata, axis, meta, scale):
        """Forward transform using inverse matrix transform."""
        cdata, gdata = self.check_arrays(cdata, gdata, axis, scale)
        # Apply transform matrices
        _, backward_mat = self._mmt_setup(gdata.shape[axis], cdata.shape[axis])
        apply_matrix(backward_mat, cdata, axis, out=gdata)
        return gdata

    @CachedAttribute
    def Integrate(self):
        """Build integration class."""

        class IntegrateLegendre(operators.Integrate, operators.Coupled):
            name = 'integ_{}'.format(self.name)
            basis = self

            def matrix_form(self):
                # Call class method with arg metadata
                return self._cls_matrix_form(self.args[0].meta[self.axis])

            @classmethod
            def _cls_matrix_form(cls, arg_basis_meta):
                # Call cached constructor
                return cls._integ_matrix()

            @classmethod
            def _cls_vector_form(cls, arg_basis_meta):
                # Call cached constructor
                return cls._integ_vector()

            @classmethod
            @CachedMethod
            def _integ_matrix(cls):
                """Legendre integration: int(P_n) = 2 δ(n,0)"""
                size = cls.basis.coeff_size
                matrix = sparse.lil_matrix((size, size), dtype=cls.basis.coeff_dtype)
                matrix[0,:] = cls._integ_vector()
                return matrix.tocsr()

            @classmethod
            @CachedMethod
            def _integ_vector(cls):
                """Legendre integration: int(P_n) = 2 δ(n,0)"""
                vector = np.zeros(cls.basis.coeff_size, dtype=cls.basis.coeff_dtype)
                vector[0] = 2
                vector *= cls.basis._grid_stretch
                return vector

        return IntegrateLegendre

    @CachedAttribute
    def Interpolate(self):
        """Buld interpolation class."""

        class InterpolateLegendre(operators.Interpolate, operators.Coupled):
            name = 'interp_{}'.format(self.name)
            basis = self

            def matrix_form(self):
                # Call class method with arg metadata
                return self._cls_matrix_form(self.args[0].meta[self.axis], self.position)

            @classmethod
            def _cls_matrix_form(cls, arg_basis_meta, position):
                # Call cached constructor
                return cls._interp_matrix(position)

            @classmethod
            def _cls_vector_form(cls, arg_basis_meta, position):
                # Call cached constructor
                return cls._interp_vector(position)

            @classmethod
            @CachedMethod
            def _interp_matrix(cls, position):
                size = cls.basis.coeff_size
                matrix = sparse.lil_matrix((size, size), dtype=cls.basis.coeff_dtype)
                matrix[0,:] = cls._interp_vector(position)
                return matrix.tocsr()

            @classmethod
            @CachedMethod
            def _interp_vector(cls, position):
                if position == 'left':
                    xn = -1
                elif position == 'right':
                    xn = 1
                elif position == 'center':
                    xn = 0
                else:
                    xn = cls.basis._native_coord(position)
                xn = np.array([float(xn)])
                functions = cls.basis._evaluate_basis_functions(cls.basis.coeff_size, xn)
                return functions[:, 0]

        return InterpolateLegendre

    @CachedAttribute
    def Differentiate(self):
        """Build differentiation class."""

        class DifferentiateLegendre(operators.Differentiate, operators.Coupled):
            name = 'd' + self.name
            basis = self

            def matrix_form(self):
                # Call class method with arg metadata
                return self._cls_matrix_form(self.args[0].meta[self.axis])

            @classmethod
            def _cls_matrix_form(cls, arg_basis_meta):
                # Call cached constructor
                return cls._diff_matrix()

            @classmethod
            @CachedMethod
            def _diff_matrix(cls):
                """Legendre differentiation: d_x(P_n) = (2*n - 1) * P_(n-1) + d_x(P_(n-2))"""
                size = cls.basis.coeff_size
                dtype = cls.basis.coeff_dtype
                stretch = cls.basis._grid_stretch
                matrix = sparse.lil_matrix((size, size), dtype=dtype)
                matrix[0, 1] = 1 / stretch
                for j in range(2, size):
                    matrix[j-1, j] = (2*j - 1) / stretch
                    for i in range(j-3, -1, -2):
                        matrix[i, j] = matrix[i, j-2]
                return matrix.tocsr()

            def explicit_form(self, input, output, axis):
                """Differentiation by recursion on coefficients."""
                shape = input.shape
                # Currently setup just for last axis
                if axis != -1:
                    if axis != (len(shape) - 1):
                        raise NotImplementedError("Legendre derivative only implemented for last axis.")
                # Create 2D views of arrays
                reduced_shape = (int(np.prod(shape[:-1])), shape[-1])
                input_view = input.reshape(reduced_shape)
                output_view = output.reshape(reduced_shape)
                # Call cythonized derivative
                polynomials.legendre_derivative_2d(input_view, output_view)
                # Scale for interval
                output /= self.basis._grid_stretch

        return DifferentiateLegendre

    @CachedAttribute
    def Precondition(self):
        """
        Preconditioning matrix.
            2 * (2*n + 1) * P[n] = (n + 2) * J11[n] - n * J11[n-2]
        """
        size = self.coeff_size
        # Construct sparse matrix
        Pre = sparse.lil_matrix((size, size), dtype=self.coeff_dtype)
        for n in range(size):
            Pre[n, n] = (n + 2) / 2 / (2*n + 1)
            if n >= 2:
                Pre[n-2, n] = - n / 2 / (2*n + 1)
        return Pre.tocsr()

    @CachedAttribute
    def Dirichlet(self):
        """
        Dirichlet recombination matrix.

        D[0] = P[0]
        D[1] = P[1]
        D[n] = P[n] - P[n-2]

        <P[i]|D[j]> = <P[i]|T[j]> - <P[i]|T[j-2]>
                    = δ(i,j) - δ(i,j-2)
        """
        size = self.coeff_size
        # Construct sparse matrix
        Dir = sparse.lil_matrix((size, size), dtype=self.coeff_dtype)
        for n in range(size):
            Dir[n, n] = 1
            if n > 1:
                Dir[n-2, n] = -1
        return Dir.tocsr()

    def Multiply(self, p, ncc_basis_meta, arg_basis_meta):
        # No meta dependency -- just cache on p
        return self._Multiply(p)

    @CachedMethod
    def _Multiply(self, n):
        # Return slice of extended matrix
        N = self.coeff_size
        return self._Multiply_ext(n)[:N, :N]

    @CachedMethod
    def _Multiply_ext(self, n):
        # P[n] = (2*n - 1) / n * x * P[n-1] - (n-1) / n * P[n-2]
        # Recurse on Jacobi matrix
        # Use size 2*N to avoid truncation issues
        N = self.coeff_size
        J = self._Jacobi_matrix(2*N)
        if n == 0:
            return (J**0).tocsr()
        elif n == 1:
            return J.copy()
        else:
            P1 = self._Multiply_ext(n - 1)
            P2 = self._Multiply_ext(n - 2)
            return (2*n - 1) / n * J * P1 - (n-1) / n * P2


class Hermite(ImplicitBasis):
    """
    Hermite function/polynomial basis.

    Interval:
        The functions live on (-inf, inf) but are centered and scaled by the
        affine transformation from (-1, 1) to the specified interval.

    Hermite functions (with envelope, default):
        hf[n](xp) = exp(-xn^2/2) H[n](xn) / N[n]
        N[n]**2 = π^(1/2) 2^n n!
        integ hf[n] hf[m] dxn = δ[n,m]

    Hermite polynomials (without envelope):
        hp[n](xp) = H[n](xn)
        integ hp[n] hp[m] exp(-xn**2) dx = δ[n,m] N[n]**2

    """

    element_label = 'h'
    separable = False
    coupled = True

    def __init__(self, name, base_grid_size, center=0.0, stretch=1.0, dealias=1, tau_after_pre=False):

        # Coordinate transformation
        self.center = center
        self.stretch = stretch
        self._grid_stretch = stretch
        self._native_coord = lambda xp: (xp - center) / stretch
        self._problem_coord = lambda xn: center + (xn * stretch)
        oriented_interval = self._problem_coord(np.array([-np.inf, np.inf]))

        # Attributes
        self.subbases = [self]
        self.name = name
        self.element_name = 'h' + name
        self.base_grid_size = base_grid_size
        self.interval = sorted(tuple(oriented_interval))
        self.dealias = dealias
        self.tau_after_pre = tau_after_pre
        self.library = 'MMT'
        self.operators = (self.Integrate,
                          self.Interpolate,
                          self.Differentiate)

    def default_meta(self):
        return {'constant': False,
                'dirichlet': False,
                'envelope': True}

    @CachedMethod
    def grid(self, scale=1.):
        """Build Hermite polynomial roots grid."""
        N = self.grid_size(scale)
        native_grid, _ = special.roots_hermite(N)
        return self._problem_coord(native_grid)

    @CachedMethod
    def grid_array_object(self, domain, axis):
        """Grid array object."""
        grid = super().grid_array_object(domain, axis)
        grid.meta[axis]['envelope'] = False
        return grid

    def set_dtype(self, grid_dtype):
        """Determine coefficient properties from grid dtype."""
        # Transform retains data type
        self.grid_dtype = np.dtype(grid_dtype)
        self.coeff_dtype = self.grid_dtype
        # Same number of modes and grid points
        self.coeff_size = self.base_grid_size
        self.elements = np.arange(self.coeff_size)
        return self.coeff_dtype

    @staticmethod
    def _recursion(N, X, envelope):
        # Implement recursion relation from Boyd A.6
        # Preallocate list for outputs
        Y = [None] * N
        # Use powers to get identity element of X type and copy X
        if envelope:
            Y[0] = np.pi**(-1/4) * np.exp(-0.5*X**2)
            Y[1] = np.sqrt(2) * X * Y[0]
            for n in range(1, N-1):
                Y[n+1] = np.sqrt(2/(n+1)) * X * Y[n] - np.sqrt(n/(n+1)) * Y[n-1]
        else:
            Y[0] = X**0
            Y[1] = 2 * X
            for n in range(1, N-1):
                Y[n+1] = 2 * X * Y[n] - 2 * n * Y[n-1]
        return Y

    @classmethod
    @CachedMethod
    def _Jacobi_matrix(cls, N):
        # X * Y[n] = 1 / 2 * Y[n+1] + n * Y[n-1]
        J = sparse.lil_matrix((N, N))
        for n in range(N):
            if n > 0:
                J[n-1, n] = n
            if n < N - 1:
                J[n+1, n] = 0.5
        return J.tocsr()

    @classmethod
    def _evaluate_basis_functions(cls, N, xn, envelope):
        """Evaluate basis functions on a specified native grid."""
        # Run recursion in higher precision
        h = cls._recursion(N, xn.astype(np.longdouble), envelope)
        return np.array(h).astype(xn.dtype)

    @CachedMethod
    def _mmt_setup(self, gsize, csize, envelope):
        """Build transform matrices."""
        logger.debug("Building Hermite MMT matrices for (gsize, csize, env) = (%s, %s, %s)" %(gsize, csize, envelope))
        # Get grid and weights
        native_grid, weights = special.roots_hermite(gsize)
        # Evaluate polynomials on grid
        functions = self._evaluate_basis_functions(csize, native_grid, envelope)
        backward_mat = functions.T.copy()
        # Forward transform:
        # sum w H[n] H[m] = int H[n] H[m] exp(-x^2) dx
        #                 = δ[m,n] N[n]^2
        # sum w hp[n] hp[m] = δ[m,n] N[n]^2
        # sum w exp(x^2) hf[n] hf[m] = δ[m,n]
        forward_mat = weights * functions
        forward_mat[gsize:] = 0
        if envelope:
            forward_mat *= np.exp(native_grid**2)
        else:
            n_int = np.arange(csize, dtype=int)
            n_long = np.longdouble(n_int)
            N2 = np.pi**(1/2) * 2**n_long * special.factorial(n_int, exact=True).astype(np.longdouble)
            N2i = 1 / N2
            forward_mat *= N2i.astype(np.float64)[:, None]
        return forward_mat, backward_mat

    def _forward_mmt(self, gdata, cdata, axis, meta, scale):
        """Forward transform using matrix transform."""
        cdata, gdata = self.check_arrays(cdata, gdata, axis, scale)
        # Apply transform matrices
        forward_mat, _ = self._mmt_setup(gdata.shape[axis], cdata.shape[axis], meta['envelope'])
        apply_matrix(forward_mat, gdata, axis, out=cdata)
        return cdata

    def _backward_mmt(self, cdata, gdata, axis, meta, scale):
        """Forward transform using inverse matrix transform."""
        cdata, gdata = self.check_arrays(cdata, gdata, axis, scale)
        # Apply transform matrices
        _, backward_mat = self._mmt_setup(gdata.shape[axis], cdata.shape[axis], meta['envelope'])
        apply_matrix(backward_mat, cdata, axis, out=gdata)
        return gdata

    @CachedAttribute
    def Integrate(self):
        """Build integration class."""

        class IntegrateHermite(operators.Integrate, operators.Coupled):
            name = 'integ_{}'.format(self.name)
            basis = self

            def meta_envelope(self, axis):
                if axis == self.axis:
                    # Check current envelope
                    if self.args[0].meta[axis]['envelope']:
                        # Integral is a constant
                        return False
                    else:
                        raise ValueError("Hermite polynomials are non-integrable.")
                else:
                    # Preserve envelope
                    return self.args[0].meta[axis]['envelope']

            def matrix_form(self):
                # Call class method with arg metadata
                return self._cls_matrix_form(self.args[0].meta[self.axis])

            @classmethod
            def _cls_matrix_form(cls, arg_basis_meta):
                # Call cached constructor with required metadata
                return cls._integ_matrix(arg_basis_meta['envelope'])

            @classmethod
            def _cls_vector_form(cls, arg_basis_meta):
                # Call cached constructor with required metadata
                return cls._integ_vector(arg_basis_meta['envelope'])

            @classmethod
            @CachedMethod
            def _integ_matrix(cls, envelope):
                """Hermite integration."""
                size = cls.basis.coeff_size
                matrix = sparse.lil_matrix((size, size), dtype=cls.basis.coeff_dtype)
                matrix[0,:] = cls._integ_vector(envelope)
                return matrix.tocsr()

            @classmethod
            def _integ_vector(cls, envelope):
                """
                Hermite integration:
                    dx(hf[n]) = sqrt(n/2) hf[n-1] - sqrt((n+1)/2) hf[n+1]
                    int(hf[n]) = sqrt((n-1)/n) int(hf[n-2])
                    int(hf[0]) = sqrt(2) pi^(1/4)
                """
                if not envelope:
                    raise ValueError("Hermite polynomials are non-integrable.")
                vector = np.zeros(cls.basis.coeff_size, dtype=cls.basis.coeff_dtype)
                vector[0] = (4*np.pi)**(1/4)
                for n in range(2, cls.basis.coeff_size, 2):
                    vector[n] = np.sqrt((n-1)/n) * vector[n-2]
                vector *= np.abs(cls.basis._grid_stretch)
                return vector

        return IntegrateHermite

    @CachedAttribute
    def Interpolate(self):
        """Buld interpolation class."""

        class InterpolateHermite(operators.Interpolate, operators.Coupled):
            name = 'interp_{}'.format(self.name)
            basis = self

            def meta_envelope(self, axis):
                if axis == self.axis:
                    # Interpolation is a constant
                    return False
                else:
                    # Preserve envelope
                    return self.args[0].meta[axis]['envelope']

            def matrix_form(self):
                # Call class method with arg metadata
                return self._cls_matrix_form(self.args[0].meta[self.axis], self.position)

            @classmethod
            def _cls_matrix_form(cls, arg_basis_meta, position):
                # Call cached constructor with required metadata
                return cls._interp_matrix(arg_basis_meta['envelope'], position)

            @classmethod
            def _cls_vector_form(cls, arg_basis_meta, position):
                # Call cached constructor with required metadata
                return cls._interp_vector(arg_basis_meta['envelope'], position)

            @classmethod
            @CachedMethod
            def _interp_matrix(cls, envelope, position):
                size = cls.basis.coeff_size
                matrix = sparse.lil_matrix((size, size), dtype=cls.basis.coeff_dtype)
                matrix[0,:] = cls._interp_vector(envelope,  position)
                return matrix.tocsr()

            @classmethod
            @CachedMethod
            def _interp_vector(cls, envelope, position):
                """Hermite interpolation."""
                if position == 'left':
                    # Corresponds to xn = -inf
                    if envelope:
                        return np.zeros(cls.basis.coeff_size)
                    else:
                        raise ValueError("Hermite polynomials blow up at infinity.")
                elif position == 'right':
                    # Corresponds to xn = inf
                    if envelope:
                        return np.zeros(cls.basis.coeff_size)
                    else:
                        raise ValueError("Hermite polynomials blow up at infinity.")
                elif position == 'center':
                    xn = 0
                else:
                    xn = cls.basis._native_coord(position)
                xn = np.array([xn])
                functions = cls.basis._evaluate_basis_functions(cls.basis.coeff_size, xn, envelope)
                return functions[:, 0]

        return InterpolateHermite

    @CachedAttribute
    def Differentiate(self):
        """Build differentiation class."""

        class DifferentiateHermite(operators.Differentiate, operators.Coupled):
            name = 'd' + self.name
            basis = self

            def meta_envelope(self, axis):
                # Preserve envelope
                return self.args[0].meta[axis]['envelope']

            @CachedMethod
            def matrix_form(self):
                # Call class method with arg metadata
                return self._cls_matrix_form(self.args[0].meta[self.axis])

            @classmethod
            def _cls_matrix_form(cls, arg_basis_meta):
                # Call cached constructor with required metadata
                return cls._diff_matrix(arg_basis_meta['envelope'])

            @classmethod
            @CachedMethod
            def _diff_matrix(cls, envelope):
                """
                Hermite differentiation:
                    dx(hf[n]) = sqrt(n/2) hf[n-1] - sqrt((n+1)/2) hf[n+1]
                    dx(hp[n]) = dx(H[n]) = 2 n H[n-1] = 2 n hp[n-1]
                """
                size = cls.basis.coeff_size
                dtype = cls.basis.coeff_dtype
                stretch = cls.basis._grid_stretch
                matrix = sparse.lil_matrix((size, size), dtype=dtype)
                if envelope:
                    for n in range(size):
                        if n > 0:
                            matrix[n-1, n] = np.sqrt(n/2) / stretch
                        if n < (size-1):
                            matrix[n+1, n] = -np.sqrt((n+1)/2) / stretch
                else:
                    for n in range(size):
                        if n > 0:
                            matrix[n-1, n] = 2 * n / stretch
                return matrix.tocsr()

        return DifferentiateHermite

    @CachedAttribute
    def Precondition(self):
        """Preconditioning matrix."""
        size = self.coeff_size
        return sparse.identity(size, format='csr')

    @CachedAttribute
    def Dirichlet(self):
        """Dirichlet recombination matrix."""
        size = self.coeff_size
        return sparse.identity(size, format='csr')

    def Multiply(self, p, ncc_basis_meta, arg_basis_meta):
        """Hermite multiplication matrix."""
        # Only multiply by polynomials
        if ncc_basis_meta['envelope']:
            raise ValueError("Cannot use enveloped functions for NCCs.")
        # Dispatch based on arg envelope
        if arg_basis_meta['envelope']:
            return self._Multiply_poly_func(p)
        else:
            return self._Multiply_poly_poly(p)

    @CachedMethod
    def _Multiply_poly_poly(self, n):
        # Return slice of extended matrix
        N = self.coeff_size
        return self._Multiply_poly_poly_ext(n)[:N, :N]

    @CachedMethod
    def _Multiply_poly_poly_ext(self, n):
        """
        Multiplication matrix:
            hp[i] hp[j] = Mpp[i][k,j] hp[k]

        Mpp constructed from recursion on Jacobi matrix.
        """
        # H[n] = 2 * x * H[n-1] - 2 * (n - 1) * H[n-2]
        # Recurse on Jacobi matrix
        # Use size 2*N to avoid truncation issues
        N = self.coeff_size
        J = self._Jacobi_matrix(2*N)
        if n == 0:
            return (J**0).tocsr()
        elif n == 1:
            return 2 * J
        else:
            L1 = self._Multiply_poly_poly_ext(n - 1)
            L2 = self._Multiply_poly_poly_ext(n - 2)
            return 2 * J * L1 - 2 * (n - 1) * L2

    @CachedMethod
    def _Multiply_poly_func(self, p):
        """
        Multiplication matrix:
            hp[p] hf[n] = Mpf[p][k,n] hf[k]
                        = exp(-x^2/2) / N[n] hp[p] hp[n]
                        = exp(-x^2/2) / N[n] Mpp[p][k,n] hp[k]
                        = N[k] / N[n] Mpp[p][k,n] hf[k]
            Mpf[p][k,n] = N[k] / N[n] Mpp[p][k,n]
        """
        # Reweight poly-poly matrix
        n_int = np.arange(self.coeff_size, dtype=int)
        n_long = np.longdouble(n_int)
        N = np.pi**(1/4) * 2**(n_long/2) * special.factorial(n_int, exact=True).astype(np.longdouble)**(1/2)
        Ni = 1 / N
        Narr = sparse.diags(N.astype(np.float64))
        Ninv = sparse.diags(Ni.astype(np.float64))
        return Narr @ self._Multiply_poly_poly(p) @ Ninv


class Laguerre(ImplicitBasis):
    """
    Laguerre function/polynomial basis.

    Interval:
        The functions live on (0, inf) but are centered and scaled by the
        affine transformation from (0, 1) to the specified interval.

    Laguerre functions (with envelope, default):
        gn[n](xp) = exp(-x/2) L[n](xn)
        integ gn[n] gn[m] dxn = δ[n,m]

    Laguerre polynomials (without envelope):
        gp[n](xp) = L[n](xn)
        integ gp[n] gp[m] exp(-xn) dx = δ[n,m]

    """

    element_label = 'g'
    separable = False
    coupled = True

    def __init__(self, name, base_grid_size, edge=0.0, stretch=1.0, dealias=1, tau_after_pre=False):

        # Coordinate transformation
        self.edge = edge
        self.stretch = stretch
        self._grid_stretch = stretch
        self._native_coord = lambda xp: (xp - edge) / stretch
        self._problem_coord = lambda xn: edge + stretch * xn
        oriented_interval = self._problem_coord(np.array([0, np.inf]))

        # Attributes
        self.subbases = [self]
        self.name = name
        self.element_name = 'g' + name
        self.base_grid_size = base_grid_size
        self.interval = sorted(tuple(oriented_interval))
        self.dealias = dealias
        self.tau_after_pre = tau_after_pre
        self.library = 'MMT'
        self.operators = (self.Integrate,
                          self.Interpolate,
                          self.Differentiate)

    def default_meta(self):
        return {'constant': False,
                'dirichlet': DIRICHLET_PRECONDITIONING(),
                'envelope': True}

    @CachedMethod
    def grid(self, scale=1.):
        """Build Laguerre polynomial roots grid."""
        N = self.grid_size(scale)
        native_grid, _ = special.roots_laguerre(N)
        if self.stretch < 0:
            native_grid = native_grid[::-1].copy()
        return self._problem_coord(native_grid)

    @CachedMethod
    def grid_array_object(self, domain, axis):
        """Grid array object."""
        grid = super().grid_array_object(domain, axis)
        grid.meta[axis]['envelope'] = False
        return grid

    def set_dtype(self, grid_dtype):
        """Determine coefficient properties from grid dtype."""
        # Transform retains data type
        self.grid_dtype = np.dtype(grid_dtype)
        self.coeff_dtype = self.grid_dtype
        # Same number of modes and grid points
        self.coeff_size = self.base_grid_size
        self.elements = np.arange(self.coeff_size)
        return self.coeff_dtype

    @staticmethod
    def _recursion(N, X, envelope):
        # Implement recursion relation from Boyd A.8
        # Preallocate list for outputs
        Y = [None] * N
        # Use powers to get identity element of X type and copy X
        if envelope:
            Y[0] = np.exp(-X / 2)
        else:
            Y[0] = X**0
        Y[1] = (X**0 - X) * Y[0]
        for n in range(1, N-1):
            Y[n+1] = ((2*n + 1)*X**0 - X) / (n + 1) * Y[n] - n / (n + 1) * Y[n-1]
        return Y

    @classmethod
    @CachedMethod
    def _Jacobi_matrix(cls, N):
        # X * Y[n] = (2*n + 1) * Y[n] - (n + 1) Y[n+1] - n Y[n-1]
        J = sparse.lil_matrix((N, N))
        for n in range(N):
            if n > 0:
                J[n-1, n] = - n
            J[n, n] = 2*n + 1
            if n < N - 1:
                J[n+1, n] = - (n + 1)
        return J.tocsr()

    @classmethod
    def _evaluate_basis_functions(cls, N, xn, envelope):
        """Evaluate basis functions on a specified native grid."""
        # Run recursion in higher precision
        g = cls._recursion(N, xn.astype(np.longdouble), envelope)
        return np.array(g).astype(xn.dtype)

    @CachedMethod
    def _mmt_setup(self, gsize, csize, envelope):
        """Build transform matrices."""
        logger.debug("Building Laguerre MMT matrices for (gsize, csize, env) = (%s, %s, %s)" %(gsize, csize, envelope))
        # Get grid and weights
        native_grid, weights = special.roots_laguerre(gsize)
        if self.stretch < 0:
            native_grid = native_grid[::-1]
            weights = weights[::-1]
        # Evaluate polynomials on grid
        functions = self._evaluate_basis_functions(csize, native_grid, envelope)
        backward_mat = functions.T.copy()
        # Forward transform:
        # sum w L[n] L[m] = int L[n] L[m] exp(-x) dx
        #                 = δ[m,n]
        # sum w gp[n] gp[m] = δ[m,n]
        # sum w exp(x) gf[n] gf[m] = δ[m,n]
        forward_mat = weights * functions
        forward_mat[gsize:] = 0
        if envelope:
            forward_mat *= np.exp(native_grid)
        return forward_mat, backward_mat

    def _forward_mmt(self, gdata, cdata, axis, meta, scale):
        """Forward transform using matrix transform."""
        cdata, gdata = self.check_arrays(cdata, gdata, axis, scale)
        # Apply transform matrices
        forward_mat, _ = self._mmt_setup(gdata.shape[axis], cdata.shape[axis], meta['envelope'])
        apply_matrix(forward_mat, gdata, axis, out=cdata)
        return cdata

    def _backward_mmt(self, cdata, gdata, axis, meta, scale):
        """Forward transform using inverse matrix transform."""
        cdata, gdata = self.check_arrays(cdata, gdata, axis, scale)
        # Apply transform matrices
        _, backward_mat = self._mmt_setup(gdata.shape[axis], cdata.shape[axis], meta['envelope'])
        apply_matrix(backward_mat, cdata, axis, out=gdata)
        return gdata

    @CachedAttribute
    def Integrate(self):
        """Build integration class."""

        class IntegrateLaguerre(operators.Integrate, operators.Coupled):
            name = 'integ_{}'.format(self.name)
            basis = self

            def meta_envelope(self, axis):
                if axis == self.axis:
                    # Check current envelope
                    if self.args[0].meta[axis]['envelope']:
                        # Integral is a constant
                        return False
                    else:
                        raise ValueError("Laguerre polynomials are non-integrable.")
                else:
                    # Preserve envelope
                    return self.args[0].meta[axis]['envelope']

            def matrix_form(self):
                # Call class method with arg metadata
                return self._cls_matrix_form(self.args[0].meta[self.axis])

            @classmethod
            def _cls_matrix_form(cls, arg_basis_meta):
                # Call cached constructor with required metadata
                return cls._integ_matrix(arg_basis_meta['envelope'])

            @classmethod
            def _cls_vector_form(cls, arg_basis_meta):
                # Call cached constructor with required metadata
                return cls._integ_vector(arg_basis_meta['envelope'])

            @classmethod
            @CachedMethod
            def _integ_matrix(cls, envelope):
                """Laguerre integration."""
                size = cls.basis.coeff_size
                matrix = sparse.lil_matrix((size, size), dtype=cls.basis.coeff_dtype)
                matrix[0,:] = cls._integ_vector(envelope)
                return matrix.tocsr()

            @classmethod
            @CachedMethod
            def _integ_vector(cls, envelope):
                """
                Laguerre integration:
                    dx(g[n]) = -(1/2) g[n] - sum_{i=0}^{n-1} g[i]
                    dx(g[n+1]) - dx(g[n]) = -(1/2) g[n+1] - (1/2) g[n]
                    integ(g[n+1]) = - integ(g[n])
                    integ(g[0]) = 2
                """
                if not envelope:
                    raise ValueError("Laguerre polynomials are non-integrable.")
                vector = np.zeros(cls.basis.coeff_size, dtype=cls.basis.coeff_dtype)
                vector[0::2] = 2
                vector[1::2] = -2
                vector *= np.abs(cls.basis._grid_stretch)
                return vector

        return IntegrateLaguerre

    @CachedAttribute
    def Interpolate(self):
        """Buld interpolation class."""

        class InterpolateLaguerre(operators.Interpolate, operators.Coupled):
            name = 'interp_{}'.format(self.name)
            basis = self

            def meta_envelope(self, axis):
                if axis == self.axis:
                    # Interpolation is a constant
                    return False
                else:
                    # Preserve envelope
                    return self.args[0].meta[axis]['envelope']

            def matrix_form(self):
                # Call class method with arg metadata
                return self._cls_matrix_form(self.args[0].meta[self.axis], self.position)

            @classmethod
            def _cls_matrix_form(cls, arg_basis_meta, position):
                # Call cached constructor with required metadata
                return cls._interp_matrix(arg_basis_meta['envelope'], position)

            @classmethod
            def _cls_vector_form(cls, arg_basis_meta, position):
                # Call cached constructor with required metadata
                return cls._interp_vector(arg_basis_meta['envelope'], position)

            @classmethod
            @CachedMethod
            def _interp_matrix(cls, envelope, position):
                size = cls.basis.coeff_size
                matrix = sparse.lil_matrix((size, size), dtype=cls.basis.coeff_dtype)
                matrix[0,:] = cls._interp_vector(envelope, position)
                return matrix.tocsr()

            @classmethod
            @CachedMethod
            def _interp_vector(cls, envelope, position):
                stretch = cls.basis._grid_stretch
                if (position == 'left' and stretch > 0) or (position == 'right' and stretch < 0):
                    xn = 0
                elif (position == 'right' and stretch > 0) or (position == 'left' and stretch < 0):
                    # Corresponds to xn = inf
                    if envelope:
                        return np.zeros(cls.basis.coeff_size)
                    else:
                        raise ValueError("Laguerre polynomials blow up at infinity.")
                elif position == 'center':
                    raise ValueError("No center in semi-infinite interval.")
                else:
                    xn = cls.basis._native_coord(position)
                xn = np.array([float(xn)])
                functions = cls.basis._evaluate_basis_functions(cls.basis.coeff_size, xn, envelope)
                return functions[:, 0]

        return InterpolateLaguerre

    @CachedAttribute
    def Differentiate(self):
        """Build differentiation class."""

        class DifferentiateLaguerre(operators.Differentiate, operators.Coupled):
            name = 'd' + self.name
            basis = self

            def meta_envelope(self, axis):
                # Preserve envelope
                return self.args[0].meta[axis]['envelope']

            def matrix_form(self):
                # Call class method with arg metadata
                return self._cls_matrix_form(self.args[0].meta[self.axis])

            @classmethod
            def _cls_matrix_form(cls, arg_basis_meta):
                # Call cached constructor with required metadata
                return cls._diff_matrix(arg_basis_meta['envelope'])

            @classmethod
            @CachedMethod
            def _diff_matrix(cls, envelope):
                """
                Laguerre differentiation:
                    dx(gf[n]) = -(1/2) gf[n] + exp(-x/2) dx(L[n])
                              = -(1/2) gf[n] - sum_{i=0}^{n-1} gf[i]
                    dx(gp[n]) = - sum_{i=0}^{n-1} gp[i]
                """
                size = cls.basis.coeff_size
                dtype = cls.basis.coeff_dtype
                stretch = cls.basis._grid_stretch
                matrix = sparse.lil_matrix((size, size), dtype=dtype)
                if envelope:
                    for n in range(size):
                        if n > 0:
                            matrix[0:n, n] = -1 / stretch
                        matrix[n, n] = -0.5 / stretch
                else:
                    for n in range(size):
                        if n > 0:
                            matrix[0:n, n] = -1 / stretch
                return matrix.tocsr()

        return DifferentiateLaguerre

    @CachedAttribute
    def Precondition(self):
        """
        Preconditioning matrix.
            L[n;1] = sum_{i=0}^{n} L[n;0]
            L[n;0] = L[n;1] - L[n-1;1]
        """
        size = self.coeff_size
        # Construct sparse matrix
        Pre = sparse.lil_matrix((size, size), dtype=self.coeff_dtype)
        for n in range(size):
            Pre[n, n] = 1
            if n > 0:
                Pre[n-1, n] = -1
        return Pre.tocsr()

    @CachedAttribute
    def Dirichlet(self):
        """
        Dirichlet recombination matrix.

        G[0] = g[0]
        G[n] = g[n] - g[n-1]

        <g[i]|G[j]> = <g[i]|g[j]> - <g[i]|g[j-1]>
                    = δ(i,j) - δ(i,j-1)
        """
        size = self.coeff_size
        # Construct sparse matrix
        Dir = sparse.lil_matrix((size, size), dtype=self.coeff_dtype)
        for n in range(size):
            Dir[n, n] = 1
            if n > 0:
                Dir[n-1, n] = -1
        return Dir.tocsr()

    def Multiply(self, p, ncc_basis_meta, arg_basis_meta):
        """Laguerre multiplication matrix."""
        # Only multiply by polynomials
        if ncc_basis_meta['envelope']:
            raise ValueError("Cannot use enveloped functions for NCCs.")
        # Dispatch based on arg envelope
        if arg_basis_meta['envelope']:
            return self._Multiply_poly_func(p)
        else:
            return self._Multiply_poly_poly(p)

    def _Multiply_poly_func(self, p):
        """
        Multiplication matrix:
            gp[p] gf[n] = Mpf[p][k,n] gf[k]
                        = exp(-x/2) gp[p] gp[n]
                        = exp(-x/2) Mpp[p][k,n] gp[k]
                        = Mpp[p][k,n] gf[k]
            Mpf[p][k,n] = Mpp[p][k,n]
        """
        # Same as poly-poly matrix
        return self._Multiply_poly_poly(p)

    @CachedMethod
    def _Multiply_poly_poly(self, n):
        # Return slice of extended matrix
        N = self.coeff_size
        return self._Multiply_poly_poly_ext(n)[:N, :N]

    @CachedMethod
    def _Multiply_poly_poly_ext(self, n):
        """
        Multiplication matrix:
            gp[i] gp[j] = Mpp[i][k,j] gp[k]

        Mpp constructed from recursion on Jacobi matrix.
        """
        # L[n] = ((2*n - 1) - x) / n * L[n-1] - (n - 1) / n * L[n-2]
        # Recurse on Jacobi matrix
        # Use size 2*N to avoid truncation issues
        N = self.coeff_size
        J = self._Jacobi_matrix(2*N)
        if n == 0:
            return (J**0).tocsr()
        elif n == 1:
            return J**0 - J
        else:
            L1 = self._Multiply_poly_poly_ext(n - 1)
            L2 = self._Multiply_poly_poly_ext(n - 2)
            return ((2*n - 1)*J**0 - J) / n * L1 - (n - 1) / n * L2


class Fourier(TransverseBasis):
    """Fourier complex exponential basis."""

    separable = True
    coupled = False
    element_label= 'k'

    def __init__(self, name, base_grid_size, interval=(0,2*pi), dealias=1):

        # Coordinate transformation
        # Native interval: (0, 2π)
        start = interval[0]
        length = interval[1] - interval[0]
        self._grid_stretch = length / (2*pi)
        self._native_coord = lambda xp: (2*pi) * (xp - start) / length
        self._problem_coord = lambda xn: start + (xn / (2*pi) * length)

        # Attributes
        self.name = name
        self.element_name = 'k' + name
        self.base_grid_size = base_grid_size
        self.interval = tuple(interval)
        self.dealias = dealias
        self.library = DEFAULT_LIBRARY
        self.operators = (self.Integrate,
                          self.Interpolate,
                          self.Differentiate,
                          self.HilbertTransform)

    def default_meta(self):
        return {'constant': False}

    @CachedMethod
    def grid(self, scale=1.):
        """Build evenly spaced Fourier grid."""
        N = self.grid_size(scale)
        native_grid = np.linspace(0, 2*pi, N, endpoint=False)
        return self._problem_coord(native_grid)

    @CachedMethod
    def grid_spacing(self, scale=1.):
        """Build Fourier grid spacing."""
        N = self.grid_size(scale)
        native_spacing = 2 * pi / N * np.ones(N)
        return native_spacing * self._grid_stretch

    def set_dtype(self, grid_dtype):
        """Determine coefficient properties from grid dtype."""
        # Tranform produces complex coefficients
        self.grid_dtype = np.dtype(grid_dtype)
        self.coeff_dtype = np.dtype(np.complex128)
        # Build native wavenumbers, discarding any Nyquist mode
        kmax = (self.base_grid_size - 1) // 2
        if self.grid_dtype == np.float64:
            native_wavenumbers = np.arange(0, kmax+1)
        elif self.grid_dtype == np.complex128:
            native_wavenumbers = np.roll(np.arange(-kmax, kmax+1), -kmax)
        # Scale native wavenumbers
        self.elements = self.wavenumbers = native_wavenumbers / self._grid_stretch
        self.coeff_size = self.elements.size
        return self.coeff_dtype

    def _resize_real_coeffs(self, cdata_in, cdata_out, axis, grid_size):
        """Resize coefficient data by padding/truncation."""
        size_in = cdata_in.shape[axis]
        size_out = cdata_out.shape[axis]
        # Find maximum wavenumber (excluding Nyquist mode for even sizes)
        kmax = min((grid_size-1)//2, size_in-1, size_out-1)
        posfreq = axslice(axis, 0, kmax+1)
        badfreq = axslice(axis, kmax+1, None)
        # Copy modes up through kmax
        # For size_in < size_out, this pads with higher wavenumbers
        # For size_in > size_out, this truncates higher wavenumbers
        # For size_in = size_out, this copies the data (dropping Nyquist)
        np.copyto(cdata_out[posfreq], cdata_in[posfreq])
        np.copyto(cdata_out[badfreq], 0)

    def _resize_complex_coeffs(self, cdata_in, cdata_out, axis, *args):
        """Resize coefficient data by padding/truncation."""
        size_in = cdata_in.shape[axis]
        size_out = cdata_out.shape[axis]
        # Find maximum wavenumber (excluding Nyquist mode for even sizes)
        kmax = (min(size_in, size_out) - 1) // 2
        posfreq = axslice(axis, 0, kmax+1)
        badfreq = axslice(axis, kmax+1, -kmax)
        negfreq = axslice(axis, -kmax, None)
        # Copy modes up through +- kmax
        # For size_in < size_out, this pads with higher wavenumbers and conjugates
        # For size_in > size_out, this truncates higher wavenumbers and conjugates
        # For size_in = size_out, this copies the data (dropping Nyquist)
        np.copyto(cdata_out[posfreq], cdata_in[posfreq])
        np.copyto(cdata_out[badfreq], 0)
        np.copyto(cdata_out[negfreq], cdata_in[negfreq])

    def _forward_scipy(self, gdata, cdata, axis, meta, scale):
        """Forward transform using numpy RFFT / scipy FFT."""
        cdata, gdata = self.check_arrays(cdata, gdata, axis, scale)
        grid_size = gdata.shape[axis]
        if gdata.dtype == np.float64:
            # Numpy RFFT (scipy RFFT uses real packing)
            temp = np.fft.rfft(gdata, axis=axis)
            # Pad / truncate coefficients
            self._resize_real_coeffs(temp, cdata, axis, grid_size)
        elif gdata.dtype == np.complex128:
            # Scipy FFT (faster than numpy FFT)
            temp = fftpack.fft(gdata, axis=axis)
            # Pad / truncate coefficients
            self._resize_complex_coeffs(temp, cdata, axis)
        # Scale as Fourier amplitudes
        cdata *= 1 / grid_size
        return cdata

    def _backward_scipy(self, cdata, gdata, axis, meta, scale):
        """Backward transform using numpy IRFFT / scipy IFFT."""
        cdata, gdata = self.check_arrays(cdata, gdata, axis, scale)
        grid_size = gdata.shape[axis]
        if gdata.dtype == np.float64:
            # Pad / truncate coefficients
            shape = np.copy(gdata.shape)
            shape[axis] = grid_size//2 + 1
            temp = np.zeros(shape, dtype=np.complex128)
            self._resize_real_coeffs(cdata, temp, axis, grid_size)
            # Numpy IRFFT
            temp = np.fft.irfft(temp, n=grid_size, axis=axis)
        elif gdata.dtype == np.complex128:
            # Pad / truncate coefficients
            # Store in gdata for memory efficiency (transform preserves shape/dtype)
            self._resize_complex_coeffs(cdata, gdata, axis)
            # Scipy IFFT
            temp = fftpack.ifft(gdata, axis=axis)
        # Undo built-in scaling
        np.multiply(temp, grid_size, out=gdata)
        return gdata

    @CachedMethod
    def _fftw_setup(self, dtype, gshape, axis):
        """Build FFTW plans and temporary arrays."""
        # Note: regular method used to cache through basis instance
        logger.debug("Building FFTW FFT plan for (dtype, gshape, axis) = (%s, %s, %s)" %(dtype, gshape, axis))
        flags = ['FFTW_'+FFTW_RIGOR.upper()]
        plan = fftw.FourierTransform(dtype, gshape, axis, flags=flags)
        temp = fftw.create_array(plan.cshape, np.complex128)
        if dtype == np.float64:
            resize_coeffs = self._resize_real_coeffs
        elif dtype == np.complex128:
            resize_coeffs = self._resize_complex_coeffs
        return plan, temp, resize_coeffs

    def _forward_fftw(self, gdata, cdata, axis, meta, scale):
        """Forward transform using FFTW FFT."""
        cdata, gdata = self.check_arrays(cdata, gdata, axis, scale)
        plan, temp, resize_coeffs = self._fftw_setup(gdata.dtype, gdata.shape, axis)
        # Execute FFTW plan
        plan.forward(gdata, temp)
        # Scale FFT output to mode amplitudes
        temp *= 1 / gdata.shape[axis]
        # Pad / truncate coefficients
        resize_coeffs(temp, cdata, axis, gdata.shape[axis])
        return cdata

    def _backward_fftw(self, cdata, gdata, axis, meta, scale):
        """Backward transform using FFTW IFFT."""
        cdata, gdata = self.check_arrays(cdata, gdata, axis, scale)
        plan, temp, resize_coeffs = self._fftw_setup(gdata.dtype, gdata.shape, axis)
        # Pad / truncate coefficients
        resize_coeffs(cdata, temp, axis, gdata.shape[axis])
        # Execute FFTW plan
        plan.backward(temp, gdata)
        return gdata

    @CachedAttribute
    def Integrate(self):
        """Build integration class."""

        class IntegrateFourier(operators.Integrate, operators.Separable):
            name = 'integ_{}'.format(self.name)
            basis = self

            @classmethod
            @CachedMethod
            def vector_form(cls):
                """Fourier integration: int(Fn) = 2 π δ(n,0)"""
                vector = np.zeros(cls.basis.coeff_size, dtype=cls.basis.coeff_dtype)
                vector[0] = 2 * np.pi * cls.basis._grid_stretch
                return vector

        return IntegrateFourier

    @CachedAttribute
    def Interpolate(self):
        """Build interpolation class."""

        class InterpolateFourier(operators.Interpolate, operators.Coupled):
            name = 'interp_{}'.format(self.name)
            basis = self

            def check_conditions(self):
                arg0, = self.args
                axis = self.axis
                if self.basis.grid_dtype == np.float64:
                    # Must be in coeff+local layout
                    is_coeff = not arg0.layout.grid_space[axis]
                    is_local = arg0.layout.local[axis]
                    # Subsequent basis must be in grid space for proper interpolation symmetry
                    if axis == (self.domain.dim - 1):
                        subs_grid = True
                    else:
                        subs_grid = arg0.layout.grid_space[axis+1]
                    return (is_coeff and is_local and subs_grid)
                elif self.basis.grid_dtype == np.complex128:
                    # Must be in coeff+local layout
                    is_coeff = not arg0.layout.grid_space[axis]
                    is_local = arg0.layout.local[axis]
                    return (is_coeff and is_local)

            def operate(self, out):
                arg0, = self.args
                axis = self.axis
                if self.basis.grid_dtype == np.float64:
                    # Subsequent basis must be in grid space for proper interpolation symmetry
                    if axis != (self.domain.dim - 1):
                        arg0.require_grid_space(axis+1)
                # Require coeff+local layout
                arg0.require_coeff_space(axis)
                arg0.require_local(axis)
                out.layout = arg0.layout
                # Attempt forms
                try:
                    self.explicit_form(arg0.data, out.data, axis)
                except NotImplementedError:
                    self.apply_matrix_form(out)

            def explicit_form(self, input, output, axis):
                dim = self.domain.dim
                weights = reshape_vector(self._interp_vector(self.position), dim=dim, axis=axis)
                if self.basis.grid_dtype == np.float64:
                    # Halve mean-mode weight (will be added twice)
                    weights.flat[0] /= 2
                    pos_interp = np.sum(input * weights, axis=axis, keepdims=True)
                    interp = pos_interp + pos_interp.conj()
                elif self.basis.grid_dtype == np.complex128:
                    interp = np.sum(input * weights, axis=axis, keepdims=True)
                np.copyto(output[axslice(axis, 0, 1)], interp)
                np.copyto(output[axslice(axis, 1, None)], 0)

            @classmethod
            def _interp_vector(cls, position):
                """Fourier interpolation: Fn(x) = exp(i kn x)"""
                if position == 'left':
                    position = cls.basis.interval[0]
                elif position == 'right':
                    position = cls.basis.interval[1]
                elif position == 'center':
                    position = (cls.basis.interval[0] + cls.basis.interval[1]) / 2
                x = position - cls.basis.interval[0]
                return np.exp(1j * cls.basis.wavenumbers * x)

        return InterpolateFourier

    @CachedAttribute
    def Differentiate(self):
        """Build differentiation class."""

        class DifferentiateFourier(operators.Differentiate, operators.Separable):
            name = 'd' + self.name
            basis = self

            @classmethod
            @CachedMethod
            def vector_form(cls):
                """Fourier differentiation: dx(Fn) = i kn Fn"""
                return 1j * cls.basis.wavenumbers

        return DifferentiateFourier

    @CachedAttribute
    def HilbertTransform(self):
        """Build Hilbert transform class."""

        class HilbertTransformFourier(operators.HilbertTransform, operators.Separable):
            name = 'H' + self.name
            basis = self

            @classmethod
            @CachedMethod
            def vector_form(cls):
                """Hilbert transform: Hx(Fn) = -i sgn(kn) Fn"""
                return -1j * np.sign(cls.basis.wavenumbers)

        return HilbertTransformFourier


class SinCos(TransverseBasis):
    """Sin/Cos series basis."""

    element_label = 'k'
    separable = True
    coupled = False

    def __init__(self, name, base_grid_size, interval=(0,pi), dealias=1):

        # Coordinate transformation
        # Native interval: (0, π)
        start = interval[0]
        length = interval[1] - interval[0]
        self._grid_stretch = length / pi
        self._native_coord = lambda xp: pi * (xp - start) / length
        self._problem_coord = lambda xn: start + (xn / pi * length)

        # Attributes
        self.name = name
        self.element_name = 'k' + name
        self.base_grid_size = base_grid_size
        self.interval = tuple(interval)
        self.dealias = dealias
        self.name = name
        self.library = DEFAULT_LIBRARY
        self.operators = (self.Integrate,
                          self.Interpolate,
                          self.Differentiate,
                          self.HilbertTransform)

    def default_meta(self):
        return {'constant': False,
                'parity': None}

    @CachedMethod
    def grid(self, scale=1.):
        """Evenly spaced interior grid: cos(Nx) = 0"""
        N = self.grid_size(scale)
        native_grid = pi * (np.arange(N) + 1/2) / N
        return self._problem_coord(native_grid)

    @CachedMethod
    def grid_spacing(self, scale=1.):
        """Build cosine grid spacing."""
        N = self.grid_size(scale)
        native_spacing = pi / N * np.ones(N)
        return native_spacing * self._grid_stretch

    @CachedMethod
    def grid_array_object(self, domain, axis):
        """Grid array object."""
        grid = super().grid_array_object(domain, axis)
        grid.meta[axis]['parity'] = 1
        return grid

    def set_dtype(self, grid_dtype):
        """Determine coefficient properties from grid dtype."""
        # Tranform retains data type
        self.grid_dtype = np.dtype(grid_dtype)
        self.coeff_dtype = self.grid_dtype
        # Build native wavenumbers
        native_wavenumbers = np.arange(self.base_grid_size)
        # Scale native wavenumbers
        self.elements = self.wavenumbers = native_wavenumbers / self._grid_stretch
        self.coeff_size = self.elements.size
        return self.coeff_dtype

    @staticmethod
    def _resize_coeffs(cdata_in, cdata_out, axis):
        """Resize coefficient data by padding/truncation."""
        size_in = cdata_in.shape[axis]
        size_out = cdata_out.shape[axis]
        if size_in < size_out:
            # Pad with higher order modes at end of data
            np.copyto(cdata_out[axslice(axis, 0, size_in)], cdata_in)
            np.copyto(cdata_out[axslice(axis, size_in, None)], 0)
        elif size_in > size_out:
            # Truncate higher order modes at end of data
            np.copyto(cdata_out, cdata_in[axslice(axis, 0, size_out)])
        else:
            np.copyto(cdata_out, cdata_in)

    @staticmethod
    def _forward_dct_scaling(pdata, axis):
        """Scale DCT output to sinusoid coefficients."""
        # Scale as sinusoid amplitudes
        pdata *= 1 / pdata.shape[axis]
        pdata[axslice(axis, 0, 1)] *= 0.5

    @staticmethod
    def _forward_dst_scaling(pdata, axis):
        """Scale DST output to sinusoid coefficients."""
        # Shift data, adding zero mode and dropping Nyquist
        N = pdata.shape[axis]
        start = pdata[axslice(axis, 0, N-1)]
        shift = pdata[axslice(axis, 1, N)]
        np.copyto(shift, start)
        pdata[axslice(axis, 0, 1)] = 0
        # Scale as sinusoid amplitudes
        pdata *= 1 / N

    @staticmethod
    def _backward_dct_scaling(pdata, axis):
        """Scale sinusoid coefficients to IDCT input."""
        # Scale from sinusoid amplitudes
        pdata[axslice(axis, 1, None)] *= 0.5

    @staticmethod
    def _backward_dst_scaling(pdata, axis):
        """Scale sinusoid coefficients to IDST input."""
        # Scale from sinusoid amplitudes
        pdata *= 0.5
        # Unshift data, adding Nyquist mode and dropping zero
        N = pdata.shape[axis]
        start = pdata[axslice(axis, 0, N-1)]
        shift = pdata[axslice(axis, 1, N)]
        np.copyto(start, shift)
        pdata[axslice(axis, N-1, N)] = 0

    def _forward_scipy(self, gdata, cdata, axis, meta, scale):
        """Forward transform using scipy DCT/DST."""
        cdata, gdata = self.check_arrays(cdata, gdata, axis, scale)
        complex = (gdata.dtype == np.complex128)
        # View complex data as interleaved real data
        if complex:
            cdata_complex = cdata
            gdata = interleaved_view(gdata)
            cdata = interleaved_view(cdata)
        # Scipy transforms and scalings
        if meta['parity'] == 0:
            temp = np.zeros_like(gdata)
        elif meta['parity'] == 1:
            temp = fftpack.dct(gdata, type=2, axis=axis)
            self._forward_dct_scaling(temp, axis)
        elif meta['parity'] == -1:
            temp = fftpack.dst(gdata, type=2, axis=axis)
            self._forward_dst_scaling(temp, axis)
        else:
            raise UndefinedParityError()
        # Pad / truncate coefficients
        self._resize_coeffs(temp, cdata, axis)
        if complex:
            cdata = cdata_complex
        return cdata

    def _backward_scipy(self, cdata, gdata, axis, meta, scale):
        """Backward transform using scipy IDCT/IDST."""
        cdata, gdata = self.check_arrays(cdata, gdata, axis, scale)
        complex = (gdata.dtype == np.complex128)
        # Pad / truncate coefficients
        # Store in gdata for memory efficiency (transform preserves shape/dtype)
        self._resize_coeffs(cdata, gdata, axis)
        # View complex data as interleaved real data
        if complex:
            gdata_complex = gdata
            gdata = interleaved_view(gdata)
        # Scipy transforms and scalings
        if meta['parity'] == 0:
            temp = np.zeros_like(gdata)
        elif meta['parity'] == 1:
            self._backward_dct_scaling(gdata, axis)
            temp = fftpack.dct(gdata, type=3, axis=axis)
        elif meta['parity'] == -1:
            self._backward_dst_scaling(gdata, axis)
            temp = fftpack.dst(gdata, type=3, axis=axis)
        else:
            raise UndefinedParityError()
        np.copyto(gdata, temp)
        if complex:
            gdata = gdata_complex
        return gdata

    @CachedMethod
    def _fftw_dct_setup(self, dtype, gshape, axis):
        """Build FFTW DCT plan and temporary array."""
        flags = ['FFTW_'+FFTW_RIGOR.upper()]
        logger.debug("Building FFTW DCT plan for (dtype, gshape, axis) = (%s, %s, %s)" %(dtype, gshape, axis))
        plan = fftw.DiscreteCosineTransform(dtype, gshape, axis, flags=flags)
        temp = fftw.create_array(gshape, dtype)
        return plan, temp

    @CachedMethod
    def _fftw_dst_setup(self, dtype, gshape, axis):
        """Build FFTW DST plan and temporary array."""
        flags = ['FFTW_'+FFTW_RIGOR.upper()]
        logger.debug("Building FFTW DST plan for (dtype, gshape, axis) = (%s, %s, %s)" %(dtype, gshape, axis))
        plan = fftw.DiscreteSineTransform(dtype, gshape, axis, flags=flags)
        temp = fftw.create_array(gshape, dtype)
        return plan, temp

    def _forward_fftw(self, gdata, cdata, axis, meta, scale):
        """Forward transform using FFTW DCT."""
        cdata, gdata = self.check_arrays(cdata, gdata, axis, scale)
        if meta['parity'] == 0:
            cdata.fill(0)
        elif meta['parity'] == 1:
            plan, temp = self._fftw_dct_setup(gdata.dtype, gdata.shape, axis)
            plan.forward(gdata, temp)
            self._forward_dct_scaling(temp, axis)
            self._resize_coeffs(temp, cdata, axis)
        elif meta['parity'] == -1:
            plan, temp = self._fftw_dst_setup(gdata.dtype, gdata.shape, axis)
            plan.forward(gdata, temp)
            self._forward_dst_scaling(temp, axis)
            self._resize_coeffs(temp, cdata, axis)
        else:
            raise UndefinedParityError()
        return cdata

    def _backward_fftw(self, cdata, gdata, axis, meta, scale):
        """Backward transform using FFTW IDCT."""
        cdata, gdata = self.check_arrays(cdata, gdata, axis, scale)
        if meta['parity'] == 0:
            gdata.fill(0)
        elif meta['parity'] == 1:
            plan, temp = self._fftw_dct_setup(gdata.dtype, gdata.shape, axis)
            self._resize_coeffs(cdata, temp, axis)
            self._backward_dct_scaling(temp, axis)
            plan.backward(temp, gdata)
        elif meta['parity'] == -1:
            plan, temp = self._fftw_dst_setup(gdata.dtype, gdata.shape, axis)
            self._resize_coeffs(cdata, temp, axis)
            self._backward_dst_scaling(temp, axis)
            plan.backward(temp, gdata)
        else:
            raise UndefinedParityError()
        return gdata

    @CachedAttribute
    def Integrate(self):
        """Build integration class."""

        class IntegrateSinCos(operators.Integrate, operators.Coupled):
            name = 'integ_{}'.format(self.name)
            basis = self

            def meta_parity(self, axis):
                if axis == self.axis:
                    # Integral is a scalar (even parity)
                    return 1
                else:
                    # Preserve parity
                    return self.args[0].meta[axis]['parity']

            def explicit_form(self, input, output, axis):
                dim = self.domain.dim
                weights = reshape_vector(self._integ_vector(), dim=dim, axis=axis)
                interp = np.sum(input * weights, axis=axis, keepdims=True)
                np.copyto(output[axslice(axis, 0, 1)], interp)
                np.copyto(output[axslice(axis, 1, None)], 0)

            def _integ_vector(self):
                """Fourier interpolation: Fn(x) = exp(i kn x)"""
                arg_parity = self.args[0].meta[self.axis]['parity']
                integ = np.zeros(self.basis.coeff_size)
                if arg_parity == 1:
                    integ[0] = np.pi * self.basis._grid_stretch
                    return integ
                elif arg_parity == -1:
                    integ[1::2] = 2 / np.arange(1, integ.size, 2)
                    return integ
                else:
                    raise UndefinedParityError()

        return IntegrateSinCos

    @CachedAttribute
    def Interpolate(self):
        """Build interpolation class."""

        class InterpolateSinCos(operators.Interpolate, operators.Coupled):
            name = 'interp_{}'.format(self.name)
            basis = self

            def meta_parity(self, axis):
                if axis == self.axis:
                    # Interpolation is a scalar (even parity)
                    return 1
                else:
                    # Preserve parity
                    return self.args[0].meta[axis]['parity']

            def explicit_form(self, input, output, axis):
                dim = self.domain.dim
                weights = reshape_vector(self._interp_vector(self.position), dim=dim, axis=axis)
                interp = np.sum(input * weights, axis=axis, keepdims=True)
                np.copyto(output[axslice(axis, 0, 1)], interp)
                np.copyto(output[axslice(axis, 1, None)], 0)

            def _interp_vector(self, position):
                """Fourier interpolation: Fn(x) = exp(i kn x)"""
                if position == 'left':
                    position = self.basis.interval[0]
                elif position == 'right':
                    position = self.basis.interval[1]
                elif position == 'center':
                    position = (self.basis.interval[0] + self.basis.interval[1]) / 2
                x = position - self.basis.interval[0]

                arg_parity = self.args[0].meta[self.axis]['parity']
                if arg_parity == 1:
                    return np.cos(self.basis.wavenumbers * x)
                elif arg_parity == -1:
                    return np.sin(self.basis.wavenumbers * x)
                else:
                    raise UndefinedParityError()

        return InterpolateSinCos

    @CachedAttribute
    def Differentiate(self):
        """Build differentiation class."""

        class DifferentiateSinCos(operators.Differentiate, operators.Separable):
            name = 'd' + self.name
            basis = self

            def meta_parity(self, axis):
                parity0 = self.args[0].meta[axis]['parity']
                if axis == self.axis:
                    # Flip parity
                    return (-1) * parity0
                else:
                    # Preserve parity
                    return parity0

            @CachedMethod
            def vector_form(self):
                """Sinusoid differentiation."""
                # arg parity == -1: dx(sin(kx)) =  k cos(kx)
                # arg parity ==  1: dx(cos(kx)) = -k sin(kx)
                arg_parity = self.args[0].meta[self.axis]['parity']
                return -arg_parity * self.basis.wavenumbers

        return DifferentiateSinCos

    @CachedAttribute
    def HilbertTransform(self):
        """Build Hilbert transform class."""

        class HilbertTransformSinCos(operators.HilbertTransform, operators.Separable):
            name = 'H' + self.name
            basis = self

            def meta_parity(self, axis):
                parity0 = self.args[0].meta[axis]['parity']
                if axis == self.axis:
                    # Flip parity
                    return (-1) * parity0
                else:
                    # Preserve parity
                    return parity0

            @CachedMethod
            def vector_form(self):
                # arg parity == -1:  Hx(sin(kx)) = -sgn(k) cos(kx)
                # arg parity ==  1:  Hx(cos(kx)) =  sgn(k) sin(kx)
                arg_parity = self.args[0].meta[self.axis]['parity']
                return arg_parity * np.sign(self.basis.wavenumbers)

        return HilbertTransformSinCos


class Compound(ImplicitBasis):
    """Compound basis joining adjascent subbases."""

    separable = False
    coupled = True

    def __init__(self, name, subbases, dealias=1):

        # Check intervals
        for i in range(len(subbases)-1):
            if subbases[i].interval[1] != subbases[i+1].interval[0]:
                raise ValueError("Subbases not adjascent.")

        # Attributes
        self.subbases = subbases
        self.element_label = "(%s)" %",".join([basis.element_label for basis in self.subbases])
        self.base_grid_size = sum(basis.base_grid_size for basis in subbases)
        self.interval = (subbases[0].interval[0], subbases[-1].interval[-1])
        self.name = name
        self.dealias = dealias
        # Overwrite subbases dealias levels
        for sb in subbases:
            sb.dealias = dealias

        self.operators = (self.Integrate,
                          self.Interpolate,
                          self.Differentiate)

    def default_meta(self):
        meta = {}
        for sb in self.subbases:
            meta.update(sb.default_meta())
        return meta

    @property
    def library(self):
        return tuple(basis.library for basis in self.subbases)

    @library.setter
    def library(self, value):
        for basis in self.subbases:
            basis.library = value

    @CachedMethod
    def grid(self, scale=1.):
        """Build compound grid."""
        return np.concatenate([basis.grid(scale) for basis in self.subbases])

    @CachedMethod
    def grid_spacing(self, scale=1.):
        """Build compound grid spacing."""
        return np.concatenate([basis.grid_spacing(scale) for basis in self.subbases])

    @CachedMethod
    def grid_array_object(self, domain, axis):
        """Grid array object."""
        grid = super().grid_array_object(domain, axis)
        grid.meta[axis]['envelope'] = False
        return grid

    def set_dtype(self, grid_dtype):
        """Determine coefficient properties from grid dtype."""
        # Ensure subbases return same coeff dtype
        coeff_dtypes = list(basis.set_dtype(grid_dtype) for basis in self.subbases)
        if len(set(coeff_dtypes)) > 1:
            raise ValueError("Bases returned different coeff_dtypes.")
        self.grid_dtype = np.dtype(grid_dtype)
        self.coeff_dtype = coeff_dtypes[0]
        # Sum subbasis coeff sizes
        self.coeff_size = sum(basis.coeff_size for basis in self.subbases)
        self.elements = np.arange(self.coeff_size)
        return self.coeff_dtype

    def coeff_start(self, index):
        return sum(b.coeff_size for b in self.subbases[:index])

    def grid_start(self, index, scale):
        return sum(b.grid_size(scale) for b in self.subbases[:index])

    def sub_gdata(self, gdata, index, axis):
        """Retreive gdata corresponding to one subbasis."""
        # Infer scale from gdata size
        scale = gdata.shape[axis] / self.base_grid_size
        start = self.grid_start(index, scale)
        end = self.grid_start(index+1, scale)
        return gdata[axslice(axis, start, end)]

    def sub_cdata(self, cdata, index, axis):
        """Retrieve cdata corresponding to one subbasis."""
        start = self.coeff_start(index)
        end = self.coeff_start(index+1)
        return cdata[axslice(axis, start, end)]

    def forward(self, gdata, cdata, axis, meta, scale):
        """Forward transforms."""
        cdata, gdata = self.check_arrays(cdata, gdata, axis, scale)
        gdata = gdata.copy()
        for index, basis in enumerate(self.subbases):
            # Transform continuous copy of subbasis gdata
            # (Transforms generally require continuous data)
            temp = fftw.create_copy(self.sub_gdata(gdata, index, axis))
            temp = basis.forward(temp, None, axis, meta, scale)
            np.copyto(self.sub_cdata(cdata, index, axis), temp)

        return cdata

    def backward(self, cdata, gdata, axis, meta, scale):
        """Backward transforms."""
        cdata, gdata = self.check_arrays(cdata, gdata, axis, scale)
        # Copy cdata so we can write into gdata without overwriting subsequent coefficients
        cdata = cdata.copy()
        for index, basis in enumerate(self.subbases):
            # Transform continuous copy of subbasis cdata
            # (Transforms generally require continuous data)
            temp = fftw.create_copy(self.sub_cdata(cdata, index, axis))
            temp = basis.backward(temp, None, axis, meta, scale)
            np.copyto(self.sub_gdata(gdata, index, axis), temp)

        return gdata

    @CachedAttribute
    def Integrate(self):
        """Build integration class."""

        class IntegrateCompound(operators.Integrate, operators.Coupled):
            name = 'integ_{}'.format(self.name)
            basis = self

            def meta_envelope(self, axis):
                if axis == self.axis:
                    # Check current envelope
                    if self.args[0].meta[axis]['envelope']:
                        # Integral is a constant
                        return False
                    else:
                        raise ValueError("Laguerre polynomials are non-integrable.")
                else:
                    # Preserve envelope
                    return self.args[0].meta[axis]['envelope']

            @CachedMethod
            def matrix_form(self):
                # Call class method with arg metadata
                return self._cls_matrix_form(self.args[0].meta[self.axis])

            @classmethod
            def _cls_matrix_form(cls, arg_basis_meta):
                size = cls.basis.coeff_size
                matrix = sparse.lil_matrix((size, size), dtype=cls.basis.coeff_dtype)
                integ_vector = cls._cls_vector_form(arg_basis_meta)
                # Copy vector to each subbasis constant row
                for i,sb in enumerate(cls.basis.subbases):
                    sb0 = cls.basis.coeff_start(i)
                    matrix[sb0,:] = integ_vector
                return matrix.tocsr()

            @classmethod
            def _cls_vector_form(cls, arg_basis_meta):
                # Concatenate subbases vectors
                subvectors = [b.Integrate._cls_vector_form(arg_basis_meta) for b in cls.basis.subbases]
                return np.concatenate(subvectors)

        return IntegrateCompound

    @CachedAttribute
    def Interpolate(self):
        """Buld interpolation class."""

        class InterpolateCompound(operators.Interpolate, operators.Coupled):
            name = 'interp_{}'.format(self.name)
            basis = self

            def meta_envelope(self, axis):
                if axis == self.axis:
                    # Interpolation is a constant
                    return False
                else:
                    # Preserve envelope
                    return self.args[0].meta[axis]['envelope']

            @CachedMethod
            def matrix_form(self):
                # Call class method with arg metadata
                return self._cls_matrix_form(self.args[0].meta[self.axis], self.position)

            @classmethod
            def _cls_matrix_form(cls, arg_basis_meta, position):
                size = cls.basis.coeff_size
                matrix = sparse.lil_matrix((size, size), dtype=cls.basis.coeff_dtype)
                interp_vector = cls._cls_vector_form(arg_basis_meta, position)
                # Copy vector to each subbases constant row
                for i,sb in enumerate(cls.basis.subbases):
                    sb0 = cls.basis.coeff_start(i)
                    matrix[sb0,:] = interp_vector
                return matrix.tocsr()

            @classmethod
            def _cls_vector_form(cls, arg_basis_meta, position):
                """Chebyshev interpolation: Tn(xn) = cos(n * acos(xn))"""
                # Construct dense row vector
                interp_vector = np.zeros(cls.basis.coeff_size, dtype=cls.basis.coeff_dtype)
                if position == 'left':
                    sb_index = 0
                elif position == 'right':
                    sb_index = len(cls.basis.subbases) - 1
                else:
                    if position == 'center':
                        position = (cls.basis.interval[0] + cls.basis.interval[1]) / 2
                    # Find containing subbasis
                    for sb_index, sb in enumerate(cls.basis.subbases):
                        if sb.interval[0] <= position <= sb.interval[1]:
                            break
                    else:
                        raise ValueError("Position outside any subbasis interval.")
                # Use subbasis interpolation
                sb = cls.basis.subbases[sb_index]
                start = cls.basis.coeff_start(sb_index)
                end = cls.basis.coeff_start(sb_index+1)
                interp_vector[start:end] = sb.Interpolate._cls_vector_form(arg_basis_meta, position)
                return interp_vector

        return InterpolateCompound

    @CachedAttribute
    def Differentiate(self):
        """Build differentiation class."""

        class DifferentiateCompound(operators.Differentiate, operators.Coupled):
            name = 'd' + self.name
            basis = self

            def meta_envelope(self, axis):
                # Preserve envelope
                return self.args[0].meta[axis]['envelope']

            @CachedMethod
            def matrix_form(self):
                # Call class method with arg metadata
                return self._cls_matrix_form(self.args[0].meta[self.axis])

            @classmethod
            def _cls_matrix_form(cls, arg_basis_meta):
                """Compound differentiation matrix."""
                sub_blocks = [sb.Differentiate._cls_matrix_form(arg_basis_meta) for sb in cls.basis.subbases]
                matrix = sparse.block_diag(sub_blocks)
                return matrix.tocsr()

            # def explicit_form(self, input, output, axis):
            #     """Explicit differentiation."""
            #     # BROKEN
            #     for i,b in enumerate(self.basis.subbases):
            #         b_cdata = self.basis.sub_cdata(input, i, axis)
            #         b_cderiv = self.basis.sub_cdata(output, i, axis)
            #         b.differentiate(b_cdata, b_cderiv, axis)

        return DifferentiateCompound

    @CachedAttribute
    def Precondition(self):
        Pre = sparse.block_diag([b.Precondition for b in self.subbases])
        return Pre.tocsr()

    @CachedAttribute
    def Dirichlet(self):
        Dir = sparse.block_diag([b.Dirichlet for b in self.subbases])
        return Dir.tocsr()

    def Multiply(self, subindex, p, ncc_basis_meta, arg_basis_meta):
        size = self.coeff_size
        Mult = sparse.lil_matrix((size, size), dtype=self.coeff_dtype)
        start = self.coeff_start(subindex)
        end = self.coeff_start(subindex+1)
        subMult = self.subbases[subindex].Multiply(p, ncc_basis_meta, arg_basis_meta)
        Mult[start:end, start:end] = subMult
        return Mult.tocsr()

    def NCC(self, ncc_basis_meta, arg_basis_meta, coeffs, cutoff, max_terms):
        """Build NCC multiplication matrix."""
        if max_terms is None:
            max_terms = self.coeff_size
        n_terms = max_term = matrix = 0
        for index, basis in enumerate(self.subbases):
            n_terms_i = max_term_i = 0
            subcoeffs = self.sub_cdata(coeffs, index, axis=0)
            for p in range(min(max_terms, subcoeffs.size)):
                if abs(subcoeffs[p]) >= cutoff:
                    matrix = matrix + subcoeffs[p]*self.Multiply(index, p, ncc_basis_meta, arg_basis_meta)
                    n_terms_i += 1
                    max_term_i = p
            n_terms = max(n_terms, n_terms_i)
            max_term = max(max_term, max_term_i)
        return n_terms, max_term, matrix

    @CachedMethod
    def DropTau(self, n_tau):
        """Matrix dropping tau+match rows."""
        blocks = [subbasis.DropTau(1) for subbasis in self.subbases]
        blocks[-1] = self.subbases[-1].DropTau(n_tau)
        return sparse.block_diag(blocks, format='csr')

    @CachedAttribute
    def DropNonfirst(self):
        """Matrix dropping non-first rows."""
        N = self.coeff_size
        return sparse.eye(1, N, dtype=self.coeff_dtype, format='csr')

    @CachedAttribute
    def DropNonconstant(self):
        """Matrix dropping non-constant rows."""
        blocks = [subbasis.DropNonconstant for subbasis in self.subbases]
        return sparse.block_diag(blocks, format='csr')

    @CachedAttribute
    def DropMatch(self):
        """Matrix dropping last row from each subbasis except the last."""
        blocks = [subbasis.DropTau(1) for subbasis in self.subbases]
        blocks[-1] = self.subbases[-1].DropMatch
        return sparse.block_diag(blocks, format='csr')

    @CachedMethod
    def PreconditionDropTau(self, n_tau):
        """Preconditioning and tau+match filtering."""
        blocks = [subbasis.PreconditionDropTau(1) for subbasis in self.subbases]
        blocks[-1] = self.subbases[-1].PreconditionDropTau(n_tau)
        return sparse.block_diag(blocks, format='csr')

    @CachedAttribute
    def PreconditionDropMatch(self):
        """Preconditioning and match filtering."""
        blocks = [subbasis.PreconditionDropTau(1) for subbasis in self.subbases]
        blocks[-1] = self.subbases[-1].PreconditionDropMatch
        return sparse.block_diag(blocks, format='csr')

    @CachedAttribute
    def MatchRows(self):
        """Matrix matching subbases."""
        nsub = len(self.subbases)
        size = self.coeff_size
        Match = sparse.lil_matrix((nsub-1, size), dtype=self.coeff_dtype)
        # Use default metadata
        # This is a bit of a hack, but it's ok b/c interpolation depends on envelope
        # metadata, but the envelope is 1 at the Laguerre endpoint
        arg_basis_meta = self.default_meta()
        for i in range(len(self.subbases) - 1):
            basis1 = self.subbases[i]
            basis2 = self.subbases[i+1]
            s1 = self.coeff_start(i)
            e1 = s2 = self.coeff_start(i+1)
            e2 = self.coeff_start(i+2)
            Match[i, s1:e1] = basis1.Interpolate._cls_vector_form(arg_basis_meta, 'right')
            Match[i, s2:e2] = -basis2.Interpolate._cls_vector_form(arg_basis_meta, 'left')
        return Match.tocsr()
