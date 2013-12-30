"""
Spectral bases.

"""

import math
import numpy as np
from scipy import sparse
from scipy import fftpack

from ..tools.cache import CachedAttribute
from ..tools.cache import CachedMethod
from ..tools.array import interleaved_view
from ..tools.array import reshape_vector
from ..tools.array import axslice


class Basis:
    """
    Base class for spectral bases.

    Parameters
    ----------
    grid_size : int
        Number of grid points
    interval : tuple of floats
        Spatial domain of basis
    dealias : float, optional
        Fraction of modes to keep after dealiasing (default: 1.)

    Attributes
    ----------
    grid_embed : int
        Padded number of grid points for transform
    grid_dtype : dtype
        Grid data type
    coeff_size : int
        Number of spectral coefficients
    coeff_embed : int
        Padded number of spectral coefficients for transform
    coeff_dtype : dtype
        Coefficient data type

    """

    def set_transforms(self, grid_dtype):
        """Set transforms based on grid data type."""

        raise NotImplementedError()

    def pad_grid(self, gdata, pgdata, axis):
        """Pad grid data before forward transform."""

        raise NotImplementedError()

    def unpad_grid(self, pgdata, gdata, axis):
        """Unpad grid data after backward transform."""

        raise NotImplementedError()

    def pad_coeff(self, cdata, pcdata, axis):
        """Pad coefficient data before backward transform."""

        raise NotImplementedError()

    def unpad_coeff(self, pcdata, cdata, axis):
        """Unpad coefficient data after forward transfrom."""

    def forward(self, pgdata, pcdata, axis):
        """Grid-to-coefficient transform."""

        raise NotImplementedError()

    def backward(self, pcdata, pgdata, axis):
        """Coefficient-to-grid transform."""

        raise NotImplementedError()

    def differentiate(self, cdata, cderiv, axis):
        """Differentiate using coefficients."""

        raise NotImplementedError()

    def integrate(self, cdata, axis):
        """Integrate over interval using coefficients."""

        raise NotImplementedError()


class TransverseBasis(Basis):
    """Base class for bases supporting transverse differentiation."""

    def trans_diff(self, i):
        """Transverse differentation constant for i-th term."""

        raise NotImplementedError()


class ImplicitBasis(Basis):
    """
    Base class for bases supporting implicit methods.

    These bases define the following matrices encoding the respective linear
    functions acting on a series represented by its spectral coefficients:

    Linear operators (square matrices):
        Pre     : preconditioning
        Diff    : differentiation
        Mult(p) : multiplication by p-th basis element

    Linear functionals (vectors):
        left_vector  : left-endpoint evaluation
        right_vector : right-endpoint evaluation
        int_vector   : integration over interval

    Additionally, they define a vector `bc_vector` indicating which
    coefficient's Galerkin constraint is to be replaced by the boundary
    condition on a differential equation (i.e. the order of the tau term).

    """

    def integrate(self, cdata, axis):
        """Integrate over interval using coefficients."""

        # Contract coefficients with basis function integrals
        integral = np.tensordot(cdata, self.int_vector, (axis, 0))

        return integral

    @CachedAttribute
    def Pre(self):
        """Preconditioning matrix."""

        # Default to identity matrix
        Pre = sparse.identity(self.coeff_size, dtype=self.coeff_dtype)

        return Pre.tocsr()

    @CachedAttribute
    def Diff(self):
        """Differentiation matrix."""

        raise NotImplementedError()

    @CachedMethod
    def Mult(self, p):
        """p-element multiplication matrix."""

        raise NotImplementedError()

    @CachedAttribute
    def Left(self):
        """Left-endpoint matrix."""

        # Take outer product of boundary-row and left-endpoint vectors
        Left = sparse.kron(self.bc_vector, self.left_vector)

        return Left

    @CachedAttribute
    def Right(self):
        """Right-endpoint matrix."""

        # Take outer product of boundary-row and right-endpoint vectors
        Right = sparse.kron(self.bc_vector, self.right_vector)

        return Right

    @CachedAttribute
    def Int(self):
        """Integral matrix."""

        # Take outer product of boundary-row and integral vectors
        Int = sparse.kron(self.bc_vector, self.int_vector)

        return Int

    @CachedAttribute
    def left_vector(self):
        """Left-endpoint row vector."""

        raise NotImplementedError()

    @CachedAttribute
    def right_vector(self):
        """Right-endpoint row vector."""

        raise NotImplementedError()

    @CachedAttribute
    def int_vector(self):
        """Integral row vector."""

        raise NotImplementedError()

    @CachedAttribute
    def bc_vector(self):
        """Boundary-row column vector."""

        raise NotImplementedError()


class Chebyshev(ImplicitBasis):
    """Chebyshev polynomial basis on the extrema grid."""

    def __init__(self, grid_size, interval=(-1., 1.), dealias=1.):

        # Initial attributes
        self.grid_size = grid_size
        self.interval = tuple(interval)
        self.dealias = dealias

        # Grid-sized embedding
        self.grid_embed = grid_size
        self.coeff_embed = grid_size

        # Maximum number of coefficients below dealiasing threshold
        self.coeff_size = math.floor(dealias * grid_size)

        # Extrema grid
        radius = (interval[1] - interval[0]) / 2.
        center = (interval[1] + interval[0]) / 2.
        i = np.arange(grid_size)
        N = grid_size - 1
        native_grid = np.cos(np.pi * i / N)
        self.grid = center + radius * native_grid
        self._grid_stretch = radius

    def set_transforms(self, grid_dtype):
        """Set transforms based on grid data type."""

        # Transform retains data type
        self.grid_dtype = grid_dtype
        self.coeff_dtype = grid_dtype

        # Dispatch transform functions
        if grid_dtype == np.float64:
            self.forward = self._forward_r2r
            self.backward = self._backward_r2r
        elif grid_dtype == np.complex128:
            self.forward = self._forward_c2c
            self.backward = self._backward_c2c
        else:
            raise ValueError("Unsupported grid_dtype.")

        return self.coeff_dtype

    def pad_grid(self, gdata, pgdata, axis):
        """Pad grid data before forward transform."""

        # No padding
        np.copyto(pgdata, gdata)

    def unpad_grid(self, pgdata, gdata, axis):
        """Unpad grid data after backward transform."""

        # No padding
        np.copyto(gdata, pgdata)

    def pad_coeff(self, cdata, pcdata, axis):
        """Pad coefficient data before backward transform."""

        size = self.coeff_size

        # Pad with higher order polynomials at end of data
        np.copyto(pcdata[axslice(axis, 0, size)], cdata)
        np.copyto(pcdata[axslice(axis, size, None)], 0.)

    def unpad_coeff(self, pcdata, cdata, axis):
        """Unpad coefficient data after forward transfrom."""

        size = self.coeff_size

        # Discard higher order polynomials at end of data
        np.copyto(cdata, pcdata[axslice(axis, 0, size)])

    def _forward_r2r(self, pgdata, pcdata, axis):
        """Scipy-based DCT on real data."""

        # Scipy DCT
        np.copyto(pcdata, fftpack.dct(pgdata, type=1, norm=None, axis=axis))

        # Normalize as true mode amplitudes
        pcdata /= (self.grid_size - 1)
        pcdata[axslice(axis, 0, 1)] /= 2.
        pcdata[axslice(axis, -1, None)] /= 2.

    def _backward_r2r(self, pcdata, pgdata, axis):
        """Scipy-based IDCT on real data."""

        # Renormalize in output to avoid modifying input
        np.copyto(pgdata, pcdata)
        pgdata[axslice(axis, 1, -1)] /= 2.

        # Scipy DCT
        np.copyto(pgdata, fftpack.dct(pgdata, type=1, norm=None, axis=axis))

    def _forward_c2c(self, pgdata, pcdata, axis):
        """Scipy-based DCT on complex data."""

        # Call real transform on interleaved views of data
        pgdata_iv = interleaved_view(pgdata)
        pcdata_iv = interleaved_view(pcdata)
        self._forward_r2r(pgdata_iv, pcdata_iv, axis)

    def _backward_c2c(self, pcdata, pgdata, axis):
        """Scipy-based IDCT on complex data."""

        # Call real transform on interleaved views of data
        pcdata_iv = interleaved_view(pcdata)
        pgdata_iv = interleaved_view(pgdata)
        self._backward_r2r(pcdata_iv, pgdata_iv, axis)

    def differentiate(self, cdata, cderiv, axis):
        """Differentiation by recursion on coefficients."""

        # Currently setup just for last axis
        if axis != -1:
            if axis != (len(cdata.shape) - 1):
                raise NotImplementedError()

        # Referencess
        a = cdata
        b = cderiv
        N = self.coeff_size - 1

        # Apply recursive differentiation
        b[..., N] = 0.
        b[..., N-1] = 2. * N * a[..., N]
        for i in range(N-2, 0, -1):
            b[..., i] = 2 * (i+1) * a[..., i+1] + b[..., i+2]
        b[..., 0] = a[..., 1] + b[..., 2] / 2.

        # Scale for grid
        cderiv /= self._grid_stretch

    @CachedAttribute
    def Pre(self):
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
    def Diff(self):
        """
        Differentiation matrix.

        d_x(T_n) / n = 2 T_n + d_x(T_(n-2)) / (n-2)

        """

        size = self.coeff_size

        # Construct sparse matrix
        Diff = sparse.lil_matrix((size, size), dtype=self.coeff_dtype)
        for i in range(size-1):
            for j in range(i+1, size, 2):
                if i == 0:
                    Diff[i, j] = j / self._grid_stretch
                else:
                    Diff[i, j] = 2. * j / self._grid_stretch

        return Diff.tocsr()

    @CachedMethod
    def Mult(self, p):
        """
        p-element multiplication matrix

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

    @CachedAttribute
    def left_vector(self):
        """
        Left-endpoint row vector.

        T_n(-1) = (-1)**n

        """

        # Construct dense row vector
        left_vector = np.ones(self.coeff_size, dtype=self.coeff_dtype)
        left_vector[1::2] = -1.

        return left_vector

    @CachedAttribute
    def right_vector(self):
        """
        Right-endpoint row vector.

        T_n(1) = 1

        """

        # Construct dense row vector
        right_vector = np.ones(self.coeff_size, dtype=self.coeff_dtype)

        return right_vector

    @CachedAttribute
    def int_vector(self):
        """
        Integral row vector.

        int(T_n) = (1 + (-1)^n) / (1 - n^2)

        """

        # Construct dense row vector
        int_vector = np.zeros(self.coeff_size, dtype=self.coeff_dtype)
        for n in range(0, self.coeff_size, 2):
            int_vector[n] = 2. / (1. - n*n)
        int_vector *= self._grid_stretch

        return int_vector

    @CachedAttribute
    def bc_vector(self):
        """Last-row column vector for boundary conditions."""

        # Construct dense column vector
        bc_vector = np.zeros((self.coeff_size, 1), dtype=self.coeff_dtype)
        bc_vector[-1, 0] = 1.

        return bc_vector


def odd_floor(x):
    f = math.floor(x)
    if f % 2 == 0:
        f -= 1
    return f


class Fourier(TransverseBasis, ImplicitBasis):
    """Fourier complex exponential basis."""

    def __init__(self, grid_size, interval=(0., 2.*np.pi), cut=1.):

        # Initial attributes
        self.interval = tuple(interval)
        self.coeff_size = odd_floor(cut * grid_size)
        self.coeff_embed = grid_size
        self.grid_size = grid_size
        self.grid_embed = grid_size

        # Grid
        length = interval[1] - interval[0]
        start = interval[0]
        native_grid = np.linspace(0., 1., grid_size, endpoint=False)
        self.grid = start + length * native_grid
        self._grid_stretch = length / (2. * np.pi)

    def set_transforms(self, dtype):
        """Specify datatypes."""

        # Set datatypes
        self.grid_dtype = dtype
        self.coeff_dtype = np.complex128

        # Set transforms
        kmax = self.coeff_size // 2
        if dtype == np.float64:
            self.coeff_size = self.coeff_size // 2 + 1
            self.coeff_embed = self.coeff_embed // 2 + 1
            self.forward = self._forward_r2c
            self.backward = self._backward_c2r
            wavenumbers = np.arange(0, kmax+1)
            self.pad_coeff = self._pad_c2r
            self.unpad_coeff = self._unpad_r2c
        elif dtype == np.complex128:
            self.forward = self._forward_c2c
            self.backward = self._backward_c2c
            wavenumbers = np.arange(-kmax, kmax+1)
            wavenumbers = np.roll(wavenumbers, -kmax)
            self.pad_coeff = self._pad_c2c
            self.unpad_coeff = self._unpad_c2c
        else:
            raise ValueError("Unsupported dtype.")

        self.wavenumbers = wavenumbers / self._grid_stretch

        return self.coeff_dtype

    def _pad_c2r(self, cdata, pdata, axis):
        """Pad out coefficients."""

        size = self.coeff_size

        # Copy data and zero pad
        np.copyto(pdata[axslice(axis, 0, size)], cdata)
        np.copyto(pdata[axslice(axis, size, None)], 0.)

    def _unpad_r2c(self, pdata, cdata, axis):
        """Unpad coefficients."""

        size = self.coeff_size

        # Copy data
        np.copyto(cdata, pdata[axslice(axis, 0, size)])

    def _pad_c2c(self, cdata, pdata, axis):
        """Pad out coefficients."""

        raise NotImplementedError()

    def _unpad_c2c(self, pdata, cdata, axis):
        """Unpad coefficients."""

        raise NotImplementedError()

    def _forward_r2c(self, gdata, cdata, axis):
        """Scipy R2C FFT"""

        cdata[:] = np.fft.rfft(gdata, axis=axis)
        cdata /= self.grid_size

    def _forward_c2c(self, gdata, cdata, axis):
        """Scipy C2C FFT."""

        cdata[:] = fftpack.fft(gdata, axis=axis)
        cdata /= self.grid_size

    def _backward_c2r(self, cdata, gdata, axis):
        """Scipy C2R IFFT"""

        gdata[:] = np.fft.irfft(cdata, n=self.grid_size, axis=axis)
        gdata *= self.grid_size

    def _backward_c2c(self, cdata, gdata, axis):
        """Scipy C2C IFFT."""

        gdata[:] = fftpack.ifft(cdata, axis=axis)
        gdata *= self.grid_size

    def differentiate(self, cdata, cderiv, axis):
        """Differentiation by wavenumber multiplication."""

        # Wavenumber array
        dim = len(cdata.shape)
        ik = 1j * reshape_vector(self.wavenumbers, dim=dim, axis=axis)

        # Multiplication
        cderiv[:] = cdata * ik

    @CachedAttribute
    def Diff(self):
        """
        Differentiation matrix.

        d_x(F_n) = i k_n F_n

        """

        size = self.coeff_size

        # Initialize sparse matrix
        Diff = sparse.lil_matrix((size, size), dtype=self.coeff_dtype)

        # Add elements
        for i in range(size):
            Diff[i, i] = 1j * self.wavenumbers[i]

        return Diff.tocsr()

    @CachedAttribute
    def int_vector(self):
        """
        Integral-evaluation row vector.

        int(F_n) = 2 pi    if n = 0
                 = 0       otherwise

        """

        # Construct dense row vector
        int_vector = np.zeros(self.coeff_size, dtype=self.coeff_dtype)
        int_vector[0] = 2. * np.pi
        int_vector *= self._grid_stretch

        return int_vector

    @CachedAttribute
    def bc_vector(self):
        """First-row column vector for boundary conditions."""

        # Construct dense column vector
        bc_vector = np.zeros((self.coeff_size, 1), dtype=self.coeff_dtype)
        bc_vector[0, :] = 1.

        return bc_vector

    def trans_diff(self, i):
        """Transverse differentation constant for i-th term."""

        return 1j * self.wavenumbers[i]


class NoOp(Basis):
    """No-operation test basis."""

    def __init__(self, grid_size, interval=(0., 1.), cut=1.):

        # Initial attributes
        self.interval = tuple(interval)
        self.coeff_size = math.floor(cut * size)
        self.coeff_embed = grid_size
        self.grid_size = grid_size
        self.grid_embed = grid_size

        # Grid
        length = interval[1] - interval[0]
        start = interval[0]
        native_grid = np.linspace(0., 1., grid_size, endpoint=True)
        self.grid = start + length * native_grid
        self._grid_stretch = length

    def differentiate(self, cdata, cderiv, axis):
        """Differentiate using coefficients."""

        raise NotImplementedError()

    def integrate(self, cdata, axis):
        """Integrate over interval using coefficients."""

        raise NotImplementedError()

    def set_transforms(self, dtype):
        """Specify datatypes."""

        # Set datatypes
        self.grid_dtype = dtype
        self.coeff_dtype = dtype

        return self.coeff_dtype

    def forward(self, gdata, cdata, axis):
        """Grid-to-coefficient transform."""

        cdata[:] = gdata

    def backward(self, cdata, gdata, axis):
        """Coefficient-to-grid transform."""

        gdata[:] = cdata

