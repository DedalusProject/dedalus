"""
Abstract and built-in classes for spectral bases.

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

    These classes define methods for transforming, differentiating, and
    integrating corresponding series represented by their spectral coefficients.

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

    def set_transforms(self, grid_dtype):
        """Set transforms based on grid data type."""

        raise NotImplementedError()

    def pad_coeff(self, cdata, pdata, axis):
        """Pad coefficient data before backward transform."""

        raise NotImplementedError()

    def unpad_coeff(self, pdata, cdata, axis):
        """Unpad coefficient data after forward transfrom."""

    def forward(self, gdata, pdata, axis):
        """Grid-to-coefficient transform."""

        raise NotImplementedError()

    def backward(self, pdata, gdata, axis):
        """Coefficient-to-grid transform."""

        raise NotImplementedError()

    def differentiate(self, cdata, cderiv, axis):
        """Differentiate using coefficients."""

        raise NotImplementedError()

    def integrate(self, cdata, cint, axis):
        """Integrate over interval using coefficients."""

        raise NotImplementedError()

    def integrate(self, cdata, cint, position, axis):
        """Interpolate in interval using coefficients."""

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
        Pre     : preconditioning (default: identity)
        Diff    : differentiation
        Mult(p) : multiplication by p-th basis element

    Linear functionals (vectors):
        left_vector   : left-endpoint evaluation
        right_vector  : right-endpoint evaluation
        integ_vector  : integration over interval
        interp_vector : interpolation in interval

    Additionally, they define a column vector `bc_vector` indicating which
    coefficient's Galerkin constraint is to be replaced by the boundary
    condition on a differential equation (i.e. the order of the tau term).

    """

    def integrate(self, cdata, cint, axis):
        """Integrate over interval using coefficients."""

        # Contract coefficients with basis function integrals
        dim = len(cdata.shape)
        weights = reshape_vector(self.integ_vector, dim=dim, axis=axis)
        integral = np.sum(cdata * weights, axis=axis, keepdims=True)

        cint.fill(0)
        np.copyto(cint[axslice(axis, 0, 1)], integral)

    def interpolate(self, cdata, cint, position, axis):
        """Integrate over interval using coefficients."""

        # Contract coefficients with basis function evaluations
        dim = len(cdata.shape)
        weights = reshape_vector(self.interp_vector(position), dim=dim, axis=axis)
        interpolation = np.sum(cdata * weights, axis=axis, keepdims=True)

        cint.fill(0)
        np.copyto(cint[axslice(axis, 0, 1)], interpolation)

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

        # Outer product of boundary-row and left-endpoint vectors
        Left = sparse.kron(self.bc_vector, self.left_vector)

        return Left

    @CachedAttribute
    def Right(self):
        """Right-endpoint matrix."""

        # Outer product of boundary-row and right-endpoint vectors
        Right = sparse.kron(self.bc_vector, self.right_vector)

        return Right

    @CachedAttribute
    def Int(self):
        """Integral matrix."""

        # Outer product of boundary-row and integral vectors
        Int = sparse.kron(self.bc_vector, self.integ_vector)

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
    def integ_vector(self):
        """Integral row vector."""

        raise NotImplementedError()

    @CachedMethod
    def interp_vector(self, position):
        """Interpolation row vector."""

        raise NotImplementedError()

    @CachedAttribute
    def bc_vector(self):
        """Boundary-row column vector."""

        raise NotImplementedError()


class Chebyshev(ImplicitBasis):
    """Chebyshev polynomial basis on the extrema grid."""

    def __init__(self, grid_size, interval=(-1., 1.), dealias=1., name=None):

        # Initial attributes
        self.name = name
        self.grid_size = grid_size
        self.interval = tuple(interval)
        self.dealias = dealias

        # Retain maximum number of coefficients below threshold
        self.coeff_embed = grid_size
        self.coeff_size = math.floor(dealias * grid_size)

        # Extrema grid
        radius = (interval[1] - interval[0]) / 2.
        center = (interval[1] + interval[0]) / 2.
        self._grid_stretch = radius
        self._basis_coord = lambda X: (X - center) / radius
        self._problem_coord = lambda x: center + (x * radius)

        i = np.arange(grid_size)
        N = grid_size - 1
        native_grid = np.cos(np.pi * i / N)
        self.grid = self._problem_coord(native_grid)

    def set_transforms(self, grid_dtype):
        """Set transforms based on grid data type."""

        # Transform retains data type
        self.grid_dtype = grid_dtype
        self.coeff_dtype = grid_dtype

        # Dispatch transform methods
        if grid_dtype == np.float64:
            self.forward = self._forward_r2r
            self.backward = self._backward_r2r
        elif grid_dtype == np.complex128:
            self.forward = self._forward_c2c
            self.backward = self._backward_c2c
        else:
            raise ValueError("Unsupported grid_dtype.")

        # Basis elements
        self.elements = np.arange(self.coeff_size)

        return self.coeff_dtype

    def pad_coeff(self, cdata, pdata, axis):
        """Pad coefficient data before backward transform."""

        size = self.coeff_size

        # Pad with higher order polynomials at end of data
        np.copyto(pdata[axslice(axis, 0, size)], cdata)
        np.copyto(pdata[axslice(axis, size, None)], 0.)

    def unpad_coeff(self, pdata, cdata, axis):
        """Unpad coefficient data after forward transfrom."""

        size = self.coeff_size

        # Discard higher order polynomials at end of data
        np.copyto(cdata, pdata[axslice(axis, 0, size)])

    def _forward_r2r(self, gdata, pdata, axis):
        """Scipy-based DCT on real data."""

        # Scipy DCT
        np.copyto(pdata, fftpack.dct(gdata, type=1, norm=None, axis=axis))

        # Normalize as true mode amplitudes
        pdata /= (self.grid_size - 1)
        pdata[axslice(axis, 0, 1)] /= 2.
        pdata[axslice(axis, -1, None)] /= 2.

    def _backward_r2r(self, pdata, gdata, axis):
        """Scipy-based IDCT on real data."""

        # Renormalize in output to avoid modifying input
        np.copyto(gdata, pdata)
        gdata[axslice(axis, 1, -1)] /= 2.

        # Scipy DCT
        np.copyto(gdata, fftpack.dct(gdata, type=1, norm=None, axis=axis))

    def _forward_c2c(self, gdata, pdata, axis):
        """Scipy-based DCT on complex data."""

        # Call real transform on interleaved views of data
        gdata_iv = interleaved_view(gdata)
        pdata_iv = interleaved_view(pdata)
        self._forward_r2r(gdata_iv, pdata_iv, axis)

    def _backward_c2c(self, pdata, gdata, axis):
        """Scipy-based IDCT on complex data."""

        # Call real transform on interleaved views of data
        pdata_iv = interleaved_view(pdata)
        gdata_iv = interleaved_view(gdata)
        self._backward_r2r(pdata_iv, gdata_iv, axis)

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

        # Scale for interval
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
    def integ_vector(self):
        """
        Integral row vector.

        int(T_n) = (1 + (-1)^n) / (1 - n^2)

        """

        # Construct dense row vector
        integ_vector = np.zeros(self.coeff_size, dtype=self.coeff_dtype)
        for n in range(0, self.coeff_size, 2):
            integ_vector[n] = 2. / (1. - n*n)
        integ_vector *= self._grid_stretch

        return integ_vector

    @CachedMethod
    def interp_vector(self, position):
        """
        Interpolation row vector.

        T_n(x) = cos(n * acos(x))

        """

        # Construct dense row vector
        theta = np.arccos(self._basis_coord(position))
        interp_vector = np.cos(self.elements * theta)

        return interp_vector

    @CachedAttribute
    def bc_vector(self):
        """
        Last-row column vector for boundary conditions. This sets the tau term
        proportional to the highest-order-retained polynomial.

        """

        # Construct dense column vector
        bc_vector = np.zeros((self.coeff_size, 1), dtype=self.coeff_dtype)
        bc_vector[-1, 0] = 1.

        return bc_vector


class Fourier(TransverseBasis, ImplicitBasis):
    """Fourier complex exponential basis."""

    def __init__(self, grid_size, interval=(0., 2.*np.pi), dealias=1., name=None):

        # Initial attributes
        self.name = name
        self.grid_size = grid_size
        self.interval = tuple(interval)
        self.dealias = dealias

        # Retain maximum odd number of coefficients below threshold, since the
        # highest order mode retained is in general not the Nyquist mode.
        self.coeff_embed = grid_size
        self.coeff_size = math.floor(dealias * grid_size)
        if self.coeff_size % 2 == 0:
            self.coeff_size -= 1

        # Evenly spaced grid
        length = interval[1] - interval[0]
        start = interval[0]
        self._grid_stretch = length / (2. * np.pi)
        self._basis_coord = lambda X: (X - start) / length
        self._problem_coord = lambda x: start + (x * length)

        native_grid = np.linspace(0., 1., grid_size, endpoint=False)
        self.grid = self._problem_coord(native_grid)

    def set_transforms(self, dtype):
        """Set transforms based on grid data type."""

        # Transform always produces complex coefficients
        self.grid_dtype = dtype
        self.coeff_dtype = np.complex128

        # Dispatch transform and dealiasing methods
        if dtype == np.float64:
            self.forward = self._forward_r2c
            self.backward = self._backward_c2r
            self.pad_coeff = self._pad_c2r
            self.unpad_coeff = self._unpad_r2c
        elif dtype == np.complex128:
            self.forward = self._forward_c2c
            self.backward = self._backward_c2c
            self.pad_coeff = self._pad_c2c
            self.unpad_coeff = self._unpad_c2c
        else:
            raise ValueError("Unsupported grid_dtype.")

        # Construct wavenumbers
        kmax = self.coeff_size // 2
        if dtype == np.float64:
            # Exclude (redundant) negative wavenumbers
            self.coeff_size = self.coeff_size // 2 + 1
            self.coeff_embed = self.coeff_embed // 2 + 1
            # Positive wavenumbers only
            wavenumbers = np.arange(0, kmax+1)
        elif dtype == np.complex128:
            # Positive then negative wavenumbers
            wavenumbers = np.arange(-kmax, kmax+1)
            wavenumbers = np.roll(wavenumbers, -kmax)

        # Scale native (integer) wavenumbers
        self.wavenumbers = wavenumbers / self._grid_stretch
        self.elements = self.wavenumbers

        return self.coeff_dtype

    def _pad_c2r(self, cdata, pdata, axis):
        """Pad coefficient data before backward transform."""

        size = self.coeff_size

        # Pad with higher wavenumbers at end of data
        np.copyto(pdata[axslice(axis, 0, size)], cdata)
        np.copyto(pdata[axslice(axis, size, None)], 0.)

    def _unpad_r2c(self, pdata, cdata, axis):
        """Unpad coefficient data after forward transfrom."""

        size = self.coeff_size

        # Discard higher wavenumbers at end of data
        np.copyto(cdata, pdata[axslice(axis, 0, size)])

    def _pad_c2c(self, cdata, pdata, axis):
        """Pad coefficient data before backward transform."""

        kmax = self.coeff_size // 2
        posfreq = axslice(axis, 0, kmax+1)
        negfreq = axslice(axis, -kmax, None)

        # Pad with higher wavenumbers and conjugates
        np.copyto(pdata[posfreq], cdata[posfreq])
        np.copyto(pdata[axslice(axis, kmax+1, -kmax)], 0.)
        np.copyto(pdata[negfreq], cdata[negfreq])

    def _unpad_c2c(self, pdata, cdata, axis):
        """Unpad coefficient data after forward transfrom."""

        kmax = self.coeff_size // 2
        posfreq = axslice(axis, 0, kmax+1)
        negfreq = axslice(axis, -kmax, None)

        # Discard higher wavenumbers and conjugates
        np.copyto(cdata[posfreq], pdata[posfreq])
        np.copyto(cdata[negfreq], pdata[negfreq])

    def _forward_r2c(self, gdata, pdata, axis):
        """Numpy-based R2C FFT."""

        # Numpy RFFT
        np.copyto(pdata, np.fft.rfft(gdata, axis=axis))

        # Normalize as true mode amplitudes
        pdata /= self.grid_size

    def _backward_c2r(self, pdata, gdata, axis):
        """Numpy-based C2R IFFT."""

        # Numpy IRFFT
        np.copyto(gdata, np.fft.irfft(pdata, n=self.grid_size, axis=axis))

        # Renormalize
        gdata *= self.grid_size

    def _forward_c2c(self, gdata, pdata, axis):
        """Numpy-based C2C FFT."""

        # Numpy FFT
        np.copyto(pdata, fftpack.fft(gdata, axis=axis))

        # Normalize as true mode amplitudes
        pdata /= self.grid_size

    def _backward_c2c(self, pdata, gdata, axis):
        """Numpy-based C2C IFFT."""

        # Numpy IFFT
        np.copyto(gdata, fftpack.ifft(pdata, axis=axis))

        # Renormalize
        gdata *= self.grid_size

    def differentiate(self, cdata, cderiv, axis):
        """Differentiation by wavenumber multiplication."""

        # Wavenumber array
        dim = len(cdata.shape)
        ik = 1j * reshape_vector(self.wavenumbers, dim=dim, axis=axis)

        # Multiplication
        np.multiply(cdata, ik, out=cderiv)

    @CachedAttribute
    def Diff(self):
        """
        Differentiation matrix.

        d_x(F_n) = i k_n F_n

        """

        size = self.coeff_size

        # Construct sparse matrix
        Diff = sparse.lil_matrix((size, size), dtype=self.coeff_dtype)
        for i in range(size):
            Diff[i, i] = 1j * self.wavenumbers[i]

        return Diff.tocsr()

    @CachedMethod
    def Mult(self, p):
        """
        p-element multiplication matrix

        F_0 * F_n = F_n

        """

        if p == 0:
            # Identity matrix
            Mult = sparse.identity(self.coeff_size, dtype=self.coeff_dtype)

            return Mult.tocsr()
        else:
            raise NotImplementedError()

    @CachedAttribute
    def integ_vector(self):
        """
        Integral row vector.

        int(F_n) = 2 pi    if n = 0
                 = 0       otherwise

        """

        # Construct dense row vector
        integ_vector = np.zeros(self.coeff_size, dtype=self.coeff_dtype)
        integ_vector[0] = 2. * np.pi
        integ_vector *= self._grid_stretch

        return integ_vector

    def interpolate(self, cdata, cint, position, axis):
        """Integrate over interval using coefficients."""

        # Contract coefficients with basis function evaluations
        dim = len(cdata.shape)
        weights = reshape_vector(self.interp_vector(position), dim=dim, axis=axis)
        if self.grid_dtype == np.float64:
            pos_interp = np.sum(cdata * weights, axis=axis, keepdims=True)
            interpolation = pos_interp + pos_interp.conj()
        elif self.grid_dtype == np.complex128:
            interpolation = np.sum(cdata * weights, axis=axis, keepdims=True)

        cint.fill(0)
        np.copyto(cint[axslice(axis, 0, 1)], interpolation)

    @CachedMethod
    def interp_vector(self, position):
        """
        Interpolation row vector.

        F_n(x) = exp(i k_n x)

        """

        # Construct dense row vector
        x = position - self.interval[0]
        interp_vector = np.exp(1j * self.wavenumbers * x)
        if self.grid_dtype == np.float64:
            interp_vector[0] /= 2

        return interp_vector

    @CachedAttribute
    def bc_vector(self):
        """
        First-row column vector for boundary conditions. This allows the
        constant term to be varied to satisfy integral conditions."""

        # Construct dense column vector
        bc_vector = np.zeros((self.coeff_size, 1), dtype=self.coeff_dtype)
        bc_vector[0, :] = 1.

        return bc_vector

    def trans_diff(self, i):
        """Transverse differentation constant for i-th term."""

        return 1j * self.wavenumbers[i]


class NoOp(Basis):
    """No-operation test basis."""

    def __init__(self, grid_size, interval=(0., 1.), dealias=1.):

        # Initial attributes
        self.grid_size = grid_size
        self.interval = tuple(interval)
        self.dealias = dealias

        # Retain maximum number of coefficients below threshold
        self.coeff_embed = grid_size
        self.coeff_size = math.floor(dealias * grid_size)

        # Grid
        length = interval[1] - interval[0]
        start = interval[0]
        native_grid = np.linspace(0., 1., grid_size, endpoint=True)
        self.grid = start + length * native_grid
        self._grid_stretch = length

    def set_transforms(self, dtype):
        """Set transforms based on grid data type."""

        # No-op transform retains data type
        self.grid_dtype = dtype
        self.coeff_dtype = dtype

        return self.coeff_dtype

    def pad_coeff(self, cdata, pdata, axis):
        """Pad coefficient data before backward transform."""

        size = self.coeff_size

        # Pad with zeros at end of data
        np.copyto(pdata[axslice(axis, 0, size)], cdata)
        np.copyto(pdata[axslice(axis, size, None)], 0.)

    def unpad_coeff(self, pdata, cdata, axis):
        """Unpad coefficient data after forward transfrom."""

        size = self.coeff_size

        # Discard zeros at end of data
        np.copyto(cdata, pdata[axslice(axis, 0, size)])

    def forward(self, gdata, pdata, axis):
        """Grid-to-coefficient transform."""

        # Copy data
        np.copyto(pdata, gdata)

    def backward(self, pdata, gdata, axis):
        """Coefficient-to-grid transform."""

        # Copy data
        np.copyto(gdata, pdata)

