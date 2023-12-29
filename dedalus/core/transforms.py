"""Spectral transform classes."""

# TODO: implement transforms as cached classes

import numpy as np
import scipy
import scipy.fft
import scipy.fftpack
from ..libraries import dedalus_sphere
from math import prod

from . import basis
from ..libraries.fftw import fftw_wrappers as fftw
from ..tools import jacobi
from ..tools.array import apply_matrix, apply_dense, axslice, solve_upper_sparse, apply_sparse
from ..tools.cache import CachedAttribute
from ..tools.cache import CachedMethod

import logging
logger = logging.getLogger(__name__.split('.')[-1])

from ..tools.config import config
GET_FFTW_RIGOR = lambda: config['transforms-fftw'].get('PLANNING_RIGOR')
GET_DEALIAS_BEFORE_CONVERTING = lambda: config['transforms'].getboolean('DEALIAS_BEFORE_CONVERTING')


def register_transform(basis, name):
    """Decorator to add transform to basis class dictionary."""
    def wrapper(cls):
        basis.transforms[name] = cls
        return cls
    return wrapper


class Transform:
    """Abstract base class for all transforms."""
    pass


class SeparableTransform(Transform):
    """Abstract base class for transforms that only apply to one dimension, independent of all others."""

    def forward(self, gdata, cdata, axis):
        """Apply forward transform along specified axis."""
        # Subclasses must implement
        raise NotImplementedError("%s has not implemented 'forward' method" %type(self))

    def backward(self, cdata, gdata, axis):
        """Apply backward transform along specified axis."""
        # Subclasses must implement
        raise NotImplementedError("%s has not implemented 'backward' method" %type(self))


class SeparableMatrixTransform(SeparableTransform):
    """Abstract base class for separable matrix-multiplication transforms."""

    def forward(self, gdata, cdata, axis):
        """Apply forward transform along specified axis."""
        apply_dense(self.forward_matrix, gdata, axis=axis, out=cdata)

    def backward(self, cdata, gdata, axis):
        """Apply backward transform along specified axis."""
        apply_dense(self.backward_matrix, cdata, axis=axis, out=gdata)

    @CachedAttribute
    def forward_matrix(self):
        """Build forward transform matrix."""
        # Subclasses must implement
        raise NotImplementedError("%s has not implemented 'forward_matrix' method" %type(self))

    @CachedAttribute
    def backward_matrix(self):
        """Build backward transform matrix."""
        # Subclasses must implement
        raise NotImplementedError("%s has not implemented 'backward_matrix' method" %type(self))


class JacobiTransform(SeparableTransform):
    """
    Abstract base class for Jacobi polynomial transforms.

    Parameters
    ----------
    grid_size : int
        Grid size (N) along transform dimension.
    coeff_size : int
        Coefficient size (M) along transform dimension.
    a : int
        Jacobi "a" parameter for polynomials.
    b : int
        Jacobi "b" parameters for polynomials.
    a0 : int
        Jacobi "a" parameter for the quadrature grid.
    b0 : int
        Jacobi "b" parameter for the quadrature grid.

    Notes
    -----
    TODO: We need to define the normalization we use here.
    """

    def __init__(self, grid_size, coeff_size, a, b, a0, b0, dealias_before_converting=None):
        self.N = grid_size
        self.M = coeff_size
        self.a = a
        self.b = b
        self.a0 = a0
        self.b0 = b0
        if dealias_before_converting is None:
            dealias_before_converting = GET_DEALIAS_BEFORE_CONVERTING()
        self.dealias_before_converting = dealias_before_converting


@register_transform(basis.Jacobi, 'matrix')
class JacobiMMT(JacobiTransform, SeparableMatrixTransform):
    """Jacobi polynomial MMTs."""

    @CachedAttribute
    def forward_matrix(self):
        """Build forward transform matrix."""
        N, M = self.N, self.M
        a, a0 = self.a, self.a0
        b, b0 = self.b, self.b0
        # Gauss quadrature with base (a0, b0) polynomials
        base_grid = jacobi.build_grid(N, a=a0, b=b0)
        base_polynomials = jacobi.build_polynomials(max(M, N), a0, b0, base_grid)
        base_weights = jacobi.build_weights(N, a=a0, b=b0)
        base_transform = (base_polynomials * base_weights)
        # Zero higher coefficients for transforms with grid_size < coeff_size
        base_transform[N:, :] = 0
        if self.dealias_before_converting:
            # Truncate to specified coeff_size
            base_transform = base_transform[:M, :]
        # Spectral conversion
        if (a == a0) and (b == b0):
            forward_matrix = base_transform
        else:
            conversion = jacobi.conversion_matrix(base_transform.shape[0], a0, b0, a, b)
            forward_matrix = conversion @ base_transform
        if not self.dealias_before_converting:
            # Truncate to specified coeff_size
            forward_matrix = forward_matrix[:M, :]
        # Ensure C ordering for fast dot products
        return np.asarray(forward_matrix, order='C')

    @CachedAttribute
    def backward_matrix(self):
        """Build backward transform matrix."""
        N, M = self.N, self.M
        a, a0 = self.a, self.a0
        b, b0 = self.b, self.b0
        # Construct polynomials on the base grid
        base_grid = jacobi.build_grid(N, a=a0, b=b0)
        polynomials = jacobi.build_polynomials(M, a, b, base_grid)
        # Zero higher polynomials for transforms with grid_size < coeff_size
        polynomials[N:, :] = 0
        # Transpose and ensure C ordering for fast dot products
        return np.asarray(polynomials.T, order='C')


class ComplexFourierTransform(SeparableTransform):
    """
    Abstract base class for complex-to-complex Fourier transforms.

    Parameters
    ----------
    grid_size : int
        Grid size (N) along transform dimension.
    coeff_size : int
        Coefficient size (M) along transform dimension.

    Notes
    -----
    Let KN = (N - 1) // 2 be the maximum fully resolved (non-Nyquist) mode on the grid.
    Let KM = (M - 1) // 2 be the maximum retained mode in coeff space.
    Then K = min(KN, KM) is the maximum wavenumber used in the transforms.
    A unit-amplitude normalization is used.

    Forward transform:
        if abs(k) <= K:
            F(k) = (1/N) \sum_{x=0}^{N-1} f(x) \exp(-2 \pi i k x / N)
        else:
            F(k) = 0

    Backward transform:
        f(x) = \sum_{k=-K}^{K} F(k) \exp(2 \pi i k x / N)

    Coefficient ordering:
        If M is odd, the ordering is [0, 1, 2, ..., KM, KM+1, -KM, -KM+1, ..., -1],
        where the Nyquist mode k = KM + 1 is zeroed in both directions.
        If M is even, the ordering is [0, 1, 2, ..., KM, -KM, -KM+1, ..., -1].
    """

    def __init__(self, grid_size, coeff_size):
        self.N = grid_size
        self.M = coeff_size
        self.KN = (self.N - 1) // 2
        self.KM = (self.M - 1) // 2
        self.Kmax = min(self.KN, self.KM)

    @property
    def wavenumbers(self):
        """One-dimensional global wavenumber array."""
        M = self.M
        KM = self.KM
        k = np.arange(M)
        # Wrap around Nyquist mode
        return (k + KM) % M - KM


@register_transform(basis.ComplexFourier, 'matrix')
class ComplexFourierMMT(ComplexFourierTransform, SeparableMatrixTransform):
    """Complex-to-complex Fourier MMT."""

    @CachedAttribute
    def forward_matrix(self):
        """Build forward transform matrix."""
        K = self.wavenumbers[:, None]
        X = np.arange(self.N)[None, :]
        dX = self.N / 2 / np.pi
        quadrature = np.exp(-1j*K*X/dX) / self.N
        # Zero Nyquist and higher modes for transforms with grid_size <= coeff_size
        quadrature *= np.abs(K) <= self.Kmax
        # Ensure C ordering for fast dot products
        return np.asarray(quadrature, order='C')

    @CachedAttribute
    def backward_matrix(self):
        """Build backward transform matrix."""
        K = self.wavenumbers[None, :]
        X = np.arange(self.N)[:, None]
        dX = self.N / 2 / np.pi
        functions = np.exp(1j*K*X/dX)
        # Zero Nyquist and higher modes for transforms with grid_size <= coeff_size
        functions *= np.abs(K) <= self.Kmax
        # Ensure C ordering for fast dot products
        return np.asarray(functions, order='C')


class ComplexFFT(ComplexFourierTransform):
    """Abstract base class for complex-to-complex FFTs."""

    def resize_coeffs(self, data_in, data_out, axis, rescale):
        """Resize and rescale coefficients in standard FFT format by intermediate padding/truncation."""
        M = self.M
        Kmax = self.Kmax
        if Kmax == 0:
            posfreq = axslice(axis, 0, 1)
            badfreq = axslice(axis, 1, None)
            if rescale is None:
                np.copyto(data_out[posfreq], data_in[posfreq])
                data_out[badfreq] = 0
            else:
                np.multiply(data_in[posfreq], rescale, data_out[posfreq])
                data_out[badfreq] = 0
        else:
            posfreq = axslice(axis, 0, Kmax+1)
            badfreq = axslice(axis, Kmax+1, -Kmax)
            negfreq = axslice(axis, -Kmax, None)
            if rescale is None:
                np.copyto(data_out[posfreq], data_in[posfreq])
                data_out[badfreq] = 0
                np.copyto(data_out[negfreq], data_in[negfreq])
            else:
                np.multiply(data_in[posfreq], rescale, data_out[posfreq])
                data_out[badfreq] = 0
                np.multiply(data_in[negfreq], rescale, data_out[negfreq])


@register_transform(basis.ComplexFourier, 'scipy')
class ScipyComplexFFT(ComplexFFT):
    """Complex-to-complex FFT using scipy.fft."""

    def forward(self, gdata, cdata, axis):
        """Apply forward transform along specified axis."""
        # Call FFT
        temp = scipy.fft.fft(gdata, axis=axis) # Creates temporary
        # Resize and rescale for unit-amplitude normalization
        self.resize_coeffs(temp, cdata, axis, rescale=1/self.N)

    def backward(self, cdata, gdata, axis):
        """Apply backward transform along specified axis."""
        # Resize and rescale for unit-amplitude normalization
        # Need temporary to avoid overwriting problems
        temp = np.empty_like(gdata) # Creates temporary
        self.resize_coeffs(cdata, temp, axis, rescale=self.N)
        # Call FFT
        temp = scipy.fft.ifft(temp, axis=axis, overwrite_x=True) # Creates temporary
        np.copyto(gdata, temp)


class FFTWBase:
    """Abstract base class for FFTW transforms."""

    def __init__(self, *args, rigor=None, **kw):
        if rigor is None:
            rigor = GET_FFTW_RIGOR()
        self.rigor = rigor
        super().__init__(*args, **kw)


@register_transform(basis.ComplexFourier, 'fftw')
class FFTWComplexFFT(FFTWBase, ComplexFFT):
    """Complex-to-complex FFT using FFTW."""

    @CachedMethod
    def _build_fftw_plan(self, gshape, axis):
        """Build FFTW plans and temporary arrays."""
        dtype = np.complex128
        logger.debug("Building FFTW FFT plan for (dtype, gshape, axis) = (%s, %s, %s)" %(dtype, gshape, axis))
        flags = ['FFTW_'+self.rigor.upper()]
        plan = fftw.FourierTransform(dtype, gshape, axis, flags=flags)
        temp = fftw.create_array(plan.cshape, np.complex128)
        return plan, temp

    def forward(self, gdata, cdata, axis):
        """Apply forward transform along specified axis."""
        plan, temp = self._build_fftw_plan(gdata.shape, axis)
        # Execute FFTW plan
        plan.forward(gdata, temp)
        # Resize and rescale for unit-amplitude normalization
        self.resize_coeffs(temp, cdata, axis, rescale=1/self.N)

    def backward(self, cdata, gdata, axis):
        """Apply backward transform along specified axis."""
        plan, temp = self._build_fftw_plan(gdata.shape, axis)
        # Resize and rescale for unit-amplitude normalization
        self.resize_coeffs(cdata, temp, axis, rescale=None)
        # Execute FFTW plan
        plan.backward(temp, gdata)


class RealFourierTransform(SeparableTransform):
    """
    Abstract base class for real-to-real Fourier transforms.

    Parameters
    ----------
    grid_size : int
        Grid size (N) along transform dimension.
    coeff_size : int
        Coefficient size (M) along transform dimension.

    Notes
    -----
    Let KN = (N - 1) // 2 be the maximum fully resolved (non-Nyquist) mode on the grid.
    Let KM = (M - 1) // 2 be the maximum retained mode in coeff space.
    Then K = min(KN, KM) is the maximum wavenumber used in the transforms.
    A unit-amplitude normalization is used.

    Forward transform:
        if k == 0:
            a(k) = (1/N) \sum_{x=0}^{N-1} f(x)
            b(k) = 0
        elif k <= K:
            a(k) =  (2/N) \sum_{x=0}^{N-1} f(x) \cos(-2 \pi k x / N)
            b(k) = -(2/N) \sum_{x=0}^{N-1} f(x) \sin(-2 \pi k x / N)
        else:
            a(k) = 0
            b(k) = 0

    Backward transform:
        f(x) = \sum_{k=0}^{K} a(k) \cos(2 \pi k x / N) - b(k) \sin(2 \pi k x / N)

    Coefficient ordering:
        The cosine and minus-sine coefficients are interleaved as
        [a(0), b(0), a(1), b(1), a(2), b(2), ..., a(KM), b(KM)]
        where the k = 0 minus-sine mode is zeroed in both directions.
    """

    def __init__(self, grid_size, coeff_size):
        if coeff_size % 2 != 0:
            pass#raise ValueError("coeff_size must be even.")
        self.N = grid_size
        self.M = coeff_size
        self.KN = (self.N - 1) // 2
        self.KM = (self.M - 1) // 2
        self.Kmax = min(self.KN, self.KM)

    @property
    def wavenumbers(self):
        """One-dimensional global wavenumber array."""
        # Repeat k's for cos and msin parts
        return np.repeat(np.arange(self.KM+1), 2)


@register_transform(basis.RealFourier, 'matrix')
class RealFourierMMT(RealFourierTransform, SeparableMatrixTransform):
    """Real-to-real Fourier MMT."""

    @CachedAttribute
    def forward_matrix(self):
        """Build forward transform matrix."""
        N = self.N
        M = max(2, self.M) # Account for sin and cos parts of m=0
        Kmax = self.Kmax
        K = self.wavenumbers[::2, None]
        X = np.arange(N)[None, :]
        dX = N / 2 / np.pi
        quadrature = np.zeros((M, N))
        quadrature[0::2] = (2 / N) * np.cos(K*X/dX)
        quadrature[1::2] = -(2 / N) * np.sin(K*X/dX)
        quadrature[0] = 1 / N
        # Zero Nyquist and higher modes for transforms with grid_size <= coeff_size
        quadrature *= self.wavenumbers[:,None] <= self.Kmax
        # Ensure C ordering for fast dot products
        return np.asarray(quadrature, order='C')

    @CachedAttribute
    def backward_matrix(self):
        """Build backward transform matrix."""
        N = self.N
        M = max(2, self.M) # Account for sin and cos parts of m=0
        Kmax = self.Kmax
        K = self.wavenumbers[None, ::2]
        X = np.arange(N)[:, None]
        dX = N / 2 / np.pi
        functions = np.zeros((N, M))
        functions[:, 0::2] = np.cos(K*X/dX)
        functions[:, 1::2] = -np.sin(K*X/dX)
        # Zero Nyquist and higher modes for transforms with grid_size <= coeff_size
        functions *= self.wavenumbers[None, :] <= self.Kmax
        # Ensure C ordering for fast dot products
        return np.asarray(functions, order='C')


@register_transform(basis.RealFourier, 'fftpack')
class FFTPACKRealFFT(RealFourierTransform):
    """Real-to-real FFT using scipy.fftpack."""

    def forward(self, gdata, cdata, axis):
        """Apply forward transform along specified axis."""
        N = self.N
        Kmax = self.Kmax
        # Call RFFT
        temp = scipy.fftpack.rfft(gdata, axis=axis) # Creates temporary
        # Scale k = 0 cos data
        meancos = axslice(axis, 0, 1)
        np.multiply(temp[meancos], (1 / N), cdata[meancos])
        # Zero k = 0 sin data
        cdata[axslice(axis, 1, 2)] = 0
        # Shift and scale 1 < k <= Kmax data
        temp_posfreq = temp[axslice(axis, 1, 2*(Kmax+1)-1)]
        cdata_posfreq = cdata[axslice(axis, 2, 2*(Kmax+1))]
        np.multiply(temp_posfreq, (2 / N), cdata_posfreq)
        # Zero k > Kmax data
        cdata[axslice(axis, 2*(Kmax+1), None)] = 0

    def backward(self, cdata, gdata, axis):
        """Apply backward transform along specified axis."""
        N = self.N
        Kmax = self.Kmax
        # Need temporary to avoid overwriting problems
        temp = np.empty_like(gdata) # Creates temporary
        # Scale k = 0 cos data
        meancos = axslice(axis, 0, 1)
        np.multiply(cdata[meancos], N, temp[meancos])
        # Shift and scale 1 < k <= Kmax data
        temp_posfreq = temp[axslice(axis, 1, 2*(Kmax+1)-1)]
        cdata_posfreq = cdata[axslice(axis, 2, 2*(Kmax+1))]
        np.multiply(cdata_posfreq, (N / 2), temp_posfreq)
        # Zero k > Kmax data
        temp[axslice(axis, 2*(Kmax+1)-1, None)] = 0
        # Call IRFFT
        temp = scipy.fftpack.irfft(temp, axis=axis, overwrite_x=True) # Creates temporary
        np.copyto(gdata, temp)


class RealFFT(RealFourierTransform):
    """Abstract base class for real-to-real FFTs using real-to-complex algorithms."""

    def unpack_rescale(self, temp, cdata, axis, rescale):
        """Unpack complex coefficients and rescale for unit-amplitude normalization."""
        Kmax = self.Kmax
        # Scale k = 0 cos data
        meancos = axslice(axis, 0, 1)
        np.multiply(temp[meancos].real, rescale, cdata[meancos])
        # Zero k = 0 msin data
        cdata[axslice(axis, 1, 2)] = 0
        # Unpack and scale 1 < k <= Kmax data
        temp_posfreq = temp[axslice(axis, 1, Kmax+1)]
        cdata_posfreq_cos = cdata[axslice(axis, 2, 2*(Kmax+1), 2)]
        cdata_posfreq_msin = cdata[axslice(axis, 3, 2*(Kmax+1), 2)]
        np.multiply(temp_posfreq.real, 2*rescale, cdata_posfreq_cos)
        np.multiply(temp_posfreq.imag, 2*rescale, cdata_posfreq_msin)
        # Zero k > Kmax data
        cdata[axslice(axis, 2*(Kmax+1), None)] = 0

    def repack_rescale(self, cdata, temp, axis, rescale):
        """Repack into complex coefficients and rescale for unit-amplitude normalization."""
        Kmax = self.Kmax
        # Scale k = 0 data
        meancos = axslice(axis, 0, 1)
        if rescale is None:
            np.copyto(temp[meancos], cdata[meancos])
        else:
            np.multiply(cdata[meancos], rescale, temp[meancos])
        # Repack and scale 1 < k <= Kmax data
        temp_posfreq = temp[axslice(axis, 1, Kmax+1)]
        cdata_posfreq_cos = cdata[axslice(axis, 2, 2*(Kmax+1), 2)]
        cdata_posfreq_msin = cdata[axslice(axis, 3, 2*(Kmax+1), 2)]
        if rescale is None:
            np.multiply(cdata_posfreq_cos, (1 / 2), temp_posfreq.real)
            np.multiply(cdata_posfreq_msin, (1 / 2), temp_posfreq.imag)
        else:
            np.multiply(cdata_posfreq_cos, (rescale / 2), temp_posfreq.real)
            np.multiply(cdata_posfreq_msin, (rescale / 2), temp_posfreq.imag)
        # Zero k > Kmax data
        temp[axslice(axis, Kmax+1, None)] = 0


@register_transform(basis.RealFourier, 'scipy')
class ScipyRealFFT(RealFFT):
    """Real-to-real FFT using scipy.fft."""

    def forward(self, gdata, cdata, axis):
        """Apply forward transform along specified axis."""
        # Call RFFT
        temp = scipy.fft.rfft(gdata, axis=axis) # Creates temporary
        # Unpack from complex form and rescale
        self.unpack_rescale(temp, cdata, axis, rescale=1/self.N)

    def backward(self, cdata, gdata, axis):
        """Apply backward transform along specified axis."""
        N = self.N
        # Rescale all modes and combine into complex form
        shape = list(gdata.shape)
        shape[axis] = N // 2 + 1
        temp = np.empty(shape=shape, dtype=np.complex128) # Creates temporary
        # Repack into complex form and rescale
        self.repack_rescale(cdata, temp, axis, rescale=N)
        # Call IRFFT
        temp = scipy.fft.irfft(temp, axis=axis, n=N, overwrite_x=True) # Creates temporary
        np.copyto(gdata, temp)


@register_transform(basis.RealFourier, 'fftw')
class FFTWRealFFT(FFTWBase, RealFFT):
    """Real-to-real FFT using FFTW."""

    @CachedMethod
    def _build_fftw_plan(self, gshape, axis):
        """Build FFTW plans and temporary arrays."""
        dtype = np.float64
        logger.debug("Building FFTW FFT plan for (dtype, gshape, axis) = (%s, %s, %s)" %(dtype, gshape, axis))
        flags = ['FFTW_'+self.rigor.upper()]
        plan = fftw.FourierTransform(dtype, gshape, axis, flags=flags)
        temp = fftw.create_array(plan.cshape, np.complex128)
        return plan, temp

    def forward(self, gdata, cdata, axis):
        """Apply forward transform along specified axis."""
        plan, temp = self._build_fftw_plan(gdata.shape, axis)
        # Execute FFTW plan
        plan.forward(gdata, temp)
        # Unpack from complex form and rescale
        self.unpack_rescale(temp, cdata, axis, rescale=1/self.N)

    def backward(self, cdata, gdata, axis):
        """Apply backward transform along specified axis."""
        plan, temp = self._build_fftw_plan(gdata.shape, axis)
        # Repack into complex form and rescale
        self.repack_rescale(cdata, temp, axis, rescale=None)
        # Execute FFTW plan
        plan.backward(temp, gdata)


@register_transform(basis.RealFourier, 'fftw_hc')
class FFTWHalfComplexFFT(FFTWBase, RealFourierTransform):
    """Real-to-real FFT using FFTW half-complex DFT."""

    @CachedMethod
    def _build_fftw_plan(self, dtype, gshape, axis):
        """Build FFTW plans and temporary arrays."""
        logger.debug("Building FFTW R2HC plan for (dtype, gshape, axis) = (%s, %s, %s)" %(dtype, gshape, axis))
        flags = ['FFTW_'+self.rigor.upper()]
        plan = fftw.R2HCTransform(dtype, gshape, axis, flags=flags)
        temp = fftw.create_array(gshape, dtype)
        return plan, temp

    def unpack_rescale(self, temp, cdata, axis, rescale):
        """Unpack halfcomplex coefficients and rescale for unit-amplitude normalization."""
        Kmax = self.Kmax
        # Scale k = 0 cos data
        meancos = axslice(axis, 0, 1)
        np.multiply(temp[meancos], rescale, cdata[meancos])
        # Zero k = 0 msin data
        cdata[axslice(axis, 1, 2)] = 0
        # Unpack and scale 1 < k <= Kmax data
        temp_posfreq_cos = temp[axslice(axis, 1, Kmax+1)]
        temp_posfreq_msin = temp[axslice(axis, -1, -(Kmax+1), -1)]
        cdata_posfreq_cos = cdata[axslice(axis, 2, 2*(Kmax+1), 2)]
        cdata_posfreq_msin = cdata[axslice(axis, 3, 2*(Kmax+1), 2)]
        np.multiply(temp_posfreq_cos, 2*rescale, cdata_posfreq_cos)
        np.multiply(temp_posfreq_msin, 2*rescale, cdata_posfreq_msin)
        # Zero k > Kmax data
        cdata[axslice(axis, 2*(Kmax+1), None)] = 0

    def repack(self, cdata, temp, axis):
        """Repack into complex coefficients and rescale for unit-amplitude normalization."""
        Kmax = self.Kmax
        # Copy k = 0 data
        meancos = axslice(axis, 0, 1)
        np.copyto(temp[meancos], cdata[meancos])
        # Repack 1 < k <= Kmax data
        temp_posfreq_cos = temp[axslice(axis, 1, Kmax+1)]
        temp_posfreq_msin = temp[axslice(axis, -1, -(Kmax+1), -1)]
        cdata_posfreq_cos = cdata[axslice(axis, 2, 2*(Kmax+1), 2)]
        cdata_posfreq_msin = cdata[axslice(axis, 3, 2*(Kmax+1), 2)]
        np.multiply(cdata_posfreq_cos, (1 / 2), temp_posfreq_cos)
        np.multiply(cdata_posfreq_msin, (1 / 2), temp_posfreq_msin)
        # Zero k > Kmax data
        temp[axslice(axis, Kmax+1, -Kmax)] = 0

    def forward(self, gdata, cdata, axis):
        """Apply forward transform along specified axis."""
        plan, temp = self._build_fftw_plan(gdata.dtype, gdata.shape, axis)
        # Execute FFTW plan
        plan.forward(gdata, temp)
        # Unpack from halfcomplex form and rescale
        self.unpack_rescale(temp, cdata, axis, rescale=1/self.N)

    def backward(self, cdata, gdata, axis):
        """Apply backward transform along specified axis."""
        plan, temp = self._build_fftw_plan(gdata.dtype, gdata.shape, axis)
        # Repack into halfcomplex form
        self.repack(cdata, temp, axis)
        # Execute FFTW plan
        plan.backward(temp, gdata)


class CosineTransform(SeparableTransform):
    """
    Abstract base class for cosine transforms.

    Parameters
    ----------
    grid_size : int
        Grid size (N) along transform dimension.
    coeff_size : int
        Coefficient size (M) along transform dimension.

    Notes
    -----
    Let KN = (N - 1) be the maximum (Nyquist) mode on the grid.
    Let KM = (M - 1) be the maximum retained mode in coeff space.
    Then K = min(KN, KM) is the maximum wavenumber used in the transforms.
    A unit-amplitude normalization is used.

    Forward transform:
        if k == 0:
            a(k) = (1/N) \sum_{x=0}^{N-1} f(x)
        elif k <= K:
            a(k) =  (2/N) \sum_{x=0}^{N-1} f(x) \cos(\pi k x / N)
        else:
            a(k) = 0

    Backward transform:
        f(x) = \sum_{k=0}^{K} a(k) \cos(\pi k x / N)

    Coefficient ordering:
        The cosine coefficients are ordered simply as
        [a(0), a(1), a(2), ..., a(KM)]
    """

    def __init__(self, grid_size, coeff_size):
        self.N = grid_size
        self.M = coeff_size
        self.KN = (self.N - 1)
        self.KM = (self.M - 1)
        self.Kmax = min(self.KN, self.KM)

    @property
    def wavenumbers(self):
        """One-dimensional global wavenumber array."""
        return np.arange(self.KM + 1)


#@register_transform(basis.Cosine, 'matrix')
class CosineMMT(CosineTransform, SeparableMatrixTransform):
    """Cosine MMT."""

    @CachedAttribute
    def forward_matrix(self):
        """Build forward transform matrix."""
        N = self.N
        M = self.M
        Kmax = self.Kmax
        K = self.wavenumbers[:, None]
        X = np.arange(N)[None, :]
        dX = N / np.pi
        quadrature = (2 / N) * np.cos(K*X/dX)
        quadrature[0] = 1 / N
        # Zero higher modes for transforms with grid_size < coeff_size
        quadrature *= (K <= self.Kmax)
        # Ensure C ordering for fast dot products
        return np.asarray(quadrature, order='C')

    @CachedAttribute
    def backward_matrix(self):
        """Build backward transform matrix."""
        N = self.N
        M = self.M
        Kmax = self.Kmax
        K = self.wavenumbers[None, :]
        X = np.arange(N)[:, None] + 1/2
        dX = N / np.pi
        functions = np.cos(K*X/dX)
        # Zero higher modes for transforms with grid_size < coeff_size
        functions *= (K <= self.Kmax)
        # Ensure C ordering for fast dot products
        return np.asarray(functions, order='C')


class FastCosineTransform(CosineTransform):
    """Abstract base class for fast cosine transforms."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        # Standard scaling factors for unit-amplitude normalization from DCT-II
        self.forward_rescale_zero = 1 / self.N / 2
        self.forward_rescale_pos = 1 / self.N
        self.backward_rescale_zero = 1
        self.backward_rescale_pos = 1 / 2

    def resize_rescale_forward(self, data_in, data_out, axis, Kmax):
        """Resize by padding/trunction and rescale to unit amplitude."""
        zerofreq = axslice(axis, 0, 1)
        np.multiply(data_in[zerofreq], self.forward_rescale_zero, data_out[zerofreq])
        if Kmax > 0:
            posfreq = axslice(axis, 1, Kmax+1)
            np.multiply(data_in[posfreq], self.forward_rescale_pos, data_out[posfreq])
            if self.KM > Kmax:
                badfreq = axslice(axis, Kmax+1, None)
                data_out[badfreq] = 0

    def resize_rescale_backward(self, data_in, data_out, axis, Kmax):
        """Resize by padding/trunction and rescale to unit amplitude."""
        zerofreq = axslice(axis, 0, 1)
        np.multiply(data_in[zerofreq], self.backward_rescale_zero, data_out[zerofreq])
        if Kmax > 0:
            posfreq = axslice(axis, 1, Kmax+1)
            np.multiply(data_in[posfreq], self.backward_rescale_pos, data_out[posfreq])
            if self.KN > Kmax:
                badfreq = axslice(axis, Kmax+1, None)
                data_out[badfreq] = 0


#@register_transform(basis.Cosine, 'scipy')
class ScipyDCT(FastCosineTransform):
    """Fast cosine transform using scipy.fft."""

    def forward(self, gdata, cdata, axis):
        """Apply forward transform along specified axis."""
        # Call DCT
        temp = scipy.fft.dct(gdata, type=2, axis=axis) # Creates temporary
        # Resize and rescale for unit-ampltidue normalization
        self.resize_rescale_forward(temp, cdata, axis, self.Kmax)

    def backward(self, cdata, gdata, axis):
        """Apply backward transform along specified axis."""
        # Resize and rescale for unit-amplitude normalization
        # Need temporary to avoid overwriting problems
        temp = np.empty_like(gdata) # Creates temporary
        self.resize_rescale_backward(cdata, temp, axis, self.Kmax)
        # Call IDCT
        temp = scipy.fft.dct(temp, type=3, axis=axis, overwrite_x=True) # Creates temporary
        np.copyto(gdata, temp)


#@register_transform(basis.Cosine, 'fftw')
class FFTWDCT(FFTWBase, FastCosineTransform):
    """Fast cosine transform using FFTW."""

    @CachedMethod
    def _build_fftw_plan(self, dtype, gshape, axis):
        """Build FFTW plans and temporary arrays."""
        logger.debug("Building FFTW DCT plan for (dtype, gshape, axis) = (%s, %s, %s)" %(dtype, gshape, axis))
        flags = ['FFTW_'+self.rigor.upper()]
        plan = fftw.DiscreteCosineTransform(dtype, gshape, axis, flags=flags)
        temp = fftw.create_array(gshape, dtype)
        return plan, temp

    def forward(self, gdata, cdata, axis):
        """Apply forward transform along specified axis."""
        plan, temp = self._build_fftw_plan(gdata.dtype, gdata.shape, axis)
        # Execute FFTW plan
        plan.forward(gdata, temp)
        # Resize and rescale for unit-ampltidue normalization
        self.resize_rescale_forward(temp, cdata, axis, self.Kmax)

    def backward(self, cdata, gdata, axis):
        """Apply backward transform along specified axis."""
        plan, temp = self._build_fftw_plan(gdata.dtype, gdata.shape, axis)
        # Resize and rescale for unit-amplitude normalization
        self.resize_rescale_backward(cdata, temp, axis, self.Kmax)
        # Execute FFTW plan
        plan.backward(temp, gdata)


class FastChebyshevTransform(JacobiTransform):
    """
    Abstract base class for fast Chebyshev transforms including ultraspherical conversion.
    Subclasses should inherit from this class, then a FastCosineTransform subclass.
    """

    def __init__(self, grid_size, coeff_size, a, b, a0, b0, **kw):
        if not a0 == b0 == -1/2:
            raise ValueError("Fast Chebshev transform requires a0 == b0 == -1/2.")
        # Jacobi initialization
        super().__init__(grid_size, coeff_size, a, b, a0, b0, **kw)
        # DCT initialization to set scaling factors
        if a != a0 or b != b0:
            # Modify coeff_size to avoid truncation before conversion
            super(JacobiTransform, self).__init__(grid_size, grid_size)
        else:
            super(JacobiTransform, self).__init__(grid_size, coeff_size)
        # Make other attributes for M since they're overwritten by DCT initialization
        self.M_orig = coeff_size
        self.KM_orig = (self.M_orig - 1)
        self.Kmax_orig = min(self.KN, self.KM_orig)
        # Modify scaling factors to match Jacobi normalizations
        self.forward_rescale_zero *= np.sqrt(np.pi)
        self.forward_rescale_pos *= np.sqrt(np.pi / 2)
        self.backward_rescale_zero /= np.sqrt(np.pi)
        self.backward_rescale_pos /= np.sqrt(np.pi / 2)
        # Dispatch resize/rescale based on conversion
        if a == a0 and b == b0:
            self.resize_rescale_forward = self._resize_rescale_forward
            self.resize_rescale_backward = self._resize_rescale_backward
        else:
            # Conversion matrices
            if self.dealias_before_converting and (self.M_orig < self.N): # truncate prior to conversion matrix
                self.forward_conversion = jacobi.conversion_matrix(self.M_orig, a0, b0, a, b).tocsr()
            else: # input to conversion matrix not truncated
                self.forward_conversion = jacobi.conversion_matrix(self.N, a0, b0, a, b)
                self.forward_conversion.resize(self.M_orig, self.N)
                self.forward_conversion = self.forward_conversion.tocsr()
            self.backward_conversion = jacobi.conversion_matrix(self.M_orig, a0, b0, a, b).tocsr()
            self.backward_conversion.sum_duplicates() # for faster solve_upper
            self.resize_rescale_forward = self._resize_rescale_forward_convert
            self.resize_rescale_backward = self._resize_rescale_backward_convert

    def _resize_rescale_forward(self, data_in, data_out, axis, Kmax):
        """Resize by padding/trunction and rescale to unit amplitude."""
        # DCT resize/rescale
        super().resize_rescale_forward(data_in, data_out, axis, Kmax)
        # Change sign of odd modes
        if Kmax > 0:
            posfreq_odd = axslice(axis, 1, Kmax+1, 2)
            data_out[posfreq_odd] *= -1

    def _resize_rescale_backward(self, data_in, data_out, axis, Kmax):
        """Resize by padding/trunction and rescale to unit amplitude."""
        # Change sign of odd modes
        if Kmax > 0:
            posfreq_odd = axslice(axis, 1, Kmax+1, 2)
            data_in[posfreq_odd] *= -1
        # DCT resize/rescale
        super().resize_rescale_backward(data_in, data_out, axis, Kmax)

    def _resize_rescale_forward_convert(self, data_in, data_out, axis, Kmax_DCT):
        """Resize by padding/trunction and rescale to unit amplitude."""
        # DCT rescale in place
        super().resize_rescale_forward(data_in, data_in, axis, Kmax_DCT)
        # Change sign of odd modes
        if Kmax_DCT > 0:
            posfreq_odd = axslice(axis, 1, Kmax_DCT+1, 2)
            data_in[posfreq_odd] *= -1
        # Ultraspherical conversion
        if self.dealias_before_converting and self.M_orig < self.N: # truncate data
            goodfreq = axslice(axis, 0, self.M_orig)
            data_in = data_in[goodfreq]
        apply_sparse(self.forward_conversion, data_in, axis, out=data_out)

    def _resize_rescale_backward_convert(self, data_in, data_out, axis, Kmax_DCT):
        """Resize by padding/trunction and rescale to unit amplitude."""
        Kmax_orig = self.Kmax_orig
        badfreq = axslice(axis, Kmax_orig+1, None)
        if self.M_orig > self.N:
            # Truncate input before conversion
            data_in[badfreq] = 0
        # Ultraspherical conversion
        solve_upper_sparse(self.backward_conversion, data_in, axis, out=data_in)
        # Change sign of odd modes
        if Kmax_orig > 0:
            posfreq_odd = axslice(axis, 1, Kmax_orig+1, 2)
            data_in[posfreq_odd] *= -1
        # DCT resize/rescale
        super().resize_rescale_backward(data_in, data_out, axis, Kmax_orig)


@register_transform(basis.Jacobi, 'scipy_dct')
class ScipyFastChebyshevTransform(FastChebyshevTransform, ScipyDCT):
    """Fast ultraspherical transform using scipy.fft and spectral conversion."""
    pass  # Implementation is complete via inheritance


@register_transform(basis.Jacobi, 'fftw_dct')
class FFTWFastChebyshevTransform(FastChebyshevTransform, FFTWDCT):
    """Fast ultraspherical transform using scipy.fft and spectral conversion."""
    pass  # Implementation is complete via inheritance


# class ScipyDST(PolynomialTransform):

#     def forward_reduced(self):
#         # DST-II transform from interior points
#         temp = fftpack.dst(self.gdata_reduced, type=2, axis=1)
#         # Rescale as sinusoid amplitudes
#         temp[:, -1, :] *= 0.5
#         temp *= (1 / self.N1G)
#         # Resize
#         self.resize_reduced(temp, self.cdata_reduced)

#     def backward_reduced(self):
#         # Resize into gdata for memory efficiency
#         self.resize_reduced(self.cdata_reduced, self.gdata_reduced)
#         # Rescale from sinusoid amplitudes
#         self.gdata_reduced[:, :-1, :] *= 0.5
#         # DST-III transform to interior points
#         temp = fftpack.dst(self.gdata_reduced, type=3, axis=1)
#         np.copyto(self.gdata_reduced, temp)


# #@register_transform(basis.Sine, 'scipy')
# class ScipySineTransform(ScipyDST):

#     def forward_reduced(self):
#         super().forward_reduced()
#         # Shift data, adding zero mode and dropping Nyquist
#         start = self.cdata_reduced[:, :-1, :]
#         shift = self.cdata_reduced[:, 1:, :]
#         np.copyto(shift, start)
#         self.cdata_reduced[:, 0, :] = 0

#     def backward_reduced(self):
#         # Unshift data, adding Nyquist mode and dropping zero
#         start = self.cdata_reduced[:, :-1, :]
#         shift = self.cdata_reduced[:, 1:, :]
#         np.copyto(start, shift)
#         self.cdata_reduced[:, -1, :] = 0
#         super().backward_reduced()


# @register_transform(basis.ChebyshevU, 'scipy')
# class ScipyChebyshevUTransform(ScipyDST):
#     """
#     f(x) = c_n U_n(x)
#     f(x) sin(θ) = c_n sin(θ) U_n(x)
#                 = c_n sin((n+1)θ)

#     c_n are the U_n coefficients of f(x)
#                 sine coefficients of f(x) sin(θ)
#     """

#     def forward_reduced(self):
#         self.gdata_reduced *= sin_θ
#         super().forward_reduced()

#     def backward_reduce(self):
#         super().backward_reduced()
#         self.gdata_reduced *= (1 / sin_θ)


# cdef class FFTWSineTransform:

#     @CachedMethod
#     def _fftw_dst_setup(self, dtype, gshape, axis):
#         """Build FFTW DST plan and temporary array."""
#         flags = ['FFTW_'+FFTW_RIGOR.upper()]
#         logger.debug("Building FFTW DST plan for (dtype, gshape, axis) = (%s, %s, %s)" %(dtype, gshape, axis))
#         plan = fftw.DiscreteSineTransform(dtype, gshape, axis, flags=flags)
#         temp = fftw.create_array(gshape, dtype)
#         return plan, temp

#     def _forward_fftw(self, gdata, cdata, axis, scale):
#         """Forward transform using FFTW DCT."""
#         cdata, gdata = self.check_arrays(cdata, gdata, axis)
#         plan, temp = self._fftw_dst_setup(gdata.dtype, gdata.shape, axis)
#         plan.forward(gdata, temp)
#         self._forward_dst_scaling(temp, axis)
#         self._resize_coeffs(temp, cdata, axis)
#         return cdata

#     def _backward_fftw(self, cdata, gdata, axis, scale):
#         """Backward transform using FFTW IDCT."""
#         cdata, gdata = self.check_arrays(cdata, gdata, axis, scale)
#         plan, temp = self._fftw_dst_setup(gdata.dtype, gdata.shape, axis)
#         self._resize_coeffs(cdata, temp, axis)
#         self._backward_dst_scaling(temp, axis)
#         plan.backward(temp, gdata)
#         return gdata


#@register_transform(basis.Cosine, 'scipy')
# class ScipyDCT(PolynomialTransform):

#     def forward_reduced(self):
#         # DCT-II transform from interior points
#         temp = fftpack.dct(self.gdata_reduced, type=2, axis=1)
#         # Rescale as sinusoid amplitudes
#         temp[:, 0, :] *= 0.5
#         temp *= (1 / self.N1G)
#         # Resize
#         self.resize_reduced(temp, self.cdata_reduced)

#     def backward_reduced(self):
#         # Resize into gdata for memory efficiency
#         self.resize_reduced(self.cdata_reduced, self.gdata_reduced)
#         # Rescale from sinusoid amplitudes
#         self.gdata_reduced[:, 1:, :] *= 0.5
#         # DCT-III transform to interior points
#         temp = fftpack.dct(self.gdata_reduced, type=3, axis=1)
#         np.copyto(self.gdata_reduced, temp)


# @register_transform(basis.ChebyshevT, 'scipy')
# class ScipyChebyshevTTransform(ScipyDCT):

#     def forward_reduced(self):
#         super().forward_reduced()
#         # Negate odd modes for natural grid ordering
#         self.cdata_reduced[:, 1::2, :] *= -1

#     def backward_reduced(self):
#         # Negate odd modes for natural grid ordering
#         self.cdata_reduced[:, 1::2, :] *= -1
#         super().backward_reduced()


# class FFTWCosine:

#     @CachedMethod
#     def _fftw_dct_setup(self, dtype, gshape, axis):
#         """Build FFTW DCT plan and temporary array."""
#         flags = ['FFTW_'+FFTW_RIGOR.upper()]
#         logger.debug("Building FFTW DCT plan for (dtype, gshape, axis) = (%s, %s, %s)" %(dtype, gshape, axis))
#         plan = fftw.DiscreteCosineTransform(dtype, gshape, axis, flags=flags)
#         temp = fftw.create_array(gshape, dtype)
#         return plan, temp

#     def _forward_fftw(self, gdata, cdata, axis, scale):
#         """Forward transform using FFTW DCT."""
#         cdata, gdata = self.check_arrays(cdata, gdata, axis)
#         plan, temp = self._fftw_dct_setup(gdata.dtype, gdata.shape, axis)
#         plan.forward(gdata, temp)
#         self._forward_dct_scaling(temp, axis)
#         self._resize_coeffs(temp, cdata, axis)
#         return cdata

#     def _backward_fftw(self, cdata, gdata, axis, scale):
#         """Backward transform using FFTW IDCT."""
#         cdata, gdata = self.check_arrays(cdata, gdata, axis, scale)
#         plan, temp = self._fftw_dct_setup(gdata.dtype, gdata.shape, axis)
#         self._resize_coeffs(cdata, temp, axis)
#         self._backward_dct_scaling(temp, axis)
#         plan.backward(temp, gdata)
#         return gdata





def reduced_view_3(data, axis):
    shape = data.shape
    N0 = prod(shape[:axis])
    N1 = shape[axis]
    N2 = prod(shape[axis+1:])
    return data.reshape((N0, N1, N2))


def reduced_view_4(data, axis):
    shape = data.shape
    N0 = prod(shape[:axis])
    N1 = shape[axis]
    N2 = shape[axis+1]
    N3 = prod(shape[axis+2:])
    return data.reshape((N0, N1, N2, N3))

def reduced_view_5(data, axis):
    shape = data.shape
    N0 = prod(shape[:axis])
    N1 = shape[axis]
    N2 = shape[axis+1]
    N3 = shape[axis+2]
    N4 = prod(shape[axis+3:])
    return data.reshape((N0, N1, N2, N3, N4))

class PolynomialTransform(Transform):

    def __init__(self, grid_size, coeff_size):
        self.grid_size = self.N1G = grid_size
        self.coeff_size = self.N1C = coeff_size

    # def __init__(self, basis, coeff_shape, dtype, axis, scale):
    #     self.basis = basis
    #     self.dtype = dtype
    #     self.coeff_shape = coeff_shape
    #     self.axis = axis
    #     self.scale = scale

    #     # Treat complex arrays as higher dimensional real arrays
    #     if self.dtype == np.complex128:
    #         coeff_shape = list(coeff_shape) + [2]

    #     self.N0 = N0 = prod(coeff_shape[:axis])
    #     self.N1C = N1C = coeff_shape[axis]
    #     self.N1G = N1G = int(self.N1C * scale)
    #     self.N2 = N2 = prod(coeff_shape[axis+1:])

    #     self.gdata_reduced = np.zeros(shape=[N0, N1G, N2], dtype=np.float64)
    #     self.cdata_reduced = np.zeros(shape=[N0, N1C, N2], dtype=np.float64)


    # def check_arrays(self, cdata, gdata, axis, scale=None):
    #     """
    #     Verify provided arrays sizes and dtypes are correct.
    #     Build compliant arrays if not provided.

    #     """

    #     if cdata is None:
    #         # Build cdata
    #         cshape = list(gdata.shape)
    #         cshape[axis] = self.coeff_size
    #         cdata = fftw.create_array(cshape, self.coeff_dtype)
    #     else:
    #         # Check cdata
    #         if cdata.shape[axis] != self.space.coeff_size:
    #             raise ValueError("cdata does not match coeff_size")
    #         if cdata.dtype != self.domain.dtype:
    #             raise ValueError("cdata does not match coeff_dtype")

    #     if scale:
    #         grid_size = self.space.grid_size(scale)

    #     if gdata is None:
    #         # Build gdata
    #         gshape = list(cdata.shape)
    #         gshape[axis] = grid_size
    #         gdata = fftw.create_array(gshape, self.grid_dtype)
    #     else:
    #         # Check gdata
    #         if scale and (gdata.shape[axis] != grid_size):
    #             raise ValueError("gdata does not match scaled grid_size")
    #         if gdata.dtype != self.domain.dtype:
    #             raise ValueError("gdata does not match grid_dtype")

    #     return cdata, gdata

    @staticmethod
    def resize_reduced(data_in, data_out):
        """Resize data by padding/truncation."""
        size_in = data_in.shape[1]
        size_out = data_out.shape[1]
        if size_in < size_out:
            # Pad with zeros at end of data
            np.copyto(data_out[:, :size_in, :], data_in)
            np.copyto(data_out[:, size_in:, :], 0)
        elif size_in > size_out:
            # Truncate higher order modes at end of data
            np.copyto(data_out, data_in[:, :size_out, :])
        else:
            np.copyto(data_out, data_in)

    def forward(self, gdata, cdata, axis):
        # Make reduced view into input arrays
        self.gdata_reduced = reduced_view_3(gdata, axis)
        self.cdata_reduced = reduced_view_3(cdata, axis)
        #self.gdata_reduced.data = gdata
        #self.cdata_reduced.data = cdata
        # Transform reduced arrays
        self.forward_reduced()

    def backward(self, cdata, gdata, axis):
        # Make reduced view into input arrays
        self.gdata_reduced = reduced_view_3(gdata, axis)
        self.cdata_reduced = reduced_view_3(cdata, axis)
        # self.cdata_reduced.data = cdata
        # self.gdata_reduced.data = gdata
        # Transform reduced arrays
        self.backward_reduced()



def reduce_array(data, axis):
    """Return reduced 3D view of array collapsed above and below specified axis."""
    N0 = prod(data.shape[:axis])
    N1 = data.shape[axis]
    N2 = prod(data.shape[axis+1:])
    return data.reshape((N0, N1, N2))

def forward_DFT(gdata, cdata, axis):
    gdata_reduced = reduce_array(gdata, axis)
    cdata_reduced = reduce_array(cdata, axis)
    # Raw transform
    temp = np.fft.fft(gdata_reduced, axis=1)
    PolynomialTransform.resize_reduced(temp, cdata_reduced)
    # Rescale to sinusoid amplitudes
    cdata_reduced /= gdata_reduced.shape[1]

def backward_DFT(cdata, gdata, axis):
    gdata_reduced = reduce_array(gdata, axis)
    cdata_reduced = reduce_array(cdata, axis)
    # Rescale from sinusoid amplitudes
    cdata_reduced *= gdata_reduced.shape[1]
    # Raw transform
    PolynomialTransform.resize_reduced(cdata_reduced, gdata_reduced)
    temp = np.fft.ifft(gdata_reduced, axis=1)
    np.copyto(gdata_reduced, temp)


class NonSeparableTransform(Transform):

    def __init__(self, grid_shape, coeff_size, axis, dtype):

        self.N2g = grid_shape[axis]
        self.N2c = coeff_size

#    @staticmethod
#    def resize_reduced(data_in, data_out):
#        """Resize data by padding/truncation."""
#        size_in = data_in.shape[2]
#        size_out = data_out.shape[2]
#        if size_in < size_out:
#            # Pad with zeros at end of data
#            np.copyto(data_out[:, :, :size_in, :], data_in)
#            np.copyto(data_out[:, :, size_in:, :], 0)
#        elif size_in > size_out:
#            # Truncate higher order modes at end of data
#            np.copyto(data_out, data_in[:, :, :size_out, :])
#        else:
#            np.copyto(data_out, data_in)

    def forward(self, gdata, cdata, axis):
        # Make reduced view into input arrays
        gdata = reduced_view_4(gdata, axis-1)
        cdata = reduced_view_4(cdata, axis-1)
        # Transform reduced arrays
        self.forward_reduced(gdata, cdata)

    def backward(self, cdata, gdata, axis):
        # Make reduced view into input arrays
        cdata = reduced_view_4(cdata, axis-1)
        gdata = reduced_view_4(gdata, axis-1)
        # Transform reduced arrays
        self.backward_reduced(cdata, gdata)


@register_transform(basis.SphereBasis, 'matrix')
class SWSHColatitudeTransform(NonSeparableTransform):

    def __init__(self, Ntheta, Lmax, m_maps, s):
        self.Ntheta = Ntheta
        self.Lmax = Lmax
        self.m_maps = m_maps
        self.s = s

    def forward_reduced(self, gdata, cdata):
        # local_m = self.local_m
        # if gdata.shape[1] != len(local_m): # do we want to do this check???
        #     raise ValueError("gdata.shape[1]: %i, len(local_m): %i" %(gdata.shape[1], len(local_m)))
        m_matrices = self._forward_SWSH_matrices
        Lmax = self.Lmax
        for m, mg_slice, mc_slice, ell_slice in self.m_maps:
            # Skip transforms when |m| > Lmax
            if abs(m) <= Lmax:
                # Use rectangular transform matrix, padded with zeros when Lmin > abs(m)
                grm = gdata[:, mg_slice, :, :]
                crm = cdata[:, mc_slice, ell_slice, :]
                apply_matrix(m_matrices[m], grm, axis=2, out=crm)

    def backward_reduced(self, cdata, gdata):
        # local_m = self.local_m
        # if gdata.shape[1] != len(local_m): # do we want to do this check???
        #     raise ValueError("gdata.shape[1]: %i, len(local_m): %i" %(gdata.shape[1], len(local_m)))
        m_matrices = self._backward_SWSH_matrices
        Lmax = self.Lmax
        for m, mg_slice, mc_slice, ell_slice in self.m_maps:
            if abs(m) > Lmax:
                # Write zeros because they'll be used by the inverse azimuthal transform
                gdata[:, mg_slice, :, :] = 0
            else:
                # Use rectangular transform matrix, padded with zeros when Lmin > abs(m)
                grm = gdata[:, mg_slice, :, :]
                crm = cdata[:, mc_slice, ell_slice, :]
                apply_matrix(m_matrices[m], crm, axis=2, out=grm)

    @CachedAttribute
    def _quadrature(self):
        return dedalus_sphere.sphere.quadrature(self.Ntheta-1)

    @CachedAttribute
    def _forward_SWSH_matrices(self):
        """Build transform matrix for single m and s."""
        # Get functions from sphere library
        cos_grid, weights = self._quadrature
        Lmax = self.Lmax
        m_matrices = {}
        for m, _, _, _ in self.m_maps:
            if m in m_matrices:
                continue
            if m > Lmax:
                # Don't make matrices for m's that will be dropped after transform
                m_matrices[m] = None
            else:
                Y = dedalus_sphere.sphere.harmonics(Lmax, m, self.s, cos_grid)  # shape (Nc-Lmin, Ng)
                # Pad to shape (Nc-|m|, Ng) so transforms don't depend on Lmin
                Lmin = max(abs(m), abs(self.s))
                Yfull = np.zeros((Lmax+1-abs(m), self.Ntheta))
                Yfull[Lmin-abs(m):, :] = (Y*weights).astype(np.float64)
                # Zero higher coefficients than can be correctly computed with base Gauss quadrature
                Yfull[self.Ntheta-abs(m):, :] = 0
                m_matrices[m] = np.asarray(Yfull, order='C')
        return m_matrices

    @CachedAttribute
    def _backward_SWSH_matrices(self):
        """Build transform matrix for single m and s."""
        # Get functions from sphere library
        cos_grid, weights = self._quadrature
        Lmax = self.Lmax
        m_matrices = {}
        for m, _, _, _ in self.m_maps:
            if m in m_matrices:
                continue
            if m > Lmax:
                # Don't make matrices for m's that will be dropped after transform
                m_matrices[m] = None
            else:
                Y = dedalus_sphere.sphere.harmonics(Lmax, m, self.s, cos_grid) # shape (Nc-Lmin, Ng)
                # Pad to shape (Nc-|m|, Ng) so transforms don't depend on Lmin
                Lmin = max(abs(m), abs(self.s))
                Yfull = np.zeros((self.Ntheta, Lmax+1-abs(m)))
                Yfull[:, Lmin-abs(m):] = Y.T.astype(np.float64)
                # Zero higher coefficients than can be correctly computed with base Gauss quadrature
                Yfull[:, self.Ntheta-abs(m):] = 0
                m_matrices[m] = np.asarray(Yfull, order='C')
        return m_matrices

@register_transform(basis.DiskBasis, 'matrix')
class DiskRadialTransform(NonSeparableTransform):
    """
    TODO:
        - Remove dependence on grid_shape?
    """

    def __init__(self, grid_shape, basis_shape, axis, m_maps, s, k, alpha, dtype=np.complex128, dealias_before_converting=None):
        self.Nphi = basis_shape[0]
        self.Nmax = basis_shape[1] - 1
        super().__init__(grid_shape, self.Nmax+1, axis, dtype)
        self.m_maps = m_maps
        self.s = s
        self.k = k
        self.alpha = alpha
        if dealias_before_converting is None:
            dealias_before_converting = GET_DEALIAS_BEFORE_CONVERTING()
        self.dealias_before_converting = dealias_before_converting

    def forward_reduced(self, gdata, cdata):
        # local_m = self.local_m
        # if gdata.shape[1] != len(local_m): # do we want to do this check???
        #     raise ValueError("gdata.shape[1]: %i, len(local_m): %i" %(gdata.shape[1], len(local_m)))
        m_matrices = self._forward_matrices
        Nphi = self.Nphi
        Nmax = self.Nmax
        for m, mg_slice, mc_slice, n_slice in self.m_maps:
            # Skip transforms when |m| > 2*Nmax
            if abs(m) <= 2*Nmax:
                # Use rectangular transform matrix, padded with zeros when Nmin > abs(m)//2
                grm = gdata[:, mg_slice, :, :]
                crm = cdata[:, mc_slice, n_slice, :]
                apply_matrix(m_matrices[m], grm, axis=2, out=crm)

    def backward_reduced(self, cdata, gdata):
        # local_m = self.local_m
        # if gdata.shape[1] != len(local_m): # do we want to do this check???
        #     raise ValueError("gdata.shape[1]: %i, len(local_m): %i" %(gdata.shape[1], len(local_m)))
        m_matrices = self._backward_matrices
        Nphi = self.Nphi
        Nmax = self.Nmax
        for m, mg_slice, mc_slice, n_slice in self.m_maps:
            if abs(m) > 2*Nmax:
                # Write zeros because they'll be used by the inverse azimuthal transform
                gdata[:, mg_slice, :, :] = 0
            else:
                # Use rectangular transform matrix, padded with zeros when Nmin > abs(m)//2
                grm = gdata[:, mg_slice, :, :]
                crm = cdata[:, mc_slice, n_slice, :]
                apply_matrix(m_matrices[m], crm, axis=2, out=grm)

    @CachedAttribute
    def _quadrature(self):
        # get grid and weights from sphere library
        return dedalus_sphere.zernike.quadrature(2, self.N2g, k=self.alpha)

    @CachedAttribute
    def _forward_matrices(self):
        """Build transform matrix for single l and r."""
        # Get functions from sphere library
        z_grid, weights = self._quadrature
        m_list = tuple(map[0] for map in self.m_maps)
        m_matrices = {}
        for m in m_list:
            if m not in m_matrices:
                # Gauss quadrature with base (k=0) polynomials
                Nmin = dedalus_sphere.zernike.min_degree(abs(m))
                Nc = max(max(self.N2g, self.N2c) - Nmin, 0)
                W = dedalus_sphere.zernike.polynomials(2, Nc, self.alpha, abs(m + self.s), z_grid) # shape (N2c-Nmin, Ng)
                W = W * weights
                # Zero higher coefficients than can be correctly computed with base Gauss quadrature
                dN = abs(m + self.s) // 2
                W[max(self.N2g-dN,0):] = 0
                if self.dealias_before_converting:
                    # Truncate to specified coeff_size
                    W = W[:max(self.N2c-Nmin,0)]
                # Spectral conversion
                if self.k > 0:
                    conversion = dedalus_sphere.zernike.operator(2, 'E')(+1)**self.k
                    W = conversion(W.shape[0], self.alpha, abs(m + self.s)) @ W
                if not self.dealias_before_converting:
                    # Truncate to specified coeff_size
                    W = W[:max(self.N2c-Nmin,0)]
                m_matrices[m] = np.asarray(W.astype(np.float64), order='C')
        return m_matrices

    @CachedAttribute
    def _backward_matrices(self):
        """Build transform matrix for single l and r."""
        # Get functions from sphere library
        z_grid, weights = self._quadrature
        m_list = tuple(map[0] for map in self.m_maps)
        m_matrices = {}
        for m in m_list:
            if m not in m_matrices:
                # Construct polynomials on the base grid
                Nmin = dedalus_sphere.zernike.min_degree(abs(m))
                Nc = max(self.N2c - Nmin, 0)
                W = dedalus_sphere.zernike.polynomials(2, Nc, self.k + self.alpha, abs(m + self.s), z_grid)
                # Zero higher coefficients than can be correctly computed with base Gauss quadrature
                dN = abs(m + self.s) // 2
                W[max(self.N2g-dN,0):] = 0
                # Transpose and ensure C ordering for fast dot products
                m_matrices[m] = np.asarray(W.T.astype(np.float64), order='C')
        return m_matrices


@register_transform(basis.BallRadialBasis, 'matrix')
@register_transform(basis.BallBasis, 'matrix')
class BallRadialTransform(Transform):

    def __init__(self, grid_shape, coeff_size, axis, ell_maps, regindex, regtotal, k, alpha, dtype=np.complex128, dealias_before_converting=None):
        self.N3g = grid_shape[axis]
        self.N3c = coeff_size
        self.ell_maps = ell_maps
        self.intertwiner = lambda l: dedalus_sphere.spin_operators.Intertwiner(l, indexing=(-1,+1,0))
        self.regindex = regindex
        self.regtotal = regtotal
        self.k = k
        self.alpha = alpha
        if dealias_before_converting is None:
            dealias_before_converting = GET_DEALIAS_BEFORE_CONVERTING()
        self.dealias_before_converting = dealias_before_converting

    def forward(self, gdata, cdata, axis):
        # Make reduced view into input arrays
        gdata = reduced_view_5(gdata, axis-2)
        cdata = reduced_view_5(cdata, axis-2)
        # Transform reduced arrays
        self.forward_reduced(gdata, cdata)

    def backward(self, cdata, gdata, axis):
        # Make reduced view into input arrays
        cdata = reduced_view_5(cdata, axis-2)
        gdata = reduced_view_5(gdata, axis-2)
        # Transform reduced arrays
        self.backward_reduced(cdata, gdata)

    def forward_reduced(self, gdata, cdata):
        #if gdata.shape[1] != len(local_l): # do we want to do this check???
        #    raise ValueError("Local l must match size of %i axis." %(self.axis-1) )
        # Apply transform for each l
        ell_matrices = self._forward_GSZP_matrix
        for ell, m_ind, ell_ind in self.ell_maps:
            Nmin = dedalus_sphere.zernike.min_degree(ell)
            grl = gdata[:, m_ind, ell_ind, :, :]
            crl = cdata[:, m_ind, ell_ind, Nmin:, :]
            apply_matrix(ell_matrices[ell], grl, axis=3, out=crl)

    def backward_reduced(self, cdata, gdata):
        #if gdata.shape[1] != len(local_l): # do we want to do this check???
        #    raise ValueError("Local l must match size of %i axis." %(self.axis-1) )
        # Apply transform for each l
        ell_matrices = self._backward_GSZP_matrix
        for ell, m_ind, ell_ind in self.ell_maps:
            Nmin = dedalus_sphere.zernike.min_degree(ell)
            grl = gdata[:, m_ind, ell_ind, :, :]
            crl = cdata[:, m_ind, ell_ind, Nmin:, :]
            apply_matrix(ell_matrices[ell], crl, axis=3, out=grl)

    @CachedAttribute
    def _quadrature(self):
        # get grid and weights from sphere library
        return dedalus_sphere.zernike.quadrature(3, self.N3g, k=self.alpha)

    @CachedAttribute
    def _forward_GSZP_matrix(self):
        """Build transform matrix for single l and r."""
        z_grid, weights = self._quadrature
        ell_list = tuple(map[0] for map in self.ell_maps)
        ell_matrices = {}
        Rb = np.array([-1, 1, 0], dtype=int)
        for ell in ell_list:
            if ell not in ell_matrices:
                Nmin = dedalus_sphere.zernike.min_degree(ell)
                if self.regindex != () and self.intertwiner(ell).forbidden_regularity(Rb[np.array(self.regindex)]):
                    ell_matrices[ell] = np.zeros((self.N3c-Nmin, self.N3g))
                else:
                    # Gauss quadrature with base (k=0) polynomials

                    Nc = max(max(self.N3g, self.N3c) - Nmin, 0)
                    W = dedalus_sphere.zernike.polynomials(3, Nc, self.alpha, ell+self.regtotal, z_grid) # shape (Nc-Nmin, Ng)
                    W = W * weights
                    # Zero higher coefficients than can be correctly computed with base Gauss quadrature
                    dN = (ell + self.regtotal) // 2
                    W[max(self.N3g-dN,0):] = 0
                    if self.dealias_before_converting:
                        # Truncate to specified coeff_size
                        W = W[:max(self.N3c-Nmin,0)]
                    # Spectral conversion
                    if self.k > 0:
                        conversion = dedalus_sphere.zernike.operator(3, 'E')(+1)**self.k
                        W = conversion(W.shape[0], self.alpha, ell+self.regtotal) @ W
                    if not self.dealias_before_converting:
                        # Truncate to specified coeff_size
                        W = W[:max(self.N3c-Nmin,0)]
                    # Ensure C ordering for fast dot products
                    ell_matrices[ell] = np.asarray(W.astype(np.float64), order='C')
        return ell_matrices

    @CachedAttribute
    def _backward_GSZP_matrix(self):
        """Build transform matrix for single l and r."""
        z_grid, weights = self._quadrature
        ell_list = tuple(map[0] for map in self.ell_maps)
        ell_matrices = {}
        Rb = np.array([-1, 1, 0], dtype=int)
        for ell in ell_list:
            if ell not in ell_matrices:
                Nmin = dedalus_sphere.zernike.min_degree(ell)
                if self.regindex != () and self.intertwiner(ell).forbidden_regularity(Rb[np.array(self.regindex)]):
                    ell_matrices[ell] = np.zeros((self.N3g, self.N3c-Nmin))
                else:
                    # Construct polynomials on the base grid
                    Nc = max(self.N3c - Nmin, 0)
                    W = dedalus_sphere.zernike.polynomials(3, Nc, self.alpha+self.k, ell+self.regtotal, z_grid)
                    # Zero higher coefficients than can be correctly computed with base Gauss quadrature
                    dN = (ell + self.regtotal) // 2
                    W[max(self.N3g-dN,0):] = 0
                    # Transpose and ensure C ordering for fast dot products
                    ell_matrices[ell] = np.asarray(W.T.astype(np.float64), order='C')
        return ell_matrices


## Disk transforms

def forward_disk(gdata, cdata, axis, k0, k, s, local_m):
    """Apply forward radial transform to data with fixed s and varying m."""
    # Build reduced views
    gdata_reduced = reduced_view_4(gdata, axis-1)
    cdata_reduced = reduced_view_4(cdata, axis-1)
    if gdata_reduced.shape[1] != len(local_m):
        raise ValueError("Local m must match axis-1 size.")
    # Apply transform for each m
    Ng = gdata.shape[axis]
    Nc = cdata.shape[axis]
    for dm, m in enumerate(local_m):
        m_matrix = _forward_disk_matrix(Ng, Nc, k0, k, m+s)
        grm = gdata_reduced[:, dm, :, :]
        crm = cdata_reduced[:, dm, :, :]
        apply_matrix(m_matrix, grm, axis=1, out=crm)

def _forward_disk_matrix(Ng, Nc, k0, k, m):
    """Build forward transform matrix for Q[k,m,n](r[k0])."""
    # Get base grid and weights
    z_grid, weights = dedalus_sphere.disk128.quadrature(Ng-1, k=k0, niter=3)
    # Get functions
    Nc_max = Nc
    logger.warning("No truncation")
    Q = dedalus_sphere.disk128.polynomials(Nc_max-1, k=k, m=m, z=z_grid)
    # Pad to square transform
    Qfull = np.zeros((Nc, Ng))
    Qfull[:Nc, :] = Q.astype(np.float64)
    return Qfull

def backward_disk(cdata, gdata, axis, k, s, local_m):
    """Apply bakward radial transform to data with fixed s and varying m."""
    # Build reduced views
    gdata_reduced = reduced_view_4(gdata, axis-1)
    cdata_reduced = reduced_view_4(cdata, axis-1)
    if gdata_reduced.shape[1] != len(local_m):
        raise ValueError("Local m must match axis-1 size.")
    # Apply transform for each m
    for dm, m in enumerate(local_m):
        m_matrix = _backward_SWSH_matrix(N2c, N2g, k, m+s)
        grm = gdata_reduced[:, dm, :, :]
        crm = cdata_reduced[:, dm, :, :]
        apply_matrix(m_matrix, crm, axis=1, out=grm)

def _backward_disk_matrix(Nc, Ng, k0, k, m):
    """Build backward transform matrix for Q[k,m,n](r[k0])."""
    # Get base grid and weights
    z_grid, weights = dedalus_sphere.disk128.quadrature(Ng-1, k=k0, niter=3)
    # Get functions
    Nc_max = Nc
    logger.warning("No truncation")
    Q = dedalus_sphere.disk128.polynomials(Nc_max-1, k=k, m=m, z=z_grid)
    # Pad to square transform
    Qfull = np.zeros((Nc, Ng))
    Qfull[:Nc, :] = Q.astype(np.float64)
    return Qfull

