"""
Abstract and built-in classes for spectral bases.

"""

import math
import numpy as np
from numpy import pi
from scipy import sparse
from scipy import fftpack


from ..tools.config import config
from ..tools.cache import CachedAttribute
from ..tools.cache import CachedMethod
from ..tools.array import interleaved_view
from ..tools.array import reshape_vector
from ..tools.array import axslice
try:
    from ..libraries.fftw import fftw_wrappers as fftw
    fftw.fftw_mpi_init()
except ImportError:
    logger.error("Don't forget to buid using 'python3 setup.py build_ext --inplace'")
    raise

DEFAULT_LIBRARY = config['transforms'].get('DEFAULT_LIBRARY')
FFTW_RIGOR = config['transforms'].get('FFTW_RIGOR')


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

    def set_dtype(self, grid_dtype):
        """Set transforms based on grid data type."""

        raise NotImplementedError()

    def forward(self, gdata, cdata, axis):
        """Grid-to-coefficient transform."""

        raise NotImplementedError()

    def backward(self, cdata, gdata, axis):
        """Coefficient-to-grid transform."""

        raise NotImplementedError()

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
        """Interpolate in interval using coefficients."""

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
    def Mult(self, p, subindex):
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
    """Chebyshev polynomial basis on the roots grid."""

    element_label = 'T'

    def __init__(self, grid_size, interval=(-1,1), dealias=1, name=None):

        # Coordinate transformation
        # Native interval: (-1, 1)
        radius = (interval[1] - interval[0]) / 2
        center = (interval[1] + interval[0]) / 2
        self._grid_stretch = radius / 1
        self._native_coord = lambda xp: (xp - center) / radius
        self._problem_coord = lambda xn: center + (xn * radius)

        # Attributes
        self.grid_size = grid_size
        self.interval = tuple(interval)
        self.dealias = dealias
        self.name = name
        self.library = DEFAULT_LIBRARY

    @CachedMethod
    def grid(self, grid_size):
        """Build Chebyshev roots grid."""

        i = np.arange(grid_size)
        theta = pi * (i + 1/2) / grid_size
        native_grid = -np.cos(theta)
        return self._problem_coord(native_grid)

    def set_dtype(self, grid_dtype):
        """Determine coefficient properties from grid dtype."""

        # Transform retains data type
        self.grid_dtype = np.dtype(grid_dtype)
        self.coeff_dtype = self.grid_dtype
        # Same number of modes and grid points
        self.coeff_size = self.grid_size
        self.elements = np.arange(self.coeff_size)

        return self.coeff_dtype

    @staticmethod
    def _resize_coeffs(cdata_in, cdata_out, axis):
        """Resize coefficient data by padding/truncation."""

        size_in = cdata_in.shape[axis]
        size_out = cdata_out.shape[axis]

        if size_in < size_out:
            # Pad with higher order polynomials at end of data
            np.copyto(cdata_out[axslice(axis, 0, size_in)], cdata_in)
            np.copyto(cdata_out[axslice(axis, size_in, None)], 0.)
        elif size_in > size_out:
            # Truncate higher order polynomials at end of data
            np.copyto(cdata_out, cdata_in[axslice(axis, 0, size_out)])
        else:
            np.copyto(cdata_out, cdata_in)

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

    def _forward_scipy(self, gdata, cdata, axis):
        """Forward tranform using scipy DCT."""

        # View complex data as interleaved real data
        if gdata.dtype == np.complex128:
            gdata = interleaved_view(gdata)
            cdata = interleaved_view(cdata)
        # Scipy out-of-place DCT to preserve gdata
        pdata = fftpack.dct(gdata, type=2, axis=axis, overwrite_x=False)
        # Scale DCT output to Chebyshev coefficients
        self._forward_scaling(pdata, axis)
        # Pad / truncate coefficients
        self._resize_coeffs(pdata, cdata, axis)

    def _backward_scipy(self, cdata, gdata, axis):
        """Backward transform using scipy IDCT."""

        # Pad / truncate coefficients
        # Store in gdata for memory efficiency (same shape/dtype as pdata)
        self._resize_coeffs(cdata, gdata, axis)
        # Scale Chebyshev coefficients to IDCT input
        self._backward_scaling(gdata, axis)
        # View complex data as interleaved real data
        if gdata.dtype == np.complex128:
            gdata = interleaved_view(gdata)
        # Scipy in-place IDCT
        fftpack.dct(gdata, type=3, axis=axis, overwrite_x=True)

    @CachedMethod
    def _fftw_setup(self, dtype, gshape, axis):
        """Build FFTW plans and temporary arrays."""
        # Note: regular method used to cache through basis instance

        flags = ['FFTW_'+FFTW_RIGOR.upper()]
        plan = fftw.DiscreteCosineTransform(dtype, gshape, axis, flags=flags)
        pdata = fftw.create_array(gshape, dtype)

        return plan, pdata

    def _forward_fftw(self, gdata, cdata, axis):
        """Forward transform using FFTW DCT."""

        plan, pdata = self._fftw_setup(gdata.dtype, gdata.shape, axis)
        # Execute FFTW plan
        plan.forward(gdata, pdata)
        # Scale DCT output to Chebyshev coefficients
        self._forward_scaling(pdata, axis)
        # Pad / truncate coefficients
        self._resize_coeffs(pdata, cdata, axis)

    def _backward_fftw(self, cdata, gdata, axis):
        """Backward transform using FFTW IDCT."""

        plan, pdata = self._fftw_setup(gdata.dtype, gdata.shape, axis)
        # Pad / truncate coefficients
        self._resize_coeffs(cdata, pdata, axis)
        # Scale Chebyshev coefficients to IDCT input
        self._backward_scaling(pdata, axis)
        # Execute FFTW plan
        plan.backward(pdata, gdata)

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

        d_x(T_n) / n = 2 T_(n-1) + d_x(T_(n-2)) / (n-2)

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
    def Mult(self, p, subindex):
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

    element_label = 'k'

    def __init__(self, grid_size, interval=(0,2*pi), dealias=1, name=None):

        # Coordinate transformation
        # Native interval: (0, 2π)
        start = interval[0]
        length = interval[1] - interval[0]
        self._grid_stretch = length / (2*pi)
        self._native_coord = lambda xp: (2*pi) * (xp - start) / length
        self._problem_coord = lambda xn: start + (xn / (2*pi) * length)

        # Attributes
        self.grid_size = grid_size
        self.interval = tuple(interval)
        self.dealias = dealias
        self.name = name
        self.library = DEFAULT_LIBRARY

    @CachedMethod
    def grid(self, grid_size):
        """Build evenly spaced Fourier grid."""

        native_grid = np.linspace(0, 2*pi, grid_size, endpoint=False)
        return self._problem_coord(native_grid)

    def set_dtype(self, grid_dtype):
        """Determine coefficient properties from grid dtype."""

        # Tranform produces complex coefficients
        self.grid_dtype = np.dtype(grid_dtype)
        self.coeff_dtype = np.dtype(np.complex128)
        # Build native wavenumbers, discarding any Nyquist mode
        kmax = (self.grid_size - 1) // 2
        if self.grid_dtype == np.float64:
            native_wavenumbers = np.arange(0, kmax+1)
        elif self.grid_dtype == np.complex128:
            native_wavenumbers = np.roll(np.arange(-kmax, kmax+1), -kmax)
        # Scale native wavenumbers
        self.coeff_size = native_wavenumbers.size
        self.elements = self.wavenumbers = native_wavenumbers / self._grid_stretch
        # Set methods
        if self.grid_dtype == np.float64:
            self._resize_coeffs = self._resize_coeffs_real
        elif self.grid_dtype == np.complex128:
            self._resize_coeffs = self._resize_coeffs_complex

        return self.coeff_dtype

    def _resize_coeffs_real(self, cdata_in, cdata_out, axis):
        """Resize coefficient data by padding/truncation."""

        size_in = cdata_in.shape[axis]
        size_out = cdata_out.shape[axis]

        if size_in < size_out:
            # Pad with higher wavenumbers at end of data
            np.copyto(cdata_out[axslice(axis, 0, size_in)], cdata_in)
            np.copyto(cdata_out[axslice(axis, size_in, None)], 0.)
        elif size_in > size_out:
            # Truncate higher wavenumbers at end of data
            np.copyto(cdata_out, cdata_in[axslice(axis, 0, size_out)])
        else:
            np.copyto(cdata_out, cdata_in)

    def _resize_coeffs_complex(self, cdata_in, cdata_out, axis):
        """Resize coefficient data by padding/truncation."""

        size_in = cdata_in.shape[axis]
        size_out = cdata_out.shape[axis]
        kmax = min(size_in, size_out) // 2
        posfreq = axslice(axis, 0, kmax+1)
        padfreq = axslice(axis, kmax+1, -kmax)
        negfreq = axslice(axis, -kmax, None)

        if size_in < size_out:
            # Pad with higher wavenumbers and conjugates
            np.copyto(cdata_out[posfreq], cdata_in[posfreq])
            np.copyto(cdata_out[padfreq], 0)
            np.copyto(cdata_out[negfreq], cdata_in[negfreq])
        elif size_in > size_out:
            # Truncate higher wavenumbers and conjugates
            np.copyto(cdata_out[posfreq], cdata_in[posfreq])
            np.copyto(cdata_out[negfreq], cdata_in[negfreq])
        else:
            np.copyto(cdata_out, cdata_in)

    def _forward_transform_scipy(self, gdata, pdata, axis):
        """Scipy FFT."""

        grid_size = gdata.shape[axis]
        if gdata.dtype == np.float64:
            np.copyto(pdata, np.fft.rfft(gdata, n=grid_size, axis=axis))
        elif gdata.dtype == np.complex128:
            np.copyto(pdata, fftpack.fft(gdata, n=grid_size, axis=axis))
        # Scale as Fourier amplitudes
        pdata *= 1 / grid_size

    def _backward_transform_scipy(self, pdata, gdata, axis):
        """Scipy IFFT."""

        grid_size = gdata.shape[axis]
        if gdata.dtype == np.float64:
            np.copyto(gdata, np.fft.irfft(pdata, n=grid_size, axis=axis))
        elif gdata.dtype == np.complex128:
            np.copyto(gdata, fftpack.ifft(pdata, n=grid_size, axis=axis))
        # Undo built-in scaling
        gdata *= grid_size

    @CachedMethod
    def _fftw_plan(self, gdtype, gshape, axis):
        """Build FFTW plans."""

        flags = ['FFTW_'+FFTW_RIGOR.upper()]
        return fftw.FourierTransform(gdtype, gshape, axis, flags=flags)

    def _forward_transform_fftw(self, gdata, pdata, axis):
        """FFTW FFT."""

        plan = self._fftw_plan(gdata.dtype, gdata.shape, axis)
        plan.forward(gdata, pdata)
        # Scale as Fourier amplitudes
        pdata *= 1 / gdata.shape[axis]

    def _backward_transform_fftw(self, pdata, gdata, axis):
        """FFTW IFFT."""

        plan = self._fftw_plan(gdata.dtype, gdata.shape, axis)
        plan.backward(pdata, gdata)

    def differentiate(self, cdata, cderiv, axis):
        """Differentiation by wavenumber multiplication."""

        dim = len(cdata.shape)
        ik = 1j * reshape_vector(self.wavenumbers, dim=dim, axis=axis)
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
    def Mult(self, p, subindex):
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
        """Interpolate in interval using coefficients."""

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


class Compound(ImplicitBasis):
    """Chebyshev polynomial basis on the extrema grid."""

    def __init__(self, subbases, name=None):

        # Initial attributes
        self.subbases = subbases
        self.name = name

        # Check intervals
        for i in range(len(subbases)-1):
            if subbases[i].interval[1] != subbases[i+1].interval[0]:
                raise ValueError("Subbases not adjascent.")

        # Get cumulative sizes
        grid_cu = np.cumsum([b.grid_size for b in subbases])
        pad_cu = np.cumsum([b.coeff_embed for b in subbases])
        coeff_cu = np.cumsum([b.coeff_size for b in subbases])

        # Compute starting indices
        self.grid_start = np.concatenate(([0,], grid_cu))
        self.pad_start = np.concatenate(([0,], pad_cu))
        self.coeff_start = np.concatenate(([0,], coeff_cu))

        # Sizes
        self.grid_size = grid_cu[-1]
        self.coeff_embed = pad_cu[-1]
        self.coeff_size = coeff_cu[-1]

        self.interval = (subbases[0].interval[0], subbases[-1].interval[-1])
        self.grid = np.concatenate([b.grid for b in subbases])

    def set_transforms(self, grid_dtype):
        """Set transforms based on grid data type."""

        coeff_dtypes = [b.set_transforms(grid_dtype) for b in self.subbases]
        if len(set(coeff_dtypes)) > 1:
            raise ValueError("Bases returned different dtypes")

        # Transform retains data type
        self.grid_dtype = grid_dtype
        self.coeff_dtype = coeff_dtypes[0]

        self.fftw_plan = None

        # Basis elements
        self.elements = np.concatenate([b.elements for b in self.subbases])
        self.element_label = "+".join([b.element_label for b in self.subbases])

        return self.coeff_dtype

    def grid_subdata(self, gdata, index, axis):
        start = self.grid_start[index]
        end = self.grid_start[index+1]
        return gdata[axslice(axis, start, end)]

    def pad_subdata(self, pdata, index, axis):
        start = self.pad_start[index]
        end = self.pad_start[index+1]
        return pdata[axslice(axis, start, end)]

    def coeff_subdata(self, cdata, index, axis):
        start = self.coeff_start[index]
        end = self.coeff_start[index+1]
        return cdata[axslice(axis, start, end)]

    def pad_coeff(self, cdata, pdata, axis):

        for i,b in enumerate(self.subbases):
            b_cdata = self.coeff_subdata(cdata, i, axis)
            b_pdata = self.pad_subdata(pdata, i, axis)
            b.pad_coeff(b_cdata, b_pdata, axis)

    def unpad_coeff(self, pdata, cdata, axis):

        for i,b in enumerate(self.subbases):
            b_pdata = self.pad_subdata(pdata, i, axis)
            b_cdata = self.coeff_subdata(cdata, i, axis)
            b.unpad_coeff(b_pdata, b_cdata, axis)

    def forward(self, gdata, pdata, axis):

        for i,b in enumerate(self.subbases):
            b_gdata = self.grid_subdata(gdata, i, axis)
            b_pdata = self.pad_subdata(pdata, i, axis)

            b_gdata_cont = np.copy(b_gdata)
            b_pdata_cont = np.zeros_like(b_pdata)

            b.forward(b_gdata_cont, b_pdata_cont, axis)
            np.copyto(b_pdata, b_pdata_cont)

    def backward(self, pdata, gdata, axis):

        for i,b in enumerate(self.subbases):
            b_pdata = self.pad_subdata(pdata, i, axis)
            b_gdata = self.grid_subdata(gdata, i, axis)

            b_pdata_cont = np.copy(b_pdata)
            b_gdata_cont = np.zeros_like(b_gdata)

            b.backward(b_pdata_cont, b_gdata_cont, axis)
            np.copyto(b_gdata, b_gdata_cont)

    def differentiate(self, cdata, cderiv, axis):

        for i,b in enumerate(self.subbases):
            b_cdata = self.coeff_subdata(cdata, i, axis)
            b_cderiv = self.coeff_subdata(cderiv, i, axis)
            b.differentiate(b_cdata, b_cderiv, axis)

    @CachedAttribute
    def Pre(self):

        Pre = sparse.block_diag([b.Pre for b in self.subbases])
        return Pre.tocsr()

    @CachedAttribute
    def Diff(self):

        Diff = sparse.block_diag([b.Diff for b in self.subbases])
        return Diff.tocsr()

    @CachedMethod
    def Mult(self, p, subindex):

        size = self.coeff_size
        Mult = sparse.lil_matrix((size, size), dtype=self.coeff_dtype)
        start = self.coeff_start[subindex]
        end = self.coeff_start[subindex+1]
        subMult = self.subbases[subindex].Mult(p, 0)
        Mult[start:end, start:end] = subMult

        return Mult.tocsr()

    @CachedAttribute
    def left_vector(self):

        # Construct dense column vector
        left_vector = np.zeros(self.coeff_size, dtype=self.coeff_dtype)
        # Use first basis for BC
        start = self.coeff_start[0]
        end = self.coeff_start[1]
        left_vector[start:end] = self.subbases[0].left_vector

        return left_vector

    @CachedAttribute
    def right_vector(self):

        # Construct dense column vector
        right_vector = np.zeros(self.coeff_size, dtype=self.coeff_dtype)
        # Use last basis for BC
        start = self.coeff_start[-2]
        end = self.coeff_start[-1]
        right_vector[start:] = self.subbases[-1].right_vector

        return right_vector

    @CachedAttribute
    def integ_vector(self):

        integ_vector = np.concatenate([b.integ_vector for b in self.subbases])
        return integ_vector

    @CachedMethod
    def interp_vector(self, position):

        # Construct dense row vector
        interp_vector = np.zeros(self.coeff_size, dtype=self.coeff_dtype)
        # Take first basis with position in interval
        for i,b in enumerate(self.subbases):
            if b.interval[0] <= position <= b.interval[1]:
                start = self.coeff_start[i]
                end = self.coeff_start[i+1]
                interp_vector[start:end] = b.interp_vector(position)
                return interp_vector
        raise ValueError("Position outside any subbasis interval.")

    @CachedAttribute
    def bc_vector(self):

        # Construct dense column vector
        bc_vector = np.zeros((self.coeff_size, 1), dtype=self.coeff_dtype)
        # Use last basis spot for BC
        start = self.coeff_start[-2]
        end = self.coeff_start[-1]
        bc_vector[start:end] = self.subbases[-1].bc_vector

        return bc_vector

    @CachedAttribute
    def match_vector(self):

        # Construct dense column vector
        match_vector = np.zeros((self.coeff_size, 1), dtype=self.coeff_dtype)
        # Use all but last basis spots for matching
        for i,b in enumerate(self.subbases[:-1]):
            start = self.coeff_start[i]
            end = self.coeff_start[i+1]
            match_vector[start:end] = b.bc_vector

        return match_vector

    @CachedAttribute
    def Match(self):

        size = self.coeff_size
        Match = sparse.lil_matrix((size, size), dtype=self.coeff_dtype)
        for i in range(len(self.subbases) - 1):
            basis1 = self.subbases[i]
            basis2 = self.subbases[i+1]
            s1 = self.coeff_start[i]
            e1 = self.coeff_start[i+1]
            s2 = e1
            e2 = self.coeff_start[i+2]

            k1 = sparse.kron(basis1.bc_vector, basis1.right_vector)
            Match[s1:e1, s1:e1] = sparse.kron(basis1.bc_vector, basis1.right_vector)
            Match[s1:e1, s2:e2] = -sparse.kron(basis1.bc_vector, basis2.left_vector)

        return Match.tocsr()


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

