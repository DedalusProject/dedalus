

import numpy as np
from scipy import sparse
from scipy import fftpack

from ..tools.general import CachedAttribute, CachedMethod, interleaved_view


class Basis:
    """Base class for all bases."""

    def __init__(self, grid_size, interval):

        # Initial attributes
        self.grid_size = grid_size
        self.interval = tuple(interval)

    def set_dtype(self, grid_dtype):
        """Specify datatypes."""

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

    def integrate(self, cdata, axis):
        """Integrate over interval using coefficients."""

        raise NotImplementedError()


class TransverseBasis(Basis):
    """Base class for bases supporting transverse differentiation."""

    def trans_diff(self, i):
        """Transverse differentation constant for i-th term."""

        raise NotImplementedError()


class TauBasis(Basis):
    """Base class for bases supporting Tau solves."""

    def integrate(self, cdata, axis):
        """Integrate over interval using coefficients."""

        # Dot coefficients with basis function integrals
        integral = np.tensordot(cdata, self.int_vector, (axis, 0))

        return integral

    @CachedAttribute
    def Pre(self):
        """Preconditioning matrix."""

        # Construct sparse identity matrix
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
        """Left-endpoint-evaluation matrix."""

        # Sparse kronecker BC column vector with left row vector
        Left = sparse.kron(self.bc_vector, self.left_vector)

        return Left

    @CachedAttribute
    def Right(self):
        """Right-endpoint-evaluation matrix."""

        # Sparse kronecker BC column vector with right row vector
        Right = sparse.kron(self.bc_vector, self.right_vector)

        return Right

    @CachedAttribute
    def Int(self):
        """Integral-evaluation matrix."""

        # Sparse kronecker BC column vector with int row vector
        Int = sparse.kron(self.bc_vector, self.int_vector)

        return Int

    @CachedAttribute
    def left_vector(self):
        """Left-endpoint-evaluation row vector."""

        raise NotImplementedError()

    @CachedAttribute
    def right_vector(self):
        """Right-endpoint-evaluation row vector."""

        raise NotImplementedError()

    @CachedAttribute
    def int_vector(self):
        """Integral-evaluation row vector."""

        raise NotImplementedError()

    @CachedAttribute
    def bc_vector(self):
        """Boundary-row column vector."""

        raise NotImplementedError()


class Chebyshev(TauBasis):
    """Chebyshev polynomial basis on the extrema grid."""

    def __init__(self, grid_size, interval=(-1., 1.)):

        # Inherited initialization
        Basis.__init__(self, grid_size, interval)

        # Initial attributes
        self.coeff_size = grid_size
        self.N = grid_size - 1

        # Grid
        radius = (interval[1] - interval[0]) / 2.
        center = (interval[1] + interval[0]) / 2.
        i = np.arange(self.N + 1)
        native_grid = np.cos(np.pi * i / self.N)
        self.grid = center + radius * native_grid
        self._grid_stretch = radius

    def set_dtype(self, grid_dtype):
        """Specify datatypes."""

        # Set datatypes
        self.grid_dtype = grid_dtype
        self.coeff_dtype = grid_dtype

        # Set transforms
        if grid_dtype is np.float64:
            self.forward = self._forward_r2r
            self.backward = self._backward_r2r
        elif grid_dtype is np.complex128:
            self.forward = self._forward_c2c
            self.backward = self._backward_c2c
        else:
            raise ValueError("Unsupported grid_dtype.")

        return self.coeff_dtype

    def _forward_r2r(self, gdata, cdata, axis):
        """Scipy DCT on real data."""

        # Currently setup just for last axis
        if axis != -1:
            if axis != (len(gdata.shape) - 1):
                raise NotImplementedError()

        # DCT with adjusted coefficients
        N = self.N
        cdata[:] = fftpack.dct(gdata, type=1, norm=None, axis=axis)
        cdata /= N
        cdata[..., 0] /= 2.
        cdata[..., N] /= 2.

    def _backward_r2r(self, cdata, gdata, axis):
        """Scipy IDCT on real data."""

        # Currently setup just for last axis
        if axis != -1:
            if axis != (len(cdata.shape) - 1):
                raise NotImplementedError()

        # DCT with adjusted coefficients
        N = self.N
        gdata[:] = cdata
        gdata[..., 1:N] /= 2.
        gdata[:] = fftpack.dct(gdata, type=1, norm=None, axis=axis)

    def _forward_c2c(self, gdata, cdata, axis):
        """Scipy DCT on complex data."""

        # Handle negative axes
        if axis < 0:
            axis += len(gdata.shape)

        # Currently setup just for last axis
        if axis != (len(gdata.shape) - 1):
            raise NotImplementedError()

        # Create interleaved view
        cdata_iv = interleaved_view(cdata)

        # DCT with adjusted coefficients
        N = self.N
        cdata[:] = gdata
        cdata_iv[:] = fftpack.dct(cdata_iv, type=1, norm=None, axis=axis)
        cdata /= N
        cdata[..., 0] /= 2.
        cdata[..., N] /= 2.

    def _backward_c2c(self, cdata, gdata, axis):
        """Scipy IDCT on complex data."""

        # Handle negative axes
        if axis < 0:
            axis += len(gdata.shape)

        # Currently setup just for last axis
        if axis != (len(gdata.shape) - 1):
            raise NotImplementedError()

        # Create interleaved view
        gdata_iv = interleaved_view(gdata)

        # DCT with adjusted coefficients
        N = self.N
        gdata[:] = cdata
        gdata[..., 1:N] /= 2.
        gdata_iv[:] = fftpack.dct(gdata_iv, type=1, norm=None, axis=axis)

    def differentiate(self, cdata, cderiv, axis):
        """Differentiation by recursion on coefficients."""

        # Currently setup just for last axis
        if axis != -1:
            if axis != (len(cdata.shape) - 1):
                raise NotImplementedError()

        # Referencess
        a = cdata
        b = cderiv
        N = self.N

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

        # Initialize sparse matrix
        Pre = sparse.lil_matrix((size, size), dtype=self.coeff_dtype)

        # Add elements
        for n in range(size):

            # Diagonal
            if n == 0:
                Pre[n, n] = 1.
            else:
                Pre[n, n] = 0.5

            # 2nd superdiagonal
            if n >= 2:
                Pre[n-2, n] = -0.5

        return Pre.tocsr()

    @CachedAttribute
    def Diff(self):
        """
        Differentiation matrix.

        d_x(T_n) / n = 2 T_n + d_x(T_(n-2)) / (n-2)

        """

        size = self.coeff_size

        # Initialize sparse matrix
        Diff = sparse.lil_matrix((size, size), dtype=self.coeff_dtype)

        # Add elements
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

        # Add elements
        for n in range(size):

            # Upper product
            i = n + p
            if i < size:
                Mult[i, n] += 0.5

            # Lower product
            i = abs(n - p)
            if i < size:
                Mult[i, n] += 0.5

        return Mult.tocsr()

    @CachedAttribute
    def left_vector(self):
        """
        Left-endpoint-evaluation row vector.

        T_n(-1) = (-1)**n

        """

        # Construct dense row vector
        left_vector = np.ones(self.coeff_size, dtype=self.coeff_dtype)
        left_vector[1::2] = -1.

        return left_vector

    @CachedAttribute
    def right_vector(self):
        """
        Right-endpoint-evaluation row vector.

        T_n(1) = 1

        """

        # Construct dense row vector
        right_vector = np.ones(self.coeff_size, dtype=self.coeff_dtype)

        return right_vector

    @CachedAttribute
    def int_vector(self):
        """
        Integral-evaluation row vector.

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
        bc_vector[-1, :] = 1.

        return bc_vector


class Fourier(TransverseBasis, TauBasis):
    """Fourier complex exponential basis."""

    def __init__(self, grid_size, interval=(0., 2.*np.pi)):

        # Inherited initialization
        Basis.__init__(self, grid_size, interval)

        # Grid
        length = interval[1] - interval[0]
        start = interval[0]
        native_grid = np.linspace(0., 1., grid_size, endpoint=False)
        self.grid = start + length * native_grid
        self._grid_stretch = length / (2. * np.pi)

    def set_dtype(self, dtype):
        """Specify datatypes."""

        # Set datatypes
        self.grid_dtype = dtype
        self.coeff_dtype = np.complex128

        # Set transforms
        n = self.grid_size
        if dtype is np.float64:
            self.forward = self._forward_r2c
            self.backward = self._backward_c2r
            self.coeff_size = n//2 + 1
            wavenumbers = np.arange(0, n//2 + 1)
        elif dtype is np.complex128:
            self.forward = self._forward_c2c
            self.backward = self._backward_c2c
            self.coeff_size = n
            wavenumbers = np.hstack((np.arange(0, n//2+1),
                                     np.arange((-n)//2+1, 0)))
        else:
            raise ValueError("Unsupported dtype.")

        self.wavenumbers = wavenumbers / self._grid_stretch

        return self.coeff_dtype

    def _forward_r2c(self, gdata, cdata, axis):
        """Scipy R2C FFT"""

        cdata[:] = fftpack.rfft(gdata, axis=axis)
        cdata /= self.grid_size

    def _forward_c2c(self, gdata, cdata, axis):
        """Scipy C2C FFT."""

        cdata[:] = fftpack.fft(gdata, axis=axis)
        cdata /= self.grid_size

    def _backward_c2r(self, cdata, gdata, axis):
        """Scipy C2R IFFT"""

        gdata[:] = fftpack.irfft(cdata, axis=axis)
        gdata *= self.grid_size

    def _backward_c2c(self, cdata, gdata, axis):
        """Scipy C2C IFFT."""

        gdata[:] = fftpack.ifft(cdata, axis=axis)
        gdata *= self.grid_size

    def differentiate(self, cdata, cderiv, axis):
        """Differentiation by wavenumber multiplication."""

        # Wavenumber array
        shape = [1] * len(cdata.shape)
        shape[axis] = self.coeff_size
        ik = 1j * self.wavenumbers.reshape(shape)

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

