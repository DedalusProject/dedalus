

import numpy as np
from scipy import sparse
from scipy.sparse import linalg
from scipy import fftpack as fft


class Basis(object):
    """Base class for all bases."""

    def __init__(self, size, range):

        # Inputs
        self.size = size
        self.range = range


class TauBasis(Basis):
    """Base class for bases supporting Tau solves."""

    def __init__(self, size, range):

        # Inherited initialization
        Basis.__init__(self, size, range)

    def build_matrices(self):

        # Build Tau matrices
        self.Eval = self._build_Eval()
        self.Deriv = self._build_Deriv()
        self.Left = self._build_Left()
        self.Right = self._build_Right()
        self.last = self._build_last()
        self.InvEval = linalg.inv(self.Eval.tocsc())

    def _build_last(self):
        """Last-element vector"""

        # Construct dense vector
        size = self.size
        last = np.zeros(size, dtype=np.complex128)
        last[size-1] = 1.

        return last


class Chebyshev(TauBasis):
    """Chebyshev polynomial basis on the extrema grid."""

    diff_space = 'k'

    def __init__(self, size, range=[-1., 1.]):

        # Inherited initialization
        TauBasis.__init__(self, size, range)

        # Parameters
        self.N = size - 1

        # Grid
        radius = (range[1] - range[0]) / 2.
        center = (range[1] + range[0]) / 2.
        self._diff_scale = 1. / radius

        i = np.arange(self.N + 1)
        self.grid = np.cos(np.pi * i / self.N)
        self.grid *= radius
        self.grid += center

        # Math array
        self._math = np.zeros(size, dtype=np.complex128)

    def forward(self, xdata, kdata):
        """Grid values to coefficients transform"""

        # DCT with adjusted coefficients
        N = self.N
        kdata.real = fft.dct(xdata.real, type=1, norm=None, axis=-1)
        kdata.imag = fft.dct(xdata.imag, type=1, norm=None, axis=-1)
        kdata /= N
        kdata[0] /= 2.
        kdata[N] /= 2.

    def backward(self, kdata, xdata):
        """Coefficient to grid values transform"""

        # DCT with adjusted coefficients
        N = self.N
        self._math[:] = kdata
        self._math[1:N] /= 2.
        xdata.real = fft.dct(self._math.real, type=1, norm=None, axis=-1)
        xdata.imag = fft.dct(self._math.imag, type=1, norm=None, axis=-1)

    def differentiate(self, kdata, kderiv):
        """Diffentiation by recursion on coefficients"""

        # Referencess
        a = kdata
        b = kderiv
        N = self.N

        # Apply recursive differentiation
        b[N] = 0.
        b[N-1] = 2. * N * a[N]
        for i in xrange(N-2, 0, -1):
            b[i] = 2 * (i+1) * a[i+1] + b[i+2]
        b[0] = a[1] + b[2] / 2.

        # Scale for grid
        kderiv *= self._diff_scale

    def _build_Eval(self):
        """
        T-to-U evaluation matrix.

        T_n = (U_n - U_(n-2)) / 2

        """

        size = self.size

        # Construct sparse matrix
        Eval = sparse.lil_matrix((size, size), dtype=np.complex128)

        # Add elements
        for n in xrange(size):

            # Diagonal
            if n == 0:
                Eval[n, n] = 1.
            else:
                Eval[n, n] = 0.5

            # 2nd superdiagonal
            if n >= 2:
                Eval[n-2, n] = -0.5

        return Eval.tocsr()

    def _build_Deriv(self):
        """
        T-to-U differentiation matrix.

        d_x(T_n) = n U_(n-1)

        """

        size = self.size

        # Construct sparse matrix
        Deriv = sparse.lil_matrix((size, size), dtype=np.complex128)

        # Add elements
        for n in xrange(1, size):
            Deriv[n-1, n] = n * self._diff_scale

        return Deriv.tocsr()

    def _build_Left(self):
        """
        Left boundary evaluation in last row for boundary condition.

        T_n(-1) = (-1)**n

        """

        size = self.size

        # Construct sparse matrix
        Left = sparse.lil_matrix((size, size), dtype=np.complex128)

        # Add elements
        for n in xrange(size):
            if (n % 2) == 0:
                Left[size-1, n] = 1.
            else:
                Left[size-1, n] = -1.

        return Left.tocsr()

    def _build_Right(self):
        """
        Right boundary evaluation in last row for boundary condition.

        T_n(1) = 1

        """

        size = self.size

        # Construct sparse matrix
        Right = sparse.lil_matrix((size, size), dtype=np.complex128)

        # Add elements
        for n in xrange(size):

            # Last row entries
            Right[size-1, n] = 1.

        return Right.tocsr()

    def _build_Mult(self, p):
        """
        U-times-T_p multiplication matrix.

        T_p * U_n = (U_(n+p) + U_(n-p)) / 2
        U_(n-p) = -U_(p-n-2)
        U_(-1) = 0

        """

        size = self.size

        # Construct sparse matrix
        Mult = sparse.lil_matrix((size, size), dtype=np.complex128)

        # Add elements
        for n in xrange(size):

            # Upper product
            i = n + p
            if i < size:
                Mult[i, n] += 0.5

            # Lower product
            i = n - p
            if i > -1:
                Mult[i, n] += 0.5
            elif i < -1:
                Mult[-i-2, n] -= 0.5

        return Mult.tocsr()

    def _build_Mult1(self, p):
        """
        T-times-T_p multiplication matrix

        T_p * T_n = (T_(n+p) + T_(n-p)) / 2
        T_(-n) = T_n

        """

        size = self.size

        # Construct sparse matrix
        Mult = sparse.lil_matrix((size, size), dtype=np.complex128)

        # Add elements
        for n in xrange(size):

            # Upper product
            i = n + p
            if i < size:
                Mult[i, n] += 0.5

            # Lower product
            i = n - p
            if i >= 0:
                Mult[i, n] += 0.5
            else:
                Mult[-i, n] += 0.5

        return Mult.tocsr()


class Fourier(Basis):
    """Fourier complex exponential basis."""

    diff_space = 'k'

    def __init__(self, size, range=[0., 2*np.pi]):

        # Inherited initialization
        Basis.__init__(self, size, range)

        # Grid
        length = range[1] - range[0]
        start = range[0]
        self._diff_scale = 2. * np.pi / length

        self.grid = np.linspace(0., 1., size, endpoint=False)
        self.grid *= length
        self.grid += start

        # Math array
        self._math = np.zeros(size, dtype=np.complex128)

    def forward(self, xdata, kdata):
        """Grid values to coefficients transform"""

        # FFT with mode-amplitude weighting
        kdata[:] = fft.fft(xdata, axis=-1)
        kdata /= self.size

    def backward(self, kdata, xdata):
        """Coefficient to grid values transform"""

        # FFT with mode-amplitude weighting
        xdata[:] = fft.ifft(kdata, axis=-1)
        xdata *= self.size

    def differentiate(self, kdata, kderiv):
        """Diffentiation wavenumber multiplication."""

        # Wavenumber multiplication
        k = 1j * np.arange(self.size) * self._diff_scale
        kderiv[:] = kdata * k


# class PiecewiseBasis(object):

#     def __init__(self, bases):

#         self.bases = bases
#         self.grid = np.hstack([b.grid for b in bases])
#         self.last = np.hstack([])

