

import numpy as np
from scipy import sparse
from scipy import fftpack as fft


class PrimaryBasis(object):
    """Base class for primary bases"""

    def _build_last(self):
        """Last-element vector"""

        # Construct dense vector
        size = self.size
        last = np.zeros(size, dtype=np.complex128)
        last[size-1] = 1.

        return last


# class PiecewiseBasis(object):

#     def __init__(self, bases):

#         self.bases = bases
#         self.grid = np.hstack([b.grid for b in bases])
#         self.last = np.hstack([])


class Chebyshev(PrimaryBasis):
    """Chebyshev polynomial basis on the extrema grid"""

    diff_space = 'k'

    def __init__(self, size, range=[-1., 1.]):

        # Input parameters
        self.size = size
        self.N = size - 1
        self.range = range

        # Grid
        radius = (range[1] - range[0]) / 2.
        center = (range[1] + range[0]) / 2.
        self._diff_scale = 1. / radius

        i = np.arange(self.N + 1)
        self.grid = np.cos(np.pi * i / self.N)
        self.grid *= radius
        self.grid += center

        # Tau matrices
        self.Eval = self._build_Eval()
        self.Deriv = self._build_Deriv()
        self.Left = self._build_Left()
        self.Right = self._build_Right()
        self.last = self._build_last()

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
        T-to-U evaluation matrix with last row empty for boundary condition.

        T_n = (U_n - U_(n-2)) / 2

        """

        size = self.size

        # Diagonal entries
        i1 = np.arange(0, size-1, dtype=np.int64)
        j1 = np.arange(0, size-1, dtype=np.int64)
        v1 = np.ones(size-1, dtype=np.complex128) * 0.5
        v1[0] = 1.

        # 2nd superdiagonal entries
        i2 = np.arange(0, size-2, dtype=np.int64)
        j2 = np.arange(2, size, dtype=np.int64)
        v2 = np.ones(size-2, dtype=np.complex128) * (-0.5)

        # Combine entries
        i = np.hstack((i1, i2))
        j = np.hstack((j1, j2))
        v = np.hstack((v1, v2))

        # Construct sparse matrix
        Eval = sparse.coo_matrix((v, (i, j)), shape=(size, size), dtype=np.complex128)

        return Eval.tocsr()

    def _build_Deriv(self):
        """
        T-to-U differentiation matrix (last row naturally empty).

        d_x(T_n) = n U_(n-1)

        """

        size = self.size

        # Superdiagonal entries
        i = np.arange(0, size-1, dtype=np.int64)
        j = np.arange(1, size, dtype=np.int64)
        v = np.arange(1, size, dtype=np.complex128) * self._diff_scale

        # Construct sparse matrix
        Deriv = sparse.coo_matrix((v, (i, j)), shape=(size, size), dtype=np.complex128)

        return Deriv.tocsr()

    def _build_Left(self):
        """
        Left boundary evaluation in last row for boundary condition.

        T_n(-1) = (-1)**n

        """

        size = self.size

        # Last row entries
        i = np.ones(size, dtype=np.int64) * (size - 1)
        j = np.arange(0, size, dtype=np.int64)
        v = np.ones(size, dtype=np.complex128)
        v[1::2] = -1.

        # Construct sparse matrix
        Left = sparse.coo_matrix((v, (i, j)), shape=(size, size), dtype=np.complex128)

        return Left.tocsr()

    def _build_Right(self):
        """
        Right boundary evaluation in last row for boundary condition.

        T_n(1) = 1

        """

        size = self.size

        # Last row entries
        i = np.ones(size, dtype=np.int64) * (size - 1)
        j = np.arange(0, size, dtype=np.int64)
        v = np.ones(size, dtype=np.complex128)

        # Construct sparse matrix
        Right = sparse.coo_matrix((v, (i, j)), shape=(size, size), dtype=np.complex128)

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
        for n in xrange(0, size):
            # Upper product
            i = n + p
            if i < size:
                Mult[i, n] += 0.5
            # Lower product
            i = n - p
            # Handle negatives
            if i == -1:
                continue
            elif i < -1:
                Mult[-i-2, n] -= 0.5
            else:
                Mult[i, n] += 0.5

        # Empty last row
        Mult[size-1, :] = 0

        return Mult.tocsr()

class Fourier(PrimaryBasis):
    """Fourier complex exponential basis"""

    def __init__(self, range=[0., 2*np.pi]):

        pass

