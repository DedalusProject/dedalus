"""
Abstract and built-in classes for spectral bases.

"""

import math
import numpy as np
from scipy import sparse

from . import operators
from ..tools.array import axslice
from ..tools.cache import CachedAttribute
from ..tools.cache import CachedMethod
from ..tools.cache import CachedClass
from ..tools import jacobi
from ..tools import clenshaw

import logging
logger = logging.getLogger(__name__.split('.')[-1])

from ..tools.config import config
DEFAULT_LIBRARY = config['transforms'].get('DEFAULT_LIBRARY')


class Basis:
    """Base class for spectral bases."""

    def __init__(self, space, library=DEFAULT_LIBRARY):
        self.space = space
        self.domain = space.domain
        self.axes = space.axes
        self.library = library

    def __repr__(self):
        return '<%s %i>' %(self.__class__.__name__, id(self))

    def __str__(self):
        return '%s.%s' %(self.space.name, self.__class__.__name__)

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __getitem__(self, mode):
        """Return field populated by one mode."""
        # if not self.compute_mode(mode):
        #     raise ValueError("Basis does not contain specified mode.")
        from .field import Field
        axis = self.space.axis
        out = Field(bases=[self], layout='c')
        data = np.zeros(out.global_shape, dtype=out.dtype)
        if mode < 0:
            mode += self.space.coeff_size
        data[axslice(axis, mode, mode+1)] = 1
        out.set_global_data(data)
        return out

    @property
    def library(self):
        """Current transform library."""
        return self._library

    @library.setter
    def library(self, value):
        """Clear transform cache and set new library."""
        self._library = value.lower()
        self.transform_plan.cache.clear()

    @CachedMethod
    def transform_plan(self, coeff_shape, axis, scale):
        """Build and cache transform plan."""
        transform_class = self.transforms[self.library]
        return transform_class(self, coeff_shape, axis, scale)

    @CachedAttribute
    def inclusion_flags(self):
        return np.array([self.include_mode(i) for i in range(self.space.coeff_size)])

    @CachedAttribute
    def inclusion_matrix(self):
        diag = self.inclusion_flags.astype(float)
        return sparse.diags(diag, 0, format='csr')

    @CachedAttribute
    def modes(self):
        return np.arange(self.space.coeff_size)[self.inclusion_flags]

    @CachedAttribute
    def n_modes(self):
        return self.modes.size

    def mode_map(self, group):
        flags = self.inclusion_flags
        matrix = self.inclusion_matrix
        # Restrict to group elements
        if group is not None:
            n0 = group * self.space.group_size
            n1 = n0 + self.space.group_size
            matrix = matrix[n0:n1, n0:n1]
            flags = flags[n0:n1]
        # Discard empty rows
        return matrix[flags, :]

    def ncc_matrix(self, arg_basis, coeffs):
        """Build NCC matrix via direct summation."""
        print('in Basis.ncc_matrix')
        N = len(coeffs)
        cutoff = 1e-6
        total = 0
        for i in range(N):
            coeff = coeffs[i]
            if abs(coeff) > cutoff:
                matrix = self.product_matrix(arg_basis, i)
                total = total + sparse.kron(matrix, coeff)
                print('add term', i)
        return total

    def product_matrix(self, arg_basis, i):
        if arg_basis is None:
            N = self.space.coeff_size
            return sparse.coo_matrix(([1],([i],[0])), shape=(N,1)).tocsr()
        else:
            raise NotImplementedError()


class Jacobi(Basis, metaclass=CachedClass):
    """Jacobi polynomial basis."""

    def __init__(self, space, da, db, library='matrix'):
        super().__init__(space, library=library)
        self.da = da
        self.db = db
        self.a = space.a + da
        self.b = space.b + db
        self.const = 1 / np.sqrt(jacobi.mass(self.a, self.b))

    def include_mode(self, mode):
        return (0 <= mode < self.space.coeff_size)

    def __str__(self):
        space = self.space
        cls = self.__class__
        return '%s.%s(%s,%s)' %(space.name, cls.__name__, self.a, self.b)

    def __add__(self, other_basis):
        if other_basis is None:
            return self
        elif self.space == other_basis.space:
            # Add in highest {a,b} basis
            da = max(self.da, other_basis.da)
            db = max(self.db, other_basis.db)
            return Jacobi(self.space, da, db)
        else:
            return NotImplemented

    def __mul__(self, other_basis):
        if other_basis is None:
            return self
        elif self.space == other_basis.space:
            # Put product in highest {a,b} basis
            da = max(self.da, other_basis.da)
            db = max(self.db, other_basis.db)
            return Jacobi(self.space, da, db)
        else:
            return NotImplemented

    def ncc_matrix(self, arg_basis, coeffs):
        """Build NCC matrix via Clenshaw algorithm."""
        CUTOFF = 1e-6
        if arg_basis is None:
            return super().ncc_matrix(arg_basis, coeffs)
        # Kronecker Clenshaw on argument Jacobi matrix
        N = self.space.coeff_size
        J = jacobi.jacobi_matrix(N, arg_basis.a, arg_basis.b)
        A, B = clenshaw.jacobi_recursion(N, self.a, self.b, J)
        f0 = self.const * sparse.identity(N)
        total = clenshaw.kronecker_clenshaw(coeffs, A, B, f0, cutoff=CUTOFF)
        # Conversion matrix
        input_basis = arg_basis
        output_basis = (self * arg_basis)
        conversion = ConvertJacobiJacobi._build_subspace_matrix(self.space, input_basis, output_basis)
        return (conversion @ total)


class ConvertJacobiJacobi(operators.Convert):

    input_basis_type = Jacobi
    output_basis_type = Jacobi
    separable = False

    @staticmethod
    def _build_subspace_matrix(space, input_basis, output_basis):
        N = space.coeff_size
        a0, b0 = input_basis.a, input_basis.b
        a1, b1 = output_basis.a, output_basis.b
        matrix = jacobi.conversion_matrix(N, a0, b0, a1, b1)
        return matrix.tocsr()


class DifferentiateJacobi(operators.Differentiate):

    input_basis_type = Jacobi
    separable = False

    @staticmethod
    def output_basis(space, input_basis):
        da, db = input_basis.da, input_basis.db
        return Jacobi(space, da+1, db+1)

    @staticmethod
    def _build_subspace_matrix(space, input_basis):
        N = space.coeff_size
        a, b = input_basis.a, input_basis.b
        matrix = jacobi.differentiation_matrix(N, a, b)
        return (matrix.tocsr() / space.COV.stretch)


class InterpolateJacobi(operators.Interpolate):

    input_basis_type = Jacobi

    @staticmethod
    def _build_subspace_matrix(space, input_basis, position):
        N = space.coeff_size
        a, b = input_basis.a, input_basis.b
        x = space.COV.native_coord(position)
        return jacobi.interpolation_vector(N, a, b, x)


class IntegrateJacobi(operators.Integrate):

    input_basis_type = Jacobi

    @staticmethod
    def _build_subspace_matrix(space, input_basis):
        N = space.coeff_size
        a, b = input_basis.a, input_basis.b
        vector = jacobi.integration_vector(N, a, b)
        return (vector * space.COV.stretch)


class ChebyshevT(Basis, metaclass=CachedClass):
    """Chebyshev-T polynomial basis."""

    element_label = 'T'

    def include_mode(self, mode):
        return (0 <= mode < self.space.coeff_size)

    def __add__(self, other):
        space = self.space
        if other is None:
            return space.ChebyshevT
        elif other is space.ChebyshevT:
            return space.ChebyshevT
        elif other is space.ChebyshevU:
            return space.ChebyshevU
        else:
            return NotImplemented

    def __mul__(self, other):
        space = self.space
        if other is None:
            return space.ChebyshevT
        elif other is space.ChebyshevT:
            return space.ChebyshevT
        elif other is space.ChebyshevU:
            return space.ChebyshevT
        else:
            return NotImplemented

    def __pow__(self, other):
        return self.space.ChebyshevT

    def Multiply(self, p, basis):
        """p-element multiplication matrix"""
        size = self.space.coeff_size
        # Construct sparse matrix
        if basis is None:
            Mult = sparse.lil_matrix((size, 1), dtype=self.space.domain.dtype)
            Mult[p, 0] = 1
        elif basis is self.space.ChebyshevT:
            # T[p] * T[n] = (T[n+p] + T[n-p]) / 2
            # T[-n] = T[n]
            Mult = sparse.lil_matrix((size, size), dtype=self.space.domain.dtype)
            for n in range(size):
                if (p+n) < size:
                    Mult[p+n, n] += 0.5
                if (p-n) < 0:
                    Mult[n-p, n] -= 0.5
                else:
                    Mult[p-n, n] += 0.5
        else:
            raise TypeError()
        return Mult.tocsr()


class ChebyshevU(Basis, metaclass=CachedClass):
    """Chebyshev-U polynomial basis."""

    element_label = 'U'

    def include_mode(self, mode):
        return (0 <= mode < self.space.coeff_size)

    def __add__(self, other):
        space = self.space
        if other is None:
            return space.ChebyshevU
        elif other is space.ChebyshevU:
            return space.ChebyshevU
        elif other is space.ChebyshevT:
            return space.ChebyshevU
        else:
            return NotImplemented

    def __mul__(self, other):
        space = self.space
        if other is None:
            return space.ChebyshevU
        elif other is space.ChebyshevU:
            return space.ChebyshevT
        elif other is space.ChebyshevT:
            return space.ChebyshevU
        else:
            return NotImplemented

    def __pow__(self, other):
        return self.space.ChebyshevT

    def Multiply(self, p, basis):
        """p-element multiplication matrix"""
        size = self.space.coeff_size
        # Construct sparse matrix
        if basis is None:
            Mult = sparse.lil_matrix((size, 1), dtype=self.space.domain.dtype)
            Mult[p, 0] = 1
        elif basis is self.space.ChebyshevT:
            # U[p] * T[n] = (U[p+n] + U[p-n]) / 2
            # U[-n] = -U[n-2]
            Mult = sparse.lil_matrix((size, size), dtype=self.space.domain.dtype)
            for n in range(size):
                if (p+n) < size:
                    Mult[p+n, n] += 0.5
                if (p-n) < 0:
                    Mult[n-p-2, n] -= 0.5
                else:
                    Mult[p-n, n] += 0.5
        else:
            raise TypeError()
        return Mult.tocsr()


class InterpolateChebyshevT(operators.Interpolate):

    input_basis_type = ChebyshevT

    @classmethod
    def entry(cls, j, space, position):
        """Tn(x) = cos(n*acos(x))"""
        x = space.COV.native_coord(position)
        theta = math.acos(x)
        return math.cos(j*theta)


class InterpolateChebyshevU(operators.Interpolate):

    input_basis_type = ChebyshevU

    @classmethod
    def entry(cls, j, space, position):
        """Un(x) = sin((n+1)*acos(x)) / sin(acos(x))"""
        x = space.COV.native_coord(position)
        theta = math.acos(x)
        if theta == 0:
            return (j + 1)
        elif theta == np.pi:
            return (j + 1) * (-1)**j
        else:
            return math.sin((j+1)*theta) / math.sin(theta)


class IntegrateChebyshevT(operators.Integrate):

    input_basis_type = ChebyshevT

    @classmethod
    def entry(cls, j, space):
        """Integral(T_n(x), -1, 1) = (1 + (-1)**n) / (1 - n**2)"""
        if (j % 2):
            return 0
        else:
            return 2 / (1 - j**2) * space.COV.stretch


class IntegrateChebyshevU(operators.Integrate):

    input_basis_type = ChebyshevU

    @classmethod
    def entry(cls, j, space):
        """Integral(U_n(x), -1, 1) = (1 + (-1)**n) / (1 + n)"""
        if (j % 2):
            return 0
        else:
            return 2 / (1 + j) * space.COV.stretch


class DifferentiateChebyshevT(operators.Differentiate):

    input_basis_type = ChebyshevT
    output_basis_type = ChebyshevU
    bands = [0, 1, 2]
    separable = False

    @classmethod
    def entry(cls, i, j, space):
        """dx(T_n) = n * U_{n-1}"""
        if i == (j - 1):
            return j / space.COV.stretch
        else:
            return 0


class ConvertChebyshevTChebyshevU(operators.Convert):

    input_basis_type = ChebyshevT
    output_basis_type = ChebyshevU
    bands = [0, 2]
    separable = False

    @classmethod
    def entry(cls, i, j, space, basis):
        """
        T_n = (U_n - U_(n-2)) / 2
        U_(-n) = -U_(n-2)
        """
        if i == j:
            if j == 0:
                return 1
            else:
                return 0.5
        elif i == (j - 2):
            return -0.5


class Fourier(Basis, metaclass=CachedClass):
    """Fourier sine/cosine series basis."""

    element_label= 'k'

    def include_mode(self, mode):
        if mode == 1:
            return False
        else:
            k = mode // 2
            return (0 <= k <= self.space.kmax)

    def __add__(self, other):
        space = self.space
        if other is None:
            return space.Fourier
        elif other is space.Fourier:
            return space.Fourier
        else:
            return NotImplemented

    def __mul__(self, other):
        space = self.space
        if other is None:
            return space.Fourier
        elif other is space.Fourier:
            return space.Fourier
        else:
            return NotImplemented

    def __pow__(self, other):
        return self.space.Fourier


class InterpolateFourier(operators.Interpolate):

    input_basis_type = Fourier

    @classmethod
    def entry(cls, j, space, position):
        """
        cos(n*x)
        sin(n*x)
        """
        n = j // 2
        x = space.COV.native_coord(position)
        if (j % 2) == 0:
            return math.cos(n*x)
        else:
            return math.sin(n*x)


class IntegrateFourier(operators.Integrate):

    input_basis_type = Fourier

    @classmethod
    def entry(cls, j, space):
        """
        Integral(cos(n*x), 0, 2*pi) = 2 * pi * δ(n, 0)
        Integral(sin(n*x), 0, 2*pi) = 0
        """
        if j == 0:
            return 2 * np.pi * space.COV.stretch
        else:
            return 0


class DifferentiateFourier(operators.Differentiate):

    input_basis_type = Fourier
    output_basis_type = Fourier
    bands = [-1, 1]
    separable = True

    @classmethod
    def entry(cls, i, j, space):
        """
        dx(cos(n*x)) = -n*sin(n*x)
        dx(sin(n*x)) = n*cos(n*x)
        """
        n = j // 2
        if n == 0:
            return 0
        elif (j % 2) == 0:
            # dx(cos(n*x)) = -n*sin(n*x)
            if i == (j + 1):
                return (-n) / space.COV.stretch
            else:
                return 0
        else:
            # dx(sin(n*x)) = n*cos(n*x)
            if i == (j - 1):
                return n / space.COV.stretch
            else:
                return 0


class HilbertTransformFourier(operators.HilbertTransform):

    input_basis_type = Fourier
    output_basis_type = Fourier
    bands = [-1, 1]
    separable = True

    @classmethod
    def entry(cls, i, j, space):
        """
        Hx(cos(n*x)) = sin(n*x)
        Hx(sin(n*x)) = -cos(n*x)
        """
        n = j // 2
        if n == 0:
            return 0
        elif (j % 2) == 0:
            # dx(cos(n*x)) = -n*sin(n*x)
            if i == (j + 1):
                return 1
            else:
                return 0
        else:
            # dx(sin(n*x)) = n*cos(n*x)
            if i == (j - 1):
                return (-1)
            else:
                return 0


class Sine(Basis, metaclass=CachedClass):
    """Sine series basis."""

    element_label = 'k'

    def include_mode(self, mode):
        k = mode
        return (1 <= k <= self.space.kmax)

    def __add__(self, other):
        space = self.space
        if other is space.Sine:
            return space.Sine
        else:
            return NotImplemented

    def __mul__(self, other):
        space = self.space
        if other is None:
            return space.Sine
        elif other is space.Sine:
            return space.Cosine
        elif other is space.Cosine:
            return space.Sine
        else:
            return NotImplemented

    def __pow__(self, other):
        space = self.space
        if (other % 2) == 0:
            return space.Cosine
        elif (other % 2) == 1:
            return space.Sine
        else:
            return NotImplemented


class Cosine(Basis, metaclass=CachedClass):
    """Cosine series basis."""

    element_label = 'k'

    def include_mode(self, mode):
        k = mode
        return (0 <= k <= self.space.kmax)

    def __add__(self, other):
        space = self.space
        if other is None:
            return space.Cosine
        elif other is space.Cosine:
            return space.Cosine
        else:
            return NotImplemented

    def __mul__(self, other):
        space = self.space
        if other is None:
            return space.Cosine
        elif other is space.Sine:
            return space.Sine
        elif other is space.Cosine:
            return space.Cosine
        else:
            return NotImplemented

    def __pow__(self, other):
        return self.space.Cosine


class InterpolateSine(operators.Interpolate):

    input_basis_type = Sine

    @classmethod
    def entry(self, j, space, position):
        """sin(n*x)"""
        x = space.COV.native_coord(position)
        return math.sin(j*x)


class InterpolateCosine(operators.Interpolate):

    input_basis_type = Cosine

    @classmethod
    def entry(cls, j, space, position):
        """cos(n*x)"""
        x = space.COV.native_coord(position)
        return math.cos(j*x)


class IntegrateSine(operators.Integrate):

    input_basis_type = Sine

    @classmethod
    def entry(cls, j, space):
        """Integral(sin(n*x), 0, pi) = (2 / n) * (n % 2)"""
        if (j % 2):
            return 0
        else:
            return (2 / j) * space.COV.stretch


class IntegrateCosine(operators.Integrate):

    input_basis_type = Cosine

    @classmethod
    def entry(cls, j, space):
        """Integral(cos(n*x), 0, pi) = pi * δ(n, 0)"""
        if j == 0:
            return np.pi * space.COV.stretch
        else:
            return 0


class DifferentiateSine(operators.Differentiate):

    input_basis_type = Sine
    output_basis_type = Cosine
    bands = [0]
    separable = True

    @classmethod
    def entry(cls, i, j, space):
        """dx(sin(n*x)) = n*cos(n*x)"""
        if i == j:
            return j / space.COV.stretch
        else:
            return 0


class DifferentiateCosine(operators.Differentiate):

    input_basis_type = Cosine
    output_basis_type = Sine
    bands = [0]
    separable = True

    @classmethod
    def entry(cls, i, j, space):
        """dx(cos(n*x)) = -n*sin(n*x)"""
        if i == j:
            return (-j) / space.COV.stretch
        else:
            return 0


class HilbertTransformSine(operators.HilbertTransform):

    input_basis_type = Sine
    output_basis_type = Cosine
    bands = [0]
    separable = True

    @classmethod
    def entry(cls, i, j, space):
        """Hx(sin(n*x)) = -cos(n*x)"""
        if i == j:
            return (-1)
        else:
            return 0


class HilbertTransformCosine(operators.HilbertTransform):

    input_basis_type = Cosine
    output_basis_type = Sine
    bands = [0]
    separate = True

    @classmethod
    def entry(cls, i, j, space):
        """Hx(cos(n*x)) = sin(n*x)"""
        if i == j:
            return 1
        else:
            return 0

