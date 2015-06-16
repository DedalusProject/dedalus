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

import logging
logger = logging.getLogger(__name__.split('.')[-1])

from ..tools.config import config
DEFAULT_LIBRARY = config['transforms'].get('DEFAULT_LIBRARY')


class Basis:
    """Base class for spectral bases."""

    def __init__(self, space):
        self.space = space
        self.domain = space.domain
        self.library = DEFAULT_LIBRARY

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
    def transform_plan(self, coeff_shape, dtype, axis, scale):
        """Build and cache transform plan."""
        transform_class = self.transforms[self._library]
        return transform_class(coeff_shape, dtype, axis, scale)

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


class ChebyshevT(Basis):
    """Chebyshev polynomial basis on the roots grid."""

    element_label = 'T'

    def __init__(self, space):
        super().__init__(space)
        #self.modes = np.arange(self.space.coeff_size)

    def include_mode(self, mode):
        return (0 <= mode < self.space.coeff_size)

    # def group_size(self, group):
    #     if (0 <= group < self.space.coeff_size):
    #         return 1
    #     else:
    #         return 0

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

    @CachedAttribute
    def Integrate(self):
        """Build integration class."""

        class IntegrateChebyshevT(operators.Integrate):
            name = 'integ_{}'.format(self.space.name)
            input_basis = self

            @classmethod
            def entry(cls, mode):
                """Integral(T_n(x), -1, 1) = (1 + (-1)**n) / (1 - n**2)"""
                n = mode
                if (n % 2):
                    return 0
                else:
                    return 2 / (1 - n**2)

        return IntegrateChebyshevT

    @CachedAttribute
    def Interpolate(self):
        """Buld interpolation class."""

        class InterpolateChebyshev(operators.Interpolate):
            name = 'interp_{}'.format(self.space.name)
            input_basis = self

            @classmethod
            def entry(cls, mode, position):
                """Tn(x) = cos(n*acos(x))"""
                n = mode
                x = cls.input_basis.space.COV.native_coord(position)
                theta = math.acos(x)
                return math.cos(n*theta)

        return InterpolateChebyshev

    @CachedAttribute
    def Differentiate(self):
        """Build differentiation class."""

        class DifferentiateChebyshev(operators.Differentiate):
            name = 'd' + self.space.name
            input_basis = self
            output_basis = self.space.ChebyshevU
            axis = self.space.axis
            bands = [0, 1, 2]
            separable = False

            @classmethod
            def entry(cls, mode_out, mode_in):
                """dx(T_n) = n * U_{n-1}"""
                n = mode_in
                if mode_out == (mode_in - 1):
                    return n
                else:
                    return 0

        return DifferentiateChebyshev

    @CachedAttribute
    def Convert(self):
        """Build conversion class."""

        class ConvertChebyshevT(operators.Convert):
            input_basis = self
            output_basis = self.space.ChebyshevU
            axis = self.space.axis
            bands = [0, 2]
            separable = False

            @classmethod
            def entry(cls, mode_out, mode_in):
                """
                T_n = (U_n - U_(n-2)) / 2
                U_(-n) = -U_(n-2)
                """
                if mode_in == mode_out:
                    if mode_in == 0:
                        return 1
                    else:
                        return 0.5
                elif mode_out == (mode_in - 2):
                    return -0.5

        return ConvertChebyshevT

    @CachedAttribute
    def Filter(self):

        class FilterChebyshevT(operators.Filter):
            input_basis = self

        return FilterChebyshevT

    def Multiply(self, p):
        """
        p-element multiplication matrix

        T_p * T_n = (T_(n+p) + T_(n-p)) / 2
        T_(-n) = T_n

        """
        size = self.space.coeff_size
        # Construct sparse matrix
        Mult = sparse.lil_matrix((size, size), dtype=self.space.domain.dtype)
        for n in range(size):
            upper = n + p
            if upper < size:
                Mult[upper, n] += 0.5
            lower = abs(n - p)
            if lower < size:
                Mult[lower, n] += 0.5
        return Mult.tocsr()


class ChebyshevU(Basis):
    """Chebyshev polynomial basis on the roots grid."""

    element_label = 'U'

    def __init__(self, space):
        super().__init__(space)
        #self.modes = np.arange(self.space.coeff_size)

    def include_mode(self, mode):
        return (0 <= mode < self.space.coeff_size)

    # def group_size(self, group):
    #     if (0 <= group < self.space.coeff_size):
    #         return 1
    #     else:
    #         return 0

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
            return space.ChebyshevU
        elif other is space.ChebyshevT:
            return space.ChebyshevT
        else:
            return NotImplemented

    def __pow__(self, other):
        return self.space.ChebyshevU

    @CachedAttribute
    def Integrate(self):
        """Build integration class."""

        class IntegrateChebyshevU(operators.Integrate):
            name = 'integ_{}'.format(self.space.name)
            input_basis = self

            @classmethod
            def entry(cls, mode):
                """Integral(U_n(x), -1, 1) = (1 + (-1)**n) / (1 + n)"""
                n = mode
                if (n % 2):
                    return 0
                else:
                    return 2 / (1 + n)

        return IntegrateChebyshevU

    @CachedAttribute
    def Interpolate(self):
        """Buld interpolation class."""

        class InterpolateChebyshev(operators.Interpolate):
            name = 'interp_{}'.format(self.space.name)
            input_basis = self

            @classmethod
            def entry(cls, mode, position):
                """Un(x) = sin((n+1)*acos(x)) / sin(acos(x))"""
                n = mode
                x = cls.input_basis.space.COV.native_coord(position)
                theta = math.acos(x)
                if theta == 0:
                    return (n + 1)
                elif theta == np.pi:
                    return (n + 1) * (-1)**n
                else:
                    return math.sin((n+1)*theta) / math.sin(theta)

        return InterpolateChebyshev

    @CachedAttribute
    def Filter(self):

        class FilterChebyshevU(operators.Filter):
            input_basis = self

        return FilterChebyshevU

    def Multiply(self, p):
        """
        p-element multiplication matrix

        U_p * T_n = (U_(p+n) + U_(p-n)) / 2
        U_(-n) = -U_(n-2)

        """
        size = self.space.coeff_size
        # Construct sparse matrix
        Mult = sparse.lil_matrix((size, size), dtype=self.space.domain.dtype)
        for n in range(size):
            upper = p + n
            if upper < size:
                Mult[upper, n] += 0.5
            lower = p - n
            if lower >= 0:
                Mult[lower, n] += 0.5
            elif lower < -1:
                Mult[-lower-2, n] += 0.5
        return Mult.tocsr()


class Fourier(Basis):
    """Fourier sine/cosine series basis."""

    element_label= 'k'

    def __init__(self, space):
        super().__init__(space)
        #self.modes = np.concatenate(([0], np.arange(2, self.space.coeff_size)))

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

    def include_mode(self, mode):
        if mode == 1:
            return False
        else:
            k = mode // 2
            return (0 <= k <= self.space.kmax)

    # def group_size(self, group):
    #     if group == 0:
    #         return 1
    #     if 0 < group < self.space.kmax:
    #         return 2
    #     else:
    #         return 0

    @CachedAttribute
    def Integrate(self):
        """Build integration class."""

        class IntegrateFourier(operators.Integrate, operators.Separable):
            name = 'integ_{}'.format(self.space.name)
            input_basis = self

            @classmethod
            def entry(cls, mode):
                if mode == 0:
                    return 2*np.pi
                else:
                    return 0

        return IntegrateFourier

    @CachedAttribute
    def Interpolate(self):
        """Build interpolation class."""

        class InterpolateFourier(operators.Interpolate, operators.Coupled):
            name = 'interp_{}'.format(self.space.name)
            input_basis = self

            @classmethod
            def entry(cls, mode, position):
                n = mode // 2
                x = cls.input_basis.space.COV.native_coord(position)
                if (mode % 2) == 0:
                    return math.cos(n*x)
                else:
                    return math.sin(n*x)

        return InterpolateFourier

    @CachedAttribute
    def Differentiate(self):
        """Build differentiation class."""

        class DifferentiateFourier(operators.Differentiate):
            name = 'd' + self.space.name
            input_basis = self
            output_basis = self
            bands = [-1, 1]
            separable = True

            @classmethod
            def entry(cls, mode_out, mode_in):
                n = mode_in // 2
                if mode_in == 0:
                    return 0
                elif (mode_in % 2) == 0:
                    # dx(cos(n*x)) = -n*sin(n*x)
                    if mode_out == (mode_in + 1):
                        return (-n)
                    else:
                        return 0
                else:
                    # dx(sin(n*x)) = n*cos(n*x)
                    if mode_out == (mode_in - 1):
                        return n
                    else:
                        return 0

        return DifferentiateFourier

    @CachedAttribute
    def HilbertTransform(self):
        """Build Hilbert transform class."""

        class HilbertTransformFourier(operators.HilbertTransform):
            name = 'H' + self.space.name
            input_basis = self
            output_basis = self
            bands = [-1, 1]
            separable = True

            @classmethod
            def entry(cls, mode_out, mode_in):
                if mode_in == 0:
                    return 0
                elif (mode_in % 2) == 0:
                    # Hx(cos(n*x)) = sin(n*x)
                    if mode_out == (mode_in + 1):
                        return 1
                    else:
                        return 0
                else:
                    # Hx(sin(n*x)) = -cos(n*x)
                    if mode_out == (mode_in - 1):
                        return (-1)
                    else:
                        return 0

        return HilbertTransformFourier


class Sine(Basis):
    """Sine series basis."""

    element_label = 'k'

    def __init__(self, space):
        super().__init__(space)
        #self.modes = np.arange(1, self.space.coeff_size)
        #self.library = DEFAULT_LIBRARY

    def include_mode(self, mode):
        k = mode
        return (1 <= k <= self.space.kmax)

    # def group_size(self, group):
    #     if 1 <= group < self.space.kmax:
    #         return 2
    #     else:
    #         return 0

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

    @CachedAttribute
    def Integrate(self):
        """Build integration class."""

        class IntegrateSine(operators.Integrate):
            name = 'integ_{}'.format(self.space.name)
            input_basis = self

            @classmethod
            def entry(cls, mode):
                """Integral(sin(n*x), 0, pi) = (2 / n) * (n % 2)"""
                n = mode
                if (n % 2):
                    return 0
                else:
                    return (2 / n)

        return IntegrateSine

    @CachedAttribute
    def Interpolate(self):
        """Build interpolation class."""

        class InterpolateSine(operators.Interpolate):
            name = 'interp_{}'.format(self.space.name)
            input_basis = self

            @classmethod
            def entry(self, mode, position):
                """sin(n*x)"""
                n = mode
                x = cls.input_basis.space.COV.native_coord(position)
                return math.sin(n*x)

        return InterpolateSine

    @CachedAttribute
    def Differentiate(self):
        """Build differentiation class."""

        class DifferentiateSine(operators.Differentiate):
            name = 'd' + self.space.name
            input_basis = self
            output_basis = self.space.Cosine
            bands = [0]
            separable = True

            @classmethod
            def entry(cls, mode_out, mode_in):
                """dx(sin(n*x)) = n*cos(n*x)"""
                n = mode_in
                if mode_out == mode_in:
                    return n
                else:
                    return 0

        return DifferentiateSine

    @CachedAttribute
    def HilbertTransform(self):
        """Build Hilbert transform class."""

        class HilbertTransformSine(operators.HilbertTransform):
            name = 'H' + self.space.name
            input_basis = self
            output_basis = self.space.Cosine
            bands = [0]
            separable = True

            @classmethod
            def entry(cls, mode_out, mode_in):
                """Hx(sin(n*x)) = -cos(n*x)"""
                if mode_out == mode_in:
                    return (-1)
                else:
                    return 0

        return HilbertTransformSine


class Cosine(Basis):
    """Cosine series basis."""

    element_label = 'k'

    def __init__(self, space):
        super().__init__(space)
        #self.modes = np.arange(self.space.coeff_size)
        #self.library = DEFAULT_LIBRARY

    def include_mode(self, mode):
        k = mode
        return (0 <= k <= self.space.kmax)

    # def group_size(self, group):
    #     if 0 <= group < self.space.kmax:
    #         return 2
    #     else:
    #         return 0

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

    @CachedAttribute
    def Integrate(self):
        """Build integration class."""

        class IntegrateCosine(operators.Integrate):
            name = 'integ_{}'.format(self.space.name)
            input_basis = self

            @classmethod
            def entry(cls, mode):
                """Integral(cos(n*x), 0, pi) = pi * δ(n, 0)"""
                n = mode
                if n == 0:
                    return np.pi
                else:
                    return 0

        return IntegrateCosine

    @CachedAttribute
    def Interpolate(self):
        """Build interpolation class."""

        class InterpolateCosine(operators.Interpolate):
            name = 'interp_{}'.format(self.space.name)
            input_basis = self

            @classmethod
            def entry(cls, mode, position):
                """cos(n*x)"""
                n = mode
                x = cls.input_basis.space.COV.native_coord(position)
                return math.cos(n*x)

        return InterpolateCosine

    @CachedAttribute
    def Differentiate(self):
        """Build differentiation class."""

        class DifferentiateCosine(operators.Differentiate):
            name = 'd' + self.space.name
            input_basis = self
            output_basis = self.space.Sine
            bands = [0]
            separable = True

            @classmethod
            def entry(cls, mode_out, mode_in):
                """dx(cos(n*x)) = -n*sin(n*x)"""
                n = mode_in
                if mode_out == mode_in:
                    return (-n)
                else:
                    return 0

        return DifferentiateCosine

    @CachedAttribute
    def HilbertTransform(self):
        """Build Hilbert transform class."""

        class HilbertTransformCosine(operators.HilbertTransform):
            name = 'H' + self.space.name
            input_basis = self
            output_basis = self.space.Sine
            bands = [0]
            separate = True

            @classmethod
            def entry(cls, mode_out, mode_in):
                """Hx(cos(n*x)) = sin(n*x)"""
                if mode_out == mode_in:
                    return 1
                else:
                    return 0

        return HilbertTransformCosine

