"""
Abstract and built-in classes for spectral bases.

"""

import math
import numpy as np
from scipy import sparse

from . import operators
from ..tools.array import axslice
from ..tools.array import apply_matrix
from ..tools.cache import CachedAttribute
from ..tools.cache import CachedMethod
from ..tools.cache import CachedClass
from ..tools import jacobi
from ..tools import clenshaw

from .spaces import FiniteInterval, PeriodicInterval, ParityInterval, Sphere, Ball, Disk
from . import transforms

import logging
logger = logging.getLogger(__name__.split('.')[-1])

from ..tools.config import config
DEFAULT_LIBRARY = config['transforms'].get('DEFAULT_LIBRARY')
DEFAULT_LIBRARY = 'scipy'


class Basis:
    """Base class for spectral bases."""

    def __init__(self, space, library=DEFAULT_LIBRARY):
        self._check_space(space)
        self.space = space
        self.library = library

    # def __repr__(self):
    #     return '<%s %i>' %(self.__class__.__name__, id(self))

    # def __str__(self):
    #     return '%s.%s' %(self.space.name, self.__class__.__name__)

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

    @classmethod
    def _check_space(cls, space):
        if not isinstance(space, cls.space_type):
            raise ValueError("Invalid space type.")

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

    # @CachedAttribute
    # def n_modes(self):
    #     return self.modes.size

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

    def ncc_matrix(self, arg_basis, coeffs, cutoff=1e-6):
        """Build NCC matrix via direct summation."""
        N = len(coeffs)
        for i in range(N):
            coeff = coeffs[i]
            # Build initial matrix
            if i == 0:
                matrix = self.product_matrix(arg_basis, i)
                total = 0 * sparse.kron(matrix, coeff)
                total.eliminate_zeros()
            if len(coeff.shape) or (abs(coeff) > cutoff):
                matrix = self.product_matrix(arg_basis, i)
                total = total + sparse.kron(matrix, coeff)
        return total

    def product_matrix(self, arg_basis, i):
        if arg_basis is None:
            N = self.space.coeff_size
            return sparse.coo_matrix(([1],([i],[0])), shape=(N,1)).tocsr()
        else:
            raise NotImplementedError()


class Constant(Basis, metaclass=CachedClass):
    """Constant basis."""

    def __add__(self, other):
        if other is self:
            return self
        else:
            return NotImplemented

    def __mul__(self, other):
        if other is self:
            return self
        else:
            return NotImplemented


class Jacobi(Basis, metaclass=CachedClass):
    """Jacobi polynomial basis."""

    space_type = FiniteInterval
    dim = 1

    def __init__(self, space, da, db, library='matrix'):
        super().__init__(space, library=library)
        self.da = da
        self.db = db
        self.a = space.a + da
        self.b = space.b + db
        self.const = 1 / np.sqrt(jacobi.mass(self.a, self.b))
        self.axis = self.space.axis

    def __str__(self):
        space = self.space
        cls = self.__class__
        return '%s.%s(%s,%s)' %(space.name, cls.__name__, self.a, self.b)

    def __add__(self, other):
        if other is None:
            return self
        elif self.space is other.space:
            # Add in highest {a,b} basis
            da = max(self.da, other.da)
            db = max(self.db, other.db)
            return Jacobi(self.space, da, db)
        else:
            return NotImplemented

    def __mul__(self, other):
        if other is None:
            return self
        elif self.space is other.space:
            # Put product in highest {a,b} basis
            da = max(self.da, other.da)
            db = max(self.db, other.db)
            return Jacobi(self.space, da, db)
        else:
            return NotImplemented

    def include_mode(self, mode):
        return (0 <= mode < self.space.coeff_size)

    def ncc_matrix(self, arg_basis, coeffs, cutoff=1e-6):
        """Build NCC matrix via Clenshaw algorithm."""
        if arg_basis is None:
            return super().ncc_matrix(arg_basis, coeffs)
        # Kronecker Clenshaw on argument Jacobi matrix
        N = self.space.coeff_size
        J = jacobi.jacobi_matrix(N, arg_basis.a, arg_basis.b)
        A, B = clenshaw.jacobi_recursion(N, self.a, self.b, J)
        f0 = self.const * sparse.identity(N)
        total = clenshaw.kronecker_clenshaw(coeffs, A, B, f0, cutoff=cutoff)
        # Conversion matrix
        input_basis = arg_basis
        output_basis = (self * arg_basis)
        conversion = ConvertJacobiJacobi._subspace_matrix(self.space, input_basis, output_basis)
        # Kronecker with identity for matrix coefficients
        coeff_size = total.shape[0] // conversion.shape[0]
        if coeff_size > 1:
            conversion = sparse.kron(conversion, sparse.identity(coeff_size))
        return (conversion @ total)

    def forward_transform(self, field, axis, gdata, cdata):
        # Transform is the same for all components
        data_axis = len(field.tensorsig) + axis
        transforms.forward_jacobi(gdata, cdata, axis=data_axis, a0=self.space.a, b0=self.space.b, a=self.a, b=self.b)

    def backward_transform(self, field, axis, cdata, gdata):
        # Transform is the same for all components
        data_axis = len(field.tensorsig) + axis
        transforms.backward_jacobi(cdata, gdata, axis=data_axis, a0=self.space.a, b0=self.space.b, a=self.a, b=self.b)


class ConvertJacobiJacobi(operators.Convert):
    """Jacobi polynomial conversion."""

    input_basis_type = Jacobi
    output_basis_type = Jacobi
    separable = False

    @staticmethod
    def _subspace_matrix(space, input_basis, output_basis):
        N = space.coeff_size
        a0, b0 = input_basis.a, input_basis.b
        a1, b1 = output_basis.a, output_basis.b
        matrix = jacobi.conversion_matrix(N, a0, b0, a1, b1)
        return matrix.tocsr()


class DifferentiateJacobi(operators.Differentiate):
    """Jacobi polynomial differentiation."""

    input_basis_type = Jacobi
    separable = False

    @staticmethod
    def output_basis(space, input_basis):
        da, db = input_basis.da, input_basis.db
        return Jacobi(space, da+1, db+1)

    @staticmethod
    def _subspace_matrix(space, input_basis):
        N = space.coeff_size
        a, b = input_basis.a, input_basis.b
        matrix = jacobi.differentiation_matrix(N, a, b)
        return (matrix.tocsr() / space.COV.stretch)


class InterpolateJacobi(operators.Interpolate):
    """Jacobi polynomial interpolation."""

    input_basis_type = Jacobi

    @staticmethod
    def _subspace_matrix(space, input_basis, position):
        N = space.coeff_size
        a, b = input_basis.a, input_basis.b
        x = space.COV.native_coord(position)
        return jacobi.interpolation_vector(N, a, b, x)


class IntegrateJacobi(operators.Integrate):
    """Jacobi polynomial integration."""

    input_basis_type = Jacobi

    @staticmethod
    def _subspace_matrix(space, input_basis):
        N = space.coeff_size
        a, b = input_basis.a, input_basis.b
        vector = jacobi.integration_vector(N, a, b)
        return (vector * space.COV.stretch)


class Fourier(Basis, metaclass=CachedClass):
    """Fourier cosine/sine basis."""
    space_type = PeriodicInterval
    const = 1

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

    @CachedAttribute
    def wavenumbers(self):
        kmax = self.space.kmax
        return np.concatenate((np.arange(0, kmax+1), np.arange(-kmax, 0)))

    def include_mode(self, mode):
        k = mode // 2
        if (mode % 2) == 0:
            # Cosine modes: drop Nyquist mode
            return (0 <= k <= self.space.kmax)
        else:
            # Sine modes: drop k=0 and Nyquist mode
            return (1 <= k <= self.space.kmax)


class ComplexFourier(Basis, metaclass=CachedClass):
    """Fourier complex exponential basis."""
    space_type = PeriodicInterval
    const = 1

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
        k = mode // 2
        if (mode % 2) == 0:
            # Cosine modes: drop Nyquist mode
            return (0 <= k <= self.space.kmax)
        else:
            # Sine modes: drop k=0 and Nyquist mode
            return (1 <= k <= self.space.kmax)


class InterpolateFourier(operators.Interpolate):
    """Fourier series interpolation."""

    input_basis_type = Fourier

    @staticmethod
    def _build_subspace_entry(j, space, input_basis, position):
        # cos(n*x)
        # sin(n*x)
        n = j // 2
        x = space.COV.native_coord(position)
        if (j % 2) == 0:
            return math.cos(n*x)
        else:
            return math.sin(n*x)


class IntegrateFourier(operators.Integrate):
    """Fourier series integration."""

    input_basis_type = Fourier

    @staticmethod
    def _build_subspace_entry(j, space, input_basis):
        # integral(cos(n*x), 0, 2*pi) = 2 * pi * δ(n, 0)
        # integral(sin(n*x), 0, 2*pi) = 0
        if j == 0:
            return 2 * np.pi * space.COV.stretch
        else:
            return 0


class DifferentiateFourier(operators.Differentiate):
    """Fourier series differentiation."""

    input_basis_type = Fourier
    bands = [-1, 1]
    separable = True

    @staticmethod
    def output_basis(space, input_basis):
        return space.Fourier

    @staticmethod
    def _build_subspace_entry(i, j, space, input_basis):
        # dx(cos(n*x)) = -n*sin(n*x)
        # dx(sin(n*x)) = n*cos(n*x)
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
    """Fourier series Hilbert transform."""

    input_basis_type = Fourier
    bands = [-1, 1]
    separable = True

    @staticmethod
    def output_basis(space, input_basis):
        return space.Fourier

    @staticmethod
    def _build_subspace_entry(i, j, space, input_basis):
        # Hx(cos(n*x)) = sin(n*x)
        # Hx(sin(n*x)) = -cos(n*x)
        n = j // 2
        if n == 0:
            return 0
        elif (j % 2) == 0:
            # Hx(cos(n*x)) = sin(n*x)
            if i == (j + 1):
                return 1
            else:
                return 0
        else:
            # Hx(sin(n*x)) = -cos(n*x)
            if i == (j - 1):
                return (-1)
            else:
                return 0


class Sine(Basis, metaclass=CachedClass):
    """Sine series basis."""
    space_type = ParityInterval
    const = None
    supported_dtypes = {np.float64, np.complex128}

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

    def include_mode(self, mode):
        # Drop k=0 and Nyquist mode
        k = mode
        return (1 <= k <= self.space.kmax)


class Cosine(Basis, metaclass=CachedClass):
    """Cosine series basis."""
    space_type = ParityInterval
    const = 1

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

    def include_mode(self, mode):
        # Drop Nyquist mode
        k = mode
        return (0 <= k <= self.space.kmax)


class InterpolateSine(operators.Interpolate):
    """Sine series interpolation."""

    input_basis_type = Sine

    @staticmethod
    def _build_subspace_entry(j, space, input_basis, position):
        # sin(n*x)
        x = space.COV.native_coord(position)
        return math.sin(j*x)


class InterpolateCosine(operators.Interpolate):
    """Cosine series interpolation."""

    input_basis_type = Cosine

    @staticmethod
    def _build_subspace_entry(j, space, input_basis, position):
        # cos(n*x)
        x = space.COV.native_coord(position)
        return math.cos(j*x)


class IntegrateSine(operators.Integrate):
    """Sine series integration."""

    input_basis_type = Sine

    @staticmethod
    def _build_subspace_entry(j, space, input_basis):
        # integral(sin(n*x), 0, pi) = (2 / n) * (n % 2)
        if (j % 2):
            return 0
        else:
            return (2 / j) * space.COV.stretch


class IntegrateCosine(operators.Integrate):
    """Cosine series integration."""

    input_basis_type = Cosine

    @staticmethod
    def _build_subspace_entry(j, space, input_basis):
        # integral(cos(n*x), 0, pi) = pi * δ(n, 0)
        if j == 0:
            return np.pi * space.COV.stretch
        else:
            return 0


class DifferentiateSine(operators.Differentiate):
    """Sine series differentiation."""

    input_basis_type = Sine
    bands = [0]
    separable = True

    @staticmethod
    def output_basis(space, input_basis):
        return space.Cosine

    @staticmethod
    def _build_subspace_entry(i, j, space, input_basis):
        # dx(sin(n*x)) = n*cos(n*x)
        if i == j:
            return j / space.COV.stretch
        else:
            return 0


class DifferentiateCosine(operators.Differentiate):
    """Cosine series differentiation."""

    input_basis_type = Cosine
    bands = [0]
    separable = True

    @staticmethod
    def output_basis(space, input_basis):
        return space.Sine

    @staticmethod
    def _build_subspace_entry(i, j, space, input_basis):
        # dx(cos(n*x)) = -n*sin(n*x)
        if i == j:
            return (-j) / space.COV.stretch
        else:
            return 0


class HilbertTransformSine(operators.HilbertTransform):
    """Sine series Hilbert transform."""

    input_basis_type = Sine
    bands = [0]
    separable = True

    @staticmethod
    def output_basis(space, input_basis):
        return space.Cosine

    @staticmethod
    def _build_subspace_entry(i, j, space, input_basis):
        # Hx(sin(n*x)) = -cos(n*x)
        if i == j:
            return (-1)
        else:
            return 0


class HilbertTransformCosine(operators.HilbertTransform):
    """Cosine series Hilbert transform."""

    input_basis_type = Cosine
    bands = [0]
    separable = True

    @staticmethod
    def output_basis(space, input_basis):
        return space.Sine

    @staticmethod
    def _build_subspace_entry(i, j, space, input_basis):
        # Hx(cos(n*x)) = sin(n*x)
        if i == j:
            return 1
        else:
            return 0


class MultidimensionalBasis(Basis):

    def forward_transform(self, field, axis, gdata, cdata):
        subaxis = axis - self.axis
        return self.forward_transforms[subaxis](field, axis, gdata, cdata)

    def backward_transform(self, field, axis, cdata, gdata):
        subaxis = axis - self.axis
        return self.backward_transforms[subaxis](field, axis, cdata, gdata)


class SpinBasis(MultidimensionalBasis):

    dim = 2

    @CachedAttribute
    def local_m(self):
        domain = self.space.domain
        layout = self.space.dist.coeff_layout
        local_m_elements = layout.local_elements(domain, scales=1)[self.axis]
        print(self.azimuth_basis.wavenumbers)
        print(local_m_elements)
        return self.azimuth_basis.wavenumbers[local_m_elements]

    def forward_transform_azimuth(self, field, axis, gdata, cdata):
        # Azimuthal DFT is the same for all components
        data_axis = len(field.tensorsig) + axis
        transforms.forward_DFT(gdata, cdata, axis=data_axis)

    def backward_transform_azimuth(self, field, axis, cdata, gdata):
        # Azimuthal DFT is the same for all components
        data_axis = len(field.tensorsig) + axis
        transforms.backward_DFT(cdata, gdata, axis=data_axis)

    def spin_weights(self, tensorsig):
        # Spin-component ordering: [-, +]
        Ss = np.array([-1, 1], dtype=int)
        S = np.zeros([vs.dim for vs in tensorsig], dtype=int)
        for i, vs in enumerate(tensorsig):
            if self.space in vs.spaces:
                n = vs.get_index(self.space)
                S[axslice(i, n, n+self.dim)] += Ss
        return S

    def spin_recombination(self, tensorsig):
        """Build matrices for appling spin recombination to each tensor rank."""
        # Setup unitary spin recombination
        # [azimuth, colatitude] -> [-, +]
        Us = np.array([[-1j, 1], [1j, 1]]) / np.sqrt(2)
        # Perform unitary spin recombination along relevant tensor indeces
        U = []
        for i, vector_space in enumerate(tensorsig):
            if self.space in vector_space.spaces:
                n = vector_space.get_index(self.space)
                Ui = np.identity(vector_space.dim)
                Ui[n:n+self.dim, n:n+self.dim] = Us
                U.append(Ui)
            else:
                U.append(None)
        return U

    def forward_spin_recombination(self, tensorsig, gdata):
        """Apply component-to-spin recombination in place."""
        U = self.spin_recombination_matrices(tensorsig)
        for i, Ui in enumerate(U):
            if Ui is not None:
                # Directly apply U
                apply_matrix(Ui, gdata, axis=i, out=gdata)

    def backward_spin_recombination(self, tensorsig, gdata):
        """Apply spin-to-component recombination in place."""
        U = self.spin_recombination_matrices(tensorsig)
        for i, Ui in enumerate(U):
            if Ui is not None:
                # Apply U^H (inverse of U)
                apply_matrix(Ui.T.conj(), gdata, axis=i, out=gdata)


class DiskBasis(SpinBasis):

    space_type = Disk
    dim = 2

    def __init__(self, space, dk=0):
        self._check_space(space)
        self.space = space
        self.dk = dk
        self.k = space.k0 + dk
        self.axis = space.axis
        self.azimuth_basis = Fourier(self.space.azimuth_space)
        self.forward_transforms = [self.forward_transform_azimuth,
                                   self.forward_transform_radius]
        self.backward_transforms = [self.backward_transform_azimuth,
                                    self.backward_transform_radius]
        #self.forward_transform_azimuth = self.azimuth_basis.forward_transform
        #self.backward_transform_azimuth = self.azimuth_basis.backward_transform

    def forward_transform_radius(self, field, axis, gdata, cdata):
        # Apply spin recombination
        self.forward_spin_recombination(field.tensorsig, gdata)
        # Perform transforms component-by-component
        S = self.spin_weights(field.tensorsig)
        k0, k = self.k0, self.k
        local_m = self.local_m
        for i, s in np.ndenumerate(S):
            transforms.forward_disk(gdata[i], cdata[i], axis=axis, k0=k0, k=k, s=s, local_m=local_m)

    def backward_transform_radius(self, field, axis, cdata, gdata):
        # Perform transforms component-by-component
        S = self.spin_weights(field.tensorsig)
        k0, k = self.k0, self.k
        local_m = self.local_m
        for i, s in np.ndenumerate(S):
            transforms.backward_disk(cdata[i], gdata[i], axis=axis, k0=k0, k=k, s=s, local_m=local_m)
        # Apply spin recombination
        self.backward_spin_recombination(field.tensorsig, gdata)


class SpinWeightedSphericalHarmonics(SpinBasis):

    space_type = Sphere
    dim = 2

    def __init__(self, space):
        self._check_space(space)
        self.space = space
        self.axis = space.axis
        self.azimuth_basis = Fourier(self.space.azimuth_space)
        self.forward_transforms = [self.forward_transform_azimuth,
                                   self.forward_transform_colatitude]
        self.backward_transforms = [self.backward_transform_azimuth,
                                    self.backward_transform_colatitude]
        #self.forward_transform_azimuth = self.azimuth_basis.forward_transform
        #self.backward_transform_azimuth = self.azimuth_basis.backward_transform

    def forward_transform_radius(self, field, axis, gdata, cdata):
        # Apply spin recombination
        self.forward_spin_recombination(field.tensorsig, gdata)
        # Perform transforms component-by-component
        S = self.spin_weights(field.tensorsig)
        for i, s in np.ndenumerate(S):
            transforms.forward_disk(gdata[i], cdata[i], axis=axis, local_m=self.local_m, s=s)

    def backward_transform_radius(self, field, axis, cdata, gdata):
        # Perform transforms component-by-component
        S = self.spin_weights(field.tensorsig)
        for i, s in np.ndenumerate(S):
            transforms.backward_disk(cdata[i], gdata[i], axis=axis, local_m=self.local_m, s=s)
        # Apply spin recombination
        self.backward_spin_recombination(field.tensorsig, gdata)


SWSH = SpinWeightedSphericalHarmonics


class BallBasis(MultidimensionalBasis):

    space_type = Ball
    dim = 3

    def __init__(self, space):
        self._check_space(space)
        self.space = space
        self.axis = space.axis
        self.outer_shere_basis = SWSH(self.space.outer_sphere_space)
        self.forward_transforms = [self.forward_transform_azimuth,
                                   self.forward_transform_colatitude,
                                   self.forward_transform_radius]
        self.backward_transforms = [self.backward_transform_azimuth,
                                    self.backward_transform_colatitude,
                                    self.backward_transform_radius]
        #self.forward_transform_azimuth = self.azimuth_basis.forward_transform
        #self.backward_transform_azimuth = self.azimuth_basis.backward_transform

    @CachedAttribute
    def local_l(self):
        domain = self.space.domain
        layout = self.space.dist.coeff_layout
        return layout.local_elements(domain, scales=1)[self.axis+1]

    def forward_transform_azimuth(self, *args):
        return self.outer_shere_basis.forward_transform_azimuth(*args)

    def backward_transform_azimuth(self, *args):
        return self.outer_shere_basis.backward_transform_azimuth(*args)

    def forward_transform_colatitude(self, *args):
        return self.outer_shere_basis.forward_transform_colatitude(*args)

    def backward_transform_colatitude(self, *args):
        return self.outer_shere_basis.backward_transform_colatitude(*args)

    def radial_recombinations(self, tensorsig):
        import dedalus_sphere
        # For now only implement recombinations for Ball-only tensors
        for vs in tensorsig:
            for space in vs.spaces:
                if space is not self.space:
                    raise ValueError("Only supports tensors over ball.")
        order = len(tensorsig)
        logger.warning("Q orders not fixed")
        return [dedalus_sphere.ball128.Q(l, order) for l in self.local_l]

    def regularity_classes(self, tensorsig):
        # Regularity-component ordering: [-, +, 0]
        Rb = np.array([-1, 1, 0], dtype=int)
        R = np.zeros([vs.dim for vs in tensorsig], dtype=int)
        for i, vs in enumerate(tensorsig):
            if self.space in vs.spaces:
                n = vs.get_index(self.space)
                R[axslice(i, n, n+self.dim)] += Rb
        return R

    def forward_transform_radius(self, field, axis, gdata, cdata):
        # Apply radial recombinations
        order = len(field.tensorsig)
        if order > 0:
            Q = self.radial_recombinations(field.tensorsig)
            # Flatten tensor axes
            shape = gdata.shape
            order = len(field.tensorsig)
            temp = gdata.reshape((-1,)+shape[order:])
            # Apply Q transformations for each l to flattened tensor data
            for l_index, Q_l in enumerate(Q):
                # Here the l axis is 'axis' instead of 'axis-1' since we have one tensor axis prepended
                l_view = temp[axslice(axis, l_index, l_index+1)]
                apply_matrix(Q_l.T, l_view, axis=0, out=l_view)
        # Perform radial transforms component-by-component
        R = self.regularity_classes(field.tensorsig)
        for i, r in np.ndenumerate(R):
           transforms.forward_GSZP(gdata[i], cdata[i], axis=axis, local_l=self.local_l, r=r, alpha=self.space.alpha)

    def backward_transform_radius(self, field, axis, cdata, gdata):
        # Perform radial transforms component-by-component
        R = self.regularity_classes(field.tensorsig)
        for i, r in np.ndenumerate(R):
           transforms.backward_GSZP(cdata[i], gdata[i], axis=axis, local_l=self.local_l, r=r, alpha=self.space.alpha)
        # Apply radial recombinations
        order = len(field.tensorsig)
        if order > 0:
            Q = self.radial_recombinations(field.tensorsig)
            # Flatten tensor axes
            shape = gdata.shape
            order = len(field.tensorsig)
            temp = gdata.reshape((-1,)+shape[order:])
            # Apply Q transformations for each l to flattened tensor data
            for l_index, Q_l in enumerate(Q):
                # Here the l axis is 'axis' instead of 'axis-1' since we have one tensor axis prepended
                l_view = temp[axslice(axis, l_index, l_index+1)]
                apply_matrix(Q_l, l_view, axis=0, out=l_view)


