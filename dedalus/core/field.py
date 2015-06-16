"""
Class for data fields.

"""

import weakref
from functools import partial, reduce
from collections import defaultdict
import numpy as np
from mpi4py import MPI
from scipy import sparse
from scipy.sparse import linalg as splinalg
from numbers import Number


from ..libraries.fftw import fftw_wrappers as fftw
from ..tools.config import config
from ..tools.array import reshape_vector
from ..tools.cache import CachedMethod
from ..tools.exceptions import UndefinedParityError
from ..tools.exceptions import SymbolicParsingError
from ..tools.general import unify

import logging
logger = logging.getLogger(__name__.split('.')[-1])

# Load config options
permc_spec = config['linear algebra']['permc_spec']
use_umfpack = config['linear algebra'].getboolean('use_umfpack')


class Operand:

    def __getattr__(self, attr):
        # Intercept numpy ufunc calls
        from .operators import UnaryGridFunction
        try:
            ufunc = UnaryGridFunction.supported[attr]
            return partial(UnaryGridFunction, ufunc, self)
        except KeyError:
            raise AttributeError("%r object has no attribute %r" %(self.__class__.__name__, attr))

    ## Idea for alternate ufunc implementation based on changes coming in numpy 1.10
    # def __numpy_ufunc__(self, ufunc, method, i, inputs, **kw):
    #     from .operators import UnaryGridFunction
    #     if ufunc in UnaryGridFunction.supported:
    #         return UnaryGridFunction(ufunc, self, **kw)
    #     else:
    #         return NotImplemented

    def __call__(self, *args, **kw):
        """Interpolate field."""
        from .operators import interpolate
        return interpolate(self, *args, **kw)

    def __abs__(self):
        # Call: abs(self)
        from .operators import UnaryGridFunction
        return UnaryGridFunction(np.absolute, self)

    def __neg__(self):
        # Call: -self
        return ((-1) * self)

    def __add__(self, other):
        # Call: self + other
        if other == 0:
            return self
        from .operators import Add
        return Add(self, other)

    def __radd__(self, other):
        if other == 0:
            return self
        # Call: other + self
        from .operators import Add
        return Add(other, self)

    def __sub__(self, other):
        # Call: self - other
        return (self + (-other))

    def __rsub__(self, other):
        # Call: other - self
        return (other + (-self))

    def __mul__(self, other):
        # Call: self * other
        if other == 0:
            return 0
        if other == 1:
            return self
        from .operators import Multiply
        return Multiply(self, other)

    def __rmul__(self, other):
        # Call: other * self
        if other == 0:
            return 0
        if other == 1:
            return self
        from .operators import Multiply
        return Multiply(other, self)

    def __truediv__(self, other):
        # Call: self / other
        return (self * other**(-1))

    def __rtruediv__(self, other):
        # Call: other / self
        return (other * self**(-1))

    def __pow__(self, other):
        # Call: self ** other
        if other == 0:
            return 1
        if other == 1:
            return self
        from .operators import Power
        return Power(self, other)

    def __rpow__(self, other):
        # Call: other ** self
        from .operators import Power
        return Power(other, self)

    @staticmethod
    def parse(string, namespace, domain):
        """Build operand from a string expression."""
        expression = eval(string, namespace)
        return Operand.cast(expression, domain)

    @staticmethod
    def cast(x, domain):
        if isinstance(x, Operand):
            if x.domain is not domain:
                raise ValueError('Wrong domain')
            else:
                return x
        elif isinstance(x, Number):
            out = Field(name=str(x), bases=domain)
            out['c'] = x
            return out

    #     x = Operand.raw_cast(x)
    #     if domain:
    #         # Replace empty domains
    #         if hasattr(x, 'domain'):
    #             if x.domain != domain:
    #                 raise ValueError("Cannot cast operand to different domain.")
    #     return x

    # @staticmethod
    # def raw_cast(x):
    #     if isinstance(x, Operand):
    #         return x
    #     elif isinstance(x, str):
    #         raise ValueError("Cannot directly cast string expressions, only fields/operators/scalars.")
    #     elif np.isscalar(x):
    #         return Scalar(value=x)
    #     else:
    #         raise ValueError("Cannot cast type: {}".format(type(x)))


class Data(Operand):

    __array_priority__ = 100.

    def __repr__(self):
        return '<{} {}>'.format(self.__class__.__name__, id(self))

    def __str__(self):
        if self.name:
            return self.name
        else:
            return self.__repr__()

    def atoms(self, *types, **kw):
        if isinstance(self, types) or (not types):
            return (self,)
        else:
            return ()

    def has(self, *atoms):
        return (self in atoms)

    def expand(self, *vars):
        """Return self."""
        return self

    def canonical_linear_form(self, *vars):
        """Return self."""
        return self

    def factor(self, *vars):
        if self in vars:
            return defaultdict(int, {self: 1})
        else:
            return defaultdict(int, {1: self})

    def split(self, *vars):
        if self in vars:
            return [self, 0]
        else:
            return [0, self]

    def replace(self, old, new):
        """Replace an object in the expression tree."""
        if self == old:
            return new
        else:
            return self

    def order(self, *ops):
        return 0

    def operator_dict(self, subsystem, vars, **kw):
        if self in vars:
            if subsystem.size(self.bases) == 0:
                return {}
            else:
                return {self: self.subsystem_matrix(subsystem)}
        else:
            raise SymbolicParsingError('{} is not one of the specified variables.'.format(str(self)))

    def subsystem_matrix(self, subsystem):
        axmats = subsystem.compute_identities(self.bases)
        return reduce(sparse.kron, axmats, 1).tocsr()

    def separability(self, vars):
        return np.array([True for basis in self.bases])


    def sym_diff(self, var):
        """Symbolically differentiate with respect to var."""
        if self == var:
            return 1
        else:
            return 0

    @staticmethod
    def _create_buffer(buffer_size):
        """Create buffer for Field data."""
        if buffer_size == 0:
            # FFTW doesn't like allocating size-0 arrays
            return np.zeros((0,), dtype=np.float64)
        else:
            # Use FFTW SIMD aligned allocation
            alloc_doubles = buffer_size // 8
            return fftw.create_buffer(alloc_doubles)

    def set_scales(self, scales, keep_data=True):
        """Set new transform scales."""
        new_scales = self.domain.remedy_scales(scales)
        old_scales = self.scales
        if new_scales == old_scales:
            return
        # Set metadata
        self.scales = new_scales
        # Build new buffer
        buffer_size = self.domain.distributor.buffer_size(self.subdomain, new_scales)
        self.buffer = self._create_buffer(buffer_size)
        # Reset layout to build new data view
        self.set_layout(self.layout)

    def set_layout(self, layout):
        """Interpret buffer as data in specified layout."""
        layout = self.domain.dist.get_layout_object(layout)
        self.layout = layout
        local_shape = layout.local_shape(self.subdomain, self.scales)
        self.data = np.ndarray(shape=local_shape,
                               dtype=self.domain.dtype,
                               buffer=self.buffer)
        self.global_start = layout.start(self.subdomain, self.scales)


class Array(Data):

    def __init__(self, bases, name=None):
        from .domain import Subdomain
        self.subdomain, self.bases = Subdomain.from_bases(bases)
        self.domain = self.subdomain.domain
        self.name = name
        # Set initial scales and layout
        self.scales = None
        self.layout = self.domain.dist.grid_layout
        # Change scales to build buffer and data
        self.set_scales(self.subdomain.dealias)

    def set_global_data(self, global_data):
        slices = self.layout.slices(self.subdomain, self.scales)
        self.set_local_data(global_data[slices])

    def set_local_data(self, local_data):
        np.copyto(self.data, local_data)

    def require_scales(self, scales):
        scales = self.domain.remedy_scales(scales)
        if scales != self.scales:
            raise ValueError("Cannot change array scales.")

    def require_layout(self, layout):
        layout = self.domain.dist.get_layout.object(layout)
        if layout != self.layout:
            raise ValueError("Cannot change array layout.")

    @CachedMethod(max_size=1)
    def as_ncc_operator(self, *args, **kw):
        """Cast to field and convert to NCC operator."""
        from .future import FutureField
        ncc = FutureField.cast(self, self.domain)
        ncc = ncc.evaluate()
        if 'name' not in kw:
            kw['name'] = str(self)
        return ncc.as_ncc_operator(*args, **kw)


class Field(Data):
    """
    Scalar field over a domain.

    Parameters
    ----------
    domain : domain object
        Problem domain
    name : str, optional
        Field name (default: Python object id)

    Attributes
    ----------
    layout : layout object
        Current layout of field
    data : ndarray
        View of internal buffer in current layout

    """

    def __init__(self, bases, name=None, layout='c', scales=1):
        from .domain import Subdomain
        self.subdomain, self.bases = Subdomain.from_bases(bases)
        self.domain = self.subdomain.domain
        self.name = name
        # Set initial scales and layout
        self.scales = None
        self.layout = self.domain.dist.get_layout_object(layout)
        # Change scales to build buffer and data
        self.set_scales(scales)

    def __getitem__(self, layout):
        """Return data viewed in specified layout."""
        self.require_layout(layout)
        return self.data

    def __setitem__(self, layout, data):
        """Set data viewed in a specified layout."""
        layout = self.domain.distributor.get_layout_object(layout)
        self.set_layout(layout)
        np.copyto(self.data, data)

    @property
    def global_shape(self):
        return self.layout.global_shape(self.subdomain, self.scales)

    @property
    def dtype(self):
        return self.domain.dtype

    def set_global_data(self, global_data):
        slices = self.layout.slices(self.subdomain, self.scales)
        self.set_local_data(global_data[slices])

    def set_local_data(self, local_data):
        np.copyto(self.data, local_data)

    def require_scales(self, scales):
        """Change data to specified scales."""
        # Remedy scales
        new_scales = self.domain.remedy_scales(scales)
        old_scales = self.scales
        # Quit if new scales aren't new
        if new_scales == old_scales:
            return
        # Forward transform until remaining scales match
        for axis in reversed(range(self.domain.dim)):
            if not self.layout.grid_space[axis]:
                break
            if old_scales[axis] != new_scales[axis]:
                self.require_coeff_space(axis)
                break
        # Copy over scale change
        old_data = self.data
        self.set_scales(scales)
        np.copyto(self.data, old_data)

    def require_layout(self, layout):
        """Change data to specified layout."""
        layout = self.domain.distributor.get_layout_object(layout)
        # Transform to specified layout
        if self.layout.index < layout.index:
            while self.layout.index < layout.index:
                #self.domain.distributor.increment_layout(self)
                self.towards_grid_space()
        elif self.layout.index > layout.index:
            while self.layout.index > layout.index:
                #self.domain.distributor.decrement_layout(self)
                self.towards_coeff_space()

    def towards_grid_space(self):
        """Change to next layout towards grid space."""
        index = self.layout.index
        self.domain.dist.paths[index].increment([self])

    def towards_coeff_space(self):
        """Change to next layout towards coefficient space."""
        index = self.layout.index
        self.domain.dist.paths[index-1].decrement([self])

    def require_grid_space(self, axis=None):
        """Require one axis (default: all axes) to be in grid space."""
        if axis is None:
            while not all(self.layout.grid_space):
                self.towards_grid_space()
        else:
            while not self.layout.grid_space[axis]:
                self.towards_grid_space()

    def require_coeff_space(self, axis=None):
        """Require one axis (default: all axes) to be in coefficient space."""
        if axis is None:
            while any(self.layout.grid_space):
                self.towards_coeff_space()
        else:
            while self.layout.grid_space[axis]:
                self.towards_coeff_space()

    def require_local(self, axis):
        """Require an axis to be local."""
        # Move towards transform path, since the surrounding layouts are local
        if self.layout.grid_space[axis]:
            while not self.layout.local[axis]:
                self.towards_coeff_space()
        else:
            while not self.layout.local[axis]:
                self.towards_grid_space()

    def differentiate(self, *args, **kw):
        """Differentiate field."""
        from .operators import differentiate
        diff_op = differentiate(self, *args, **kw)
        return diff_op.evaluate()

    def integrate(self, *args, **kw):
        """Integrate field."""
        from .operators import integrate
        integ_op = integrate(self, *args, **kw)
        return integ_op.evaluate()

    def interpolate(self, *args, **kw):
        """Interpolate field."""
        from .operators import interpolate
        interp_op = interpolate(self, *args, **kw)
        return interp_op.evaluate()

    def antidifferentiate(self, basis, bc, out=None):
        """
        Antidifferentiate field by setting up a simple linear BVP.

        Parameters
        ----------
        basis : basis-like
            Basis to antidifferentiate along
        bc : (str, object) tuple
            Boundary conditions as (functional, value) tuple.
            `functional` is a string, e.g. "left", "right", "int"
            `value` is a field or scalar
        out : field, optional
            Output field

        """

        # References
        basis = self.domain.get_basis_object(basis)
        domain = self.domain
        bc_type, bc_val = bc

        # Only solve along last basis
        if basis is not domain.bases[-1]:
            raise NotImplementedError()

        from .problems import LBVP
        basis_name = basis.name
        problem = LBVP(domain, variables=['out'])
        problem.parameters['f'] = self
        problem.parameters['bc'] = bc_val
        problem.add_equation('d'+basis_name+'(out) = f')
        problem.add_bc(bc_type+'(out) = bc')

        solver = problem.build_solver()
        solver.solve()

        if not out:
            out = self.domain.new_field()

        out.set_scales(domain.dealias, keep_data=False)
        out['c'] = np.copy(solver.state['out']['c'])

        return out

    # @classmethod
    # def cast_scalar(cls, scalar, domain):
    #     out = Field(bases=domain)
    #     out['c'] = scalar.value
    #     return out

    # @classmethod
    # def cast(cls, input, domain):
    #     from .operators import FieldCopy
    #     from .future import FutureField
    #     # Cast to operand and check domain
    #     input = Operand.cast(input)
    #     if isinstance(input, (Field, FutureField)):
    #         return input
    #     elif isinstance(input, Scalar):
    #         return cls.cast_scalar(input, domain)
    #         # Cast to FutureField
    #         #return FieldCopy(input, domain)
    #     else:
    #         raise ValueError()

    @property
    def is_scalar(self):
        return all(basis is None for basis in self.bases)

    @CachedMethod(max_size=1)
    def as_ncc_operator(self, subsystem, bases, name=None, cacheid=None, cutoff=1e-10):
        """Convert to operator form representing multiplication as a NCC."""
        if name is None:
            name = str(self)
        if self.is_scalar:
            return self.data.ravel()[0]
        L = n_terms = 0
        self.require_coeff_space()
        axmats = subsystem.compute_identities(bases)
        for index, coeff in np.ndenumerate(self.data):
            if abs(coeff) >= cutoff:
                for axis, basis in enumerate(self.bases):
                    if basis is not None:
                        axmats[axis] = basis.Multiply(index[axis])
                L = L + coeff * reduce(sparse.kron, axmats, 1).tocsr()
                n_terms += 1
        logger.debug("Expanded NCC '{}' with {} terms.".format(name, n_terms))
        return L

