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
import h5py


from ..libraries.fftw import fftw_wrappers as fftw
from ..tools.config import config
from ..tools.cache import CachedMethod
from ..tools.exceptions import UndefinedParityError
from ..tools.exceptions import SymbolicParsingError
from ..tools.exceptions import NonlinearOperatorError
from ..tools.exceptions import DependentOperatorError
from ..tools.general import unify, unify_attributes, DeferredTuple, OrderedSet
from ..tools.random_arrays import ChunkedRandomArray

import logging
logger = logging.getLogger(__name__.split('.')[-1])

# Public interface
__all__ = ['Field',
           'ScalarField',
           'VectorField',
           'TensorField',
           'LockedField']


class Operand:
    """Base class for operand classes."""

    __array_priority__ = 100.

    def __array_ufunc__(self, ufunc, method, *inputs, **kw):
        from .operators import UnaryGridFunction
        if method != "__call__":
            return NotImplemented
        if kw:
            return NotImplemented
        # Dispatch unary ufuncs to ufunc operator
        if len(inputs) == 1:
            return UnaryGridFunction(ufunc, inputs[0])
        # Dispatch binary ufuncs to arithmetic operators, triggered by arithmetic with numpy scalars
        elif len(inputs) == 2:
            from . import arithmetic
            from . import operators
            if ufunc is np.add:
                return arithmetic.Add(*inputs)
            elif ufunc is np.subtract:
                return arithmetic.Add(inputs[0], (-1)*inputs[1])
            elif ufunc is np.multiply:
                return arithmetic.Multiply(*inputs)
            elif ufunc is np.divide:
                return arithmetic.Multiply(inputs[0], inputs[1]**(-1))
            elif ufunc is np.power:
                return operators.Power(*inputs)
            else:
                return NotImplemented
        else:
            return NotImplemented

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
        from .arithmetic import Add
        return Add(self, other)

    def __radd__(self, other):
        # Call: other + self
        from .arithmetic import Add
        return Add(other, self)

    def __sub__(self, other):
        # Call: self - other
        return (self + (-other))

    def __rsub__(self, other):
        # Call: other - self
        return (other + (-self))

    def __mul__(self, other):
        # Call: self * other
        from .arithmetic import Multiply
        return Multiply(self, other)

    def __rmul__(self, other):
        # Call: other * self
        from .arithmetic import Multiply
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
    def cast(arg, dist, tensorsig, dtype):
        # Check distributor for operands
        if isinstance(arg, Operand):
            if arg.domain.dist is not dist:
                raise ValueError("Mismatching distributor.")
            elif arg.tensorsig != tensorsig:
                raise ValueError("Mismatching tensorsig.")
            elif arg.dtype != dtype:
                raise ValueError("Mismatching dtype.")
            else:
                return arg
        # Cast numbers to constant fields
        elif isinstance(arg, Number):
            out = Field(dist=dist, tensorsig=tensorsig, dtype=dtype)
            out['g'] = arg  # Set in grid space arbitrarily
            out.name = str(arg)
            return out
        else:
            raise NotImplementedError("Cannot cast type %s" %type(arg))

    # def get_basis(self, coord):
    #     return self.domain.get_basis(coord)
        # space = self.domain.get_basis(coord)
        # if self.domain.spaces[space.axis] in [space, None]:
        #     return self.bases[space.axis]
        # else:
        #     raise ValueError()




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


    def atoms(self, *types):
        """Gather all leaf-operands by type."""
        raise NotImplementedError()

    def has(self, *vars):
        """Determine if tree contains any specified operands/operators."""
        raise NotImplementedError()

    def split(self, *vars):
        """Split into expressions containing and not containing specified operands/operators."""
        raise NotImplementedError()

    def replace(self, old, new):
        """Replace specified operand/operator."""
        raise NotImplementedError()

    def sym_diff(self, var):
        """Symbolically differentiate with respect to specified operand."""
        raise NotImplementedError()

    def expand(self, *vars):
        """Expand expression over specified variables."""
        raise NotImplementedError()

    # def simplify(self, *vars):
    #     """Simplify expression, except subtrees containing specified variables."""
    #     raise NotImplementedError()

    def require_linearity(self, *vars, name=None):
        """Require expression to be linear in specified operands/operators."""
        raise NotImplementedError()

    def require_independent(self, *vars, name=None):
        """Require expression to be independent of specified operands/operators."""
        if self.has(*vars):
            raise DependentOperatorError("{} is not independent of the specified variables.".format(name if name else str(self)))

    def separability(self, *vars):
        """Determine separable dimensions of expression as a linear operator on specified variables."""
        raise NotImplementedError("%s has not implemented a separability method." %type(self))

    # def operator_order(self, operator):
    #     """Determine maximum application order of an operator in the expression."""
    #     raise NotImplementedError()

    def build_ncc_matrices(self, separability, vars, **kw):
        """Precompute non-constant coefficients and build multiplication matrices."""
        raise NotImplementedError()

    def expression_matrices(self, subproblem, vars, **kw):
        """Build expression matrices for a specific subproblem and variables."""
        raise NotImplementedError()




class Current(Operand):

    def __repr__(self):
        return '<{} {}>'.format(self.__class__.__name__, id(self))

    def __str__(self):
        if self.name:
            return self.name
        else:
            return self.__repr__()

    def atoms(self, *types):
        """Gather all leaf-operands of specified types."""
        atoms = OrderedSet()
        if (not types) or isinstance(self, types):
            atoms.add(self)
        return atoms

    def has(self, *vars):
        """Determine if tree contains any specified operands/operators."""
        # Check for empty set or matching operand
        return (not vars) or (self in vars)

    def split(self, *vars):
        """Split into expressions containing and not containing specified operands/operators."""
        if self in vars:
            return (self, 0)
        else:
            return (0, self)

    def replace(self, old, new):
        """Replace specified operand/operator."""
        if self == old:
            return new
        else:
            return self

    def sym_diff(self, var):
        """Symbolically differentiate with respect to specified operand."""
        if self == var:
            return 1
        else:
            return 0

    def expand(self, *vars):
        """Expand expression over specified variables."""
        return self

    def prep_nccs(self, vars):
        if self not in vars:
            raise ValueError("This should never happen.")

    def gather_ncc_coeffs(self):
        pass

    def attempt(self, id=None):
        """Recursively attempt to evaluate operation."""
        return self

    # def simplify(self, *vars):
    #     """Simplify expression, except subtrees containing specified variables."""
    #     return self

    def require_linearity(self, *vars, name=None):
        """Require expression to be linear in specified variables."""
        if self not in vars:
            raise NonlinearOperatorError("{} is not linear in the specified variables.".format(name if name else str(self)))

    # def separability(self, *vars):
    #     """Determine separable dimensions of expression as a linear operator on specified variables."""
    #     self.require_linearity(*vars)
    #     return np.array([True for basis in self.domain.bases])

    def matrix_dependence(self, *vars):
        self.require_linearity(*vars)
        return np.array([False for axis in range(self.domain.dist.dim)])

    def matrix_coupling(self, *vars):
        self.require_linearity(*vars)
        return np.array([False for axis in range(self.domain.dist.dim)])

    # def operator_order(self, operator):
    #     """Determine maximum application order of an operator in the expression."""
    #     return 0

    def build_ncc_matrices(self, separability, vars, **kw):
        """Precompute non-constant coefficients and build multiplication matrices."""
        self.require_linearity(*vars)

    def expression_matrices(self, subproblem, vars, **kw):
        """Build expression matrices for a specific subproblem and variables."""
        self.require_linearity(*vars)
        # Build identity matrices over subproblem data
        # group_shape = subproblem.group_shape(self.domain)
        # factors = (sparse.identity(n, format='csr') for n in group_shape)
        # matrix = reduce(sparse.kron, factors, 1).tocsr()
        #size = self.domain.bases[0].field_radial_size(self, subproblem.ell)
        size = subproblem.field_size(self)
        matrix = sparse.identity(size, format='csr')
        return {self: matrix}

    # def setup_operator_matrix(self, separability, vars, **kw):
    #     """Setup operator matrix components."""
    #     self.require_linearity(*vars)
    #     # axmats = []
    #     # for seperable, basis in zip(separability, self.bases):
    #     #     # Size 1 for constant dimensions
    #     #     if basis is None:
    #     #         axmats.append(sparse.identity(1).tocsr())
    #     #     # Group size for separable dimensions
    #     #     elif separable:
    #     #         axmats.append(sparse.identity(basis.space.group_size).tocsr())
    #     #     # Coeff size for coupled dimensions
    #     #     else:
    #     #         axmats.append(sparse.identity(basis.space.coeff_size).tocsr())
    #     # # Store Kronecker product
    #     # self.operator_matrix = reduce(sparse.kron, axmats, 1).tocsr()

    def evaluate(self):
        return self

    def reinitialize(self, **kw):
        return self

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

    def preset_scales(self, scales, keep_data=True):
        """Set new transform scales."""
        new_scales = self.dist.remedy_scales(scales)
        old_scales = self.scales
        if new_scales == old_scales:
            return
        # Set metadata
        self.scales = new_scales
        # Get buffer size, floored at dealias buffer size
        buffer_size_new = self.dist.buffer_size(self.domain, new_scales, dtype=self.dtype)
        buffer_size_dealias = self.dist.buffer_size(self.domain, self.domain.dealias, dtype=self.dtype)
        buffer_size = max(buffer_size_new, buffer_size_dealias)
        # Build new buffer if current buffer is different size
        if buffer_size != self.buffer_size:
            ncomp = int(np.prod([vs.dim for vs in self.tensorsig]))
            self.buffer = self._create_buffer(buffer_size*ncomp)
            self.buffer_size = buffer_size
        # Reset layout to build new data view
        self.preset_layout(self.layout)

    def preset_layout(self, layout):
        """Interpret buffer as data in specified layout."""
        layout = self.dist.get_layout_object(layout)
        self.layout = layout
        tens_shape = [vs.dim for vs in self.tensorsig]
        local_shape = layout.local_shape(self.domain, self.scales)
        total_shape = tuple(tens_shape) + tuple(local_shape)
        self.data = np.ndarray(shape=total_shape,
                               dtype=self.dtype,
                               buffer=self.buffer)
        #self.global_start = layout.start(self.domain, self.scales)


class Field(Current):
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

    def __init__(self, dist, bases=None, name=None, tensorsig=None, dtype=None):
        if bases is None:
            bases = tuple()
        # Accept single basis in place of tuple/list
        if not isinstance(bases, (tuple, list)):
            bases = (bases,)
        if tensorsig is None:
            tensorsig = tuple()
        if dtype is None:
            if dist.dtype is None:
                raise ValueError("dtype must be specified for Distributor or Field.")
            dtype = dist.dtype
        from .domain import Domain
        self.dist = dist
        self.name = name
        self.tensorsig = tensorsig
        self.dtype = dtype
        # Build domain
        self.domain = Domain(dist, bases)
        # Set initial scales and layout
        self.scales = None
        self.buffer_size = -1
        self.layout = self.dist.get_layout_object('c')
        # Change scales to build buffer and data
        self.preset_scales((1,) * self.dist.dim)

    def __getitem__(self, layout):
        """Return data viewed in specified layout."""
        self.change_layout(layout)
        return self.data

    def __setitem__(self, layout, data):
        """Set data viewed in a specified layout."""
        layout = self.dist.get_layout_object(layout)
        self.preset_layout(layout)
        np.copyto(self.data, data)

    def get_basis(self, coord):
        return self.domain.get_basis(coord)
        #from .basis import Basis
        #from .coords import Coordinate
        #return self.domain.full_spaces[space.axis]
        # if isinstance(space, Basis):
        #     if space in self.bases:
        #         return space

        # elif isinstance(space, Coordinate):
        #     for basis in self.bases:
        #         if space is basis.coord:
        #             return basis
        # return None

    @property
    def global_shape(self):
        return self.layout.global_shape(self.domain, self.scales)

    def copy(self):
        copy = Field(self.dist, bases=self.domain.bases, tensorsig=self.tensorsig, dtype=self.dtype)
        copy.preset_scales(self.scales)
        copy[self.layout] = self.data
        return copy

    def set_global_data(self, global_data):
        elements = self.layout.local_elements(self.domain, self.scales)
        self.set_local_data(global_data[np.ix_(*elements)])

    def set_local_data(self, local_data):
        np.copyto(self.data, local_data)

    def change_scales(self, scales):
        """Change data to specified scales."""
        # Remedy scales
        new_scales = self.dist.remedy_scales(scales)
        old_scales = self.scales
        # Quit if new scales aren't new
        if new_scales == old_scales:
            return
        # Forward transform until remaining scales match
        for axis in reversed(range(self.dist.dim)):
            if not self.layout.grid_space[axis]:
                break
            if old_scales[axis] != new_scales[axis]:
                self.require_coeff_space(axis)
                break
        # Copy over scale change
        old_data = self.data
        self.preset_scales(scales)
        np.copyto(self.data, old_data)

    def change_layout(self, layout):
        """Change data to specified layout."""
        layout = self.dist.get_layout_object(layout)
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
        self.dist.paths[index].increment([self])

    def towards_coeff_space(self):
        """Change to next layout towards coefficient space."""
        index = self.layout.index
        self.dist.paths[index-1].decrement([self])

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

        out.preset_scales(domain.dealias, keep_data=False)
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
        return all(basis is None for basis in self.domain.bases)

    def local_elements(self):
        return self.layout.local_elements(self.domain, self.scales)

    # @CachedAttribute
    # def mode_mask(self):
    #     return reduce()

    def load_from_hdf5(self, file, index, task=None):
        """Load grid data from an hdf5 file. Task correpsonds to field name by default."""
        if task is None:
            task = self.name
        dset = file['tasks'][task]
        if not np.all(dset.attrs['grid_space']):
            raise ValueError("Can only load data from grid space")
        self.load_from_global_grid_data(dset, pre_slices=(index,))

    def load_from_global_grid_data(self, global_data, pre_slices=tuple()):
        """Load local grid data from array-like global grid data."""
        dim = self.dist.dim
        layout = self.dist.grid_layout
        # Set scales to match saved data
        scales = np.array(global_data.shape[-dim:]) / np.array(layout.global_shape(self.domain, scales=1))
        self.preset_scales(scales)
        # Extract local data from global data
        component_slices = tuple(slice(None) for cs in self.tensorsig)
        spatial_slices = layout.slices(self.domain, scales)
        local_slices = pre_slices + component_slices + spatial_slices
        self[layout] = global_data[local_slices]
        # Change scales back to dealias scales
        self.change_scales(self.domain.dealias)

    def allgather_data(self, layout=None):
        """Build global data on all processes."""
        # Change layout
        if layout is not None:
            self.change_layout(layout)
        # Shortcut for serial execution
        if self.dist.comm.size == 1:
            return self.data.copy()
        # Build global buffers
        tensor_shape = tuple(cs.dim for cs in self.tensorsig)
        global_shape = tensor_shape + self.layout.global_shape(self.domain, self.scales)
        local_slices = tuple(slice(None) for cs in self.tensorsig) + self.layout.slices(self.domain, self.scales)
        send_buff = np.zeros(shape=global_shape, dtype=self.dtype)
        recv_buff = np.empty_like(send_buff)
        # Combine data via allreduce -- easy but not communication-optimal
        # Should be optimized using Allgatherv if this is used past startup
        send_buff[local_slices] = self.data
        self.dist.comm.Allreduce(send_buff, recv_buff, op=MPI.SUM)
        return recv_buff

    def allreduce_data_norm(self, layout=None, order=2):
        # Change layout
        if layout is not None:
            self.change_layout(layout)
        # Compute local data
        if self.data.size == 0:
            norm = 0
        elif order == np.inf:
            norm = np.max(np.abs(self.data))
        else:
            norm = np.sum(np.abs(self.data)**order)
        # Reduce
        if order == np.inf:
            if self.dist.comm.size > 1:
                norm = self.dist.comm.allreduce(norm, op=MPI.MAX)
        else:
            if self.dist.comm.size > 1:
                norm = self.dist.comm.allreduce(norm, op=MPI.SUM)
            norm = norm ** (1 / order)
        return norm

    def allreduce_data_max(self, layout=None):
        return self.allreduce_data_norm(layout=layout, order=np.inf)

    def allreduce_L2_norm(self, normalize_volume=True):
        from . import arithmetic
        from . import operators
        # Compute local self inner product
        rank = len(self.tensorsig)
        if rank == 0:
            self_inner_product = np.conj(self) * self
        elif rank == 1:
            self_inner_product = arithmetic.dot(np.conj(self), self)
        elif rank == 2:
            self_inner_product = arithmetic.Trace(arithmetic.Dot(operators.Transpose(np.conj(self)), self))
        else:
            raise ValueError("Norms only implemented up to rank-2 tensors.")
        # Compute L2 norm
        norm_sq = operators.Integrate(self_inner_product).evaluate().allreduce_data_max()
        if normalize_volume:
            norm_sq /= self.domain.volume
        return norm_sq ** 0.5

    def normalize(self, normalize_volume=True):
        """
        Normalize field inplace using L2 norm.

        Parameters
        ----------
        normalize_volume : bool, optional
            Normalize inner product by domain volume. Default: True.
        """
        norm = self.allreduce_L2_norm(normalize_volume=normalize_volume)
        self.data /= norm

    def broadcast_ghosts(self, output_nonconst_dims):
        """Copy data over constant distributed dimensions for arithmetic broadcasting."""
        # Determine deployment dimensions
        self_const_dims = np.array(self.domain.constant)
        distributed = ~self.layout.local
        broadcast_dims = output_nonconst_dims & self_const_dims
        deploy_dims_ext = broadcast_dims & distributed
        deploy_dims = deploy_dims_ext[distributed]
        if not any(deploy_dims):
            return self.data
        # Broadcast on subgrid communicator
        comm_sub = self.domain.dist.comm_cart.Sub(remain_dims=deploy_dims)
        data = None
        if comm_sub.rank == 0:
            data = self.data
        else:
            shape = np.array(self.data.shape)
            shape[shape == 0] = 1
            data = np.empty(shape=shape, dtype=self.dtype)
        comm_sub.Bcast(data, root=0)
        return data

    def fill_random(self, layout=None, seed=None, chunk_size=2**20, distribution='standard_normal', **kw):
        """
        Fill field with random data. If a seed is specified, the global data is
        reproducibly generated for any process mesh.

        Parameters
        ----------
        layout : Layout object, 'c', or 'g', optional
            Layout for setting field data. Default: current layout.
        seed : int, optional
            RNG seed. Default: None.
        chunk_size : int, optional
            Chunk size for drawing from distribution. Should be less than locally
            available memory. Default: 2**20, corresponding to 8 MB of float64.
        distribution : str, optional
            Distribution name, corresponding to numpy random Generator method.
            Default: 'standard_normal'.
        **kw : dict
            Other keywords passed to the distribution method.
        """
        # Set layout if requested
        if layout is not None:
            self.preset_layout(layout)
        # Build global chunked random array (does not require global-sized memory)
        shape = tuple(cs.dim for cs in self.tensorsig) + self.global_shape
        if np.iscomplexobj(self):
            shape = shape + (2,)
        global_data = ChunkedRandomArray(shape, seed, chunk_size, distribution, **kw)
        # Extract local data
        component_slices = tuple(slice(None) for cs in self.tensorsig)
        spatial_slices = self.layout.slices(self.domain, self.scales)
        local_slices = component_slices + spatial_slices
        local_data = global_data[local_slices]
        if np.isrealobj(self):
            self.data[:] = local_data
        else:
            self.data.real[:] = local_data[..., 0]
            self.data.imag[:] = local_data[..., 1]

    def low_pass_filter(self, shape=None, scales=None):
        """
        Apply a spectral low-pass filter by zeroing modes above specified relative scales.
        The scales can be specified directly or deduced from a specified global grid shape.

        Parameters
        ----------
        shape : tuple of ints, optional
            Global grid shape for inferring truncation scales.
        scales : float or tuple of floats, optional
            Scale factors for truncation.
        """
        original_scales = self.scales
        # Determine scales from shape
        if shape is not None:
            if scales is not None:
                raise ValueError("Specify either shape or scales.")
            global_shape = self.dist.grid_layout.global_shape(self.domain, scales=1)
            scales = np.array(shape) / global_shape
        # Low-pass filter by changing scales
        self.change_scales(scales)
        self.require_grid_space()
        self.change_scales(original_scales)

    def high_pass_filter(self, shape=None, scales=None):
        """
        Apply a spectral high-pass filter by zeroing modes below specified relative scales.
        The scales can be specified directly or deduced from a specified global grid shape.

        Parameters
        ----------
        shape : tuple of ints, optional
            Global grid shape for inferring truncation scales.
        scales : float or tuple of floats, optional
            Scale factors for truncation.
        """
        data_orig = self['c'].copy()
        self.low_pass_filter(shape=shape, scales=scales)
        data_filt = self['c'].copy()
        self['c'] = data_orig - data_filt


ScalarField = Field


def VectorField(dist, coordsys, *args, **kw):
    tensorsig = (coordsys,)
    return Field(dist, *args, tensorsig=tensorsig, **kw)


def TensorField(dist, coordsys, *args, order=2, **kw):
    if isinstance(coordsys, (tuple, list)):
        tensorsig = coordsys
    else:
        tensorsig = (coordsys,) * order
    return Field(dist, *args, tensorsig=tensorsig, **kw)


class LockedField(Field):
    """Field locked to particular layouts, disallowing changes to other layouts."""

    def change_scales(self, scales):
        scales = self.dist.remedy_scales(scales)
        if scales != self.scales:
            raise ValueError("Cannot change locked scales.")

    def towards_grid_space(self):
        """Change to next layout towards grid space."""
        index = self.layout.index
        new_index = index + 1
        new_layout = self.dist.layouts[new_index]
        if new_layout in self.allowed_layouts:
            super().towards_grid_space()
        else:
            raise ValueError("Cannot change locked layout.")

    def towards_coeff_space(self):
        """Change to next layout towards coefficient space."""
        index = self.layout.index
        new_index = index - 1
        new_layout = self.dist.layouts[new_index]
        if new_layout in self.allowed_layouts:
            super().towards_coeff_space()
        else:
            raise ValueError("Cannot change locked layout.")

    def lock_to_layouts(self, *layouts):
        self.allowed_layouts = tuple(layouts)

    def lock_axis_to_grid(self, axis):
        self.allowed_layouts = tuple(l for l in self.dist.layouts if l.grid_space[axis])

