"""
Abstract and built-in classes defining deferred operations on fields.

"""

from collections import defaultdict
from functools import partial, reduce
import numpy as np
from scipy import sparse
from numbers import Number
from inspect import isclass
from operator import add
import dedalus_sphere

from .domain import Domain
from . import coords
from .field import Operand, Field
from .future import Future, FutureField, FutureLockedField
from ..tools.array import reshape_vector, apply_matrix, add_sparse, axindex, axslice
from ..tools.cache import CachedAttribute, CachedMethod
from ..tools.dispatch import MultiClass
from ..tools.exceptions import NonlinearOperatorError
from ..tools.exceptions import SymbolicParsingError
from ..tools.exceptions import UndefinedParityError
from ..tools.exceptions import SkipDispatchException
from ..tools.general import unify, unify_attributes


# Use simple decorator to track parseable operators
parseables = {}
prefixes = {}

def parseable(*names):
    def register_op(op):
        for name in names:
            parseables[name] = op
        return op
    return register_op

def prefix(*names):
    def register_op(op):
        for name in names:
            prefixes[name] = op
        return op
    return register_op





# class Cast(FutureField, metaclass=MultiClass):
#     """
#     Cast to field.

#     Parameters
#     ----------
#     input : Number or Operand
#     domain : Domain object

#     """

#     @classmethod
#     def _check_args(cls, input, domain):
#         return isinstance(input, cls.argtype)

#     @property
#     def base(self):
#         return Cast


# class CastNumber(Cast):
#     """Cast number to field."""

#     argtype = Number

#     def __init__(self, input, domain, out=None):
#         self.args = [input, domain]
#         self.original_args = tuple(self.args)
#         self.out = out
#         self.bases = (None,) * domain.dim
#         self.domain = domain
#         self.subdomain = Subdomain.from_bases(self.domain, self.bases)
#         self._grid_layout = self.domain.dist.grid_layout
#         self._coeff_layout = self.domain.dist.coeff_layout
#         self.last_id = None
#         self.scales = self.subdomain.dealias

#     def __str__(self):
#         return str(self.number)

#     def __eq__(self, other):
#         # Compare by value
#         return (self.number == other)

#     @property
#     def number(self):
#         return self.args[0]

#     def split(self, *vars):
#         """Split into expressions containing and not containing specified operands/operators."""
#         return (0, self.number)

#     def replace(self, old, new):
#         """Replace specified operand/operator."""
#         return self.number

#     def sym_diff(self, var):
#         """Symbolically differentiate with respect to specified operand."""
#         return 0

#     def expand(self, *vars):
#         """Expand expression over specified variables."""
#         return self.number

#     # def simplify(self, *vars):
#     #     """Simplify expression, except subtrees containing specified variables."""
#     #     return self.number

#     def require_linearity(self, *vars, name=None):
#         """Require expression to be linear in specified variables."""
#         raise NonlinearOperatorError("{} is not linear in the specified variables.".format(name if name else str(self)))

#     def separability(self, *vars):
#         """Determine separable dimensions of expression as a linear operator on specified variables."""
#         raise NonlinearOperatorError("{} is not one of the specified variables.".format(str(self)))

#     def build_ncc_matrices(self, separability, vars, **kw):
#         """Precompute non-constant coefficients and build multiplication matrices."""
#         raise NonlinearOperatorError("{} is not one of the specified variables.".format(str(self)))

#     def expression_matrices(self, subproblem, vars):
#         """Build expression matrices for a specific subproblem and variables."""
#         raise NonlinearOperatorError("{} is not one of the specified variables.".format(str(self)))

#     def check_conditions(self):
#         """Check that arguments are in a proper layout."""
#         # No conditions
#         return True

#     def enforce_conditions(self):
#         """Require arguments to be in a proper layout."""
#         # No conditions
#         pass

#     def operate(self, out):
#         """Perform operation."""
#         # Copy data
#         np.copyto(out.data, self.number)


# class CastOperand(Cast):
#     """Cast operand to field."""

#     argtype = Operand

#     def __new__(cls, input, domain):
#         # Make sure domains match
#         if input.domain is not domain:
#             raise ValueError()
#         else:
#             return input




# class FieldCopy(FutureField):
#     """Operator making a new field copy of data."""

#     name = 'FieldCopy'

#     @classmethod
#     def _preprocess_args(cls, arg, domain, **kw):
#         arg = Operand.cast(arg, domain=domain)
#         return (arg,), kw

#     @classmethod
#     def _check_args(cls, *args, **kw):
#         match = (isinstance(args[i], types) for i,types in cls.argtypes.items())
#         return all(match)

#     def __init__(self, arg, **kw):
#         super().__init__(arg, **kw)
#         self.kw = {'domain': arg.domain}

#     def _build_bases(self, arg0):
#         return arg0.bases

#     def __str__(self):
#         return str(self.args[0])

#     def check_conditions(self):
#         return True

#     def base(self):
#         return FieldCopy

#     def sym_diff(self, var):
#         """Symbolically differentiate with respect to var."""
#         return self.args[0].sym_diff(var)



# class FieldCopyArray(FieldCopy):

#     argtypes = {0: (Array, FutureArray)}

#     def operate(self, out):
#         # Copy in grid layout
#         out.set_layout(self._grid_layout)
#         np.copyto(out.data, self.args[0].data)


# class FieldCopyField(FieldCopy):

#     argtypes = {0: (Field, FutureField)}

#     def operate(self, out):
#         arg0, = self.args
#         # Copy in current layout
#         out.set_layout(arg0.layout)
#         np.copyto(out.data, arg0.data)







class NonlinearOperator(Future):

    def split(self, *vars):
        """Split into expressions containing and not containing specified operands/operators."""
        if self.has(*vars):
            return (self, 0)
        else:
            return (0, self)

    def expand(self, *vars):
        """Expand expression over specified variables."""
        return self

    def require_linearity(self, *vars, name=None):
        """Require expression to be linear in specified variables."""
        raise NonlinearOperatorError("{} is a non-linear function of the specified variables.".format(name if name else str(self)))

    def separability(self, *vars):
        """Determine separable dimensions of expression as a linear operator on specified variables."""
        raise NonlinearOperatorError("{} is a non-linear function of the specified variables.".format(str(self)))

    def build_ncc_matrices(self, separability, vars, **kw):
        """Precompute non-constant coefficients and build multiplication matrices."""
        raise NonlinearOperatorError("{} is a non-linear function of the specified variables.".format(str(self)))

    def expression_matrices(self, subproblem, vars):
        """Build expression matrices for a specific subproblem and variables."""
        raise NonlinearOperatorError("{} is a non-linear function of the specified variables.".format(str(self)))


class Power(NonlinearOperator, metaclass=MultiClass):

    name = 'Pow'
    str_op = '**'

    def _build_bases(self, arg0, arg1):
        return arg0.bases

    @classmethod
    def _preprocess_args(cls, *args, **kw):
        domain = unify_attributes(args, 'domain', require=False)
        args = tuple(Operand.cast(arg, domain) for arg in args)
        return args, kw

    @classmethod
    def _check_args(cls, *args, **kw):
        match = (isinstance(args[i], types) for i,types in cls.argtypes.items())
        arg1_constant = all(b is None for b in args[1].bases)
        return (all(match) and arg1_constant)

    @property
    def base(self):
        return Power

    def sym_diff(self, var):
        """Symbolically differentiate with respect to specified operand."""
        base, power = self.args
        return power * base**(power-1) * base.sym_diff(var)


class PowerFieldConstant(Power, FutureField):

    argtypes = {0: (Field, FutureField),
                1: (Field, FutureField)}

    def __new__(cls, arg0, arg1, *args, **kw):
        if (arg1.name is None) and (arg1.value == 0):
            return 1
        elif (arg1.name is None) and (arg1.value == 1):
            return arg0
        else:
            return object.__new__(cls)

    def __init__(self, arg0, arg1, out=None):
        super().__init__(arg0, arg1, out=out)
        for axis, b0 in enumerate(arg0.bases):
            if b0 is not None:
                self.require_grid_axis = axis
                break
        else:
            self.require_grid_axis = None

    def check_conditions(self):
        layout0 = self.args[0].layout
        layout1 = self.args[1].layout
        # Fields must be in grid layout
        if self.require_grid_axis is not None:
            axis = self.require_grid_axis
            return (layout0.grid_space[axis] and (layout0 is layout1))
        else:
            return (layout0 is layout1)

    def enforce_conditions(self):
        arg0, arg1 = self.args
        if self.require_grid_axis is not None:
            axis = self.require_grid_axis
            arg0.require_grid_space(axis=axis)
        arg1.require_layout(arg0.layout)

    def operate(self, out):
        arg0, arg1 = self.args
        # Multiply in grid layout
        out.set_layout(arg0.layout)
        if out.data.size:
            np.power(arg0.data, arg1.data, out.data)


# class PowerArrayScalar(PowerDataScalar, FutureArray):

#     argtypes = {0: (Array, FutureArray),
#                 1: Number}

#     def check_conditions(self):
#         return True

#     def operate(self, out):
#         arg0, arg1 = self.args
#         np.power(arg0.data, arg1.value, out.data)


# class PowerFieldNumber(PowerDataScalar, FutureField):

#     argtypes = {0: (Field, FutureField),
#                 1: Number}

#     def check_conditions(self):
#         # Field must be in grid layout
#         return (self.args[0].layout is self._grid_layout)

#     def operate(self, out):
#         arg0, arg1 = self.args
#         # Raise in grid layout
#         arg0.require_grid_space()
#         out.set_layout(self._grid_layout)
#         np.power(arg0.data, arg1.value, out.data)




class GeneralFunction(NonlinearOperator, FutureField):
    """
    Operator wrapping a general python function.

    Parameters
    ----------
    domain : domain object
        Domain
    layout : layout object or identifier
        Layout of function output
    func : function
        Function producing field data
    args : list
        Arguments to pass to func
    kw : dict
        Keywords to pass to func
    out : field, optional
        Output field (default: new field)

    Notes
    -----
    On evaluation, this wrapper evaluates the provided funciton with the given
    arguments and keywords, and takes the output to be data in the specified
    layout, i.e.

        out[layout] = func(*args, **kw)

    """

    def __init__(self, domain, layout, func, args=[], kw={}, out=None,):

        # Required attributes
        self.args = list(args)
        self.original_args = list(args)
        self.domain = domain
        self.out = out
        self.last_id = None
        # Additional attributes
        self.layout = domain.distributor.get_layout_object(layout)
        self.func = func
        self.kw = kw
        self._field_arg_indices = [i for (i,arg) in enumerate(self.args) if is_fieldlike(arg)]
        try:
            self.name = func.__name__
        except AttributeError:
            self.name = str(func)
        self.build_metadata()

    def build_metadata(self):
        self.constant = np.array([False] * self.domain.dim)

    def check_conditions(self):
        # Fields must be in proper layout
        for i in self._field_arg_indices:
            if self.args[i].layout is not self.layout:
                return False
        return True

    def operate(self, out):
        # Apply func in proper layout
        for i in self._field_arg_indices:
            self.args[i].require_layout(self.layout)
        out.set_layout(self.layout)
        np.copyto(out.data, self.func(*self.args, **self.kw))


class UnaryGridFunction(NonlinearOperator, metaclass=MultiClass):

    arity = 1
    supported = {ufunc.__name__: ufunc for ufunc in
        (np.absolute, np.sign, np.conj, np.exp, np.exp2, np.log, np.log2,
         np.log10, np.sqrt, np.square, np.sin, np.cos, np.tan, np.arcsin,
         np.arccos, np.arctan, np.sinh, np.cosh, np.tanh, np.arcsinh,
         np.arccosh, np.arctanh)}
    aliased = {'abs':np.absolute, 'conj':np.conjugate}
    # Add ufuncs and shortcuts to parseables
    parseables.update(supported)
    parseables.update(aliased)

    @classmethod
    def _preprocess_args(self, func, arg, **kw):
        #arg = Operand.cast(arg)
        return (func, arg), kw

    @classmethod
    def _check_args(cls, *args, **kw):
        match = (isinstance(args[i], types) for i,types in cls.argtypes.items())
        return all(match)

    def __init__(self, func, arg, **kw):
        #arg = Operand.cast(arg)
        super().__init__(arg, **kw)
        self.func = func

    @property
    def name(self):
        return self.func.__name__

    @property
    def base(self):
        return UnaryGridFunction

    def _build_bases(self, arg0):
        bases = arg0.bases
        if all(basis is None for basis in bases):
            bases = arg0.domain
        return bases

    def meta_constant(self, axis):
        # Preserves constancy
        return self.args[0].meta[axis]['constant']

    def meta_parity(self, axis):
        # Preserving constancy -> even parity
        if self.args[0].meta[axis]['constant']:
            return 1
        elif self.args[0].meta[axis]['parity'] == 1:
            return 1
        else:
            raise UndefinedParityError("Unknown action of {} on odd parity.".format(self._name))

    def sym_diff(self, var):
        """Symbolically differentiate with respect to specified operand."""
        diff_map = {np.absolute: lambda x: np.sign(x),
                    np.sign: lambda x: 0,
                    np.exp: lambda x: np.exp(x),
                    np.exp2: lambda x: np.exp2(x) * np.log(2),
                    np.log: lambda x: x**(-1),
                    np.log2: lambda x: (x * np.log(2))**(-1),
                    np.log10: lambda x: (x * np.log(10))**(-1),
                    np.sqrt: lambda x: (1/2) * x**(-1/2),
                    np.square: lambda x: 2*x,
                    np.sin: lambda x: np.cos(x),
                    np.cos: lambda x: -np.sin(x),
                    np.tan: lambda x: np.cos(x)**(-2),
                    np.arcsin: lambda x: (1 - x**2)**(-1/2),
                    np.arccos: lambda x: -(1 - x**2)**(-1/2),
                    np.arctan: lambda x: (1 + x**2)**(-1),
                    np.sinh: lambda x: np.cosh(x),
                    np.cosh: lambda x: np.sinh(x),
                    np.tanh: lambda x: np.cosh(x)**(-2),
                    np.arcsinh: lambda x: (x**2 + 1)**(-1/2),
                    np.arccosh: lambda x: (x**2 - 1)**(-1/2),
                    np.arctanh: lambda x: (1 - x**2)**(-1)}
        arg = self.args[0]
        arg_diff = arg.sym_diff(var)
        return diff_map[self.func](arg) * arg_diff


# class UnaryGridFunctionScalar(UnaryGridFunction, FutureScalar):

#     argtypes = {1: (Scalar, FutureScalar)}

#     def check_conditions(self):
#         return True

#     def operate(self, out):
#         out.value = self.func(self.args[0].value)


# class UnaryGridFunctionArray(UnaryGridFunction, FutureArray):

#     argtypes = {1: (Array, FutureArray)}

#     def check_conditions(self):
#         return True

#     def operate(self, out):
#         self.func(self.args[0].data, out=out.data)


class UnaryGridFunctionField(UnaryGridFunction, FutureField):

    argtypes = {1: (Field, FutureField)}

    def __init__(self, func, arg, **kw):
        #arg = Operand.cast(arg)
        super().__init__(func, arg, **kw)
        self.domain = arg.domain
        self.tensorsig = arg.tensorsig
        self.dtype = arg.dtype

    def check_conditions(self):
        # Field must be in grid layout
        return (self.args[0].layout is self._grid_layout)

    def enforce_conditions(self):
        self.args[0].require_grid_space()

    def operate(self, out):
        # References
        arg0, = self.args
        # Evaluate in grid layout
        out.set_layout(self._grid_layout)
        self.func(arg0.data, out=out.data)


class LinearOperator(FutureField):
    """
    Base class for linear operators.

    Subclasses must define the following attributes:

        # LinearOperator requirements
        self.operand

        # FutureField requirements
        self.domain
        self.tensorsig
        self.dtype

    """

    # def __init__(self, *args, **kw):
    #     self.coord = args[1]
    #     try:
    #         self.axis = self.coord.axis
    #     except AttributeError:
    #         self.axis = self.coord.coords[0].axis
    #     self.input_basis = args[0].domain.full_bases[self.axis]
    #     self.basis_axis = self.input_basis.axis
    #     self.tensorsig = args[0].tensorsig
    #     self.dtype = args[0].dtype
    #     super().__init__(*args, **kw)

    # @CachedAttribute
    # def bases(self):
    #     output_bases = list(self.operand.bases)  # copy input bases
    #     output_bases[self.basis_axis] = self.output_basis(self.input_basis)
    #     return tuple(output_bases)

    # @property
    # def operand(self):
    #     # Set as a property rather than an attribute so it correctly updates during evaluation
    #     return self.args[0]

    # def new_operand(self, operand):
    #     """Call operator with new operand."""
    #     args = list(self.args)
    #     args[0] = operand
    #     return self.base(*args)

    def __repr__(self):
        return '{}({})'.format(self.name, repr(self.operand))

    def __str__(self):
        return '{}({})'.format(self.name, str(self.operand))

    def reinitialize(self, **kw):
        operand = self.operand.reinitialize(**kw)
        return self.new_operand(operand)

    def new_operand(self, operand, **kw):
        # Subclasses must implement with correct arguments
        raise NotImplementedError("%s has not implemented the new_operand method." %type(self))

    def split(self, *vars):
        """Split into expressions containing and not containing specified operands/operators."""
        # Check for matching operator
        if any(isinstance(self, var) for var in vars):
            return (self, 0)
        # Distribute over split operand
        else:
            return tuple(self.new_operand(arg) for arg in self.operand.split(*vars))

    def sym_diff(self, var):
        """Symbolically differentiate with respect to specified operand."""
        # Differentiate argument
        return self.new_operand(self.operand.sym_diff(var))

    def replace(self, old, new):
        """Replace specified operand/operator."""
        # Check for entire expression match
        if self == old:
            return new
        # Check base and call with replaced arguments
        elif isinstance(self, old):
            new_operand = self.operand.replace(new, old)
            return new(new_operand)
        # Call with replaced arguments
        else:
            new_operand = self.operand.replace(old, new)
            return self.new_operand(new_operand)

    def expand(self, *vars):
        """Expand expression over specified variables."""
        from .arithmetic import Add, Multiply
        if self.has(*vars):
            # Expand operand
            operand = self.operand.expand(*vars)
            # Distribute over linear operators
            if isinstance(operand, LinearOperator):
                return self._expand_linop(operand, vars)
            # Distribute over addition
            elif isinstance(operand, Add):
                return self._expand_add(operand, vars)
            # Distribute over multiplication
            elif isinstance(operand, Multiply):
                return self._expand_multiply(operand, vars)
            # Apply to expanded operand
            else:
                return self.new_operand(operand)
        else:
            return self

    def _expand_linop(self, operand, vars):
        """Expand over linear operators."""
        # By default, perform no further expansion
        return self.new_operand(operand)

    def _expand_add(self, operand, vars):
        """Expand over addition."""
        # There are no sums of sums since addition is flattened
        # Apply to arguments and re-expand to cover sums of products
        return sum((self.new_operand(arg) for arg in operand.args)).expand(*vars)

    def _expand_multiply(self, operand, vars):
        """Expand over multiplication."""
        # There are no products of products since multiplication is flattened
        # There are no products of relevant sums since operand has been expanded
        # By default, perform no further expansion
        return self.new_operand(operand)

    def require_linearity(self, *vars, name=None):
        """Require expression to be linear in specified variables."""
        # Require operand to be linear
        self.operand.require_linearity(*vars, name=name)

    def build_ncc_matrices(self, separability, vars, **kw):
        """Precompute non-constant coefficients and build multiplication matrices."""
        # Build operand matrices
        self.operand.build_ncc_matrices(separability, vars, **kw)

    def expression_matrices(self, subproblem, vars):
        """Build expression matrices for a specific subproblem and variables."""
        # Intercept calls to compute matrices over expressions
        if self in vars:
            return Field.expression_matrices(self, subproblem, vars)
        # Build operand matrices
        operand_mats = self.operand.expression_matrices(subproblem, vars)
        # Apply operator matrix
        operator_mat = self.subproblem_matrix(subproblem)
        return {var: operator_mat @ operand_mats[var] for var in operand_mats}

    def subproblem_matrix(self, subproblem):
        """Build operator matrix for a specific subproblem."""
        raise NotImplementedError("%s has not implemented a subproblem_matrix method." %type(self))


class Lock(FutureLockedField, LinearOperator):

    name = 'Lock'

    def __init__(self, operand, layout):
        super().__init__(operand)
        # LinearOperator requirements
        self.operand = operand
        # FutureField requirements
        self.domain = operand.domain
        self.tensorsig = operand.tensorsig
        self.dtype = operand.dtype
        # Resolve layout
        self.layout = self.dist.get_layout_object(layout)

    def check_conditions(self):
        """Check that arguments are in a proper layout."""
        return (self.args[0].layout is self.layout)

    def enforce_conditions(self):
        """Require arguments to be in a proper layout."""
        self.args[0].require_layout(self.layout)

    def operate(self, out):
        """Perform operation."""
        out.set_layout(self.layout)
        np.copyto(out.data, self.args[0].data)

    def new_operand(self, operand, **kw):
        return Lock(operand, layout=self.layout, **kw)


def Grid(operand):
    return Lock(operand, 'g')


def Coeff(operand):
    return Lock(operand, 'c')


class SpectralOperator(LinearOperator):
    """
    Base class for linear operators acting on the coefficients of an individual spectral basis.

    Subclasses must define the following attributes:

        # SpectralOperator requirements
        self.coord
        self.input_basis
        self.output_basis
        self.first_axis
        self.last_axis
        self.subaxis_dependence
        self.subaxis_coupling

        # LinearOperator requirements
        self.operand

        # FutureField requirements
        self.domain
        self.tensorsig
        self.dtype

    """

    def matrix_dependence(self, *vars):
        # Assumes operand is linear in vars
        md = self.operand.matrix_dependence(*vars).copy()
        axes = slice(self.first_axis, self.last_axis+1)
        md[axes] = np.logical_or(md[axes], self.subaxis_dependence)
        return md

    def matrix_coupling(self, *vars):
        # Assumes operand is linear in vars
        mc = self.operand.matrix_coupling(*vars).copy()
        axes = slice(self.first_axis, self.last_axis+1)
        mc[axes] = np.logical_or(mc[axes], self.subaxis_coupling)
        return mc

    def check_conditions(self):
        """Check that arguments are in a proper layout."""
        arg0 = self.args[0]
        last_axis = self.last_axis
        is_coeff = not arg0.layout.grid_space[last_axis]
        is_local = arg0.layout.local[last_axis]
        # Require coefficient space along last axis
        # Require locality along last axis if non-separable
        if not self.subaxis_coupling[-1]:
            return is_coeff
        else:
            return (is_coeff and is_local)

    def enforce_conditions(self):
        """Require arguments to be in a proper layout."""
        arg0 = self.args[0]
        last_axis = self.last_axis
        # Require coefficient space along last axis
        arg0.require_coeff_space(last_axis)
        # Require locality along last axis if non-separable
        if not not self.subaxis_coupling[-1]:
            arg0.require_local(last_axis)


class SpectralOperator1D(SpectralOperator):
    """
    Base class for linear operators acting on a single coordinate.
    Arguments: operand, coordinate, others...
    """

    # def __init__(self, *args, **kw):
    #     self.coord = args[1]
    #     self.axis = self.coord.axis
    #     self.input_basis = args[0].bases[self.axis]
    #     self.tensorsig = args[0].tensorsig
    #     self.dtype = args[0].dtype
    #     super().__init__(*args, **kw)

    # @staticmethod
    # def output_basis(input_basis):
    #     # Subclasses must implement
    #     raise NotImplementedError()

    # def separability(self, *vars):
    #     """Determine separable dimensions of expression as a linear operator on specified variables."""
    #     # Start from operand separability
    #     separability = self.operand.separability(*vars).copy()
    #     if not self.separable:
    #         separability[self.last_axis] = False
    #     return separability

    def subproblem_matrix(self, subproblem):
        """Build operator matrix for a specific subproblem."""
        axis = self.last_axis
        # Build identity matrices for each axis
        subsystem_shape = subproblem.coeff_shape(self.domain)
        factors = [sparse.identity(n, format='csr') for n in subsystem_shape]
        # Substitute factor for operator axis
        if subproblem.group[axis] is None:
            factors[axis] = self.subspace_matrix
        else:
            group = subproblem.group[axis]
            group_size = self.input_basis.group_shape[0]
            argslice = outslice = slice(group*group_size, (group+1)*group_size)
            # argslice = subproblem.global_slices(self.operand.domain)[axis]
            # outslice = subproblem.global_slices(self.domain)[axis]
            factors[axis] = self.subspace_matrix[outslice, argslice]
        # Add factor for components
        comps = np.prod([cs.dim for cs in self.tensorsig], dtype=int)
        factors = [sparse.identity(comps, format='csr')] + factors
        return reduce(sparse.kron, factors, 1).tocsr()

    @CachedAttribute
    def subspace_matrix(self):
        """Build matrix operating on global subspace data."""
        return self._subspace_matrix(self.input_basis)

    @classmethod
    def _subspace_matrix(cls, basis):
        dtype = cls.dtype
        N = basis.size
        # Build matrix entry by entry over nonzero bands
        M = sparse.lil_matrix((N, N), dtype=dtype)
        for i in range(N):
            for b in cls.bands:
                j = i + b
                if (0 <= j < N):
                    Mij = cls._subspace_entry(i, j, basis)
                    if Mij:
                        M[i,j] = Mij
        return M.tocsr()

    @staticmethod
    def _subspace_entry(i, j, basis):
        raise NotImplementedError()

    def operate(self, out):
        """Perform operation."""
        arg = self.args[0]
        layout = arg.layout
        axis = self.last_axis
        matrix = self.subspace_matrix
        # Set output layout
        out.set_layout(layout)
        # Restrict subspace matrix to local elements if separable
        # OPTIMIZE: don't need to construct full matrix if separable
        if not self.subaxis_coupling[-1]:
            arg_elements = arg.local_elements()[axis]
            out_elements = out.local_elements()[axis]
            matrix = matrix[out_elements[:,None], arg_elements[None,:]]
        # Apply matrix
        data_axis = self.last_axis + len(arg.tensorsig)
        apply_matrix(matrix, arg.data, data_axis, out=out.data)


class LinearSubspaceFunctional(SpectralOperator1D):
    """Base class for linear functionals acting within a single space."""

    separable = False

    def output_basis(self, space, input_basis):
        """Determine output basis."""
        return None

    @classmethod
    def _subspace_matrix(cls, space, basis, *args):
        dtype = space.domain.dtype
        N = space.coeff_size
        # Build row entry by entry
        M = sparse.lil_matrix((1, N), dtype=dtype)
        for j in range(N):
            Mij = cls._subspace_entry(j, space, basis, *args)
            if Mij:
                M[0,j] = Mij
        return M.tocsr()

    @staticmethod
    def _subspace_entry(j, space, basis, *args):
        raise NotImplementedError()


class TimeDerivative(LinearOperator):
    """Class for representing time derivative while parsing."""

    name = 'dt'

    # def __new__(cls, arg):
    #     if isinstance(arg, (Number, Cast)):
    #         return 0
    #     else:
    #         return object.__new__(cls)

    def __init__(self, operand, out=None):
        super().__init__(operand, out=out)
        # LinearOperator requirements
        self.operand = operand
        # FutureField requirements
        self.domain = operand.domain
        self.tensorsig = operand.tensorsig
        self.dtype = operand.dtype

    @property
    def base(self):
        return TimeDerivative

    def _expand_linop(self, operand, vars):
        """Expand over linear operator."""
        # Halt expansion when hitting another time derivative
        if isinstance(operand, TimeDerivative):
            return self.new_operand(operand)
        # Commute operators and re-expand to continue propagating
        else:
            return operand.new_operand(self.new_operand(operand.operand).expand(*vars))

    def _expand_multiply(self, operand, vars):
        """Expand over multiplication."""
        args = operand.args
        # Apply product rule to factors
        partial_diff = lambda i: np.prod([self.new_operand(arg) if i==j else arg for j,arg in enumerate(args)])
        out = sum((partial_diff(i) for i in range(len(args))))
        # Re-expand to continue propagating
        return out.expand(*vars)

    # def separability(self, *vars):
    #     """Determine separable dimensions of expression as a linear operator on specified variables."""
    #     return self.operand.separability(*vars).copy()

    def matrix_dependence(self, *vars):
        return self.operand.matrix_dependence(*vars)

    def matrix_coupling(self, *vars):
        return self.operand.matrix_coupling(*vars)


@parseable('interpolate', 'interp')
def interpolate(arg, **positions):
    # Identify domain
    domain = unify_attributes((arg,)+tuple(positions), 'domain', require=False)
    # Apply iteratively
    for coord, position in positions.items():
        arg = Interpolate(arg, coord, position)
    return arg

class Interpolate(SpectralOperator, metaclass=MultiClass):
    """
    Interpolation along one dimension.

    Parameters
    ----------
    operand : number or Operand object
    space : Space object
    position : 'left', 'center', 'right', or float

    """

    name = 'interp'

    @classmethod
    def _preprocess_args(cls, operand, coord, position, out=None):
        if isinstance(operand, Number):
            raise SkipDispatchException(output=operand)
        if isinstance(coord, coords.Coordinate):
            pass
        elif isinstance(coord, str):
            coord = operand.domain.get_coord(coord)
        else:
            raise ValueError("coord must be Coordinate or str")
        return (operand, coord, position), {'out': out}

    @classmethod
    def _check_args(cls, operand, coord, position, out=None):
        # Dispatch by operand basis
        if isinstance(operand, Operand):
            basis = operand.domain.get_basis(coord)
            if isinstance(basis, cls.input_basis_type):
                return True
        return False

    def __init__(self, operand, coord, position, out=None):
        SpectralOperator.__init__(self, operand, out=out)
        self.position = position
        # SpectralOperator requirements
        self.coord = coord
        self.input_basis = operand.domain.get_basis(coord)
        self.output_basis = self._output_basis(self.input_basis, position)
        self.first_axis = self.input_basis.first_axis
        self.last_axis = self.input_basis.last_axis
        # LinearOperator requirements
        self.operand = operand
        # FutureField requirements
        self.domain = operand.domain.substitute_basis(self.input_basis, self.output_basis)
        self.tensorsig = operand.tensorsig
        self.dtype = operand.dtype

    def new_operand(self, operand, **kw):
        return Interpolate(operand, self.coord, self.position, **kw)

    @CachedAttribute
    def subspace_matrix(self):
        """Build matrix operating on global subspace data."""
        return self._subspace_matrix(self.input_basis, self.position)

    def _expand_multiply(self, operand, vars):
        """Expand over multiplication."""
        # Apply to each factor
        return np.prod([self.new_operand(arg) for arg in operand.args])


#class Interpolate(LinearSubspaceFunctional, metaclass=MultiClass):
#    """
#    Interpolation along one dimension.
#
#    Parameters
#    ----------
#    operand : number or Operand object
#    space : Space object
#    position : 'left', 'center', 'right', or float
#
#    """
#
#    @classmethod
#    def _check_args(cls, operand, coord, position, out=None):
#        # Dispatch by operand basis
#        if isinstance(operand, Operand):
#            if isinstance(operand.get_basis(coord), cls.input_basis_type):
#                if operand.domain.get_basis_subaxis(coord) == cls.input_basis_subaxis:
#                    return True
#        return False
#
#    def __init__(self, operand, coord, position, out=None):
#        self.position = position
#        super().__init__(operand, coord, position, out=out)
#
#    def _expand_multiply(self, operand, vars):
#        """Expand over multiplication."""
#        # Apply to each factor
#        return np.prod([self.new_operand(arg) for arg in operand.args])
#
#    @property
#    def base(self):
#        return Interpolate
#
#
#class InterpolateConstant(Interpolate):
#    """Constant interpolation."""
#
#    @classmethod
#    def _check_args(cls, operand, coord, position, out=None):
#        # Dispatch for numbers or constant bases
#        if isinstance(operand, Number):
#            return True
#        if isinstance(operand, Operand):
#            if operand.get_basis(coord) is None:
#                return True
#        return False
#
#    def __new__(cls, operand, coord, position, out=None):
#        return operand


@parseable('integrate', 'integ')
def integrate(arg, *spaces):
    # Identify domain
    domain = unify_attributes((arg,)+spaces, 'domain', require=False)
    # Apply iteratively
    for space in spaces:
        space = domain.get_space_object(space)
        arg = Integrate(arg, space)
    return arg


class Integrate(LinearSubspaceFunctional, metaclass=MultiClass):
    """
    Integration along one dimension.

    Parameters
    ----------
    operand : number or Operand object
    space : Space object

    """

    @classmethod
    def _check_args(cls, operand, space, out=None):
        # Dispatch by operand basis
        if isinstance(operand, Operand):
            if isinstance(operand.get_basis(space), cls.input_basis_type):
                return True
        return False

    @property
    def base(self):
        return Integrate


class IntegrateConstant(Integrate):
    """Constant integration."""

    @classmethod
    def _check_args(cls, operand, space, out=None):
        # Dispatch for numbers or constant bases
        if isinstance(operand, Number):
            return True
        if isinstance(operand, Operand):
            if operand.get_basis(space) is None:
                return True
        return False

    def __new__(cls, operand, space, out=None):
        return (space.COV.problem_length * operand)


# CHECK NEW
@parseable('filter', 'f')
def filter(arg, **modes):
    # Identify domain
    domain = unify_attributes((arg,)+tuple(modes), 'domain', require=False)
    # Apply iteratively
    for space, mode in modes.items():
        space = domain.get_space_object(space)
        arg = Filter(arg, space, mode)
    return arg


class Filter(LinearSubspaceFunctional):

    def __new__(cls, arg, space, mode):
        if isinstance(arg, Number) or (arg.get_basis(space) is None):
            if mode == 0:
                return arg
            else:
                return 0
        elif space not in arg.subdomain:
            raise ValueError("Invalid space.")
        else:
            return object.__new__(cls)

    def __init__(self, arg, space, mode):
        # Wrap initialization to define keywords
        super().__init__(arg, space, mode)

    @property
    def base(self):
        return Filter

    @staticmethod
    def _subspace_entry(j, space, basis, mode):
        """F(j,m) = δ(j,m)"""
        if j == mode:
            return 1
        else:
            return 0


@prefix('d')
@parseable('differentiate', 'diff', 'd')
def differentiate(arg, *spaces, **space_kw):
    # Parse space/order keywords into space list
    for space, order in space_kw.items():
        spaces += (space,) * order
    # Identify domain
    domain = unify_attributes((arg,)+spaces, 'domain', require=False)
    # Apply iteratively
    for space in spaces:
        space = domain.get_space_object(space)
        arg = Differentiate(arg, space)
    return arg


class Differentiate(SpectralOperator1D, metaclass=MultiClass):
    """
    Differentiation along one dimension.

    Parameters
    ----------
    operand : number or Operand object
    space : Space object

    """

    name = "Diff"

    def __init__(self, operand, coord, out=None):
        super().__init__(operand, out=out)
        # SpectralOperator requirements
        self.coord = coord
        self.input_basis = operand.domain.get_basis(coord)
        self.output_basis = self._output_basis(self.input_basis)
        self.first_axis = coord.axis
        self.last_axis = coord.axis
        self.axis = coord.axis
        # LinearOperator requirements
        self.operand = operand
        # FutureField requirements
        self.domain = operand.domain.substitute_basis(self.input_basis, self.output_basis)
        self.tensorsig = operand.tensorsig
        self.dtype = operand.dtype

    @classmethod
    def _check_args(cls, operand, coord, out=None):
        # Dispatch by operand basis
        if isinstance(operand, Operand):
            basis = operand.domain.get_basis(coord)
            if isinstance(basis, cls.input_basis_type):
                return True
        return False

    def new_operand(self, operand, **kw):
        return Differentiate(operand, self.coord, **kw)

    @staticmethod
    def _output_basis(input_basis):
        # Subclasses must implement
        raise NotImplementedError()

    def __str__(self):
        return 'd{!s}({!s})'.format(self.coord.name, self.operand)

    def _expand_multiply(self, operand, vars):
        """Expand over multiplication."""
        args = operand.args
        # Apply product rule to factors
        partial_diff = lambda i: np.prod([self.new_operand(arg) if i==j else arg for j,arg in enumerate(args)])
        return sum((partial_diff(i) for i in range(len(args))))


class DifferentiateConstant(Differentiate):
    """Constant differentiation."""

    @classmethod
    def _check_args(cls, operand, coord, out=None):
        # Dispatch for numbers of constant bases
        if isinstance(operand, Number):
            return True
        if isinstance(operand, Operand):
            if operand.domain.get_basis(coord) is None:
                return True
        return False

    def __new__(cls, operand, coord, out=None):
        return 0


@prefix('H')
@parseable('hilbert_transform', 'hilbert', 'H')
def hilbert_transform(arg, *spaces, **space_kw):
    # Parse space/order keywords into space list
    for space, order in space_kw.items():
        spaces += (space,) * order
    # Identify domain
    domain = unify_attributes((arg,)+spaces, 'domain', require=False)
    # Apply iteratively
    for space in spaces:
        space = domain.get_space_object(space)
        arg = HilbertTransform(arg, space)
    return arg


class HilbertTransform(SpectralOperator1D, metaclass=MultiClass):
    """
    Hilbert transform along one dimension.

    Parameters
    ----------
    operand : number or Operand object
    space : Space object

    """

    @classmethod
    def _check_args(cls, operand, space, out=None):
        # Dispatch by operand basis
        if isinstance(operand, Operand):
            if isinstance(operand.get_basis(space), cls.input_basis_type):
                return True
        return False

    @property
    def base(self):
        return HilbertTransform


class HilbertTransformConstant(HilbertTransform):
    """Constant Hilbert transform."""

    @classmethod
    def _check_args(cls, operand, space, out=None):
        # Dispatch for numbers of constant bases
        if isinstance(operand, Number):
            return True
        if isinstance(operand, Operand):
            if operand.get_basis(space) is None:
                return True
        return False

    def __new__(cls, operand, space, out=None):
        return 0


def convert(arg, bases):
    # Skip for numbers
    if isinstance(arg, Number):
        return arg
    # Drop Nones
    bases = [b for b in bases if b is not None]
    # if not bases:
    #     return arg
    # # Cast to operand
    # domain = unify_attributes(bases, 'domain', require=False)
    # arg = Field.cast(arg, domain=domain)

    # Apply iteratively
    for basis in bases:
        arg = Convert(arg, basis)
    return arg


class Convert(SpectralOperator, metaclass=MultiClass):
    """
    Convert bases along one dimension.

    Parameters
    ----------
    operand : Operand object
    output_basis : Basis object

    """

    name = "Convert"

    def __init__(self, operand, output_basis, out=None):
        SpectralOperator.__init__(self, operand, out=out)
        # SpectralOperator requirements
        self.coords = output_basis.coords
        self.input_basis = operand.domain.get_basis(self.coords)
        self.output_basis = output_basis
        self.first_axis = self.output_basis.first_axis
        self.last_axis = self.output_basis.last_axis
        # LinearOperator requirements
        self.operand = operand
        # FutureField requirements
        self.domain = operand.domain.substitute_basis(self.input_basis, self.output_basis)
        self.tensorsig = operand.tensorsig
        self.dtype = operand.dtype

    # @classmethod
    # def _preprocess_args(cls, operand, space, output_basis, out=None):
    #     operand = Cast(operand, space.domain)
    #     return (operand, space, output_basis), {'out': out}

    @classmethod
    def _check_args(cls, operand, output_basis, out=None):
        # Dispatch by operand and output basis
        # Require correct types
        if isinstance(operand, Operand):
            input_basis = operand.domain.get_basis(output_basis.coords)
            if input_basis == output_basis:
                return False
            if not isinstance(input_basis, cls.input_basis_type):
                return False
            if not isinstance(output_basis, cls.output_basis_type):
                return False
            return True
        return False

    def __str__(self):
        return 'C(%s)' %str(self.operand)

    def new_operand(self, operand, **kw):
        # Pass through without conversion
        return operand

    def check_conditions(self):
        """Check that arguments are in a proper layout."""
        arg0 = self.args[0]
        last_axis = self.last_axis
        is_coeff = not arg0.layout.grid_space[last_axis]
        is_local = arg0.layout.local[last_axis]
        # Allow conversion in grid space
        if not is_coeff:
            return True
        # In coeff space, require locality if non-separable
        if not self.subaxis_coupling[-1]:
            return True
        else:
            return is_local

    def enforce_conditions(self):
        """Require arguments to be in a proper layout."""
        arg0 = self.args[0]
        last_axis = self.last_axis
        is_coeff = not arg0.layout.grid_space[last_axis]
        is_local = arg0.layout.local[last_axis]
        # Require locality if non-separable and in coeff space
        if is_coeff and self.subaxis_coupling[-1]:
            self.args[0].require_local(last_axis)

    @property
    def base(self):
        return Convert

    def replace(self, old, new):
        """Replace specified operand/operator."""
        # Replace operand, skipping conversion
        return self.operand.replace(old, new)

    # def expand(self, *vars):
    #     """Expand expression over specified variables."""
    #     # Expand operand, skipping conversion
    #     return self.operand.expand(*vars)

    # def simplify(self, *vars):
    #     """Simplify expression, except subtrees containing specified variables."""
    #     # Simplify operand, skipping conversion
    #     return self.operand.simplify(*vars)

    @CachedAttribute
    def subspace_matrix(self):
        """Build matrix operating on global subspace data."""
        return self._subspace_matrix(self.input_basis, self.output_basis)

    def operate(self, out):
        """Perform operation."""
        arg = self.args[0]
        layout = arg.layout
        last_axis = self.last_axis
        # Copy for grid space
        if layout.grid_space[last_axis]:
            out.set_layout(layout)
            np.copyto(out.data, arg.data)
        # Revert to matrix application for coeff space
        else:
            super().operate(out)


class ConvertSame(Convert):
    """Identity conversion."""

    @classmethod
    def _check_args(cls, operand, output_basis, out=None):
        # Dispatch by operand and output basis
        if isinstance(operand, Operand):
            for basis in operand.domain.bases:
                if basis == output_basis:
                    raise SkipDispatchException(output=operand)
        return False

    def __new__(cls, operand, output_basis, out=None):
        if out is not None:
            raise NotImplementedError()
        return operand


# class Convert1DConstant(Convert1D):
#     """Constant conversion."""

#     separable = True
#     bands = [0]

#     @classmethod
#     def _check_args(cls, operand, space, output_basis, out=None):
#         if output_basis.const is not None:
#             if isinstance(operand, Number):
#                 return True
#             if isinstance(operand, Operand):
#                 input_basis = operand.get_basis(space)
#                 if input_basis is None:
#                     return True
#         return False

#     @staticmethod
#     def _subspace_matrix(space, input_basis, output_basis):
#         dtype = space.domain.dtype
#         N = space.coeff_size
#         # Reweight by constant-mode amplitude
#         M = sparse.lil_matrix((N, 1), dtype=dtype)
#         M[0,0] = 1 / output_basis.const
#         return M.tocsr()

#     def check_conditions(self):
#         """Check that arguments are in a proper layout."""
#         # No conditions
#         return True

#     def enforce_conditions(self):
#         """Require arguments to be in a proper layout."""
#         # No conditions
#         pass

#     def operate(self, out):
#         """Perform operation."""
#         operand = self.operand
#         axis = self.axis
#         output_basis = self.args[2]
#         # Set output layout
#         out.set_layout(operand.layout)
#         # Broadcast constant in grid space
#         if operand.layout.grid_space[axis]:
#             np.copyto(out.data, operand.data)
#         # Set constant mode in coefficient space
#         else:
#             out.data.fill(0)
#             mask = out.local_elements()[axis] == 0
#             out.data[axindex(axis, mask)] = operand.data / output_basis.const


class Trace(LinearOperator):

    name = "Trace"

    def __init__(self, operand, out=None):
        super().__init__(operand, out=out)
        # LinearOperator requirements
        self.operand = operand
        # FutureField requirements
        self.domain = operand.domain
        self.tensorsig = tuple(operand.tensorsig[2:])
        self.dtype = operand.dtype

    def new_operand(self, operand, **kw):
        return Trace(operand, **kw)

    def matrix_dependence(self, *vars):
        return self.operand.matrix_dependence(*vars)

    def matrix_coupling(self, *vars):
        return self.operand.matrix_coupling(*vars)

    def check_conditions(self):
        """Right now require grid space"""
        layout = self.args[0].layout
        return all(layout.grid_space)

    def enforce_conditions(self):
        """Require arguments to be in a proper layout."""
        self.args[0].require_grid_space()

    @property
    def base(self):
        return Trace

    def operate(self, out):
        """Perform operation."""
        arg = self.args[0]
        out.set_layout(arg.layout)
        np.einsum('ii...', arg.data, out=out.data)


class TransposeComponents(LinearOperator, metaclass=MultiClass):

    name = "TransposeComponents"

    @classmethod
    def _preprocess_args(cls, operand, indices=(0,1), out=None):
        if isinstance(operand, Number):
            raise SkipDispatchException(output=0)
        return [operand], {'indices': indices, 'out': out}

    @classmethod
    def _check_args(cls, operand, indices=(0,1), out=None):
        # Dispatch by coordinate system; not clear that this is useful (see Keaton's note)
        if isinstance(operand, Operand):
            # hack to check coordsys of first index only
            i0, i1 = indices
            if i0 < 0:
                i0 += len(operand.tensorsig)
            cs0 = operand.tensorsig[i0]
            if isinstance(cs0, cls.cs_type):
                return True
        return False

    def __init__(self, operand, indices=(0,1), out=None):
        i0, i1 = indices
        if i0 < 0:
            i0 += len(operand.tensorsig)
        if i1 < 0:
            i1 += len(operand.tensorsig)
        if max(i0, i1) > len(operand.tensorsig):
            raise ValueError("Transpose index greater than operand rank")
        if i0 == i1:
            raise ValueError("Don't transpose same indices")
        cs0 = operand.tensorsig[i0]
        cs1 = operand.tensorsig[i1]
        if cs0 != cs1:
            raise NotImplementedError("Can only transpose two indices which have the same coordinate system")
        super().__init__(operand, out=out)
        self.indices = (i0, i1)
        self.coordsys = cs0
        # LinearOperator requirements
        self.operand = operand
        # FutureField requirements
        self.domain = operand.domain
        self.tensorsig = operand.tensorsig
        self.dtype = operand.dtype

    def new_operand(self, operand, **kw):
        return TransposeComponents(operand, self.indices, **kw)

    def matrix_dependence(self, *vars):
        return self.operand.matrix_dependence(*vars)

    def matrix_coupling(self, *vars):
        return self.operand.matrix_coupling(*vars)

    @property
    def base(self):
        return TransposeComponents


class CartesianTransposeComponents(TransposeComponents):

    cs_type = coords.CartesianCoordinates

    def __init__(self, operand, indices=(0,1), out=None):
        super().__init__(operand, indices=indices, out=out)
        input_basis = self.domain.get_basis(self.coordsys)
        self.input_basis = input_basis

    def check_conditions(self):
        """Can always take the transpose"""
        return True

    def enforce_conditions(self):
        """Can always take the transpose"""
        pass

    def subproblem_matrix(self, subproblem):
        """Build operator matrix for a specific subproblem."""
        operand = self.args[0]

        indices = self.indices
        rank = len(self.tensorsig)
        tensor_shape = [cs.dim for cs in self.tensorsig]
        start_indices = list(np.ndindex(*tensor_shape))
        neworder = np.arange(rank)
        neworder[indices[0]] = indices[1]
        neworder[indices[1]] = indices[0]

        transpose = np.zeros((len(start_indices), len(start_indices)))
        for i, start_index in enumerate(start_indices):
            end_index = list(start_index)
            end_index[indices[0]] = start_index[indices[1]]
            end_index[indices[1]] = start_index[indices[0]]
            j = start_indices.index(tuple(end_index))
            transpose[j,i] = 1

        # assume all regularities have the same n_size
        eye = sparse.identity(subproblem.coeff_size(self.domain), self.dtype, format='csr')
        matrix = sparse.kron( transpose, eye)
        return matrix

    def operate(self, out):
        """Perform operation."""
        operand = self.args[0]

        # Set output layout
        layout = operand.layout
        out.set_layout(layout)
        indices = self.indices
        np.copyto(out.data, operand.data)

        axes_list = np.arange(len(out.data.shape))
        axes_list[indices[0]] = indices[1]
        axes_list[indices[1]] = indices[0]
        np.copyto(out.data,np.transpose(out.data,axes=axes_list))


class SphericalTransposeComponents(TransposeComponents):

    cs_type = coords.SphericalCoordinates

    def __init__(self, operand, indices=(0,1), out=None):
        super().__init__(operand, indices=indices, out=out)
        self.radius_axis = self.coordsys.coords[2].axis
        input_basis = self.domain.get_basis(self.coordsys)
        self.input_basis = input_basis
        self.radial_basis = self.input_basis.get_radial_basis()

    def check_conditions(self):
        """Can always take the transpose"""
        return True

    def enforce_conditions(self):
        """Can always take the transpose"""
        pass

    def subproblem_matrix(self, subproblem):
        operand = self.args[0]
        basis = self.radial_basis
        R = basis.regularity_classes(self.tensorsig)

        ell = subproblem.group[self.radius_axis - 1]

        indices = self.indices
        rank = len(self.tensorsig)
        neworder = np.arange(rank)
        neworder[indices[0]] = indices[1]
        neworder[indices[1]] = indices[0]

        matrix = []
        for regindex_out, regtotal_out in np.ndenumerate(R):
            regindex_out = np.array(regindex_out)
            matrix_row = []
            for regindex_in, regtotal_in in np.ndenumerate(R):
                if tuple(regindex_out[neworder]) == regindex_in:
                    matrix_row.append( 1 )
                else:
                    matrix_row.append( 0 )
            matrix.append(matrix_row)
        transpose = np.array(matrix)

        Q = basis.radial_recombinations(self.tensorsig,ell_list=(ell,))
        transpose = Q[ell].T @ transpose @ Q[ell]

        # assume all regularities have the same n_size
        eye = sparse.identity(basis.n_size(ell), self.dtype, format='csr')
        matrix = sparse.kron( transpose, eye)
        # Block-diag for sin/cos parts for real dtype
        if self.dtype == np.float64:
            matrix = sparse.kron(matrix, sparse.identity(2, format='csr')).tocsr()
        return matrix

    def operate(self, out):
        """Perform operation."""
        operand = self.args[0]
        basis = self.input_basis.radial_basis
        # Set output layout
        layout = operand.layout
        out.set_layout(layout)
        indices = self.indices
        np.copyto(out.data, operand.data)

        if not layout.grid_space[self.radius_axis]: # in regularity componentsinput
            basis.backward_regularity_recombination(operand.tensorsig, self.radius_axis, out.data)

        axes_list = np.arange(len(out.data.shape))
        axes_list[indices[0]] = indices[1]
        axes_list[indices[1]] = indices[0]
        np.copyto(out.data,np.transpose(out.data,axes=axes_list))

        if not layout.grid_space[self.radius_axis]: # in regularity components
            basis.forward_regularity_recombination(operand.tensorsig, self.radius_axis, out.data)


class SphericalComponent(LinearOperator):

    @classmethod
    def _preprocess_args(cls, operand, index=0, out=None):
        if isinstance(operand, Number):
            raise SkipDispatchException(output=0)
        return [operand], {'index': index, 'out': out}

    def __init__(self, operand, index=0, out=None):
        if index < 0: index += len(operand.tensorsig)
        if index >= len(operand.tensorsig):
            raise ValueError("index greater than rank")
        self.index = index
        self.coordsys = operand.tensorsig[self.index]
        if not isinstance(self.coordsys, coords.SphericalCoordinates):
            raise ValueError("Can only take the SphericalComponent of a SphericalCoordinate vector")
        super().__init__(operand, out=out)
        # LinearOperator requirements
        self.operand = operand
        # FutureField requirements
        self.domain = operand.domain
        self.tensorsig = operand.tensorsig
        self.dtype = operand.dtype

    def check_conditions(self):
        """Can always take components"""
        return True

    def enforce_conditions(self):
        """Can always take components"""
        pass

    def matrix_dependence(self, *vars):
        return self.operand.matrix_dependence(*vars)

    def matrix_coupling(self, *vars):
        return self.operand.matrix_coupling(*vars)


class RadialComponent(SphericalComponent, metaclass=MultiClass):

    @classmethod
    def _check_args(cls, operand, index=0, out=None):
        # Dispatch by coordinate system
        if isinstance(operand, Operand):
            if index < 0: index += len(operand.tensorsig)
            coordsys = operand.tensorsig[index]
            basis = operand.domain.get_basis(coordsys)
            if isinstance(basis, cls.basis_type):
                return True
        return False

    def __init__(self, operand, index=0, out=None):
        super().__init__(operand, index=index, out=out)
        tensorsig = operand.tensorsig
        self.tensorsig = tuple( tensorsig[:index] + tensorsig[index+1:] )

    def new_operand(self, operand, **kw):
        return RadialComponent(operand, self.index, **kw)


class AngularComponent(SphericalComponent, metaclass=MultiClass):

    @classmethod
    def _check_args(cls, operand, index=0, out=None):
        # Dispatch by coordinate system
        if isinstance(operand, Operand):
            if index < 0: index += len(operand.tensorsig)
            coordsys = operand.tensorsig[index]
            basis = operand.domain.get_basis(coordsys)
            if isinstance(basis, cls.basis_type):
                return True
        return False

    def __init__(self, operand, index=0, out=None):
        super().__init__(operand, index=index, out=out)
        tensorsig = operand.tensorsig
        S2coordsys = tensorsig[index].S2coordsys
        self.tensorsig = tuple( tensorsig[:index] + (S2coordsys,) + tensorsig[index+1:] )

    def new_operand(self, operand, **kw):
        return AngularComponent(operand, self.index, **kw)


class Gradient(LinearOperator, metaclass=MultiClass):

    name = "Grad"

    @classmethod
    def _preprocess_args(cls, operand, coordsys, out=None):
        if isinstance(operand, Number):
            raise SkipDispatchException(output=0)
        return [operand, coordsys], {'out': out}

    @classmethod
    def _check_args(cls, operand, cs, out=None):
        # Dispatch by coordinate system
        if isinstance(operand, Operand):
            if isinstance(cs, cls.cs_type):
                return True
        return False

    def new_operand(self, operand, **kw):
        return Gradient(operand, self.coordsys, **kw)


class CartesianGradient(Gradient):

    cs_type = coords.CartesianCoordinates

    def __init__(self, operand, coordsys, out=None):
        args = [Differentiate(operand, coord) for coord in coordsys.coords]
        bases = self._build_bases(*args)
        args = [convert(arg, bases) for arg in args]
        LinearOperator.__init__(self, *args, out=out)
        self.coordsys = coordsys
        # LinearOperator requirements
        self.operand = operand
        # FutureField requirements
        self.domain = Domain(operand.dist, bases)
        self.tensorsig = (coordsys,) + operand.tensorsig
        self.dtype = operand.dtype

    def _build_bases(self, *args):
        return sum(args).domain.bases

    def matrix_dependence(self, *vars):
        arg_vals = [arg.matrix_dependence(self, *vars) for arg in self.args]
        return np.logical_or.reduce(arg_vals)

    def matrix_coupling(self, *vars):
        arg_vals = [arg.matrix_coupling(self, *vars) for arg in self.args]
        return np.logical_or.reduce(arg_vals)

    def subproblem_matrix(self, subproblem):
        """Build operator matrix for a specific subproblem."""
        return sparse.vstack(arg.expression_matrices(subproblem, [self.operand])[self.operand] for arg in self.args)

    def check_conditions(self):
        """Check that operands are in a proper layout."""
        # Require operands to be in same layout
        layouts = [operand.layout for operand in self.args if operand]
        all_layouts_equal = (len(set(layouts)) == 1)
        return all_layouts_equal

    def enforce_conditions(self):
        """Require operands to be in a proper layout."""
        # Require operands to be in same layout
        # Take coeff layout arbitrarily
        layout = self.dist.coeff_layout
        for arg in self.args:
            if arg:
                arg.require_layout(layout)

    def operate(self, out):
        """Perform operation."""
        operands = self.args
        layouts = [operand.layout for operand in self.args if operand]
        # Set output layout
        out.set_layout(layouts[0])
        # Copy operand data to output components
        for i, comp in enumerate(operands):
            if comp:
                out.data[i] = comp.data
            else:
                out.data[i] = 0


class S2Gradient(Gradient, SpectralOperator):

    cs_type = coords.S2Coordinates

    def __init__(self, operand, coordsys, out=None):
        super().__init__(operand, out=out)
        self.coordsys = coordsys
        self.colatitude_axis = coordsys.coords[1].axis
        # SpectralOperator requirements
        self.input_basis = operand.domain.get_basis(coordsys)
        self.output_basis = self.input_basis
        self.last_axis = self.input_basis.last_axis
        # LinearOperator requirements
        self.operand = operand
        # FutureField requirements
        self.domain  = operand.domain
        self.tensorsig = (coordsys,) + operand.tensorsig
        self.dtype = operand.dtype

    def check_conditions(self):
        """Check that operands are in a proper layout."""
        # Require colatitude to be in coefficient space
        layout = self.args[0].layout
        return not layout.grid_space[self.colatitude_axis]

    def enforce_conditions(self):
        """Require operands to be in a proper layout."""
        # Require colatitude to be in coefficient space
        self.args[0].require_coeff_space(self.colatitude_axis)

    def operate(self, out):
        """Perform operation."""
        operand = self.args[0]
        basis = self.input_basis
        azimuthal_axis = self.colatitude_axis - 1
        layout = operand.layout
        # Set output layout
        out.set_layout(layout)
        # slicing local ell's
#        local_l_elements = layout.local_elements(basis.domain, scales=1)[1]
#        local_l = tuple(basis.degrees[local_l_elements])
        local_l = basis.local_l

        # Apply operator
        S = basis.spin_weights(operand.tensorsig)
        for i, s in np.ndenumerate(S):

            operand_spin = reduced_view_4(operand.data[i],azimuthal_axis)
            multiindex = (0,)+i
            out_m = reduced_view_4(out.data[multiindex],azimuthal_axis)
            multiindex = (1,)+i
            out_p = reduced_view_4(out.data[multiindex],azimuthal_axis)
            for dm, m in enumerate(basis.local_m):
                vector = basis.k_vector(-1,m,s,local_l)
                vector = reshape_vector(vector,dim=3,axis=1)
                out_m[:,dm,:,:] = vector * operand_spin[:,dm,:,:]

                vector = basis.k_vector(+1,m,s,local_l)
                vector = reshape_vector(vector,dim=3,axis=1)
                out_p[:,dm,:,:] = vector * operand_spin[:,dm,:,:]


def reduced_view_4(data, axis):
    shape = data.shape
    N0 = int(np.prod(shape[:axis]))
    N1 = shape[axis]
    N2 = shape[axis+1]
    N3 = int(np.prod(shape[axis+2:]))
    return data.reshape((N0, N1, N2, N3))


class SphericalEllOperator(SpectralOperator):

    subaxis_dependence = [False, True, True]  # Depends on ell and n
    subaxis_coupling = [False, False, True]  # Only couples n

    def __init__(self, operand, coordsys):
        self.coordsys = coordsys
        self.radius_axis = coordsys.coords[2].axis
        input_basis = operand.domain.get_basis(coordsys)
        if input_basis is None:
            input_basis = operand.domain.get_basis(coordsys.radius)
        self.radial_basis = input_basis.get_radial_basis()
        self.intertwiner = lambda l: dedalus_sphere.spin_operators.Intertwiner(l, indexing=(-1,+1,0))
        # SpectralOperator requirements
        self.input_basis = input_basis
        self.output_basis = self._output_basis(self.input_basis)
        self.first_axis = self.input_basis.first_axis
        self.last_axis = self.input_basis.last_axis
        # LinearOperator requirements
        self.operand = operand

    def operate(self, out):
        """Perform operation."""
        operand = self.args[0]
        input_basis = self.input_basis
        radial_basis = self.radial_basis
        axis = radial_basis.radial_axis
        # Set output layout
        out.set_layout(operand.layout)
        out.data[:] = 0
        # Apply operator
        R_in = radial_basis.regularity_classes(operand.tensorsig)
        slices = [slice(None) for i in range(input_basis.dist.dim)]
        for regindex_in, regtotal_in in np.ndenumerate(R_in):
            for regindex_out in self.regindex_out(regindex_in):
                comp_in = operand.data[regindex_in]
                comp_out = out.data[regindex_out]
                # Should reorder to make ell loop first, check forbidden reg, remove reg from radial_vector_3
                for ell, m_ind, ell_ind in input_basis.ell_maps:
                    allowed_in  = radial_basis.regularity_allowed(ell, regindex_in)
                    allowed_out = radial_basis.regularity_allowed(ell, regindex_out)
                    if allowed_in and allowed_out:
                        slices[axis-2] = m_ind
                        slices[axis-1] = ell_ind
                        slices[axis] = radial_basis.n_slice(ell)
                        vec_in  = comp_in[tuple(slices)]
                        vec_out = comp_out[tuple(slices)]
                        A = self.radial_matrix(regindex_in, regindex_out, ell)
                        vec_out += apply_matrix(A, vec_in, axis=axis)

    def subproblem_matrix(self, subproblem):
        operand = self.args[0]
        radial_basis = self.radial_basis
        R_in = radial_basis.regularity_classes(operand.tensorsig)
        R_out = radial_basis.regularity_classes(self.tensorsig)  # Should this use output_basis?
        ell = subproblem.group[self.last_axis - 1]
        # Loop over components
        submatrices = []
        for regindex_out, regtotal_out in np.ndenumerate(R_out):
            submatrix_row = []
            for regindex_in, regtotal_in in np.ndenumerate(R_in):
                # Build identity matrices for each axis
                subshape_in = subproblem.coeff_shape(self.operand.domain)
                subshape_out = subproblem.coeff_shape(self.domain)
                # Check if regularity component exists for this ell
                if (regindex_out in self.regindex_out(regindex_in)) and radial_basis.regularity_allowed(ell, regindex_in) and radial_basis.regularity_allowed(ell, regindex_out):
                    # Substitute factor for radial axis
                    factors = [sparse.eye(m, n, format='csr') for m, n in zip(subshape_out, subshape_in)]
                    factors[self.last_axis] = self.radial_matrix(regindex_in, regindex_out, ell)
                    comp_matrix = reduce(sparse.kron, factors, 1).tocsr()
                else:
                    # Build zero matrix
                    comp_matrix = sparse.csr_matrix((np.prod(subshape_out), np.prod(subshape_in)))
                submatrix_row.append(comp_matrix)
            submatrices.append(submatrix_row)
        matrix = sparse.bmat(submatrices)
        matrix.tocsr()
        return matrix

    def regindex_out(self, regindex_in):
        raise NotImplementedError("regindex_out not implemented for type %s" %type(self))

    def radial_matrix(regindex_in, regindex_out, ell):
        raise NotImplementedError()


class SphericalGradient(Gradient, SphericalEllOperator):

    cs_type = coords.SphericalCoordinates

    def __init__(self, operand, coordsys, out=None):
        Gradient.__init__(self, operand, out=out)
        SphericalEllOperator.__init__(self, operand, coordsys)
        # FutureField requirements
        self.domain  = operand.domain.substitute_basis(self.input_basis, self.output_basis)
        self.tensorsig = (coordsys,) + operand.tensorsig
        self.dtype = operand.dtype

    @staticmethod
    def _output_basis(input_basis):
        out = input_basis._new_k(input_basis.k + 1)
        return out

    def check_conditions(self):
        """Check that operands are in a proper layout."""
        # Require radius to be in coefficient space
        layout = self.args[0].layout
        return (not layout.grid_space[self.radius_axis]) and (layout.local[self.radius_axis])

    def enforce_conditions(self):
        """Require operands to be in a proper layout."""
        # Require radius to be in coefficient space
        self.args[0].require_coeff_space(self.radius_axis)
        self.args[0].require_local(self.radius_axis)

    def regindex_out(self, regindex_in):
        # Regorder: -, +, 0
        # Gradients hits - and +
        return ((0,) + regindex_in, (1,) + regindex_in)

    def radial_matrix(self, regindex_in, regindex_out, ell):
        radial_basis = self.radial_basis
        regtotal = radial_basis.regtotal(regindex_in)
        if regindex_out[0] != 2 and regindex_in == regindex_out[1:]:
            return self._radial_matrix(radial_basis, regindex_out[0], regtotal, ell)
        else:
            raise ValueError("This should never happen")

    @staticmethod
    @CachedMethod
    def _radial_matrix(radial_basis, regindex_out0, regtotal, ell):
        if regindex_out0 == 0:
            return radial_basis.xi(-1, ell+regtotal) * radial_basis.operator_matrix('D-', ell, regtotal)
        elif regindex_out0 == 1:
            return radial_basis.xi(+1, ell+regtotal) * radial_basis.operator_matrix('D+', ell, regtotal)
        else:
            raise ValueError("This should never happen")


class Component(LinearOperator, metaclass=MultiClass):

    name = 'Comp'

    def __init__(self, operand, index, coord, out=None):
        super().__init__(operand, out=out)
        self.index = index
        self.coord = coord
        self.coordsys = operand.tensorsig[index]
        self.coord_subaxis = self.coord.axis - self.coordsys.first_axis
        # LinearOperator requirements
        self.operand = operand
        # FutureField requirements
        self.domain = operand.domain
        self.tensorsig = operand.tensorsig[:index] + operand.tensorsig[index+1:]
        self.dtype = operand.dtype

    @classmethod
    def _check_args(cls, operand, index, coord, out=None):
        # Dispatch by coordinate system
        return isinstance(operand.tensorsig[index], cls.cs_type)

    def new_operand(self, operand, **kw):
        return Component(operand, self.index, self.coord, **kw)

    # def separability(self, *vars):
    #     return self.operand.separability(*vars)

    def matrix_dependence(self, *vars):
        return self.operand.matrix_dependence(*vars)

    def matrix_coupling(self, *vars):
        return self.operand.matrix_coupling(*vars)


class CartesianComponent(Component):

    cs_type = coords.CartesianCoordinates

    def check_conditions(self):
        """Check that operands are in a proper layout."""
        # Any layout
        return True

    def enforce_conditions(self):
        """Require operands to be in a proper layout."""
        # Any layout
        pass

    def subproblem_matrix(self, subproblem):
        # Build identities for each tangent space
        factors = [sparse.identity(cs.dim, format='csr') for cs in self.operand.tensorsig]
        factors.append(sparse.identity(subproblem.coeff_size(self.domain), format='csr'))
        # Build selection matrix for selected coord
        index_factor = np.zeros((1, self.coordsys.dim))
        index_factor[0, self.coord_subaxis] = 1
        # Replace indexed factor with selection matrix
        factors[self.index] = index_factor
        return reduce(sparse.kron, factors, 1).tocsr()

    def operate(self, out):
        """Perform operation."""
        arg0 = self.args[0]
        # Set output layout
        out.set_layout(arg0.layout)
        # Copy specified comonent
        take_comp = tuple([None] * self.index + [self.coord_subaxis])
        out.data[:] = arg0.data[take_comp]


class Divergence(LinearOperator, metaclass=MultiClass):

    name = 'Div'
    # should check that we're not taking div of a scalar'

    @classmethod
    def _preprocess_args(cls, operand, index=0, out=None):
        if isinstance(operand, Number):
            raise SkipDispatchException(output=0)
        return [operand], {'index': index, 'out': out}

    @classmethod
    def _check_args(cls, operand, index=0, out=None):
        # Dispatch by coordinate system
        if isinstance(operand, Operand):
            if isinstance(operand.tensorsig[index], cls.cs_type):
                return True
        return False

    def new_operand(self, operand, **kw):
        return Divergence(operand, index=self.index, **kw)


class CartesianDivergence(Divergence):

    cs_type = coords.CartesianCoordinates

    def __init__(self, operand, index=0, out=None):
        coordsys = operand.tensorsig[index]
        # Get components
        comps = [CartesianComponent(operand, index=index, coord=c) for c in coordsys.coords]
        comps = [Differentiate(comp, c) for comp, c in zip(comps, coordsys.coords)]
        arg = sum(comps)
        LinearOperator.__init__(self, arg, out=out)
        self.index = index
        self.coordsys = coordsys
        # LinearOperator requirements
        self.operand = operand
        # FutureField requirements
        self.domain = arg.domain
        self.tensorsig = arg.tensorsig
        self.dtype = arg.dtype

    def matrix_dependence(self, *vars):
        return self.args[0].matrix_dependence(*vars)

    def matrix_coupling(self, *vars):
        return self.args[0].matrix_coupling(*vars)

    def check_conditions(self):
        """Check that operands are in a proper layout."""
        # Any layout (addition is done)
        return True

    def enforce_conditions(self):
        """Require operands to be in a proper layout."""
        # Any layout (addition is done)
        pass

    def subproblem_matrix(self, subproblem):
        """Build operator matrix for a specific subproblem."""
        return self.args[0].expression_matrices(subproblem, [self.operand])[self.operand]

    def operate(self, out):
        """Perform operation."""
        # OPTIMIZE: this has an extra copy
        arg0 = self.args[0]
        # Set output layout
        out.set_layout(arg0.layout)
        np.copyto(out.data, arg0.data)


class SphericalDivergence(Divergence, SphericalEllOperator):

    cs_type = coords.SphericalCoordinates

    def __init__(self, operand, index=0, out=None):
        Divergence.__init__(self, operand, out=out)
        if index != 0:
            raise ValueError("Divergence only implemented along index 0.")
        self.index = index
        coordsys = operand.tensorsig[index]
        SphericalEllOperator.__init__(self, operand, coordsys)
        # FutureField requirements
        self.domain  = operand.domain.substitute_basis(self.input_basis, self.output_basis)
        self.tensorsig = operand.tensorsig[:index] + operand.tensorsig[index+1:]
        self.dtype = operand.dtype

    @staticmethod
    def _output_basis(input_basis):
        out = input_basis._new_k(input_basis.k + 1)
        return out

    def check_conditions(self):
        """Check that operands are in a proper layout."""
        # Require radius to be in coefficient space
        layout = self.args[0].layout
        return (not layout.grid_space[self.radius_axis]) and (layout.local[self.radius_axis])

    def enforce_conditions(self):
        """Require operands to be in a proper layout."""
        # Require radius to be in coefficient space
        self.args[0].require_coeff_space(self.radius_axis)
        self.args[0].require_local(self.radius_axis)

    def regindex_out(self, regindex_in):
        # Regorder: -, +, 0
        # Divergence feels - and +
        if regindex_in[0] in (0, 1):
            return (regindex_in[1:],)
        else:
            return tuple()

    def radial_matrix(self, regindex_in, regindex_out, ell):
        radial_basis = self.radial_basis
        regtotal = radial_basis.regtotal(regindex_in)
        if regindex_in[0] != 2 and regindex_in[1:] == regindex_out:
            return self._radial_matrix(radial_basis, regindex_in[0], regtotal, ell)
        else:
            raise ValueError("This should never happen")

    @staticmethod
    @CachedMethod
    def _radial_matrix(radial_basis, regindex_in0, regtotal, ell):
        if regindex_in0 == 0:
            return radial_basis.xi(-1, ell+regtotal+1) * radial_basis.operator_matrix('D+', ell, regtotal)
        elif regindex_in0 == 1:
            return radial_basis.xi(+1, ell+regtotal-1) * radial_basis.operator_matrix('D-', ell, regtotal)
        else:
            raise ValueError("This should never happen")


class Curl(LinearOperator, metaclass=MultiClass):

    name = 'Curl'

    @classmethod
    def _preprocess_args(cls, operand, index=0, out=None):
        if isinstance(operand, Number):
            raise SkipDispatchException(output=0)
        return [operand], {'index': index, 'out': out}

    @classmethod
    def _check_args(cls, operand, index=0, out=None):
        # Dispatch by coordinate system
        if isinstance(operand, Operand):
            if isinstance(operand.tensorsig[index], cls.cs_type):
                return True
        return False

    def new_operand(self, operand, **kw):
        return Curl(operand, index=self.index, **kw)


class SphericalCurl(Curl, SphericalEllOperator):

    cs_type = coords.SphericalCoordinates

    def __init__(self, operand, index=0, out=None):
        Curl.__init__(self, operand, out=out)
        if index != 0:
            raise ValueError("Curl only implemented along index 0.")
        self.index = index
        coordsys = operand.tensorsig[index]
        SphericalEllOperator.__init__(self, operand, coordsys)
        # FutureField requirements
        self.domain  = operand.domain.substitute_basis(self.input_basis, self.output_basis)
        self.tensorsig = (coordsys,) + operand.tensorsig[:index] + operand.tensorsig[index+1:]
        self.dtype = operand.dtype

    @staticmethod
    def _output_basis(input_basis):
        out = input_basis._new_k(input_basis.k + 1)
        return out

    def check_conditions(self):
        """Check that operands are in a proper layout."""
        # Require radius to be in coefficient space
        layout = self.args[0].layout
        return (not layout.grid_space[self.radius_axis]) and (layout.local[self.radius_axis])

    def enforce_conditions(self):
        """Require operands to be in a proper layout."""
        # Require radius to be in coefficient space
        self.args[0].require_coeff_space(self.radius_axis)
        self.args[0].require_local(self.radius_axis)

    def regindex_out(self, regindex_in):
        # Regorder: -, +, 0
        # - and + map to 0
        if regindex_in[0] in (0, 1):
            return ((2,) + regindex_in[1:],)
        # 0 maps to - and +
        else:
            return ((0,) + regindex_in[1:], (1,) + regindex_in[1:])

    def radial_matrix(self, regindex_in, regindex_out, ell):
        radial_basis = self.radial_basis
        regtotal_in = radial_basis.regtotal(regindex_in)
        regtotal_out = radial_basis.regtotal(regindex_out)
        if regindex_in[1:] == regindex_out[1:]:
            return self._radial_matrix(radial_basis, regindex_in[0], regindex_out[0], regtotal_in, regtotal_out, ell)
        else:
            raise ValueError("This should never happen")

    @staticmethod
    @CachedMethod
    def _radial_matrix(radial_basis, regindex_in0, regindex_out0, regtotal_in, regtotal_out, ell):
        if regindex_in0 == 0 and regindex_out0 == 2:
            return -1j * radial_basis.xi(+1, ell+regtotal_in+1) * radial_basis.operator_matrix('D+', ell, regtotal_in)
        elif regindex_in0 == 1 and regindex_out0 == 2:
            return 1j * radial_basis.xi(-1, ell+regtotal_in-1) * radial_basis.operator_matrix('D-', ell, regtotal_in)
        elif regindex_in0 == 2 and regindex_out0 == 0:
            return -1j * radial_basis.xi(+1, ell+regtotal_in) * radial_basis.operator_matrix('D-', ell, regtotal_in)
        elif regindex_in0 == 2 and regindex_out0 == 1:
            return 1j * radial_basis.xi(-1, ell+regtotal_in) * radial_basis.operator_matrix('D+', ell, regtotal_in)
        else:
            raise ValueError("This should never happen")

    def subproblem_matrix(self, subproblem):
        if self.dtype == np.complex128:
            return super().subproblem_matrix(subproblem)
        elif self.dtype == np.float64:
            operand = self.args[0]
            radial_basis = self.radial_basis
            R_in = radial_basis.regularity_classes(operand.tensorsig)
            R_out = radial_basis.regularity_classes(self.tensorsig)  # Should this use output_basis?
            ell = subproblem.group[self.last_axis - 1]
            # Loop over components
            submatrices = []
            for regindex_out, regtotal_out in np.ndenumerate(R_out):
                submatrix_row = []
                for regindex_in, regtotal_in in np.ndenumerate(R_in):
                    # Build identity matrices for each axis
                    subshape_in = subproblem.coeff_shape(self.operand.domain)
                    subshape_out = subproblem.coeff_shape(self.domain)
                    # Check if regularity component exists for this ell
                    if (regindex_out in self.regindex_out(regindex_in)) and radial_basis.regularity_allowed(ell, regindex_in) and radial_basis.regularity_allowed(ell, regindex_out):
                        factors = [sparse.eye(m, n, format='csr') for m, n in zip(subshape_out, subshape_in)]
                        radial_matrix = self.radial_matrix(regindex_in, regindex_out, ell)
                        # Real part
                        factors[self.last_axis] = radial_matrix.real
                        comp_matrix_real = reduce(sparse.kron, factors, 1).tocsr()
                        # Imaginary pary
                        m_size = subshape_in[self.first_axis]
                        mult_1j = np.array([[0, -1], [1, 0]])
                        m_blocks = sparse.eye(m_size//2, m_size//2, format='csr')
                        factors[self.first_axis] = sparse.kron(mult_1j, m_blocks)
                        factors[self.last_axis] = radial_matrix.imag
                        comp_matrix_imag = reduce(sparse.kron, factors, 1).tocsr()
                        comp_matrix = comp_matrix_real + comp_matrix_imag
                    else:
                        # Build zero matrix
                        comp_matrix = sparse.csr_matrix((np.prod(subshape_out), np.prod(subshape_in)))
                    submatrix_row.append(comp_matrix)
                submatrices.append(submatrix_row)
            matrix = sparse.bmat(submatrices)
            matrix.tocsr()
            return matrix

    def operate(self, out):
        """Perform operation."""
        if self.dtype == np.complex128:
            return super().operate(out)
        operand = self.args[0]
        input_basis = self.input_basis
        radial_basis = self.radial_basis
        axis = radial_basis.radial_axis
        # Set output layout
        out.set_layout(operand.layout)
        out.data[:] = 0
        # Apply operator
        R_in = radial_basis.regularity_classes(operand.tensorsig)
        slices = [slice(None) for i in range(input_basis.dist.dim)]
        for regindex_in, regtotal_in in np.ndenumerate(R_in):
            for regindex_out in self.regindex_out(regindex_in):
                comp_in = operand.data[regindex_in]
                comp_out = out.data[regindex_out]
                # Should reorder to make ell loop first, check forbidden reg, remove reg from radial_vector_3
                for ell, m_ind, ell_ind in input_basis.ell_maps:
                    allowed_in  = radial_basis.regularity_allowed(ell, regindex_in)
                    allowed_out = radial_basis.regularity_allowed(ell, regindex_out)
                    if allowed_in and allowed_out:
                        slices[axis-2] = m_ind
                        slices[axis-1] = ell_ind
                        slices[axis] = radial_basis.n_slice(ell)
                        cos_slice = axslice(axis-2, 0, None, 2)
                        msin_slice = axslice(axis-2, 1, None, 2)
                        vec_in_cos = comp_in[tuple(slices)][cos_slice]
                        vec_in_msin = comp_in[tuple(slices)][msin_slice]
                        vec_in_complex = vec_in_cos + 1j*vec_in_msin
                        A = self.radial_matrix(regindex_in, regindex_out, ell)
                        vec_out_complex = apply_matrix(A, vec_in_complex, axis=axis)
                        comp_out[tuple(slices)][cos_slice] += vec_out_complex.real
                        comp_out[tuple(slices)][msin_slice] += vec_out_complex.imag


class Laplacian(LinearOperator, metaclass=MultiClass):

    name = "Lap"

    @classmethod
    def _preprocess_args(cls, operand, coordsys, out=None):
        if isinstance(operand, Number):
            raise SkipDispatchException(output=0)
        return [operand, coordsys], {'out': out}

    @classmethod
    def _check_args(cls, operand, coordsys, out=None):
        # Dispatch by coordinate system
        if isinstance(operand, Operand):
            if isinstance(coordsys, cls.cs_type):
                return True
        return False

    def new_operand(self, operand, **kw):
        return Laplacian(operand, self.coordsys, **kw)


class CartesianLaplacian(Laplacian):

    cs_type = coords.CartesianCoordinates

    def __init__(self, operand, coordsys, out=None):
        parts = [Differentiate(Differentiate(operand, c), c) for c in coordsys.coords]
        arg = sum(parts)
        LinearOperator.__init__(self, arg, out=out)
        self.coordsys = coordsys
        # LinearOperator requirements
        self.operand = operand
        # FutureField requirements
        self.domain = arg.domain
        self.tensorsig = arg.tensorsig
        self.dtype = arg.dtype

    def matrix_dependence(self, *vars):
        return self.args[0].matrix_dependence(*vars)

    def matrix_coupling(self, *vars):
        return self.args[0].matrix_coupling(*vars)

    def subproblem_matrix(self, subproblem):
        """Build operator matrix for a specific subproblem."""
        return self.args[0].expression_matrices(subproblem, [self.operand])[self.operand]

    def check_conditions(self):
        """Check that operands are in a proper layout."""
        # Any layout (addition is done)
        return True

    def enforce_conditions(self):
        """Require operands to be in a proper layout."""
        # Any layout (addition is done)
        pass

    def operate(self, out):
        """Perform operation."""
        # OPTIMIZE: this has an extra copy
        arg0 = self.args[0]
        # Set output layout
        out.set_layout(arg0.layout)
        np.copyto(out.data, arg0.data)


class SphericalLaplacian(Laplacian, SphericalEllOperator):

    cs_type = coords.SphericalCoordinates

    def __init__(self, operand, coordsys, out=None):
        Laplacian.__init__(self, operand, out=out)
        SphericalEllOperator.__init__(self, operand, coordsys)
        # FutureField requirements
        self.domain  = operand.domain.substitute_basis(self.input_basis, self.output_basis)
        self.tensorsig = operand.tensorsig
        self.dtype = operand.dtype

    @staticmethod
    def _output_basis(input_basis):
        out = input_basis._new_k(input_basis.k + 2)
        return out

    def check_conditions(self):
        """Check that operands are in a proper layout."""
        # Require radius to be in coefficient space
        layout = self.args[0].layout
        return (not layout.grid_space[self.radius_axis]) and (layout.local[self.radius_axis])

    def enforce_conditions(self):
        """Require operands to be in a proper layout."""
        # Require radius to be in coefficient space
        self.args[0].require_coeff_space(self.radius_axis)
        self.args[0].require_local(self.radius_axis)

    def regindex_out(self, regindex_in):
        return (regindex_in,)

    def radial_matrix(self, regindex_in, regindex_out, ell):
        radial_basis = self.radial_basis
        regtotal = radial_basis.regtotal(regindex_in)
        if regindex_in == regindex_out:
            return self._radial_matrix(radial_basis, regtotal, ell)
        else:
            raise ValueError("This should never happen")

    @staticmethod
    @CachedMethod
    def _radial_matrix(radial_basis, regtotal, ell):
        return radial_basis.operator_matrix('L', ell, regtotal)


class SphericalEllProduct(SphericalEllOperator, metaclass=MultiClass):

    name = "SphericalEllProduct"

    @classmethod
    def _preprocess_args(cls, operand, coordsys, ell_func, out=None):
        if isinstance(operand, Number):
            raise SkipDispatchException(output=0)
        return [operand, coordsys, ell_func], {'out': out}

    def new_operand(self, operand, **kw):
        return SphericalEllProduct(operand, self.coordsys, self.ell_func, **kw)


class SphericalEllProductField(SphericalEllProduct):

    @classmethod
    def _check_args(cls, operand, coordsys, ell_func, out=None):
        return isinstance(operand, Operand)

    def __init__(self, operand, coordsys, ell_func, out=None):
        SpectralOperator.__init__(self, operand, out=out)
        SphericalEllOperator.__init__(self, operand, coordsys)
        self.ell_func = ell_func
        # FutureField requirements
        self.domain  = operand.domain
        self.tensorsig = operand.tensorsig
        self.dtype = operand.dtype

    @staticmethod
    def _output_basis(input_basis):
        return input_basis

    def check_conditions(self):
        """Check that operands are in a proper layout."""
        # Require radius to be in coefficient space
        layout = self.args[0].layout
        return (not layout.grid_space[self.radius_axis]) and (layout.local[self.radius_axis])

    def enforce_conditions(self):
        """Require operands to be in a proper layout."""
        # Require radius to be in coefficient space
        self.args[0].require_coeff_space(self.radius_axis)
        self.args[0].require_local(self.radius_axis)

    def regindex_out(self, regindex_in):
        return (regindex_in,)

    def radial_matrix(self, regindex_in, regindex_out, ell):
        radial_basis = self.radial_basis
        regtotal = radial_basis.regtotal(regindex_in)
        if regindex_in == regindex_out:
            return self._radial_matrix(ell, regtotal)
        else:
            raise ValueError("This should never happen")

    @CachedMethod
    def _radial_matrix(self, ell, regtotal):
        return self.ell_func(ell + regtotal) * self.radial_basis.operator_matrix('Id', ell, regtotal)


"""
Field has no bases built on cs:
    return zero

Field has basis in cs:
    grad_b f(s2)
    still want full vector field on b

Field has basis in cs and others:
    grad_b f(x, s2)
    same as above, with extra bases

Field has basis on full cs:
    grad_b f(b)
    makes sense

Field has basis on full cs and others:
    grad_b f(x, b)
    same as above, with extra bases



f(Fourer, Chebyshev, Chebyshev)

k(I*I), (D*I), (I*D)


"""
