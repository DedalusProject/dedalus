"""
Abstract and built-in classes defining deferred operations on fields.

"""

from collections import defaultdict
from functools import partial, reduce
import numpy as np
from scipy import sparse

from .domain import Subdomain
from .field import Operand, Data, Array, Field
from .future import Future, FutureArray, FutureField
from ..tools.array import reshape_vector, apply_matrix, add_sparse
from ..tools.dispatch import MultiClass
from ..tools.exceptions import NonlinearOperatorError
from ..tools.exceptions import SymbolicParsingError
from ..tools.exceptions import UndefinedParityError
from ..tools.general import unify, unify_attributes


class Rand:
    pass
Scalar = FutureScalar = Rand


# Use simple decorator to track parseable operators
parseables = {}

def parseable(op):
    parseables[op.name] = op
    return op

def addname(name):
    def decorator(func):
        func.name = name
        return func
    return decorator


# Other helpers
def is_integer(x):
    if isinstance(x, int):
        return True
    else:
        return x.is_integer()


class FieldCopy(FutureField, metaclass=MultiClass):
    """Operator making a new field copy of data."""

    name = 'FieldCopy'

    @classmethod
    def _preprocess_args(cls, arg, domain, **kw):
        arg = Operand.cast(arg, domain=domain)
        return (arg,), kw

    @classmethod
    def _check_args(cls, *args, **kw):
        match = (isinstance(args[i], types) for i,types in cls.argtypes.items())
        return all(match)

    def __init__(self, arg, **kw):
        super().__init__(arg, **kw)
        self.kw = {'domain': arg.domain}

    def _build_bases(self, arg0):
        if all(basis is None for basis in arg0.bases):
            return arg0.domain
        else:
            return arg0.bases

    def __str__(self):
        return str(self.args[0])

    def check_conditions(self):
        return True

    def base(self):
        return FieldCopy

    def sym_diff(self, var):
        """Symbolically differentiate with respect to var."""
        return self.args[0].sym_diff(var)


class FieldCopyScalar(FieldCopy):

    argtypes = {0: (Scalar, FutureScalar)}

    def operate(self, out):
        if self.args[0].value == 0:
            # Copy in coeff layout
            out.set_layout(self._coeff_layout)
            out.data.fill(0)
        else:
            # Copy in grid layout
            out.set_layout(self._grid_layout)
            np.copyto(out.data, self.args[0].value)

    def __eq__(self, other):
        return self.args[0].__eq__(other)


class FieldCopyArray(FieldCopy):

    argtypes = {0: (Array, FutureArray)}

    def operate(self, out):
        # Copy in grid layout
        out.set_layout(self._grid_layout)
        np.copyto(out.data, self.args[0].data)


class FieldCopyField(FieldCopy):

    argtypes = {0: (Field, FutureField)}

    def operate(self, out):
        arg0, = self.args
        # Copy in current layout
        out.set_layout(arg0.layout)
        np.copyto(out.data, arg0.data)


class Operator(Future):

    def base(self):
        return type(self)

    def order(self, *ops):
        order = max(arg.order(*ops) for arg in self.args)
        if type(self) in ops:
            order += 1
        return order


class NonlinearOperator(Operator):

    def expand(self, *vars):
        """Return self."""
        return self

    def canonical_linear_form(self, *vars):
        """Raise if arguments contain specified variables (default: None)"""
        if self.has(*vars):
            raise NonlinearOperatorError("{} is a non-linear function of the specified variables.".format(str(self)))
        else:
            return self

    def factor(self, *vars):
        """Produce operator-factor dictionary over specified variables."""
        if self.has(*vars):
            return defaultdict(int, {self: 1})
        else:
            return defaultdict(int, {1: self})

    def split(self, *vars):
        if self.has(*vars):
            return [self, 0]
        else:
            return [0, self]


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


class UnaryGridFunction(NonlinearOperator, Future, metaclass=MultiClass):

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
        self.name = func.__name__

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
            raise UndefinedParityError("Unknown action of {} on odd parity.".format(self.name))

    def sym_diff(self, var):
        """Symbolically differentiate with respect to var."""
        diffmap = {np.absolute: lambda x: np.sign(x),
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
        arg0 = self.args[0]
        diff0 = arg0.sym_diff(var)
        return diffmap[self.func](arg0) * diff0


class UnaryGridFunctionField(UnaryGridFunction, FutureField):

    argtypes = {1: (Field, FutureField)}

    def check_conditions(self):
        # Field must be in grid layout
        return (self.args[0].layout is self._grid_layout)

    def operate(self, out):
        # References
        arg0, = self.args
        # Evaluate in grid layout
        arg0.require_grid_space()
        out.set_layout(self._grid_layout)
        self.func(arg0.data, out=out.data)


class Arithmetic(Future):

    arity = 2

    def __str__(self):
        def substring(arg):
            if isinstance(arg, Arithmetic):
                return '({})'.format(arg)
            else:
                return str(arg)
        str_args = map(substring, self.args)
        return '%s' %self.str_op.join(str_args)

    def order(self, *ops):
        return max(arg.order(*ops) for arg in self.args)


class Add(Arithmetic, metaclass=MultiClass):

    name = 'Add'
    str_op = ' + '

    def __init__(self, arg0, arg1, out=None):
        typecheck((arg0, Field), (arg1, Field))

        bases = self._build_bases(arg0, arg1)
        self.subdomain, self.bases = Subdomain.from_bases(bases)
        self.domain = self.subdomain.domain
        arg0 = convert(arg0, self.bases)
        arg1 = convert(arg1, self.bases)
        super().__init__(arg0, arg1, out=out)
        #self.args = [arg0, arg1]
        #self.out = out

    def _build_bases(self, arg0, arg1):
        bases = []
        for b0, b1 in zip(arg0.bases, arg1.bases):
            if (b0 is None) and (b1 is None):
                bases.append(None)
            else:
                bases.append(b0 + b1)
        if all(basis is None for basis in bases):
            bases = arg0.domain
        return bases

    @classmethod
    def _preprocess_args(cls, *args, **kw):
        domain = unify_attributes(args, 'domain', require=False)
        args = tuple(Operand.cast(arg, domain) for arg in args)
        return args, kw

    @classmethod
    def _check_args(cls, *args, **kw):
        match = (isinstance(args[i], types) for i,types in cls.argtypes.items())
        return all(match)

    def base(self):
        return Add

    def expand(self, *vars):
        """Expand arguments containing specified variables (default: all)."""
        arg0, arg1 = self.args
        if (not vars) or arg0.has(*vars):
            arg0 = arg0.expand(*vars)
        if (not vars) or arg1.has(*vars):
            arg1 = arg1.expand(*vars)
        if (not vars) or arg0.has(*vars) or arg1.has(*vars):
            return arg0 + arg1
        else:
            return self

    def canonical_linear_form(self, *vars):
        """Ensure arguments have same dependency on specified variables."""
        arg0, arg1 = self.args
        if arg0.has(*vars) and arg1.has(*vars):
            arg0 = arg0.canonical_linear_form(*vars)
            arg1 = arg1.canonical_linear_form(*vars)
            return arg0 + arg1
        elif arg0.has(*vars) or arg1.has(*vars):
            raise NonlinearOperatorError("Cannot add dependent and independent terms.")
        else:
            return self

    def separability(self, vars):
        arg0, arg1 = self.args
        sep0 = arg0.separability(vars)
        sep1 = arg1.separability(vars)
        return (sep0 & sep1)

    def factor(self, *vars):
        """Produce operator-factor dictionary over specified variables."""
        out = defaultdict(int)
        F0 = self.args[0].factor(*vars)
        F1 = self.args[1].factor(*vars)
        for f in set().union(F0, F1):
            out[f] = F0[f] + F1[f]
        return out

    def split(self, *vars):
        S0 = self.args[0].split(*vars)
        S1 = self.args[1].split(*vars)
        return [S0[0]+S1[0], S0[1]+S1[1]]

    def operator_dict(self, subsystem, vars, **kw):
        """Produce matrix-operator dictionary over specified variables."""
        # May need to convert None bases up to self bases
        # Vars will only appear in op dicts if subsystem has None groups
        out = defaultdict(int)
        op0 = self.args[0].operator_dict(subsystem, vars, **kw)
        op1 = self.args[1].operator_dict(subsystem, vars, **kw)
        convert0 = subsystem.compute_conversion(self.args[0].bases, self.bases)
        convert1 = subsystem.compute_conversion(self.args[1].bases, self.bases)
        for var in op0:
            out[var] += convert0 * op0[var]
        for var in op1:
            out[var] += convert1 * op1[var]
        return out

    def sym_diff(self, var):
        """Symbolically differentiate with respect to var."""
        arg0, arg1 = self.args
        diff0 = arg0.sym_diff(var)
        diff1 = arg1.sym_diff(var)
        return diff0 + diff1

    def add_subdata(self, arg, out):
        # (Only called if out.data.size != 0)
        arg_slices, out_slices = [], []
        for axis in range(self.domain.dim):
            if arg.bases[axis] is out.bases[axis]:
                # (Both None or both not None)
                # Directly add all data
                arg_slices.append(slice(None))
                out_slices.append(slice(None))
            else:
                # (arg basis is None)
                if out.layout.grid_space[axis]:
                    # Broadcast addition
                    arg_slices.append(slice(None))
                    out_slices.append(slice(None))
                else:
                    # Select constant mode
                    #const_slice = arg.layout.select_global(0, axis=axis)
                    if out.global_start[axis] == 0:
                        const_slice = slice(1)
                    else:
                        const_slice = slice(0)
                    arg_slices.append(const_slice)
                    out_slices.append(const_slice)
        arg_data = arg.data[tuple(arg_slices)]
        out_data = out.data[tuple(out_slices)]
        np.add(arg_data, out_data, out=out_data)


class AddFieldField(Add, FutureField):

    argtypes = {0: (Field, FutureField),
                1: (Field, FutureField)}

    def check_conditions(self):
        # Layouts must match
        return (self.args[0].layout is self.args[1].layout)

    def enforce_conditions(self):
        arg0, arg1 = self.args
        # Add in arg0 layout (arbitrary choice)
        arg1.require_layout(arg0.layout)

    def operate(self, out):
        arg0, arg1 = self.args
        out.set_layout(arg0.layout)
        if out.data.size:
            out.data.fill(0)
            self.add_subdata(arg0, out)
            self.add_subdata(arg1, out)


class Multiply(Arithmetic, metaclass=MultiClass):

    name = 'Mul'
    str_op = '*'

    # def __init__(self, arg0, arg1, out=None):
    #     super().__init__()
    #     self.domain = unify([arg0.domain, arg1.domain])
    #     self._build_bases(arg0, arg1)
    #     self.args = [arg0, arg1]

    def _build_bases(self, arg0, arg1):
        bases = []
        for b0, b1 in zip(arg0.bases, arg1.bases):
            if (b0 is None) and (b1 is None):
                bases.append(None)
            else:
                bases.append(b0 * b1)
        if all(basis is None for basis in bases):
            bases = arg0.domain
        return bases

    @classmethod
    def _preprocess_args(cls, *args, **kw):
        domain = unify_attributes(args, 'domain', require=False)
        args = tuple(Operand.cast(arg, domain) for arg in args)
        return args, kw

    @classmethod
    def _check_args(cls, *args, **kw):
        match = (isinstance(args[i], types) for i,types in cls.argtypes.items())
        return all(match)

    def base(self):
        return Multiply

    def expand(self, *vars):
        """Distribute over sums containing specified variables (default: all)."""
        arg0, arg1 = self.args
        if (not vars) or arg0.has(*vars):
            arg0 = self.args[0].expand(*vars)
            if isinstance(arg0, Add):
                arg0a, arg0b = arg0.args
                return (arg0a*arg1 + arg0b*arg1).expand(*vars)
        if (not vars) or arg1.has(*vars):
            arg1 = self.args[1].expand(*vars)
            if isinstance(arg1, Add):
                arg1a, arg1b = arg1.args
                return (arg0*arg1a + arg0*arg1b).expand(*vars)
        return self

    def canonical_linear_form(self, *vars):
        """Eliminate nonlinear multiplications and float specified variables right."""
        arg0, arg1 = self.args
        if arg0.has(*vars) and arg1.has(*vars):
            raise NonlinearOperatorError("Cannot multiply two linear terms.")
        elif arg0.has(*vars):
            arg0 = arg0.canonical_linear_form(*vars)
            if isinstance(arg0, Multiply):
                arg0a, arg0b = arg0.args
                return (arg0a * arg1) * arg0b
            else:
                return arg1 * arg0
        elif arg1.has(*vars):
            arg1 = arg1.canonical_linear_form(*vars)
            if isinstance(arg1, Multiply):
                arg1a, arg1b = arg1.args
                return (arg0 * arg1a) * arg1b
            else:
                return arg0 * arg1
        else:
            return self

    def factor(self, *vars):
        """Produce operator-factor dictionary over specified variables."""
        out = defaultdict(int)
        F0 = self.args[0].factor(*vars)
        F1 = self.args[1].factor(*vars)
        for f0 in F0:
            for f1 in F1:
                out[f0*f1] += F0[f0] * F1[f1]
        return out

    def split(self, *vars):
        S0 = self.args[0].split(*vars)
        S1 = self.args[1].split(*vars)
        return [S0[0]*S1[0] + S0[0]*S1[1] + S0[1]*S1[0], S0[1]*S1[1]]

    def operator_dict(self, subsystem, vars, **kw):
        """Produce matrix-operator dictionary over specified variables."""
        arg0, arg1 = self.args
        out = defaultdict(int)
        op0 = arg0.as_ncc_operator(subsystem, self.bases, **kw)
        op1 = arg1.operator_dict(subsystem, vars, **kw)
        #convert0 = self.subsystem_conversion(subsystem, arg0)
        convert1 = self.subsystem_conversion(subsystem, arg1)
        for var in op1:
            out[var] = op0 * convert1 * op1[var]
        return out

    def subsystem_conversion(self, subsystem, arg):
        axmats = subsystem.compute_identities(self.bases)
        for axis, (inbasis, outbasis) in enumerate(zip(arg.bases, self.bases)):
            if (inbasis is None) and (outbasis is not None):
                axmats[axis] = axmats[axis][:, 0:1]
        return reduce(sparse.kron, axmats, 1).tocsr()

    def separability(self, vars):
        """Determine separability as linear operator over specified variables."""
        # Assume canonical linear form: arg1 linearly depends on vars
        arg0, arg1 = self.args
        # NCC multiplication is separable on constant axes
        sep0 = [basis is None for basis in arg0.bases]
        sep1 = arg1.separability(vars)
        return (sep0 & sep1)

    def sym_diff(self, var):
        """Symbolically differentiate with respect to var."""
        arg0, arg1 = self.args
        diff0 = arg0.sym_diff(var)
        diff1 = arg1.sym_diff(var)
        return (diff0*arg1 + arg0*diff1)

    # def simplify(self, retain):
    #     arg0 = self.args[0].simplify(retain)
    #     arg1 = self.arts[0].simplify(retain)

    #     if arg0 not in retain:
    #         if arg0 == 0:
    #             return 0
    #         elif arg0 == 1:
    #             return arg1
    #     if arg1 not in retain:
    #         elif arg1 == 0:
    #             return 0
    #         elif arg1 == 1:
    #             return arg0
    #     return (arg0 * arg1)

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        arg0, arg1 = self.args
        for axis, (b0, b1) in enumerate(zip(arg0.bases, arg1.bases)):
            if (b0 is not None) and (b1 is not None):
                self.require_grid_axis = axis
                break
        else:
            self.require_grid_axis = None

    def check_conditions(self):
        layout0 = self.args[0].layout
        layout1 = self.args[1].layout
        # Fields must be in grid layout
        if self.require_grid_axis:
            axis = self.require_grid_axis
            return (layout0.grid_space[axis] and (layout0 is layout1))
        else:
            return (layout0 is layout1)

    def enforce_conditions(self):
        arg0, arg1 = self.args
        layout = self.determine_layout()
        arg0.require_layout(layout)
        arg1.require_layout(layout)


        if self.require_grid_axis:
            axis = self.require_grid_axis
            arg0.require_grid_space(axis=axis)
        arg1.require_layout(arg0.layout)

    def operate(self, out):
        arg0, arg1 = self.args
        out.set_layout(arg0.layout)
        if out.data.size:
            np.multiply(arg0.data, arg1.data, out.data)


    ## Ideas for separating condition enforcement from operation to potentially
    ## trim down the boilerplate for the dispatching subclasses
    # def enforce_conditions(self):
    #     self.args[0].require_grid_space()
    #     self.args[1].require_grid_space()
    #     out.set_layout(self._grid_layout)
    # def _operate(self):
    #     np.multiply(self.args[0].data, self.args[1].data, out.data)



class Power(NonlinearOperator, Arithmetic):

    name = 'Pow'
    str_op = '**'

    def __new__(cls, arg0, arg1, out=None):
        if arg1 == 0:
            return 1
        elif arg1 == 1:
            return arg0
        else:
            return object.__new__(cls)

    def sym_diff(self, var):
        """Symbolically differentiate with respect to var."""
        arg0, arg1 = self.args
        diff0 = arg0.sym_diff(var)
        return arg1 * arg0**(arg1-1) * diff0

    def base(self):
        return Power

    def check_conditions(self):
        # Field must be in grid layout
        return (self.args[0].layout is self._grid_layout)

    def operate(self, out):
        arg0, arg1 = self.args
        # Raise in grid layout
        arg0.require_grid_space()
        out.set_layout(self._grid_layout)
        np.power(arg0.data, arg1.value, out.data)


class LinearOperator(Operator, FutureField):

    kw = {}
    entry_scaling = 1

    def expand(self, *vars):
        """Distribute over sums containing specified variables (default: all)."""
        arg0 = self.args[0]
        if (not vars) or arg0.has(*vars):
            arg0 = arg0.expand(*vars)
            if isinstance(arg0, Add):
                op = type(self)
                arg0a, arg0b = arg0.args
                return (op(arg0a, **self.kw) + op(arg0b, **self.kw)).expand(*vars)
        return self

    def canonical_linear_form(self, *vars):
        """Change argument to canonical linear form."""
        arg0 = self.args[0]
        if arg0.has(*vars):
            op = type(self)
            arg0 = arg0.canonical_linear_form(*vars)
            return op(arg0, **self.kw)
        else:
            return self

    def factor(self, *vars):
        """Produce operator-factor dictionary over specified variables."""
        if self.has(*vars):
            return defaultdict(int, {self: 1})
        else:
            return defaultdict(int, {1: self})

    def split(self, *vars):
        if self.base() in vars:
            return [self, 0]
        else:
            S0 = self.args[0].split(*vars)
            return [self.base()(S0[0], **self.kw), self.base()(S0[1], **self.kw)]

    def operator_dict(self, subsystem, vars, **kw):
        op_dict = defaultdict(int)
        arg_dict = self.args[0].operator_dict(subsystem, vars, **kw)
        submat = self.subsystem_matrix(subsystem)
        for var in arg_dict:
            op_dict[var] = submat * arg_dict[var]
        return op_dict

    def subsystem_matrix(self, subsystem):
        axis = self.input_basis.space.axis
        inslice = subsystem.global_slices(self.args[0].bases)[axis]
        outslice = subsystem.global_slices(self.bases)[axis]
        axmats = subsystem.compute_identities(self.bases)
        axmats[axis] = self.memory_matrix()[outslice, inslice]
        return reduce(sparse.kron, axmats, 1).tocsr()

    def separability(self, vars):
        sep = self.args[0].separability(vars).copy()
        if not self.separable:
            sep[self.axis] = False
        return sep

    def sym_diff(self, var):
        """Symbolically differentiate with respect to var."""
        diff0 = self.args[0].sym_diff(var)
        return self.base()(diff0, **self.kw)

    def _build_bases(self, arg0):
        bases = [b for b in arg0.bases]
        bases[self.input_basis.space.axis] = self.output_basis
        if all(basis is None for basis in bases):
            bases = arg0.domain
        return bases

    # @classmethod
    # def mode_matrix(cls):
    #     I = len(cls.output_basis.modes)
    #     J = len(cls.input_basis.modes)
    #     dtype = cls.input_basis.domain.dtype
    #     M = sparse.lil_matrix((I, J), dtype=dtype)
    #     for i, mode_i in enumerate(cls.output_basis.modes):
    #         for j, mode_j in enumerate(cls.input_basis.modes):
    #             Mij = cls.entry(mode_i, mode_j)
    #             if Mij:
    #                 M[i, j] = Mij
    #     return M.tocsr()

    # @classmethod
    # def memory_matrix(cls):
    #     input_map = cls.input_basis.memory_map
    #     output_map = cls.output_basis.memory_map
    #     return (output_map * cls.mode_matrix() * input_map.T)

    def memory_matrix(self):
        entry_scaling = self.entry_scaling
        I = self.output_basis.space.coeff_size
        J = self.input_basis.space.coeff_size
        dtype = self.input_basis.domain.dtype
        M = sparse.lil_matrix((I, J), dtype=dtype)
        for i in range(I):
            for j in range(J):
                Mij = self.entry(i, j)
                if Mij:
                    M[i, j] = entry_scaling * Mij
        return M.tocsr()

    @classmethod
    def local_matrix(cls):
        global_matrix = self.memory_matrix()
        islice = cls.input_basis.local_coeff_slice
        oslice = cls.output_basis.local_coeff_slice
        return global_matrix[oslice, islice]

    def check_conditions(self):
        layout = self.args[0].layout
        is_coeff = not layout.grid_space[self.axis]
        is_local = layout.local[self.axis]
        if self.separable:
            return is_coeff
        else:
            return (is_coeff and is_local)

    def enforce_conditions(self):
        self.args[0].require_coeff_space(self.axis)
        if not self.separable:
            self.args[0].require_local(self.axis)

    def operate(self, out):
        arg0, = self.args
        axis = self.axis
        # Apply matrix form
        out.set_layout(arg0.layout)
        matrix = self.memory_matrix()
        if self.separable:
            local_slice = arg0.layout.slices(arg0.subdomain, arg0.scales)[self.axis]
            matrix = matrix[local_slice, local_slice]
        apply_matrix(matrix, arg0.data, axis, out=out.data)


class LinearFunctional(LinearOperator):

    output_basis = None
    separable = False
    entry_scaling = 1

    def check_conditions(self):
        layout = self.args[0].layout
        is_coeff = not layout.grid_space[self.axis]
        is_local = layout.local[self.axis]
        return (is_coeff and is_local)

    def enforce_conditions(self):
        self.args[0].require_coeff_space(self.axis)
        self.args[0].require_local(self.axis)

    def matrix_form(self):
        # Dense vector
        entry_scaling = self.entry_scaling
        J = self.input_basis.space.coeff_size
        M = np.zeros((1,J), dtype=self.domain.dtype)
        for j in range(J):
            M[0,j] = entry_scaling * self.entry(j, **self.kw)
        return M

    def memory_matrix(self):
        return self.matrix_form()

    def explicit_form(self, input, output, axis):
        dim = self.domain.dim
        weights = reshape_vector(self.matrix_form(), dim=dim, axis=axis)
        interp = np.sum(input * weights, axis=axis, keepdims=True)
        np.copyto(output[axslice(axis, 0, 1)], interp)
        np.copyto(output[axslice(axis, 1, None)], 0)


class TimeDerivative(LinearOperator, FutureField):

    name = 'dt'

    def _build_bases(self, arg0):
        return arg0.bases

    def base(self):
        return TimeDerivative

    def factor(self, *vars):
        """Produce operator-factor dictionary over specified variables."""
        if type(self) in vars:
            out = defaultdict(int)
            F0 = self.args[0].factor(*vars)
            for f in F0:
                out[f*self._scalar] = F0[f]
            return out
        else:
            return defaultdict(int, {1: self})

    def operator_form(self, index):
        raise ValueError("Operator form not available for time derivative.")

    def operate(self, out):
        raise ValueError("Cannot evaluate time derivative operator.")


@parseable
class Integrate(LinearFunctional):

    name = 'integ'

    def __new__(cls, arg0, out=None):
        # Cast to operand
        arg0 = Operand.cast(arg0)
        # Check if operand depends on basis
        if (cls.basis not in arg0.domain.bases) or (arg0.meta[cls.basis.name]['constant']):
            length = cls.basis.interval[1] - cls.basis.interval[0]
            integral = arg0*length
            integral.out = out
            return integral
        else:
            return object.__new__(cls)

    def __init__(self, arg0, **kw):
        # Cast argument to field
        arg0 = Field.cast(arg0, arg0.domain)
        super().__init__(arg0, **kw)
        self.axis = self.domain.bases.index(self.basis)

    @property
    def entry_scaling(self):
        return self.input_basis.space.COV.stretch


@parseable
@addname('integ')
def integrate(arg0, *bases, out=None):
    # Cast to operand
    arg0 = Operand.cast(arg0)
    # No bases: integrate over whole domain
    if len(bases) == 0:
        bases = arg0.domain.bases
    # Multiple bases: apply recursively
    if len(bases) > 1:
        arg0 = integrate(arg0, *bases[:-1])
    # Call with single basis
    basis = arg0.domain.get_basis_object(bases[-1])
    return basis.Integrate(arg0, out=out)


@parseable
class Interpolate(LinearFunctional):

    name = 'interp'

    def __new__(cls, arg0, position, out=None):
        # Cast to operand
        #arg0 = Operand.cast(arg0)
        # Check if operand depends on basis
        if arg0 == 0:
            return 0
        elif (cls.input_basis not in arg0.bases):
            return arg0
        else:
            return object.__new__(cls)

    def __init__(self, arg0, position, out=None):
        # Cast argument to field
        arg0 = Field.cast(arg0, arg0.domain)
        super().__init__(arg0, out=out)
        self.kw = {'position': position}
        self.position = position
        self.axis = self.input_basis.space.axis

    def distribute(self):
        arg0, = self.args
        if not isinstance(arg0, Add):
            raise ValueError("Can only apply distributive rule to a sum.")
        a, b = arg0.args
        op = type(self)
        return op(a, self.position) + op(b, self.position)

    def __repr__(self):
        return 'interp(%r, %r, %r)' %(self.args[0], self.input_basis, self.position)

    def __str__(self):
        return "interp({},'{}',{})".format(self.args[0], self.input_basis, self.position)


@parseable
@addname('interp')
def interpolate(arg0, out=None, **basis_kw):
    # Cast to operand
    #arg0 = Operand.cast(arg0)
    if isinstance(arg0, (Scalar, FutureScalar)):
        return arg0
    # Require at least one basis
    if len(basis_kw) == 0:
        raise ValueError("No basis specified.")
    # Unpack bases
    bases = list(basis_kw.items())
    # Multiple bases: apply recursively
    if len(bases) > 1:
        arg0 = interpolate(arg0, **dict(bases[:-1]))
    # Call with single basis
    basis, position = bases[-1]
    basis = arg0.domain.get_space_object(basis)
    return basis.interpolate(arg0, position, out=out)


@parseable
class Filter(LinearFunctional):

    name = 'filter'

    def __new__(cls, arg0, func, out=None):
        # Cast to operand
        #arg0 = Operand.cast(arg0)
        # Check if operand depends on basis
        if arg0 == 0:
            return 0
        elif (cls.input_basis not in arg0.bases):
            return arg0
        else:
            return object.__new__(cls)

    def __init__(self, arg0, func, out=None):
        # Cast argument to field
        arg0 = Field.cast(arg0, arg0.domain)
        super().__init__(arg0, out=out)
        self.kw = {'func': func}
        self.func = func
        self.axis = self.input_basis.space.axis

    def distribute(self):
        arg0, = self.args
        if not isinstance(arg0, Add):
            raise ValueError("Can only apply distributive rule to a sum.")
        a, b = arg0.args
        op = type(self)
        return op(a, self.position) + op(b, self.position)

    def __repr__(self):
        return 'interp(%r, %r, %r)' %(self.args[0], self.input_basis, self.func)

    def __str__(self):
        return "interp({},'{}',{})".format(self.args[0], self.input_basis, self.func)

    @classmethod
    def entry(cls, mode, func):
        return func(mode)

@parseable
@addname('filter')
def filter(arg0, out=None, **basis_kw):
    # Cast to operand
    #arg0 = Operand.cast(arg0)
    if isinstance(arg0, (Scalar, FutureScalar)):
        return arg0
    # Require at least one basis
    if len(basis_kw) == 0:
        raise ValueError("No basis specified.")
    # Unpack bases
    bases = list(basis_kw.items())
    # Multiple bases: apply recursively
    if len(bases) > 1:
        arg0 = filter(arg0, **dict(bases[:-1]))
    # Call with single basis
    basis, func = bases[-1]
    basis = arg0.domain.get_space_object(basis)
    return basis.filter(arg0, func, out=out)



class Differentiate(LinearOperator):

    name = 'd'

    def __new__(cls, arg, out=None):
        # Cast to data and check bases
        arg = Data.cast(arg, cls.domain)
        arg_basis = arg.bases[cls.axis]
        if arg_basis is None:
            return 0
        elif arg_basis is cls.input_basis:
            return object.__new__(cls)
        else:
            raise ValueError("Basis mismatch.")

    def __init__(self, arg, out=None):
        # Cast arg to field
        arg = Field.cast(arg, self.domain)
        super().__init__(arg, out=out)

    @property
    def base(self):
        return self.input_basis.space.differentiate

    @property
    def entry_scaling(self):
        return (1 / self.input_basis.space.COV.stretch)

    def expand(self, *vars):
        """Distribute over sums and apply the product rule to arguments
        containing specified variables (default: all)."""
        arg, = self.args
        if (not vars) or arg.has(*vars):
            base = self.base
            arg = arg.expand(*vars)
            # Distribute over addition
            if isinstance(arg, Add):
                arg_a, arg_b = arg.args
                return (base(arg_a) + base(arg_b)).expand(*vars)
            # Apply product rule over multiplication
            if isinstance(arg, Multiply):
                arg_a, arg_b = arg.args
                return (base(arg_a)*arg_b + arg_a*base(arg_b)).expand(*vars)
        return self


@parseable
@addname('d')
def differentiate(arg, *spaces, out=None, **space_kw):
    """Differentiation factory."""
    # Parse keyword spaces
    for space, order in space_kw.items():
        spaces += (space,) * order
    # Require at least one space
    if len(spaces) == 0:
        raise ValueError("No spaces specified.")
    # Multiple spaces: apply recursively
    if len(spaces) > 1:
        arg = differentiate(arg, *spaces[:-1])
    # Single space: call space method
    space = arg.domain.get_space_object(spaces[-1])
    return space.differentiate(arg, out=out)


class Convert(LinearOperator):

    name = 'conv'

    def __new__(cls, arg0, **kw):
        # Cast to operand
        #arg0 = Operand.cast(arg0)
        # Check if operand depends on basis
        if arg0 == 0:
            return 0
        elif cls.input_basis not in arg0.bases:
            raise ValueError("arg does not depend on basis")
        else:
            return object.__new__(cls)

    def __init__(self, arg0, **kw):
        # Cast argument to field
        arg0 = Field.cast(arg0, arg0.domain)
        super().__init__(arg0, **kw)

    def expand(self, *vars):
        """Distribute over sums and apply the product rule to arguments
        containing specified variables (default: all)."""
        arg0 = self.args[0]
        if (not vars) or arg0.has(*vars):
            op = type(self)
            arg0 = arg0.expand(*vars)
            if isinstance(arg0, Add):
                arg0a, arg0b = arg0.args
                return (op(arg0a) + op(arg0b)).expand(*vars)
        return self


def convert(arg0, bases, out=None):
    # Cast to operand
    #arg0 = Operand.cast(arg0)
    # Return scalars
    if isinstance(arg0, Scalar):
        return arg0
    # Require at least one basis
    if len(bases) != arg0.domain.dim:
        raise ValueError("Full bases not specified.")
    # Convert iteratively
    for axis, out_basis in enumerate(bases):
        in_basis = arg0.bases[axis]
        if in_basis in [None, out_basis]:
            continue
        elif out_basis is None:
            raise ValueError("Cannot convert down to None")
        else:
            arg0 = in_basis.Convert(arg0)#, out_basis)
    return arg0


@parseable
class HilbertTransform(LinearOperator):

    name = 'Hilbert'

    def __new__(cls, arg0, **kw):
        # Cast to operand
        arg0 = Operand.cast(arg0)
        # Check if operand depends on basis
        if cls.basis not in arg0.domain.bases:
            return 0
        elif arg0.meta[cls.basis.name]['constant']:
            return 0
        else:
            return object.__new__(cls)

    def __init__(self, arg0, **kw):
        # Cast argument to field
        arg0 = Field.cast(arg0, arg0.domain)
        super().__init__(arg0, **kw)
        self.axis = self.domain.bases.index(self.basis)


@parseable
@addname('H')
def hilberttransform(arg0, *bases, out=None, **basis_kw):
    # Cast to operand
    arg0 = Operand.cast(arg0)
    if isinstance(arg0, (Scalar, FutureScalar)):
        return 0
    # Parse keyword bases
    for basis, order in basis_kw.items():
        bases += (basis,) * order
    # Require at least one basis
    if len(bases) == 0:
        raise ValueError("No basis specified.")
    # Multiple bases: apply recursively
    if len(bases) > 1:
        arg0 = hilberttransform(arg0, *bases[:-1])
    # Call with single basis
    basis = arg0.domain.get_basis_object(bases[-1])
    return basis.HilbertTransform(arg0, out=out)

