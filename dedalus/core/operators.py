"""
Abstract and built-in classes defining deferred operations on fields.

"""

from collections import defaultdict
from functools import partial
import numpy as np

from .field import Operand, Data, Scalar, Array, Field
from .future import Future, FutureScalar, FutureArray, FutureField
from ..tools.array import reshape_vector, apply_matrix, add_sparse
from ..tools.dispatch import MultiClass
from ..tools.exceptions import NonlinearOperatorError
from ..tools.exceptions import SymbolicParsingError
from ..tools.exceptions import UndefinedParityError


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


class Operator(Future):

    @property
    def base(self):
        return type(self)

    def order(self, *ops):
        order = max(arg.order(*ops) for arg in self.args)
        if type(self) in ops:
            order += 1
        return order


class FieldCopy(Operator, FutureField, metaclass=MultiClass):
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

    def __str__(self):
        return str(self.args[0])

    def check_conditions(self):
        return True

    def meta_constant(self, axis):
        # Preserve constancy
        return self.args[0].meta[axis]['constant']

    def meta_parity(self, axis):
        # Preserve parity
        return self.args[0].meta[axis]['parity']

    @property
    def base(self):
        return FieldCopy

    def sym_diff(self, var):
        """Symbolically differentiate with respect to var."""
        return self.args[0].sym_diff(var)

    def split(self, *vars):
        return self.args[0].split(*vars)


class FieldCopyScalar(FieldCopy):

    argtypes = {0: (Scalar, FutureScalar)}

    def operate(self, out):
        if self.args[0].value == 0:
            # Copy in coeff layout
            out.layout = self._coeff_layout
            out.data.fill(0)
        else:
            # Copy in grid layout
            out.layout = self._grid_layout
            np.copyto(out.data, self.args[0].value)

    def __eq__(self, other):
        return self.args[0].__eq__(other)


class FieldCopyArray(FieldCopy):

    argtypes = {0: (Array, FutureArray)}

    def operate(self, out):
        # Copy in grid layout
        out.layout = self._grid_layout
        np.copyto(out.data, self.args[0].data)


class FieldCopyField(FieldCopy):

    argtypes = {0: (Field, FutureField)}

    def operate(self, out):
        arg0, = self.args
        # Copy in current layout
        out.layout = arg0.layout
        np.copyto(out.data, arg0.data)


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
    On evaluation, this wrapper evaluates the provided function with the given
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
        self._field_arg_indices = [i for (i,arg) in enumerate(self.args) if isinstance(arg, (Field, FutureField))]
        try:
            self.name = func.__name__
        except AttributeError:
            self.name = str(func)

    def meta_constant(self, axis):
        # Assume no constancy
        return False

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
        out.layout = self.layout
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
        arg = Operand.cast(arg)
        return (func, arg), kw

    @classmethod
    def _check_args(cls, *args, **kw):
        match = (isinstance(args[i], types) for i,types in cls.argtypes.items())
        return all(match)

    def __init__(self, func, arg, **kw):
        arg = Operand.cast(arg)
        super().__init__(arg, **kw)
        self.func = func
        self.name = func.__name__

    @property
    def base(self):
        return partial(type(self), self.func)

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


class UnaryGridFunctionScalar(UnaryGridFunction, FutureScalar):

    argtypes = {1: (Scalar, FutureScalar)}

    def check_conditions(self):
        return True

    def operate(self, out):
        out.value = self.func(self.args[0].value)


class UnaryGridFunctionArray(UnaryGridFunction, FutureArray):

    argtypes = {1: (Array, FutureArray)}

    def check_conditions(self):
        return True

    def operate(self, out):
        self.func(self.args[0].data, out=out.data)


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
        out.layout = self._grid_layout
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

    @classmethod
    def _preprocess_args(cls, *args, **kw):
        args = tuple(Operand.cast(arg) for arg in args)
        return args, kw

    @classmethod
    def _check_args(cls, *args, **kw):
        match = (isinstance(args[i], types) for i,types in cls.argtypes.items())
        return all(match)

    @property
    def base(self):
        return Add

    def meta_constant(self, axis):
        # Logical 'and' of constancies
        constant0 = self.args[0].meta[axis]['constant']
        constant1 = self.args[1].meta[axis]['constant']
        return (constant0 and constant1)

    def meta_parity(self, axis):
        # Parities must match
        parity0 = self.args[0].meta[axis]['parity']
        parity1 = self.args[1].meta[axis]['parity']
        if parity0 != parity1:
            raise UndefinedParityError("Cannot add fields of different parities.")
        else:
            return parity0

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

    def split(self, *vars):
        S0 = self.args[0].split(*vars)
        S1 = self.args[1].split(*vars)
        return [S0[0]+S1[0], S0[1]+S1[1]]

    def operator_dict(self, index, vars, **kw):
        """Produce matrix-operator dictionary over specified variables."""
        out = defaultdict(int)
        op0 = self.args[0].operator_dict(index, vars, **kw)
        op1 = self.args[1].operator_dict(index, vars, **kw)
        for var in set().union(op0, op1):
            if (var in op0) and (var in op1):
                out[var] = add_sparse(op0[var], op1[var])
            elif (var in op0):
                out[var] = op0[var]
            else:
                out[var] = op1[var]
        return out

    def sym_diff(self, var):
        """Symbolically differentiate with respect to var."""
        arg0, arg1 = self.args
        diff0 = arg0.sym_diff(var)
        diff1 = arg1.sym_diff(var)
        return diff0 + diff1


class AddScalarScalar(Add, FutureScalar):

    argtypes = {0: (Scalar, FutureScalar),
                1: (Scalar, FutureScalar)}

    def __new__(cls, arg0, arg1, *args, **kw):
        if (arg0.name is None) and (arg1.name is None):
            value = arg0.value + arg1.value
            return Scalar(value=value)
        elif (arg0.name is None) and (arg0.value == 0):
            return arg1
        elif (arg1.name is None) and (arg1.value == 0):
            return arg0
        else:
            return object.__new__(cls)

    def check_conditions(self):
        return True

    def operate(self, out):
        arg0, arg1 = self.args
        out.value = arg0.value + arg1.value


class AddArrayArray(Add, FutureArray):

    argtypes = {0: (Array, FutureArray),
                1: (Array, FutureArray)}

    def check_conditions(self):
        return True

    def operate(self, out):
        arg0, arg1 = self.args
        np.add(arg0.data, arg1.data, out.data)


class AddFieldField(Add, FutureField):

    argtypes = {0: (Field, FutureField),
                1: (Field, FutureField)}

    def check_conditions(self):
        # Layouts must match
        return (self.args[0].layout is self.args[1].layout)

    def operate(self, out):
        arg0, arg1 = self.args
        # Add in arg0 layout (arbitrary choice)
        arg1.require_layout(arg0.layout)
        out.layout = arg0.layout
        np.add(arg0.data, arg1.data, out.data)


class AddScalarArray(Add, FutureArray):

    argtypes = {0: (Scalar, FutureScalar),
                1: (Array, FutureArray)}

    def __new__(cls, arg0, arg1, *args, **kw):
        if (arg0.name is None) and (arg0.value == 0):
            return arg1
        else:
            return object.__new__(cls)

    def check_conditions(self):
        return True

    def operate(self, out):
        arg0, arg1 = self.args
        np.add(arg0.value, arg1.data, out.data)


class AddArrayScalar(Add, FutureArray):

    argtypes = {0: (Array, FutureArray),
                1: (Scalar, FutureScalar)}

    def __new__(cls, arg0, arg1, *args, **kw):
        if (arg1.name is None) and (arg1.value == 0):
            return arg0
        else:
            return object.__new__(cls)

    def check_conditions(self):
        return True

    def operate(self, out):
        arg0, arg1 = self.args
        np.add(arg0.data, arg1.value, out.data)


class AddScalarField(Add, FutureField):

    argtypes = {0: (Scalar, FutureScalar),
                1: (Field, FutureField)}

    def __new__(cls, arg0, arg1, *args, **kw):
        if (arg0.name is None) and (arg0.value == 0):
            return arg1
        else:
            return object.__new__(cls)

    def check_conditions(self):
        # Field must be in grid layout
        return (self.args[1].layout is self._grid_layout)

    def operate(self, out):
        arg0, arg1 = self.args
        # Add in grid layout
        arg1.require_grid_space()
        out.layout = self._grid_layout
        np.add(arg0.value, arg1.data, out.data)


class AddFieldScalar(Add, FutureField):

    argtypes = {0: (Field, FutureField),
                1: (Scalar, FutureScalar)}

    def __new__(cls, arg0, arg1, *args, **kw):
        if (arg1.name is None) and (arg1.value == 0):
            return arg0
        else:
            return object.__new__(cls)

    def check_conditions(self):
        # Field must be in grid layout
        return (self.args[0].layout is self._grid_layout)

    def operate(self, out):
        arg0, arg1 = self.args
        # Add in grid layout
        arg0.require_grid_space()
        out.layout = self._grid_layout
        np.add(arg0.data, arg1.value, out.data)


class AddArrayField(Add, FutureField):

    argtypes = {0: (Array, FutureArray),
                1: (Field, FutureField)}

    def check_conditions(self):
        # Field must be in grid layout
        return (self.args[1].layout is self._grid_layout)

    def operate(self, out):
        arg0, arg1 = self.args
        # Add in grid layout
        arg1.require_grid_space()
        out.layout = self._grid_layout
        np.add(arg0.data, arg1.data, out.data)


class AddFieldArray(Add, FutureField):

    argtypes = {0: (Field, FutureField),
                1: (Array, FutureArray)}

    def check_conditions(self):
        # Field must be in grid layout
        return (self.args[0].layout is self._grid_layout)

    def operate(self, out):
        arg0, arg1 = self.args
        # Add in grid layout
        arg0.require_grid_space()
        out.layout = self._grid_layout
        np.add(arg0.data, arg1.data, out.data)


class Multiply(Arithmetic, metaclass=MultiClass):

    name = 'Mul'
    str_op = '*'

    @classmethod
    def _preprocess_args(cls, *args, **kw):
        args = tuple(Operand.cast(arg) for arg in args)
        return args, kw

    @classmethod
    def _check_args(cls, *args, **kw):
        match = (isinstance(args[i], types) for i,types in cls.argtypes.items())
        return all(match)

    @property
    def base(self):
        return Multiply

    def meta_constant(self, axis):
        # Logical 'and' of constancies
        constant0 = self.args[0].meta[axis]['constant']
        constant1 = self.args[1].meta[axis]['constant']
        return (constant0 and constant1)

    def meta_parity(self, axis):
        # Multiply parities
        parity0 = self.args[0].meta[axis]['parity']
        parity1 = self.args[1].meta[axis]['parity']
        return parity0 * parity1

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

    def split(self, *vars):
        S0 = self.args[0].split(*vars)
        S1 = self.args[1].split(*vars)
        return [S0[0]*S1[0] + S0[0]*S1[1] + S0[1]*S1[0], S0[1]*S1[1]]

    def operator_dict(self, index, vars, **kw):
        """Produce matrix-operator dictionary over specified variables."""
        out = defaultdict(int)
        op0 = self.args[0].as_ncc_operator(**kw)
        op1 = self.args[1].operator_dict(index, vars, **kw)
        for var in op1:
            out[var] = op0 * op1[var]
        return out

    def sym_diff(self, var):
        """Symbolically differentiate with respect to var."""
        arg0, arg1 = self.args
        diff0 = arg0.sym_diff(var)
        diff1 = arg1.sym_diff(var)
        return diff0*arg1 + arg0*diff1


class MultiplyScalarScalar(Multiply, FutureScalar):

    argtypes = {0: (Scalar, FutureScalar),
                1: (Scalar, FutureScalar)}

    def __new__(cls, arg0, arg1, *args, **kw):
        if (arg0.name is None) and (arg1.name is None):
            value = arg0.value * arg1.value
            return Scalar(value=value)
        elif (arg0.name is None) and (arg0.value == 0):
            return 0
        elif (arg0.name is None) and (arg0.value == 1):
            return arg1
        elif (arg1.name is None) and (arg1.value == 0):
            return 0
        elif (arg1.name is None) and (arg1.value == 1):
            return arg0
        else:
            return object.__new__(cls)

    def check_conditions(self):
        return True

    def operate(self, out):
        arg0, arg1 = self.args
        out.value = arg0.value * arg1.value


class MultiplyArrayArray(Multiply, FutureArray):

    argtypes = {0: (Array, FutureArray),
                1: (Array, FutureArray)}

    def check_conditions(self):
        return True

    def operate(self, out):
        arg0, arg1 = self.args
        np.multiply(arg0.data, arg1.data, out.data)


class MultiplyFieldField(Multiply, FutureField):

    argtypes = {0: (Field, FutureField),
                1: (Field, FutureField)}

    def check_conditions(self):
        # Fields must be in grid layout
        return ((self.args[0].layout is self._grid_layout) and
                (self.args[1].layout is self._grid_layout))

    def operate(self, out):
        arg0, arg1 = self.args
        # Multiply in grid layout
        arg0.require_grid_space()
        arg1.require_grid_space()
        out.layout = self._grid_layout
        np.multiply(arg0.data, arg1.data, out.data)

    ## Ideas for separating condition enforcement from operation to potentially
    ## trim down the boilerplate for the dispatching subclasses
    # def enforce_conditions(self):
    #     self.args[0].require_grid_space()
    #     self.args[1].require_grid_space()
    #     out.layout = self._grid_layout
    # def _operate(self):
    #     np.multiply(self.args[0].data, self.args[1].data, out.data)


class MultiplyScalarArray(Multiply, FutureArray):

    argtypes = {0: (Scalar, FutureScalar),
                1: (Array, FutureArray)}

    def __new__(cls, arg0, arg1, *args, **kw):
        if (arg0.name is None) and (arg0.value == 0):
            return 0
        elif (arg0.name is None) and (arg0.value == 1):
            return arg1
        else:
            return object.__new__(cls)

    def check_conditions(self):
        return True

    def operate(self, out):
        arg0, arg1 = self.args
        np.multiply(arg0.value, arg1.data, out.data)


class MultiplyArrayScalar(Multiply, FutureArray):

    argtypes = {0: (Array, FutureArray),
                1: (Scalar, FutureScalar)}

    def __new__(cls, arg0, arg1, *args, **kw):
        if (arg1.name is None) and (arg1.value == 0):
            return 0
        elif (arg1.name is None) and (arg1.value == 1):
            return arg0
        else:
            return object.__new__(cls)

    def check_conditions(self):
        return True

    def operate(self, out):
        arg0, arg1 = self.args
        np.multiply(arg0.data, arg1.value, out.data)


class MultiplyScalarField(Multiply, FutureField):

    argtypes = {0: (Scalar, FutureScalar),
                1: (Field, FutureField)}

    def __new__(cls, arg0, arg1, *args, **kw):
        if (arg0.name is None) and (arg0.value == 0):
            return 0
        elif (arg0.name is None) and (arg0.value == 1):
            return arg1
        else:
            return object.__new__(cls)

    def check_conditions(self):
        return True

    def operate(self, out):
        arg0, arg1 = self.args
        # Multiply in current layout
        out.layout = arg1.layout
        np.multiply(arg0.value, arg1.data, out.data)


class MultiplyFieldScalar(Multiply, FutureField):

    argtypes = {0: (Field, FutureField),
                1: (Scalar, FutureScalar)}

    def __new__(cls, arg0, arg1, *args, **kw):
        if (arg1.name is None) and (arg1.value == 0):
            return 0
        elif (arg1.name is None) and (arg1.value == 1):
            return arg0
        else:
            return object.__new__(cls)

    def check_conditions(self):
        return True

    def operate(self, out):
        arg0, arg1 = self.args
        # Multiply in current layout
        out.layout = arg0.layout
        np.multiply(arg0.data, arg1.value, out.data)


class MultiplyArrayField(Multiply, FutureField):

    argtypes = {0: (Array, FutureArray),
                1: (Field, FutureField)}

    def check_conditions(self):
        # Field must be in grid layout
        return (self.args[1].layout is self._grid_layout)

    def operate(self, out):
        arg0, arg1 = self.args
        # Multiply in grid layout
        arg1.require_grid_space()
        out.layout = self._grid_layout
        np.multiply(arg0.data, arg1.data, out.data)


class MultiplyFieldArray(Multiply, FutureField):

    argtypes = {0: (Field, FutureField),
                1: (Array, FutureArray)}

    def check_conditions(self):
        # Field must be in grid layout
        return (self.args[0].layout is self._grid_layout)

    def operate(self, out):
        arg0, arg1 = self.args
        # Multiply in grid layout
        arg0.require_grid_space()
        out.layout = self._grid_layout
        np.multiply(arg0.data, arg1.data, out.data)


class Power(NonlinearOperator, Arithmetic, metaclass=MultiClass):

    name = 'Pow'
    str_op = '**'

    @classmethod
    def _preprocess_args(cls, *args, **kw):
        args = tuple(Operand.cast(arg) for arg in args)
        return args, kw

    @classmethod
    def _check_args(cls, *args, **kw):
        match = (isinstance(args[i], types) for i,types in cls.argtypes.items())
        return all(match)

    @property
    def base(self):
        return Power


class PowerDataScalar(Power):

    argtypes = {0: (Data, Future),
                1: (Scalar, FutureScalar)}

    def __new__(cls, arg0, arg1, *args, **kw):
        if (arg1.name is None) and (arg1.value == 0):
            return 1
        elif (arg1.name is None) and (arg1.value == 1):
            return arg0
        else:
            return object.__new__(cls)

    def meta_constant(self, axis):
        # Preserves constancy
        return self.args[0].meta[axis]['constant']

    def meta_parity(self, axis):
        # Constant data keeps even parity
        constant = self.args[0].meta[axis]['constant']
        if constant:
            return 1
        # Integer exponents maintain valid parity
        parity = self.args[0].meta[axis]['parity']
        power = self.args[1].value
        if is_integer(power):
            return parity**int(power)
        # Otherwise invalid
        raise UndefinedParityError("Non-integer power of nonconstant data has undefined parity.")

    def sym_diff(self, var):
        """Symbolically differentiate with respect to var."""
        arg0, arg1 = self.args
        diff0 = arg0.sym_diff(var)
        return arg1 * arg0**(arg1-1) * diff0


class PowerScalarScalar(PowerDataScalar, FutureScalar):

    argtypes = {0: (Scalar, FutureScalar),
                1: (Scalar, FutureScalar)}

    def __new__(cls, arg0, arg1, *args, **kw):
        if (arg0.name is None) and (arg0.value == 0):
            return 0
        else:
            return super().__new__(cls, arg0, arg1, *args, **kw)

    def check_conditions(self):
        return True

    def operate(self, out):
        arg0, arg1 = self.args
        out.value = arg0.value ** arg1.value


class PowerArrayScalar(PowerDataScalar, FutureArray):

    argtypes = {0: (Array, FutureArray),
                1: (Scalar, FutureScalar)}

    def check_conditions(self):
        return True

    def operate(self, out):
        arg0, arg1 = self.args
        np.power(arg0.data, arg1.value, out.data)


class PowerFieldScalar(PowerDataScalar, FutureField):

    argtypes = {0: (Field, FutureField),
                1: (Scalar, FutureScalar)}

    def check_conditions(self):
        # Field must be in grid layout
        return (self.args[0].layout is self._grid_layout)

    def operate(self, out):
        arg0, arg1 = self.args
        # Raise in grid layout
        arg0.require_grid_space()
        out.layout = self._grid_layout
        np.power(arg0.data, arg1.value, out.data)


class LinearOperator(Operator):

    kw = {}

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

    def split(self, *vars):
        if self.base in vars:
            return [self, 0]
        else:
            S0 = self.args[0].split(*vars)
            return [self.base(S0[0], **self.kw), self.base(S0[1], **self.kw)]

    def operator_dict(self, index, vars, **kw):
        """Produce matrix-operator dictionary over specified variables."""
        out = defaultdict(int)
        ops = self.operator_form(index)
        op0 = self.args[0].operator_dict(index, vars, **kw)
        for var in op0:
            out[var] = ops * op0[var]
        return out

    def operator_form(self, index):
        raise NotImplementedError()

    def sym_diff(self, var):
        """Symbolically differentiate with respect to var."""
        diff0 = self.args[0].sym_diff(var)
        return self.base(diff0, **self.kw)


class TimeDerivative(LinearOperator, FutureField):

    name = 'dt'

    @property
    def base(self):
        return TimeDerivative

    def meta_constant(self, axis):
        # Preserves constancy
        return self.args[0].meta[axis]['constant']

    def meta_parity(self, axis):
        # Preserves parity
        return self.args[0].meta[axis]['parity']

    def operator_form(self, index):
        raise ValueError("Operator form not available for time derivative.")

    def operate(self, out):
        raise ValueError("Cannot evaluate time derivative operator.")


class Separable(LinearOperator, FutureField):

    def operator_form(self, index):
        return self.vector_form()[index[self.axis]]

    def check_conditions(self):
        arg0, = self.args
        axis = self.axis
        # Must be in coeff layout
        is_coeff = not arg0.layout.grid_space[axis]
        return is_coeff

    def operate(self, out):
        arg0, = self.args
        axis = self.axis
        # Require coeff layout
        arg0.require_coeff_space(axis)
        out.layout = arg0.layout
        # Attempt forms
        try:
            self.explicit_form(arg0.data, out.data, axis)
        except NotImplementedError:
            self.apply_vector_form(out)

    def apply_vector_form(self, out):
        arg0, = self.args
        axis = self.axis
        dim = arg0.domain.dim
        slices = arg0.layout.slices(self.domain.dealias)
        vector = self.vector_form()
        vector = vector[slices[axis]]
        vector = reshape_vector(vector, dim=dim, axis=axis)
        np.multiply(arg0.data, vector, out=out.data)

    def explicit_form(self, input, output, axis):
        raise NotImplementedError()

    def vector_form(self):
        raise NotImplementedError()


class Coupled(LinearOperator, FutureField):

    def operator_form(self, index):
        return self.matrix_form()

    def check_conditions(self):
        arg0, = self.args
        axis = self.axis
        # Must be in coeff+local layout
        is_coeff = not arg0.layout.grid_space[axis]
        is_local = arg0.layout.local[axis]
        return (is_coeff and is_local)

    def operate(self, out):
        arg0, = self.args
        axis = self.axis
        # Require coeff+local layout
        arg0.require_coeff_space(axis)
        arg0.require_local(axis)
        out.layout = arg0.layout
        # Attempt forms
        try:
            self.explicit_form(arg0.data, out.data, axis)
        except NotImplementedError:
            self.apply_matrix_form(out)

    def apply_matrix_form(self, out):
        arg0, = self.args
        axis = self.axis
        dim = arg0.domain.dim
        matrix = self.matrix_form()
        apply_matrix(matrix, arg0.data, axis, out=out.data)

    def explicit_form(self, input, output, axis):
        raise NotImplementedError()

    def matrix_form(self):
        raise NotImplementedError()

    def operator_dict(self, index, vars, **kw):
        """Produce matrix-operator dictionary over specified variables."""
        if self.basis.separable:
            raise ValueError("LHS operator {} is coupled along direction {}.".format(self.name, self.basis.name))
        else:
            return super().operator_dict(index, vars, **kw)


@parseable
class Integrate(LinearOperator):

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

    def meta_constant(self, axis):
        if axis == self.axis:
            # Integral is constant
            return True
        else:
            # Preserve constancy
            return self.args[0].meta[axis]['constant']

    def meta_parity(self, axis):
        if axis == self.axis:
            # Integral is a scalar (even parity)
            return 1
        else:
            # Preserve parity
            return self.args[0].meta[axis]['parity']


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
class Interpolate(LinearOperator):

    name = 'interp'

    def __new__(cls, arg0, position, out=None):
        # Cast to operand
        arg0 = Operand.cast(arg0)
        # Check if operand depends on basis
        if (cls.basis not in arg0.domain.bases):
            return arg0
        elif (arg0.meta[cls.basis.name]['constant']):
            return arg0
        else:
            return object.__new__(cls)

    def __init__(self, arg0, position, out=None):
        # Cast argument to field
        arg0 = Field.cast(arg0, arg0.domain)
        super().__init__(arg0, out=out)
        self.kw = {'position': position}
        self.position = position
        self.axis = self.domain.bases.index(self.basis)

    def distribute(self):
        arg0, = self.args
        if not isinstance(arg0, Add):
            raise ValueError("Can only apply distributive rule to a sum.")
        a, b = arg0.args
        op = type(self)
        return op(a, self.position) + op(b, self.position)

    def __repr__(self):
        return 'interp(%r, %r, %r)' %(self.args[0], self.basis, self.position)

    def __str__(self):
        return "interp({},'{}',{})".format(self.args[0], self.basis, self.position)

    def meta_constant(self, axis):
        if axis == self.axis:
            # Interpolant is constant
            return True
        else:
            # Preserve constancy
            return self.args[0].meta[axis]['constant']

    def meta_parity(self, axis):
        if axis == self.axis:
            # Interpolation is a scalar (even parity)
            return 1
        else:
            # Preserve parity
            return self.args[0].meta[axis]['parity']

@parseable
@addname('interp')
def interpolate(arg0, out=None, **basis_kw):
    # Cast to operand
    arg0 = Operand.cast(arg0)
    if isinstance(arg0, (Scalar, FutureScalar)):
        return arg0
    # Require at least one basis
    if len(basis_kw) == 0:
        raise ValueError("No basis specified.")
    # Unpack bases
    bases = sorted(list(basis_kw.items()))
    # Multiple bases: apply recursively
    if len(bases) > 1:
        arg0 = interpolate(arg0, **dict(bases[:-1]))
    # Call with single basis
    basis, position = bases[-1]
    basis = arg0.domain.get_basis_object(basis)
    return basis.Interpolate(arg0, position, out=out)


@parseable
@addname('left')
def left(arg0, out=None):
    """Shortcut for left interpolation along last axis."""
    basis = arg0.domain.bases[-1]
    return basis.Interpolate(arg0, 'left', out=out)


@parseable
@addname('right')
def right(arg0, out=None):
    """Shortcut for right interpolation along last axis."""
    basis = arg0.domain.bases[-1]
    return basis.Interpolate(arg0, 'right', out=out)


class Differentiate(LinearOperator):

    name = 'd'

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

    def meta_constant(self, axis):
        # Preserve constancy
        return self.args[0].meta[axis]['constant']

    def meta_parity(self, axis):
        parity0 = self.args[0].meta[axis]['parity']
        if axis == self.axis:
            # Flip parity
            return (-1) * parity0
        else:
            # Preserve parity
            return parity0

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
            if isinstance(arg0, Multiply):
                arg0a, arg0b = arg0.args
                return (op(arg0a)*arg0b + arg0a*op(arg0b)).expand(*vars)
        return self


@parseable
@addname('d')
def differentiate(arg0, *bases, out=None, **basis_kw):
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
        arg0 = differentiate(arg0, *bases[:-1])
    # Call with single basis
    basis = arg0.domain.get_basis_object(bases[-1])
    return basis.Differentiate(arg0, out=out)


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

    def meta_constant(self, axis):
        # Preserve constancy
        return self.args[0].meta[axis]['constant']

    def meta_parity(self, axis):
        parity0 = self.args[0].meta[axis]['parity']
        if axis == self.axis:
            # Flip parity
            return (-1) * parity0
        else:
            # Preserve parity
            return parity0


@parseable
@addname('Hilbert')
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

