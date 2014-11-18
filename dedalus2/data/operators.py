"""
Abstract and built-in classes defining deferred operations on fields.

"""

from functools import partial
import numpy as np
import sympy as sy

from .field import Operand, Data, Scalar, Array, Field
from .future import Future, FutureScalar, FutureArray, FutureField
from ..tools.array import reshape_vector, apply_matrix
from ..tools.dispatch import MultiClass


# Use simple decorator to track parseable operators
parseables = {}
def parseable(cls):
    parseables[cls.name] = cls
    return cls

# Other helpers
def is_integer(x):
    if isinstance(x, int):
        return True
    else:
        return x.is_integer()


class UndefinedParityError(Exception):
    pass

class NonlinearOperatorError(Exception):
    pass


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

    def as_symbolic_operator(self, fields):
        return self.args[0].as_symbolic_operator(fields)


class FieldCopyScalar(FieldCopy):

    argtypes = {0: (Scalar, FutureScalar)}

    def operate(self, out):
        # Copy in grid layout
        out.layout = self._grid_layout
        np.copyto(out.data, self.args[0].data)


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


class GeneralFunction(FutureField):

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
        out.layout = self.layout
        np.copyto(out.data, self.func(*self.args, **self.kw))


class UnaryGridFunction(Future, metaclass=MultiClass):

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

    def meta_constant(self, axis):
        # Preserves constancy
        return self.args[0].meta[axis]['constant']

    def meta_parity(self, axis):
        # Preserving constancy -> even parity
        if self.args[0].meta[axis]['constant']:
            return 1
        else:
            raise UndefinedParityError("Unknown action of {} on parity.".format(self.name))

    def as_symbolic_operator(self, vars):
        arg0_op = self.args[0].as_symbolic_operator(vars)
        if arg0_op:
            raise ValueError("Nonlinear")
        else:
            return None

    def as_symbolic_operator(self, vars):

        arg0, = self.args
        arg0_has_vars = arg0.has(*vars)
        arg0_op = arg0.as_symbolic_operator(vars)

        if arg0_has_vars:
            raise NonlinearOperatorError("Cannot linearize {}.".format(self.name))
        else:
            return self.as_ncc_symbol()


class UnaryGridFunctionScalar(UnaryGridFunction, FutureScalar):

    argtypes = {1: (Scalar, FutureScalar)}

    def check_conditions(self):
        return True

    def operate(self, out):
        return self.func(self.args[0].data, out=out.data)


class UnaryGridFunctionArray(UnaryGridFunction, FutureArray):

    argtypes = {1: (Array, FutureArray)}

    def check_conditions(self):
        return True

    def operate(self, out):
        return self.func(self.args[0].data, out=out.data)


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

    def as_symbolic_operator(self, vars):

        arg0, arg1 = self.args
        arg0_has_vars = arg0.has(*vars)
        arg1_has_vars = arg1.has(*vars)
        arg0_op = arg0.as_symbolic_operator(vars)
        arg1_op = arg1.as_symbolic_operator(vars)

        if (arg0_has_vars != arg1_has_vars):
            raise NonlinearOperatorError("Cannot add linear and nonlinear terms.")
        else:
            return arg0_op + arg1_op

    def distribute_over(self, vars):
        # Distribute arguments
        return Add(*[arg.distribute_over(vars) for arg in self.args])


class AddScalarScalar(Add, FutureScalar):

    argtypes = {0: (Scalar, FutureScalar),
                1: (Scalar, FutureScalar)}

    def check_conditions(self):
        return True

    def operate(self, out):
        arg0, arg1 = self.args
        np.add(arg0.data, arg1.data, out.data)


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

    def check_conditions(self):
        return True

    def operate(self, out):
        arg0, arg1 = self.args
        np.add(arg0.data, arg1.data, out.data)


class AddArrayScalar(Add, FutureArray):

    argtypes = {0: (Array, FutureArray),
                1: (Scalar, FutureScalar)}

    def check_conditions(self):
        return True

    def operate(self, out):
        arg0, arg1 = self.args
        np.add(arg0.data, arg1.data, out.data)


class AddScalarField(Add, FutureField):

    argtypes = {0: (Scalar, FutureScalar),
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


class AddFieldScalar(Add, FutureField):

    argtypes = {0: (Field, FutureField),
                1: (Scalar, FutureScalar)}

    def check_conditions(self):
        # Field must be in grid layout
        return (self.args[0].layout is self._grid_layout)

    def operate(self, out):
        arg0, arg1 = self.args
        # Add in grid layout
        arg0.require_grid_space()
        out.layout = self._grid_layout
        np.add(arg0.data, arg1.data, out.data)


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

    def as_symbolic_operator(self, vars):
        """Change to left-multiplication on vars."""

        arg0, arg1 = self.args
        arg0_has_vars = arg0.has(*vars)
        arg1_has_vars = arg1.has(*vars)
        arg0_op = arg0.as_symbolic_operator(vars)
        arg1_op = arg1.as_symbolic_operator(vars)

        if arg0_has_vars and arg1_has_vars:
            raise NonlinearOperatorError("Cannot multiply two linear terms.")
        elif arg0_has_vars:
            return arg1_op * arg0_op
        else:
            return arg0_op * arg1_op

    def distribute_over(self, vars):
        # Distribute arguments
        out = Multiply(*[arg.distribute_over(vars) for arg in self.args])
        # Distribute over addition
        if out.args[0].has(*vars) and isinstance(out.args[0], Add):
            return out.left_distribute().distribute_over(vars)
        elif out.args[1].has(*vars) and isinstance(out.args[1], Add):
            return out.right_distribute().distribute_over(vars)
        else:
            return out

    def left_distribute(self):
        arg0, arg1 = self.args
        if not isinstance(arg0, Add):
            raise ValueError("Can only distribute over a sum.")
        a, b = arg0.args
        return a*arg1 + b*arg1

    def right_distribute(self):
        arg0, arg1 = self.args
        if not isinstance(arg1, Add):
            raise ValueError("Can only distribute over a sum.")
        a, b = arg1.args
        return arg0*a + arg0*b


class MultiplyScalarScalar(Multiply, FutureScalar):

    argtypes = {0: (Scalar, FutureScalar),
                1: (Scalar, FutureScalar)}

    def check_conditions(self):
        return True

    def operate(self, out):
        arg0, arg1 = self.args
        np.multiply(arg0.data, arg1.data, out.data)


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

    def check_conditions(self):
        return True

    def operate(self, out):
        arg0, arg1 = self.args
        np.multiply(arg0.data, arg1.data, out.data)


class MultiplyArrayScalar(Multiply, FutureArray):

    argtypes = {0: (Array, FutureArray),
                1: (Scalar, FutureScalar)}

    def check_conditions(self):
        return True

    def operate(self, out):
        arg0, arg1 = self.args
        np.multiply(arg0.data, arg1.data, out.data)


class MultiplyScalarField(Multiply, FutureField):

    argtypes = {0: (Scalar, FutureScalar),
                1: (Field, FutureField)}

    def check_conditions(self):
        return True

    def operate(self, out):
        arg0, arg1 = self.args
        # Multiply in current layout
        out.layout = arg1.layout
        np.multiply(arg0.data, arg1.data, out.data)


class MultiplyFieldScalar(Multiply, FutureField):

    argtypes = {0: (Field, FutureField),
                1: (Scalar, FutureScalar)}

    def check_conditions(self):
        return True

    def operate(self, out):
        arg0, arg1 = self.args
        # Multiply in current layout
        out.layout = arg0.layout
        np.multiply(arg0.data, arg1.data, out.data)


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


class Power(Arithmetic, metaclass=MultiClass):

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


class PowerDataScalar(Power):

    argtypes = {0: (Data, Future),
                1: (Scalar, FutureScalar)}

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

    def as_symbolic_operator(self, vars):

        arg0, arg1 = self.args
        arg0_has_vars = arg0.has(*vars)
        arg0_op = arg0.as_symbolic_operator(vars)

        if arg0_has_vars:
            raise NonlinearOperatorError("Power is nonlinear.")
        else:
            return self.as_ncc_symbol()


class PowerScalarScalar(PowerDataScalar, FutureScalar):

    argtypes = {0: (Scalar, FutureScalar),
                1: (Scalar, FutureScalar)}

    def check_conditions(self):
        return True

    def operate(self, out):
        arg0, arg1 = self.args
        np.power(arg0.data, arg1.data, out.data)


class PowerArrayScalar(PowerDataScalar, FutureArray):

    argtypes = {0: (Array, FutureArray),
                1: (Scalar, FutureScalar)}

    def check_conditions(self):
        return True

    def operate(self, out):
        arg0, arg1 = self.args
        np.power(arg0.data, arg1.data, out.data)


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
        np.power(arg0.data, arg1.data, out.data)


class LinearOperator(Future):

    kw = {}

    def as_symbolic_operator(self, vars):
        arg0, = self.args
        arg0_has_vars = arg0.has(*vars)
        arg0_op = arg0.as_symbolic_operator(vars)
        if arg0_has_vars:
            return self.op_symbol() * arg0_op
        else:
            return self.as_ncc_symbol()

    def distribute_over(self, vars):
        # Distribute arguments
        out = type(self)(*[arg.distribute_over(vars) for arg in self.args], **self.kw)
        # Distribute over addition
        if out.args[0].has(*vars) and isinstance(out.args[0], Add):
            return out.distribute().distribute_over(vars)
        else:
            return out

    def distribute(self):
        arg0, = self.args
        if not isinstance(arg0, Add):
            raise ValueError("Can only apply distributive rule to a sum.")
        a, b = arg0.args
        op = type(self)
        return op(a) + op(b)


class Separable(LinearOperator, FutureField):

    @classmethod
    def op_symbol(cls):
        return sy.Symbol(cls.name, commutative=True)

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

    @classmethod
    def op_symbol(cls):
        if cls.basis.separable:
            raise ValueError("LHS operator {} is coupled along direction {}.".format(cls.name, cls.basis.name))
        return sy.Symbol(cls.name, commutative=False)

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


@parseable
class Integrate(LinearOperator, metaclass=MultiClass):

    name = 'integ'

    @classmethod
    def _preprocess_args(cls, arg0, *bases, **kw):
        # Cast to operand
        arg0 = Operand.cast(arg0)
        # No bases: integrate over whole domain
        if len(bases) == 0:
            bases = arg0.domain.bases
        # Multiple bases: apply recursively
        if len(bases) > 1:
            arg0 = Integrate(arg0, *bases[:-1])
        # Call with single basis
        basis = arg0.domain.get_basis_object(bases[-1])
        return (arg0, basis), kw

    @classmethod
    def _check_args(cls, arg0, basis, **kw):
        return (basis == cls.basis)

    @classmethod
    def _postprocess_args(cls, arg0, basis, **kw):
        # Drop basis
        return (arg0,), kw

    def __new__(cls, arg0, **kw):
        # Cast to operand
        arg0 = Operand.cast(arg0)
        # Instantiate if operand depends on basis
        if cls.basis in arg0.domain.bases:
            return object.__new__(cls)
        # Otherwise route through dispatch
        else:
            return Differentiate(arg0, cls, **kw)

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
class Interpolate(LinearOperator, metaclass=MultiClass):

    name = 'interp'
    store_last = True

    @classmethod
    def _preprocess_args(cls, arg0, basis, position, **kw):
        # Cast to operand
        arg0 = Operand.cast(arg0)
        basis = arg0.domain.get_basis_object(basis)
        return (arg0, basis, position), kw

    @classmethod
    def _check_args(cls, arg0, basis, position, out=None):
        return (basis == cls.basis)

    @classmethod
    def _postprocess_args(cls, arg0, basis, position, **kw):
        # Drop basis
        return (arg0, position), kw

    def __init__(self, arg0, position, out=None):
        # Cast argument to field
        arg0 = Field.cast(arg0, arg0.domain)
        super().__init__(arg0, out=out)
        self.kw = {'position': position}
        self.position = position
        self.axis = self.domain.bases.index(self.basis)

    def __repr__(self):
        return 'interp(%r, %r, %r)' %(self.args[0], self.basis, self.position)

    def __str__(self):
        return "interp({},'{}',{})".format(self.args[0], self.basis, self.position)

    @classmethod
    def at(cls, position):
        cls._position = position
        return cls

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

    def op_symbol(self):
        super().op_symbol()
        symname = '{}.at({})'.format(self.name, repr(self.position))
        return sy.Symbol(symname, commutative=False)


class InterpolateScalar(Interpolate, FutureScalar):

    basis = None

    @classmethod
    def _check_args(cls, arg0, basis, position, **kw):
        return (basis is None)

    def __new__(cls, arg0, position, **kw):
        return arg0


@parseable
class Left:

    name = 'left'

    def __new__(cls, arg0, out=None):
        basis = arg0.domain.bases[-1]
        return Interpolate(arg0, basis, 'left', out=out)


@parseable
class Right:

    name = 'right'

    def __new__(cls, arg0, out=None):
        basis = arg0.domain.bases[-1]
        return Interpolate(arg0, basis, 'right', out=out)


@parseable
class Differentiate(LinearOperator, metaclass=MultiClass):

    name = 'd'

    @classmethod
    def _preprocess_args(cls, arg0, *bases, out=None, **basis_kw):
        # Cast to operand
        arg0 = Operand.cast(arg0)
        # Parse keyword bases
        for basis, order in basis_kw.items():
            bases += (basis,) * order
        # Require at least one basis
        if len(bases) == 0:
            raise ValueError("No basis specified.")
        # Multiple bases: apply recursively
        if len(bases) > 1:
            arg0 = Differentiate(arg0, *bases[:-1])
        # Call with single basis
        basis = arg0.domain.get_basis_object(bases[-1])
        return (arg0, basis), {'out': out}

    @classmethod
    def _check_args(cls, arg0, basis, **kw):
        return (basis == cls.basis)

    @classmethod
    def _postprocess_args(cls, arg0, basis, **kw):
        # Drop basis
        return (arg0,), kw

    def __new__(cls, arg0, **kw):
        # Cast to operand
        arg0 = Operand.cast(arg0)
        # Instantiate if operand depends on basis
        if cls.basis in arg0.domain.bases:
            return object.__new__(cls)
        # Otherwise route through dispatch
        else:
            return Differentiate(arg0, cls, **kw)

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

    def distribute_over(self, vars):
        if self.args[0].has(*vars):
            out = type(self)(*[arg.distribute_over(vars) for arg in self.args], **self.kw)
            if isinstance(out.args[0], Multiply):
                return out.apply_product_rule().distribute_over(vars)
            elif isinstance(out.args[0], Add):
                return out.distribute().distribute_over(vars)
            else:
                return out
        else:
            return self

    def apply_product_rule(self):
        arg0, = self.args
        if not isinstance(arg0, Multiply):
            raise ValueError("Can only apply product rule to the derivative of a product.")
        a, b = arg0.args
        d = type(self)
        return d(a)*b + a*d(b)


class DifferentiateIndependent(Differentiate, FutureScalar):

    basis = None

    @classmethod
    def _check_args(cls, arg0, basis, **kw):
        return (basis is None)

    def __new__(cls, arg0, **kw):
        return Scalar(value=0)


@parseable
class HilbertTransform(LinearOperator, metaclass=MultiClass):

    name = 'Hilbert'

    @classmethod
    def _preprocess_args(cls, arg0, *bases, out=None, **basis_kw):
        # Cast to operand
        arg0 = Operand.cast(arg0)
        # Parse keyword bases
        for basis, order in basis_kw.items():
            bases += (basis,) * order
        # Require at least one basis
        if len(bases) == 0:
            raise ValueError("No basis specified.")
        # Multiple bases: apply recursively
        if len(bases) > 1:
            arg0 = HilbertTransform(arg0, *bases[:-1])
        # Call with single basis
        basis = arg0.domain.get_basis_object(bases[-1])
        return (arg0, basis), {'out': out}

    @classmethod
    def _check_args(cls, arg0, basis, **kw):
        return (basis == cls.basis)

    @classmethod
    def _postprocess_args(cls, arg0, basis, **kw):
        # Drop basis
        return (arg0,), kw

    def __new__(cls, arg0, **kw):
        # Cast to operand
        arg0 = Operand.cast(arg0)
        # Instantiate if operand depends on basis
        if cls.basis in arg0.domain.bases:
            return object.__new__(cls)
        # Otherwise route through dispatch
        else:
            return HilbertTransform(arg0, cls, **kw)

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


class HilbertTransformIndependent(HilbertTransform, FutureScalar):

    basis = None

    @classmethod
    def _check_args(cls, arg0, basis, **kw):
        return (basis is None)

    def __new__(cls, arg0, **kw):
        return Scalar(value=0)

