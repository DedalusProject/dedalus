"""
Abstract and built-in classes defining deferred operations on fields.

"""

from functools import reduce, partial
import numpy as np

from .future import Future
from .field import Field
from ..tools.array import reshape_vector
from ..tools.general import OrderedSet
from ..tools.dispatch import MultiClass


class Operator(Future):
    """
    Base class for deferred operations on fields.

    Parameters
    ----------
    *args : fields, operators, and numeric types
        Operands. Number must match class attribute `arity`, if present.
    out : field, optional
        Output field.  If not specified, a new field will be used.

    Notes
    -----
    Operators are stacked (i.e. provided as arguments to other operators) to
    construct trees that represent compound expressions.  Nodes are evaluated
    by first recursively evaluating their subtrees, and then calling the
    `operate` method.

    """

    name = 'Op'
    arity = None
    store_last = False

    def __init__(self, *args, domain=None, out=None):

        # Check arity
        if self.arity is not None:
            if len(args) != self.arity:
                raise ValueError("Wrong number of arguments.")
        # Infer domain from arguments
        if domain is None:
            domain = unique_domain(out, *args)
        # Required attributes
        self.args = list(args)
        self.original_args = list(args)
        self.domain = domain
        self.out = out
        self.last_id = None

    def __repr__(self):
        repr_args = map(repr, self.args)
        return '%s(%s)' %(self.name, ', '.join(repr_args))

    def __str__(self):
        str_args = map(str, self.args)
        return '%s(%s)' %(self.name, ', '.join(str_args))

    # def __getattr__(self, attr):
    #     # Intercept numpy ufunc calls
    #     if attr in UfuncWrapper.supported:
    #         ufunc = UfuncWrapper.supported[attr]
    #         return partial(UfuncWrapper, ufunc, self)
    #     else:
    #         raise AttributeError("%r object has no attribute %r" %(self.__class__.__name__, attr))

    def reset(self):
        """Restore original arguments."""

        self.args = list(self.original_args)

    def field_set(self, include_out=False):
        """Get set of field leaves."""

        # Recursively collect field arguments
        fields = OrderedSet()
        for a in self.args:
            if isinstance(a, Field):
                fields.add(a)
            elif isinstance(a, Operator):
                fields.update(a.field_set(include_out=include_out))

        # Add output field as directed
        if include_out:
            if self.out:
                fields.add(self.out)

        return fields

    def evaluate(self, id=None, force=True):
        """Recursively evaluate operation."""

        # Check storage
        if self.store_last and (id is not None):
            if id == self.last_id:
                return self.last_out
            else:
                # Clear cache to free output field
                self.last_id = None
                self.last_out = None

        # Recursively attempt evaluation of all operator arguments
        # Track evaluation success with flag
        all_eval = True
        for i, a in enumerate(self.args):
            if isinstance(a, Field):
                a.set_scales(self.domain.dealias, keep_data=True)
            if isinstance(a, Operator):
                a_eval = a.evaluate(id=id, force=force)
                # If evaluation succeeds, substitute result
                if a_eval is not None:
                    self.args[i] = a_eval
                # Otherwise change flag
                else:
                    all_eval = False

        # Return None if any arguments are not evaluable
        if not all_eval:
            return None

        # Check conditions unless forcing evaluation
        if not force:
            # Return None if operator conditions are not satisfied
            if not self.check_conditions():
                return None

        # Allocate output field if necessary
        if self.out:
            out = self.out
        else:
            out = self.domain.new_field()

        # Copy metadata
        out.meta = self.meta

        # for axis in range(self.domain.dim):
        #     out.meta[axis].update(self.meta[axis])
        # out.set_scales(self.domain.dealias, keep_data=False)

        # Perform operation
        self.operate(out)

        # Update metadata
        #out.meta
        #np.copyto(out.constant, self.constant)

        # Reset to free temporary field arguments
        self.reset()

        # Update storage
        if self.store_last and (id is not None):
            self.last_id = id
            self.last_out = out

        return out

    def attempt(self, id=None):
        """Recursively attempt to evaluate operation."""

        return self.evaluate(id=id, force=False)

    @property
    def meta(self):
        meta = [basis.default_meta() for basis in self.domain.bases]
        for axis in range(self.domain.dim):
            for key in meta[axis]:
                meta[axis][key] = getattr(self, 'meta_%s' %key)(axis)
        return meta

    def meta_scale(self, axis):
        return self.domain.bases[axis].dealias

    def check_conditions(self):
        """Check that all argument fields are in proper layouts."""

        # This method must be implemented in derived classes and should return
        # a boolean indicating whether the operation can be computed without
        # changing the layout of any of the field arguments.

        raise NotImplementedError()

    def operate(self, out):
        """Perform operation."""

        # This method must be implemented in derived classes, take an output
        # field as its only argument, and evaluate the operation into this
        # field without modifying the data of the arguments.

        raise NotImplementedError()

    @staticmethod
    def from_string(string, vars, domain):
        """Build operator tree from string expression."""

        expression = eval(string, vars)
        if isinstance(expression, Operator):
            return expression
        elif isinstance(expression, Field):
            return Cast(expression)
        else:
            return Cast(expression, domain)


class UnaryOperator(Operator):
    arity = 1
    def __init__(self, *args, **kw):
        if not is_fieldlike(args[0]):
            raise ValueError()
        super().__init__(*args, **kw)

    def meta_scale(self, value):
        return value

class BinaryOperator(Operator):
    arity = 2
    def meta_scale(self, axis, value1, value2):
        if value1 != value2:
            raise ValueError("Scales do not match")
        return


class Cast(Operator, metaclass=MultiClass):

    name = 'Cast'

    def check_conditions(self):
        return True


class CastField(Cast):

    @staticmethod
    def _check_args(*args, **kw):
        return is_field(args[0])

    def meta_constant(self, axis):
        # Preserve constancy
        return self.args[0].meta[axis]['constant']

    def meta_parity(self, axis):
        # Preserve parity
        return self.args[0].meta[axis]['parity']

    def operate(self, out):
        arg0, = self.args
        # Copy in current layout
        out.layout = arg0.layout
        np.copyto(out.data, arg0.data)


class CastArray(Cast):

    @staticmethod
    def _check_args(*args, **kw):
        return is_array(args[0])

    def meta_constant(self, axis):
        # Assume arrays are nonconstant
        return False

    def operate(self, out):
        # Copy in grid layout
        out['g'] = self.args[0]


class CastScalar(Cast):

    @staticmethod
    def _check_args(*args, **kw):
        return is_scalar(args[0])

    def meta_constant(self, axis):
        # Scalar is constant
        return True

    def meta_parity(self, axis):
        # Scalar has even parity
        return 1

    def operate(self, out):
        # Copy in grid layout
        out['g'] = self.args[0]


# class GeneralFunction(Operator):

#     def __init__(self, domain, layout, func, args=[], kw={}, out=None,):

#         # Required attributes
#         self.args = list(args)
#         self.original_args = list(args)
#         self.domain = domain
#         self.out = out
#         self.last_id = None
#         # Additional attributes
#         self.layout = domain.distributor.get_layout_object(layout)
#         self.func = func
#         self.kw = kw
#         self._field_arg_indices = [i for (i,arg) in enumerate(self.args) if is_fieldlike(arg)]
#         try:
#             self.name = func.__name__
#         except AttributeError:
#             self.name = str(func)
#         self.build_metadata()

#     def build_metadata(self):
#         self.constant = np.array([False] * self.domain.dim)

#     def check_conditions(self):
#         # Fields must be in proper layout
#         for i in self._field_arg_indices:
#             if self.args[i].layout is not self.layout:
#                 return False
#         return True

#     def operate(self, out):
#         # Apply func in proper layout
#         for i in self._field_arg_indices:
#             self.args[i].require_layout(self.layout)
#         out.layout = self.layout
#         np.copyto(out.data, self.func(*self.args, **self.kw))


# class UfuncWrapper(Operator):

#     supported = {ufunc.__name__: ufunc for ufunc in
#         (np.sign, np.conj, np.exp, np.exp2, np.log, np.log2, np.log10, np.sqrt,
#          np.square, np.sin, np.cos, np.tan, np.arcsin, np.arccos, np.arctan,
#          np.sinh, np.cosh, np.tanh, np.arcsinh, np.arccosh, np.arctanh)}

#     def __init__(self, ufunc, arg0, out=None):

#         super().__init__(arg0, out=out)


#         # Required Attributes
#         self.args = [arg0]
#         self.original_args = [arg0]
#         self.domain = arg0.domain
#         self.out = out
#         self.last_id = None
#         # Additional attributes
#         self.ufunc = ufunc
#         self.name = ufunc.__name__
#         self._grid_layout = self.domain.distributor.grid_layout
#         self.build_metadata()

#     def build_metadata(self):
#         self.constant = np.copy(self.args[0].constant)

#     def check_conditions(self):
#         # Must be in grid layout
#         return (self.args[0].layout is self._grid_layout)

#     def operate(self, out):
#         # References
#         arg0, = self.args
#         # Apply ufunc in grid layout
#         arg0.require_grid_space()
#         out.layout = self._grid_layout
#         self.ufunc(arg0.data, out=out.data)

# class Sine(operator):
#     name = 'sin'
#     ufunc = np.sin

# class Cosine(operator):
#     name = 'cos'
#     ufunc = np.cos

# class Tangent(operator):
#     name = 'tan'
#     ufunc = np.tan


class UnaryGridFunction(Operator):

    arity = 1

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._grid_layout = self.domain.distributor.grid_layout

    def meta_constant(self, axis):
        # Preserves constancy
        return self.args[0].meta[axis]['constant']

    def check_conditions(self):
        # Must be in grid layout
        return (self.args[0].layout is self._grid_layout)

    def operate(self, out):
        # References
        arg0, = self.args
        # Rectify in grid layout
        arg0.require_grid_space()
        out.layout = self._grid_layout
        self.func(arg0.data, out=out.data)


class AbsoluteValue(UnaryGridFunction):

    name = 'Abs'
    func = np.absolute

    def meta_parity(self, axis):
        # Absolute value has even parity
        return 1


class MagnitudeSquared(UnaryGridFunction):

    name = 'MagSq'

    @staticmethod
    def func(x, out):
        np.multiply(x, x.conj(), out=out)

    def meta_parity(self, axis):
        # Magnitude has even parity
        return 1


class Negative(Operator):

    name = 'Neg'
    arity = 1

    def __str__(self):
        return '(-%s)' %self.args[0]

    def meta_constant(self, axis):
        # Preserves constancy
        return self.args[0].meta[axis]['constant']

    def meta_parity(self, axis):
        # Preserves parity
        return self.args[0].meta[axis]['parity']

    def check_conditions(self):
        return True

    def operate(self, out):
        arg0, = self.args
        # Negate in current layout
        out.layout = arg0.layout
        np.negative(arg0.data, out.data)


class Arithmetic(Operator):

    arity = 2

    def __str__(self):
        str_args = map(str, self.args)
        return '(%s)' %self.str_op.join(str_args)


class Add(Arithmetic, metaclass=MultiClass):

    name = 'Add'
    str_op = ' + '


class AddFieldField(Add):

    @staticmethod
    def _check_args(*args, **kw):
        return (is_fieldlike(args[0]) and is_fieldlike(args[1]))

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

    def check_conditions(self):
        # Layouts must match
        return (self.args[0].layout is self.args[1].layout)

    def operate(self, out):
        arg0, arg1 = self.args
        # Add in arg0 layout (arbitrary choice)
        arg1.require_layout(arg0.layout)
        out.layout = arg0.layout
        np.add(arg0.data, arg1.data, out.data)


class AddFieldArray(Add):

    @staticmethod
    def _check_args(arg0, arg1):
        return (is_fieldlike(arg0) and is_array(arg1))

    def __init__(self, *args, **kw):
        Add.__init__(self, *args, **kw)
        self._grid_layout = self.domain.distributor.grid_layout

    def meta_constant(self, axis):
        # Assume arrays are nonconstant
        return False

    def meta_parity(self, axis):
        # Assume arrays have undefined parity
        raise UndefinedParityError("Arrays have undefined parity.")

    def check_conditions(self):
        # Must be in grid layout
        return (self.args[0].layout is self._grid_layout)

    def operate(self, out):
        arg0, arg1 = self.args
        # Add in grid layout
        arg0.require_grid_space()
        out.layout = self._grid_layout
        np.add(arg0.data, arg1, out.data)


class AddFieldScalar(Add):

    @staticmethod
    def _check_args(arg0, arg1):
        return (is_fieldlike(arg0) and is_scalar(arg1))

    def __init__(self, *args, **kw):
        Add.__init__(self, *args, **kw)
        self._grid_layout = self.domain.distributor.grid_layout

    def meta_constant(self, axis):
        # Preserve constancy
        return self.args[0].meta[axis]['constant']

    def meta_parity(self, axis):
        # Only add scalar to even parity
        parity0 = self.args[0].meta[axis]['parity']
        if parity0 == 1:
            return 1
        else:
            raise UndefinedParityError("Cannot add a constant to an odd field.")

    def check_conditions(self):
        # Must be in grid layout
        return (self.args[0].layout is self._grid_layout)

    def operate(self, out):
        arg0, arg1 = self.args
        # Add in grid layout
        arg0.require_grid_space()
        out.layout = self._grid_layout
        np.add(arg0.data, arg1, out.data)


# class AddNumericField(Add):

#     @staticmethod
#     def _check_args(*args, **kw):
#         return (is_numeric(args[0]) and is_fieldlike(args[1]))

#     def __init__(self, *args, **kw):
#         Add.__init__(self, *args, **kw)
#         self._grid_layout = self.domain.distributor.grid_layout

#     def build_metadata(self):
#         self.constant = numeric_constant(self.args[0], self.domain) & self.args[1].constant

#     def check_conditions(self):
#         # Must be in grid layout
#         return (self.args[1].layout is self._grid_layout)

#     def operate(self, out):
#         # References
#         arg0, arg1 = self.args
#         # Add in grid layout
#         arg1.require_grid_space()
#         out.layout = self._grid_layout
#         np.add(arg0, arg1.data, out.data)


# class Subtract(Arithmetic, metaclass=MultiClass):

#     name = 'Sub'
#     str_op = ' - '


# class SubFieldField(Subtract):

#     @staticmethod
#     def _check_args(*args, **kw):
#         return (is_fieldlike(args[0]) and is_fieldlike(args[1]))

#     def build_metadata(self):
#         self.constant = self.args[0].constant & self.args[1].constant

#     def check_conditions(self):
#         # Layouts must match
#         return (self.args[0].layout is self.args[1].layout)

#     def operate(self, out):
#         # References
#         arg0, arg1 = self.args
#         # Subtract in arg0 layout (arbitrary choice)
#         arg1.require_layout(arg0.layout)
#         out.layout = arg0.layout
#         np.subtract(arg0.data, arg1.data, out.data)


# class SubFieldNumeric(Subtract):

#     @staticmethod
#     def _check_args(arg0, arg1):
#         return (is_fieldlike(arg0) and is_numeric(arg1))

#     def __init__(self, *args, **kw):
#         Subtract.__init__(self, *args, **kw)
#         self._grid_layout = self.domain.distributor.grid_layout

#     def build_metadata(self):
#         self.constant = self.args[0].constant & numeric_constant(self.args[1], self.domain)

#     def check_conditions(self):
#         # Must be in grid layout
#         return (self.args[0].layout is self._grid_layout)

#     def operate(self, out):
#         # References
#         arg0, arg1 = self.args
#         # Subtract in grid layout
#         arg0.require_grid_space()
#         out.layout = self._grid_layout
#         np.subtract(arg0.data, arg1, out.data)


# class SubNumericField(Subtract):

#     @staticmethod
#     def _check_args(*args, **kw):
#         return (is_numeric(args[0]) and is_fieldlike(args[1]))

#     def __init__(self, *args, **kw):
#         Subtract.__init__(self, *args, **kw)
#         self._grid_layout = self.domain.distributor.grid_layout

#     def build_metadata(self):
#         self.constant = numeric_constant(self.args[0], self.domain) & self.args[1].constant

#     def check_conditions(self):
#         # Must be in grid layout
#         return (self.args[1].layout is self._grid_layout)

#     def operate(self, out):
#         # References
#         arg0, arg1 = self.args
#         # Subtract in grid layout
#         arg1.require_grid_space()
#         out.layout = self._grid_layout
#         np.subtract(arg0, arg1.data, out.data)


class Multiply(Arithmetic, metaclass=MultiClass):

    name = 'Mul'
    str_op = ' * '


class MultiplyFieldField(Multiply):

    @staticmethod
    def _check_args(*args, **kw):
        return (is_fieldlike(args[0]) and is_fieldlike(args[1]))

    def __init__(self, *args, **kw):
        Multiply.__init__(self, *args, **kw)
        self._grid_layout = self.domain.distributor.grid_layout

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

    def check_conditions(self):
        # Must be in grid layout
        return ((self.args[0].layout is self._grid_layout) and
                (self.args[1].layout is self._grid_layout))

    def operate(self, out):
        arg0, arg1 = self.args
        # Multiply in grid layout
        arg0.require_grid_space()
        arg1.require_grid_space()
        out.layout = self._grid_layout
        np.multiply(arg0.data, arg1.data, out.data)


class MultiplyFieldArray(Multiply):

    @staticmethod
    def _check_args(arg0, arg1):
        return (is_fieldlike(arg0) and is_array(arg1))

    def __init__(self, *args, **kw):
        Multiply.__init__(self, *args, **kw)
        self._grid_layout = self.domain.distributor.grid_layout

    def meta_constant(self, axis):
        # Assume arrays are nonconstant
        return False

    def meta_parity(self, axis):
        # Assume arrays have undefined parity
        raise UndefinedParityError("Arrays have undefined parity.")

    def check_conditions(self):
        # Must be in grid layout
        return (self.args[0].layout is self._grid_layout)

    def operate(self, out):
        arg0, arg1 = self.args
        # Multiply in grid layout
        arg0.require_grid_space()
        out.layout = self._grid_layout
        np.multiply(arg0.data, arg1, out.data)


class MultiplyFieldScalar(Multiply):

    @staticmethod
    def _check_args(arg0, arg1):
        return (is_fieldlike(arg0) and is_scalar(arg1))

    def meta_constant(self, axis):
        # Preserve constancy
        return self.args[0].meta[axis]['constant']

    def meta_parity(self, axis):
        # Preserve parity
        return self.args[0].meta[axis]['parity']

    def check_conditions(self):
        return True

    def operate(self, out):
        arg0, arg1 = self.args
        # Multiply in current layout
        out.layout = arg0.layout
        np.multiply(arg0.data, arg1, out.data)


# class MultScalarField(Multiply):

#     @staticmethod
#     def _check_args(arg0, arg1):
#         return (is_scalar(arg0) and is_fieldlike(arg1))

#     def build_metadata(self):
#         self.constant = np.copy(self.args[1].constant)

#     def check_conditions(self):
#         return True

#     def operate(self, out):
#         # References
#         arg0, arg1 = self.args
#         # Multiply in current layout
#         out.layout = arg1.layout
#         np.multiply(arg0, arg1.data, out.data)


# class MultArrayField(Multiply):

#     @staticmethod
#     def _check_args(arg0, arg1):
#         return (is_array(arg0) and is_fieldlike(arg1))

#     def __init__(self, *args, **kw):
#         Multiply.__init__(self, *args, **kw)
#         self._grid_layout = self.domain.distributor.grid_layout

#     def build_metadata(self):
#         self.constant = numeric_constant(self.args[0], self.domain) & self.args[1].constant

#     def check_conditions(self):
#         # Must be in grid layout
#         return (self.args[1].layout is self._grid_layout)

#     def operate(self, out):
#         # References
#         arg0, arg1 = self.args
#         # Multiply in grid layout
#         arg1.require_grid_space()
#         out.layout = self._grid_layout
#         np.multiply(arg0, arg1.data, out.data)


# class Divide(Arithmetic, metaclass=MultiClass):

#     name = 'Div'
#     str_op = ' / '


# class DivFieldField(Divide):

#     @staticmethod
#     def _check_args(*args, **kw):
#         return (is_fieldlike(args[0]) and is_fieldlike(args[1]))

#     def __init__(self, *args, **kw):
#         Divide.__init__(self, *args, **kw)
#         self._grid_layout = self.domain.distributor.grid_layout

#     def build_metadata(self):
#         self.constant = self.args[0].constant & self.args[1].constant

#     def check_conditions(self):
#         # Must be in grid layout
#         return ((self.args[0].layout is self._grid_layout) and
#                 (self.args[1].layout is self._grid_layout))

#     def operate(self, out):
#         # References
#         arg0, arg1 = self.args
#         # Divide in grid layout
#         arg0.require_grid_space()
#         arg1.require_grid_space()
#         out.layout = self._grid_layout
#         np.divide(arg0.data, arg1.data, out.data)


# class DivFieldScalar(Divide):

#     @staticmethod
#     def _check_args(arg0, arg1):
#         return (is_fieldlike(arg0) and is_scalar(arg1))

#     def build_metadata(self):
#         self.constant = np.copy(self.args[0].constant)

#     def check_conditions(self):
#         return True

#     def operate(self, out):
#         # References
#         arg0, arg1 = self.args
#         # Divide in current layout
#         out.layout = arg0.layout
#         np.divide(arg0.data, arg1, out.data)


# class DivFieldArray(Divide):

#     @staticmethod
#     def _check_args(arg0, arg1):
#         return (is_fieldlike(arg0) and is_array(arg1))

#     def __init__(self, *args, **kw):
#         Divide.__init__(self, *args, **kw)
#         self._grid_layout = self.domain.distributor.grid_layout

#     def build_metadata(self):
#         self.constant = self.args[0].constant & numeric_constant(self.args[1], self.domain)

#     def check_conditions(self):
#         # Must be in grid layout
#         return (self.args[0].layout is self._grid_layout)

#     def operate(self, out):
#         # References
#         arg0, arg1 = self.args
#         # Divide in grid layout
#         arg0.require_grid_space()
#         out.layout = self._grid_layout
#         np.divide(arg0.data, arg1, out.data)


# class DivNumericField(Divide):

#     @staticmethod
#     def _check_args(arg0, arg1):
#         return (is_numeric(arg0) and is_fieldlike(arg1))

#     def __init__(self, *args, **kw):
#         Divide.__init__(self, *args, **kw)
#         self._grid_layout = self.domain.distributor.grid_layout

#     def build_metadata(self):
#         self.constant = numeric_constant(self.args[0], self.domain) & self.args[1].constant

#     def check_conditions(self):
#         # Must be in grid layout
#         return (self.args[1].layout is self._grid_layout)

#     def operate(self, out):
#         # References
#         arg0, arg1 = self.args
#         # Divide in grid layout
#         arg1.require_grid_space()
#         out.layout = self._grid_layout
#         np.divide(arg0, arg1.data, out.data)


class Power(Arithmetic, metaclass=MultiClass):

    name = 'Pow'
    str_op = '**'


class PowerFieldScalar(Power):

    @staticmethod
    def _check_args(arg0, arg1):
        return (is_fieldlike(arg0) and is_scalar(arg1))

    def __init__(self, *args, **kw):
        Power.__init__(self, *args, **kw)
        self._grid_layout = self.domain.distributor.grid_layout

    def meta_constant(self, axis):
        # Preserve constancy
        return self.args[0].meta[axis]['constant']

    def meta_parity(self, axis):
        # Exponentiate parity
        power = self.args[1]
        if is_integer(power):
            return value**int(power)
        else:
            raise UndefinedParityError("Non-integer power of a field has undefined parity.")

    def check_conditions(self):
        # Must be in grid layout
        return (self.args[0].layout is self._grid_layout)

    def operate(self, out):
        arg0, arg1 = self.args
        # Raise in grid layout
        arg0.require_grid_space()
        out.layout = self._grid_layout
        np.power(arg0.data, arg1, out.data)


class LinearOperator(Operator):
    pass


class Separable(Operator):

    def check_conditions(self):
        arg0, = self.args
        axis = self.axis
        # Must be in coeff layout
        is_coeff = not arg0.layout.grid_spcae[axis]
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


class Coupled(Operator):

    def check_conditions(self):
        arg0, = self.args
        axis = self.axis
        # Must be in coeff+local layout
        is_coeff = not arg0.layout.grid_spcae[axis]
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
        for i in range(matrix.shape(0)):
            weights = reshape_vector(matrix[i,:], dim=dim, axis=axis)
            out.data[axslice(axis, i, i+1)] = np.sum(arg0.data*weights, axis=axis, keepdims=True)

    def explicit_form(self, input, output, axis):
        raise NotImplementedError()

    def matrix_form(self):
        raise NotImplementedError()


class Integrate(LinearOperator, metaclass=MultiClass):

    name = 'Integ'
    store_last = True

    @classmethod
    def _check_args(cls, arg0, basis, out=None):
        return (basis == cls.basis)

    def __init__(self, arg0, basis, out=None):
        # # No bases: integrate over whole domain
        # if len(bases) == 0:
        #     bases = list(arg0.domain.bases)
        # # Multiple bases: recursively integrate
        # if len(bases) > 1:
        #     arg0 = Integrate(arg0, *bases[:-1])
        super().__init__(arg0, out=out)
        self.axis = self.domain.bases.index(self.basis)

    def __repr__(self):
        return 'Integ(%r, %r)' %(self.args[0], self.basis)

    def __str__(self):
        return 'Integ(%s, %s)' %(self.args[0], self.basis)

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


class Interpolate(LinearOperator, metaclass=MultiClass):

    name = 'Interp'
    store_last = True

    @classmethod
    def _check_args(cls, arg0, basis, position, out=None):
        return (basis == cls.basis)

    def __init__(self, arg0, basis, position, out=None):
        super().__init__(arg0, position, out=out)
        self.axis = self.domain.bases.index(self.basis)

    def __repr__(self):
        return 'Interp(%r, %r, %r)' %(self.args[0], self.basis, self.args[1])

    def __str__(self):
        return 'Interp(%s, %s, %s)' %(self.args[0], self.basis, self.args[1])

    def meta_constant(self, axis):
        if axis == self.axis:
            # Integral is constant
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


class Differentiate(LinearOperator, metaclass=MultiClass):

    name = 'Diff'
    root = 'd'

    @classmethod
    def _check_args(cls, arg0, basis, out=None):
        return (basis == cls.basis)

    def __init__(self, arg0, basis=None, out=None):
        super().__init__(arg0, out=out)
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


class HilbertTransform(LinearOperator, metaclass=MultiClass):

    name = 'Hilbert'
    root = 'H'

    @classmethod
    def _check_args(cls, arg0, basis, out=None):
        return (basis == cls.basis)

    def __init__(self, arg0, basis=None, out=None):
        super().__init__(arg0, out=out)
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


class Left:

    name = 'L'

    def __new__(cls, arg0, out=None):
        basis = arg0.domain.bases[-1]
        position = basis.interval[0]
        return Interpolate(arg0, basis, position, out=out)


class Right:

    name = 'R'

    def __new__(cls, arg0, out=None):
        basis = arg0.domain.bases[-1]
        position = basis.interval[-1]
        return Interpolate(arg0, basis, position, out=out)



# Collect operators to expose to parser
op_list = [AbsoluteValue,
           MagnitudeSquared,
           Negative,
           Add,
           Multiply,
           Power,
           Integrate,
           Interpolate,
           Left,
           Right]
op_dict = {op.name: op for op in op_list}
root_list = [Differentiate,
             HilbertTransform]
root_dict = {op.root: op for op in root_list}

# Type tests
def is_scalar(arg):
    return np.isscalar(arg)

def is_array(arg):
    return isinstance(arg, np.ndarray)

def is_numeric(arg):
    return (is_scalar(arg) or is_array(arg))

def is_field(arg):
    return isinstance(arg, Field)

def is_fieldlike(arg):
    return isinstance(arg, (Field, Operator))


# Convenience functions
# def create_diff_operator(basis_, axis_):
#     """Create differentiation operator for a basis+axis."""

#     if basis_.name is not None:
#         name_ = 'd' + basis_.name
#     else:
#         name_ = 'd' + str(axis_)

#     class d_(Differentiate):
#         name = name_
#         basis = basis_
#         axis = axis_

#     return d_

def unique_domain(*args):
    """Return unique domain from a set of fields."""

    # Get set of domains
    domains = []
    for arg in args:
        if is_fieldlike(arg):
            domains.append(arg.domain)
    domain_set = set(domains)

    if len(domain_set) > 1:
        raise ValueError("Non-unique domains")
    else:
        return list(domain_set)[0]

