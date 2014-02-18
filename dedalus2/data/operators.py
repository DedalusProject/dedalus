"""
Abstract and built-in classes defining deferred-evaluation operations on fields.

"""

import numpy as np

from ..tools.general import OrderedSet
from ..tools.dispatch import MultiClass
# Bottom-import to resolve cyclic dependencies:
# from .field import Field


class Operator:
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

    def __init__(self, *args, out=None):

        # Check arity
        if self.arity is not None:
            if len(args) != self.arity:
                raise ValueError("Wrong number of arguments.")

        # Required attributes
        self.args = list(args)
        self.original_args = list(args)
        self.domain = unique_domain(out, *args)
        self.out = out

    def __repr__(self):
        repr_args = map(repr, self.args)
        return '%s(%s)' %(self.name, ', '.join(repr_args))

    def __str__(self):
        str_args = map(str, self.args)
        return '%s(%s)' %(self.name, ', '.join(str_args))

    def __neg__(self):
        return Negate(self)

    def __add__(self, other):
        return Add(self, other)

    def __radd__(self, other):
        return Add(other, self)

    def __sub__(self, other):
        return Subtract(self, other)

    def __rsub__(self, other):
        return Subtract(other, self)

    def __mul__(self, other):
        return Multiply(self, other)

    def __rmul__(self, other):
        return Multiply(other, self)

    def __truediv__(self, other):
        return Divide(self, other)

    def __rtruediv__(self, other):
        return Divide(other, self)

    def _reset(self):
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

    def evaluate(self, force=True):
        """Recursively evaluate operation."""

        # Recursively attempt evaluation of all operator arguments
        # Track evaluation success with flag
        all_eval = True
        for i, a in enumerate(self.args):
            if isinstance(a, Operator):
                a_eval = a.evaluate(force=force)
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

        # Perform operation
        self.operate(out)

        # Reset to free temporary field arguments
        self.reset()

        return out

    def attempt(self):
        """Recursively attempt to evaluate operation."""

        return self.evaluate(force=False)

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
        else:
            return Cast(expression, domain)


class Cast(Operator, metaclass=MultiClass):

    name = 'Cast'

    def __init__(self, arg0, domain, out=None):
        # Required attributes
        self.args = [arg0]
        self.original_args = [arg0]
        self.domain = domain
        self.out = out


class CastField(Cast):

    @staticmethod
    def _check_args(*args, **kw):
        return is_field(args[0])

    def __init__(self, arg0, out=None):
        # Initialize using field domain
        Cast.__init__(self, arg0, arg0.domain, out=out)

    def check_conditions(self):
        return True

    def operate(self, out):
        # References
        arg0, = self.args
        # Copy in current layout
        layout = arg0.layout
        out[layout] = arg0[layout]
        np.copyto(out.constant, arg0.constant)


class CastNumeric(Cast):

    @staticmethod
    def _check_args(*args, **kw):
        return is_numeric(args[0])

    def __init__(self, *args, **kw):
        Cast.__init__(*args, **kw)
        self._arg0_constant = numeric_constant(args[0], self.domain)

    def check_conditions(self):
        return True

    def operate(self, out):
        # Copy in grid layout
        out['g'] = self.args[0]
        out.constant = self._arg0_constant


class Integrate(Operator):

    def __init__(self, arg0, *bases, out=None):
        # No bases: integrate over whole domain
        if len(bases) == 0:
            bases = list(arg0.domain.bases)
        # Multiple bases: recursively integrate
        if len(bases) > 1:
            arg0 = Integrate(arg0, *bases[:-1])
        # Required attributes
        self.args = [arg0]
        self.original_args = [arg0]
        self.domain = arg0.domain
        self.out = out
        # Additional attributes
        self.basis = bases[-1]
        self.axis = arg0.domain.bases.index(self.basis)

    def __repr__(self):
        return 'Int(%r, %r)' %(self.args[0], self.basis)

    def __str__(self):
        return 'Int(%s, %s)' %(self.args[0], self.basis)

    def check_conditions(self):
        # References
        arg0, = self.args
        axis = self.axis
        # Must be in ceoff+local layout
        is_coeff = not arg0.layout.grid_space[axis]
        is_local = arg0.layout.local[axis]

        return (is_coeff and is_local)

    def operate(self, out):
        # References
        arg0, = self.args
        axis = self.axis
        # Integrate in coeff+local layout
        arg0.require_coeff_space(axis)
        arg0.require_local(axis)
        out.layout = arg0.layout
        # Use basis integration method
        self.basis.integrate(arg0.data, out.data, axis=axis)
        np.copyto(out.constant, arg0.constant)
        out.constant[axis] = True


class Negate(Operator):

    name = 'Neg'
    arity = 1

    def __str__(self):
        return '(-%s)' %self.args[0]

    def check_conditions(self):
        return True

    def operate(self, out):
        # References
        arg0, = self.args
        # Negate in current layout
        out.layout = arg0.layout
        np.negative(arg0.data, out.data)
        np.copyto(out.constant, arg0.constant)


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

    def check_conditions(self):
        # Layouts must match
        return (self.args[0].layout is self.args[1].layout)

    def operate(self, out):
        # References
        arg0, arg1 = self.args
        # Add in arg0 layout (arbitrary choice)
        arg1.require_layout(arg0.layout)
        out.layout = arg0.layout
        np.add(arg0.data, arg1.data, out.data)
        out.constant = arg0.constant & arg1.constant


class AddFieldNumeric(Add):

    @staticmethod
    def _check_args(arg0, arg1):
        return (is_fieldlike(arg0) and is_numeric(arg1))

    def __init__(self, *args, **kw):
        Add.__init__(*args, **kw)
        self._arg1_constant = numeric_constant(args[1], self.domain)
        self._grid_layout = self.domain.distributor.grid_layout

    def check_conditions(self):
        # Must be in grid layout
        return (self.args[0].layout is self._grid_layout)

    def operate(self, out):
        # References
        arg0, arg1 = self.args
        # Add in grid layout
        arg0.require_grid_space()
        out.layout = self._grid_layout
        np.add(arg0.data, arg1, out.data)
        out.constant = arg0.constant & self._arg1_constant


class AddNumericField(Add):

    @staticmethod
    def _check_args(*args, **kw):
        return (is_numeric(args[0]) and is_fieldlike(args[1]))

    def __init__(self, *args, **kw):
        Add.__init__(*args, **kw)
        self._arg0_constant = numeric_constant(args[0], self.domain)
        self._grid_layout = self.domain.distributor.grid_layout

    def check_conditions(self):
        # Must be in grid layout
        return (self.args[1].layout is self._grid_layout)

    def operate(self, out):
        # References
        arg0, arg1 = self.args
        # Add in grid layout
        arg1.require_grid_layout()
        out.layout = self._grid_layout
        np.add(arg0, arg1.data, out.data)
        out.constant = self._arg0_constant & arg1.constant


class Subtract(Arithmetic, metaclass=MultiClass):

    name = 'Sub'
    str_op = ' - '


class SubFieldField(Subtract):

    @staticmethod
    def _check_args(*args, **kw):
        return (is_fieldlike(args[0]) and is_fieldlike(args[1]))

    def check_conditions(self):
        # Layouts must match
        return (self.args[0].layout is self.args[1].layout)

    def operate(self, out):
        # References
        arg0, arg1 = self.args
        # Subtract in arg0 layout (arbitrary choice)
        arg1.require_layout(arg0.layout)
        out.layout = arg0.layout
        np.subtract(arg0.data, arg1.data, out.data)
        out.constant = arg0.constant & arg1.constant


class SubFieldNumeric(Subtract):

    @staticmethod
    def _check_args(arg0, arg1):
        return (is_fieldlike(arg0) and is_numeric(arg1))

    def __init__(self, *args, **kw):
        Subtract.__init__(*args, **kw)
        self._arg1_constant = numeric_constant(args[1], self.domain)
        self._grid_layout = self.domain.distributor.grid_layout

    def check_conditions(self):
        # Must be in grid layout
        return (self.args[0].layout is self._grid_layout)

    def operate(self, out):
        # References
        arg0, arg1 = self.args
        # Subtract in grid layout
        arg0.require_grid_space()
        out.layout = self._grid_layout
        np.subtract(arg0.data, arg1, out.data)
        out.constant = arg0.constant & self._arg1_constant


class SubNumericField(Subtract):

    @staticmethod
    def _check_args(*args, **kw):
        return (is_numeric(args[0]) and is_fieldlike(args[1]))

    def __init__(self, *args, **kw):
        Subtract.__init__(*args, **kw)
        self._arg0_constant = numeric_constant(args[0], self.domain)
        self._grid_layout = self.domain.distributor.grid_layout

    def check_conditions(self):
        # Must be in grid layout
        return (self.args[1].layout is self._grid_layout)

    def operate(self, out):
        # References
        arg0, arg1 = self.args
        # Subtract in grid layout
        arg1.require_grid_space()
        out.layout = self._grid_layout
        np.subtract(arg0, arg1.data, out.data)
        out.constant = self._arg0_constant & arg1.constant


class Multiply(Arithmetic, metaclass=MultiClass):

    name = 'Mult'
    str_op = ' * '


class MultFieldField(Multiply):

    @staticmethod
    def _check_args(*args, **kw):
        return (is_fieldlike(args[0]) and is_fieldlike(args[1]))

    def __init__(self, *args, **kw):
        Multiply.__init__(*args, **kw)
        self._grid_layout = self.domain.distributor.grid_layout

    def check_conditions(self):
        # Must be in grid layout
        return ((self.args[0].layout is self._grid_layout) and
                (self.args[1].layout is self._grid_layout))

    def operate(self, out):
        # References
        arg0, arg1 = self.args
        # Multiply in grid layout
        arg0.require_grid_space()
        arg1.require_grid_space()
        out.layout = self._grid_layout
        np.multiply(arg0.data, arg1.data, out.data)
        out.constant = arg0.constant & arg1.constant


class MultFieldScalar(Multiply):

    @staticmethod
    def _check_args(arg0, arg1):
        return (is_fieldlike(arg0) and is_scalar(arg1))

    def check_conditions(self):
        return True

    def operate(self, out):
        # References
        arg0, arg1 = self.args
        # Multiply in current layout
        out.layout = arg0.layout
        np.multiply(arg0.data, arg1, out.data)
        np.copyto(out.constant, arg0.constant)


class MultScalarField(Multiply):

    @staticmethod
    def _check_args(arg0, arg1):
        return (is_scalar(arg0) and is_fieldlike(arg1))

    def check_conditions(self):
        return True

    def operate(self, out):
        # References
        arg0, arg1 = self.args
        # Multiply in current layout
        out.layout = arg1.layout
        np.multiply(arg0, arg1.data, out.data)
        np.copyto(out.constant, arg1.constant)


class MultFieldArray(Multiply):

    @staticmethod
    def _check_args(arg0, arg1):
        return (is_fieldlike(arg0) and is_array(arg1))

    def __init__(self, *args, **kw):
        Multiply.__init__(*args, **kw)
        self._arg1_constant = numeric_constant(args[1], self.domain)
        self._grid_layout = self.domain.distributor.grid_layout

    def check_conditions(self):
        # Must be in grid layout
        return (self.args[0].layout is self._grid_layout)

    def operate(self, out):
        # References
        arg0, arg1 = self.args
        # Multiply in grid layout
        arg0.require_grid_space()
        out.layout = self._grid_layout
        np.multiply(arg0.data, arg1, out.data)
        out.constant = arg0.constant & self._arg1_constant


class MultArrayField(Multiply):

    @staticmethod
    def _check_args(arg0, arg1):
        return (is_array(arg0) and is_fieldlike(arg1))

    def __init__(self, *args, **kw):
        Multiply.__init__(*args, **kw)
        self._arg0_constant = numeric_constant(args[0], self.domain)
        self._grid_layout = self.domain.distributor.grid_layout

    def check_conditions(self):
        # Must be in grid layout
        return (self.args[1].layout is self._grid_layout)

    def operate(self, out):
        # References
        arg0, arg1 = self.args
        # Multiply in grid layout
        arg1.require_grid_space()
        out.layout = self._grid_layout
        np.multiply(arg0, arg1.data, out.data)
        out.constant = self._arg0_constant & arg1.constant


class Divide(Arithmetic, metaclass=MultiClass):

    name = 'Div'
    str_op = ' / '


class DivFieldField(Divide):

    @staticmethod
    def _check_args(*args, **kw):
        return (is_fieldlike(args[0]) and is_fieldlike(args[1]))

    def __init__(self, *args, **kw):
        Divide.__init__(*args, **kw)
        self._grid_layout = self.domain.distributor.grid_layout

    def check_conditions(self):
        # Must be in grid layout
        return ((self.args[0].layout is self._grid_layout) and
                (self.args[1].layout is self._grid_layout))

    def operate(self, out):
        # References
        arg0, arg1 = self.args
        # Divide in grid layout
        arg0.require_grid_space()
        arg1.require_grid_space()
        out.layout = self._grid_layout
        np.divide(arg0.data, arg1.data, out.data)
        out.constant = arg0.constant & arg1.constant


class DivFieldScalar(Divide):

    @staticmethod
    def _check_args(arg0, arg1):
        return (is_fieldlike(arg0) and is_scalar(arg1))

    def check_conditions(self):
        return True

    def operate(self, out):
        # References
        arg0, arg1 = self.args
        # Divide in current layout
        out.layout = arg0.layout
        np.divide(arg0.data, arg1, out.data)
        np.copyto(out.constant, arg0.constant)


class DivFieldArray(Divide):

    @staticmethod
    def _check_args(arg0, arg1):
        return (is_fieldlike(arg0) and is_array(arg1))

    def __init__(self, *args, **kw):
        Divide.__init__(*args, **kw)
        self._arg1_constant = numeric_constant(args[1], self.domain)
        self._grid_layout = self.domain.distributor.grid_layout

    def check_conditions(self):
        # Must be in grid layout
        return (self.args[0].layout is self._grid_layout)

    def operate(self, out):
        # References
        arg0, arg1 = self.args
        # Divide in grid layout
        arg0.require_grid_space()
        out.layout = self._grid_layout
        np.divide(arg0.data, arg1, out.data)
        out.constant = arg0.constant & self._arg1_constant


class DivNumericField(Divide):

    @staticmethod
    def _check_args(arg0, arg1):
        return (is_numeric(arg0) and is_fieldlike(arg1))

    def __init__(self, *args, **kw):
        Divide.__init__(*args, **kw)
        self._arg0_constant = numeric_constant(args[0], self.domain)
        self._grid_layout = self.domain.distributor.grid_layout

    def check_conditions(self):
        # Must be in grid layout
        return (self.args[1].layout is self._grid_layout)

    def operate(self, out):
        # References
        arg0, arg1 = self.args
        # Divide in grid layout
        arg1.require_grid_space()
        out.layout = self._grid_layout
        np.divide(arg0, arg1.data, out.data)
        out.constant = self._arg0_constant & arg1.constant


class MagSquared(Operator):

    name = 'MagSq'
    arity = 1

    def __init__(self, *args, **kw):
        Operator.__init__(self, *args, **kw)
        self._grid_layout = self.domain.distributor.grid_layout

    def check_conditions(self):
        # Must be in grid layout
        return (self.args[0].layout is self._grid_layout)

    def operate(self, out):
        # References
        arg0, = self.args
        # Multiply by complex conjugate in grid layout
        arg0.require_grid_space()
        out.layout = self._grid_layout
        np.multiply(arg0.data, arg0.data.conj(), out.data)
        np.copyto(out.constant, arg0.constant)


def create_diff_operator(basis, axis):
    """Create differentiation operator for a basis+axis."""

    if basis.name is not None:
        name = basis.name
    else:
        name = str(axis)

    class diff(Operator):

        name = 'd' + name
        arity = 1
        basis = basis
        axis = axis

        def check_conditions(self):
            # References
            arg0, = self.args
            axis = self.axis
            # Must be in ceoff+local layout
            is_coeff = not arg0.layout.grid_space[axis]
            is_local = arg0.layout.local[axis]

            return (is_coeff and is_local)

        def operate(self, out):
            # References
            arg0, = self.args
            axis = self.axis
            # Differentiate in coeff+local space
            arg0.require_coeff_space(axis)
            arg0.require_local(axis)
            out.layout = arg0.layout
            # Use basis differentiation method
            self.basis.differentiate(arg0.data, out.data, axis=axis)
            np.copyto(out.constant, arg0.constant)

    return diff


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

def numeric_constant(arg0, domain):
    if is_scalar(arg0):
        return np.array([True]*domain.dim)
    elif is_numeric(arg0):
        # BUG: this could potentially give inconsistent results across
        # processes for edge cases where layout.shape[i] == 1
        return np.less(arg0.shape, domain.distributor.grid_layout.shape)

def unique_domain(*args):
    """Check if a set of fields are defined over the same domain."""

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


# Bottom-import to resolve cyclic dependencies:
from .field import Field

