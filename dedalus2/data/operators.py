"""
Operator classes for fields.

"""

import numpy as np

from ..tools.general import OrderedSet
from ..tools.dispatch import MultiClass
# Bottom-import to resolve cyclic dependencies:
# from .field import Field


class Operator:
    """
    Base class for operations on fields.

    Parameters
    ----------
    *args : fields, operators, and numeric types
        Operands. Number must match class attribute `arity`, if present.
    out : field, optional
        Output field.  If not specified, a new field will be used.

    Notes
    -----
    Operators are stacked (i.e. provided as inputs to other operators) to
    construct trees that represent more complicated expressions.  Nodes
    are evaluated by first recursively evaluating their subtrees, and then
    calling the `operate` method.

    """

    name = 'Op'
    arity = None

    def __init__(self, *args, out=None):

        # Initial attributes
        self.args = list(args)
        self.original_args = list(args)  # For resetting
        self.out = out

        # Check arity
        if self.arity is not None:
            if len(args) != self.arity:
                raise ValueError("Wrong number of arguments.")

        # Check that domains match
        self.domain = unique_domain(self.field_set(include_out=True))
        if not self.domain:
            raise ValueError("Arguments/outputs have multiple domains.")

    def __repr__(self):

        # Represent as "name(*args)"
        repr_op = self.name
        repr_args = [a.__repr__() for a in self.args]

        return repr_op + '(' + ', '.join(repr_args) + ')'

    def __neg__(self):
        return Negation(self)

    def __add__(self, other):
        return Addition(self, other)

    def __radd__(self, other):
        return Addition(other, self)

    def __sub__(self, other):
        return Subtraction(self, other)

    def __rsub__(self, other):
        return Subtraction(other, self)

    def __mul__(self, other):
        return Multiplication(self, other)

    def __rmul__(self, other):
        return Multiplication(other, self)

    def _reset(self):

        # Restore original arguments
        self.args = list(self.original_args)

    def field_set(self, include_out=False):
        """Set of field leaves."""

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
                if a_eval:
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
        self._reset()

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

        # This method must be implemented in derived classes and should take an
        # output field as its only argument, and evaluate the operation into
        # this field without modifying the data of the arguments.

        raise NotImplementedError()


class Negation(Operator):

    name = 'Neg'
    arity = 1

    def __str__(self):
        # Print as "(-arg)"
        str_arg = self.args[0].__str__()
        return '(' + '-' + str_arg + ')'

    def check_conditions(self):
        # No conditions
        return True

    def operate(self, out):
        # Negate in current layout
        layout = self.args[0].layout
        out[layout] = -self.args[0][layout]


class Arithmetic(Operator):

    arity = 2

    def __str__(self):
        # Print as "(arg1 [] arg2)"
        str_op = self.str_op
        str_args = [a.__str__() for a in self.args]
        return '(' + str_op.join(str_args) + ')'


class Addition(Arithmetic, metaclass=MultiClass):

    name = 'Add'
    str_op = ' + '


class AddFieldField(Addition):

    @staticmethod
    def _check_args(arg0, arg1):
        return (is_field(arg0) and is_field(arg1))

    def check_conditions(self):
        # Layouts must match
        return (self.args[0].layout is self.args[1].layout)

    def operate(self, out):
        # Add in args[0] layout (arbitrary choice)
        layout = self.args[0].layout
        out[layout] = self.args[0][layout] + self.args[1][layout]


class AddScalarField(Addition):

    @staticmethod
    def _check_args(arg0, arg1):
        return (is_scalar(arg0) and is_field(arg1))

    def check_conditions(self):
        # Must be in grid space
        grid_layout = self.domain.distributor.grid_layout
        return (self.args[1].layout is grid_layout)

    def operate(self, out):
        # Add in grid space
        out['g'] = self.args[0] + self.args[1]['g']


class AddFieldScalar(Addition):

    @staticmethod
    def _check_args(arg0, arg1):
        return (is_field(arg0) and is_scalar(arg1))

    def check_conditions(self):
        # Must be in grid space
        grid_layout = self.domain.distributor.grid_layout
        return (self.args[0].layout is grid_layout)

    def operate(self, out):
        # Add in grid space
        out['g'] = self.args[0]['g'] + self.args[1]


class Subtraction(Arithmetic, metaclass=MultiClass):

    name = 'Sub'
    str_op = ' - '


class SubFieldField(Subtraction):

    @staticmethod
    def _check_args(arg0, arg1):
        return (is_field(arg0) and is_field(arg1))

    def check_conditions(self):
        # Layouts must match
        return (self.args[0].layout is self.args[1].layout)

    def operate(self, out):
        # Subtract in args[0] layout (arbitrary choice)
        layout = self.args[0].layout
        out[layout] = self.args[0][layout] - self.args[1][layout]


class SubScalarField(Subtraction):

    @staticmethod
    def _check_args(arg0, arg1):
        return (is_scalar(arg0) and is_field(arg1))

    def check_conditions(self):
        # Must be in grid space
        grid_layout = self.domain.distributor.grid_layout
        return (self.args[1].layout is grid_layout)

    def operate(self, out):
        # Subtract in grid space
        out['g'] = self.args[0] - self.args[1]['g']


class SubFieldScalar(Subtraction):

    @staticmethod
    def _check_args(arg0, arg1):
        return (is_field(arg0) and is_scalar(arg1))

    def check_conditions(self):
        # Must be in grid space
        grid_layout = self.domain.distributor.grid_layout
        return (self.args[0].layout is grid_layout)

    def operate(self, out):
        # Subtract in grid space
        out['g'] = self.args[0]['g'] - self.args[1]


class Multiplication(Arithmetic, metaclass=MultiClass):

    name = 'Mult'
    str_op = ' * '


class MultFieldField(Multiplication):

    @staticmethod
    def _check_args(arg0, arg1):
        return (is_field(arg0) and is_field(arg1))

    def check_conditions(self):
        # Must be in grid space
        grid_layout = self.domain.distributor.grid_layout
        return ((self.args[0].layout is grid_layout) and
                (self.args[1].layout is grid_layout))

    def operate(self, out):
        # Multiply in grid space
        out['g'] = self.args[0]['g'] * self.args[1]['g']


class MultScalarField(Multiplication):

    @staticmethod
    def _check_args(arg0, arg1):
        return (is_scalar(arg0) and is_field(arg1))

    def check_conditions(self):
        # No conditions
        return True

    def operate(self, out):
        # Multiply in current layout
        layout = self.args[1].layout
        out[layout] = self.args[0] * self.args[1][layout]


class MultFieldScalar(Multiplication):

    @staticmethod
    def _check_args(arg0, arg1):
        return (is_field(arg0) and is_scalar(arg1))

    def check_conditions(self):
        # No conditions
        return True

    def operate(self, out):
        # Multiply in current layout
        layout = self.args[0].layout
        out[layout] = self.args[0][layout] * self.args[1]


class MagSquared(Operator):

    name = 'MagSq'
    arity = 1

    def check_conditions(self):
        # Must be in grid space
        grid_layout = self.domain.distributor.grid_layout
        return (self.args[0].layout is grid_layout)

    def operate(self, out):
        # Multiply by complex conjugate in grid space
        out['g'] = self.args[0]['g'] * self.args[0]['g'].conj()


def create_diff_operators(domain):

    ops = []

    for i, b in enumerate(domain.bases):

        class diff(Operator):

            name = 'D' + str(i)
            arity = 1

            index = i
            basis = domain.bases[i]

            def check_conditions(self):
                # Must be in ceoff space and local
                is_coeff = not self.args[0].layout.grid_space[self.index]
                is_local = self.args[0].layout.local[self.index]
                return (is_coeff and is_local)

            def operate(self, out):
                # Differentiate in proper space
                self.args[0].require_coeff_space(self.index)
                self.args[0].require_local(self.index)
                out.layout = self.args[0].layout
                self.basis.differentiate(self.args[0].data,
                                         out.data,
                                         axis=self.index)

        ops.append(diff)

    return ops


def is_field(arg):
    """Check if an object is a field or an operator (resolves to a field)."""

    return isinstance(arg, (Field, Operator))


def is_scalar(arg):
    """Check if an object is a scalar."""

    return np.isscalar(arg)


def unique_domain(fields):
    """Check if a set of fields are defined over the same domain."""

    # Get set of domains
    domains = set(f.domain() for f in fields)

    # Return domain if unique
    if len(domains) == 1:
        return list(domains)[0]
    # Otherwise return None
    else:
        return None


# Bottom-import to resolve cyclic dependencies:
from .field import Field

