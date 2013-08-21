

import numpy as np

# Bottom of module:
# # Import after definitions to resolve cyclic dependencies
# from .field import field_manager, Field


class Operator:

    name = 'Op'
    arity = None

    def __init__(self, *args, out=None):

        # Inputs
        self.args = list(args)
        self.out = out

        # Store original arguments for resetting
        self.original_args = list(args)

        # Check number of arguments
        if self.arity is not None:
            if len(args) != self.arity:
                raise ValueError("Wrong number of arguments.")

        # Check that domains match
        self.domain = unique_domain(self.field_set(include_out=True))
        if not self.domain:
            raise ValueError("Arguments / outputs have multiple domains.")

    def __repr__(self):

        # Represent as "name(args)"
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

    def reset(self):

        # Restore original arguments
        self.args = list(self.original_args)

    def field_set(self, include_out=False):

        # Recursively collect field arguments
        fields = set()
        for a in self.args:
            if isinstance(a, Field):
                fields.add(a)
            elif isinstance(a, Operator):
                fields.update(a.field_set(include_out=include_out))

        # Add output field if requested
        if include_out:
            if self.out:
                fields.add(self.out)

        return fields

    def evaluate(self):

        # Create flag to track if all arguments are evaluable
        arg_flag = True

        # Recursively attempt evaluation of operator arguments
        # Note: We use a flag in order to attempt evaluation of all operator
        #       arguments, i.e. not just returning None after reaching the
        #       first unevaluable operator argument.
        for i, a in enumerate(self.args):
            if isinstance(a, Operator):
                a_eval = a.evaluate()
                # If argument evaluates, replace it with its result
                if a_eval:
                    self.args[i] = a_eval
                # Otherwise change argument flag
                else:
                    arg_flag = False

        # Return None if any arguments are not evaluable
        if not arg_flag:
            return None

        # Return None if field arguments have different layouts
        layout = unique_layout(self.field_set())
        if not layout:
            return None

        # Return None if operator conditions are not satisfied
        conditions = self.conditions(layout)
        if not conditions:
            return None

        # Allocate output field if necessary
        if self.out:
            out = self.out
        else:
            out = field_manager.get_field(self.domain)

        # Set output layout to argument layout
        out.layout = layout

        # Perform operation
        result = self.operation(out)

        # Reset self to free field arguments
        self.reset()

        return result

    def conditions(self, layout):

        return True

    def operation(self, out):

        # This method must be implemented in derived classes and should:
        #   - take an output field as its only argument, and return this field
        #   - assume all operator arguments have been evaluated to fields
        #   - not modify arguments

        raise NotImplementedError()


class Negation(Operator):

    name = 'Neg'
    arity = 1

    def __str__(self):

        # Print as "(-arg)"
        str_arg = self.args[0].__str__()

        return '(' + '-' + str_arg + ')'

    def operation(self, out):

        out.data[:] = -self.args[0].data

        return out


class Arithmetic(Operator):

    arity = 2

    def __str__(self):

        # Print as "(arg1 [] arg2)"
        str_op = self.str_op
        str_args = [a.__str__() for a in self.args]

        return '(' + str_op.join(str_args) + ')'

    def get_data(self, arg):

        if isinstance(arg, Field):
            return arg.data
        elif np.isscalar(arg):
            return arg
        else:
            raise TypeError("Unsupported type: %s" %type(arg).__name__)


class Addition(Arithmetic):

    name = 'Add'
    str_op = ' + '

    def operation(self, out):

        out.data[:] = self.get_data(self.args[0]) + self.get_data(self.args[1])

        return out


class Subtraction(Arithmetic):

    name = 'Sub'
    str_op = ' - '

    def operation(self, out):

        out.data[:] = self.get_data(self.args[0]) - self.get_data(self.args[1])

        return out


class Multiplication(Arithmetic):

    name = 'Mult'
    str_op = ' * '

    def conditions(self, layout):

        flag = True
        for a in self.args:
            if isinstance(a, Field):
                if a.space[0] == 'k':
                    flag = False
        return flag

    def operation(self, out):

        out.data[:] = self.get_data(self.args[0]) * self.get_data(self.args[1])

        return out


class MagSquared(Operator):

    name = 'MagSq'
    arity = 1

    def conditions(self):

        flag = True
        for a in self.args:
            if isinstance(a, Field):
                if a.space[0] == 'k':
                    flag = False
        return flag

    def operation(self, out):

        out.data[:] = self.args[0].data * self.args[0].data.conj()

        return out


def unique_domain(fields):

    # Get set of domains
    domains = set(f.domain for f in fields)

    # Return domain if unique
    if len(domains) == 1:
        return list(domains)[0]
    # Otherwise return None
    else:
        return None


def unique_layout(fields):

    # Get set of layouts
    layouts = set(f.layout for f in fields)

    # Return layout if unique
    if len(layouts) == 1:
        return list(layouts)[0]
    # Otherwise return None
    else:
        return None


# Import after definitions to resolve cyclic dependencies
from .field import field_manager, Field

