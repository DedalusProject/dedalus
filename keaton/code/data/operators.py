

import numpy as np

# Bottom of module:
# # Import after definitions to resolve cyclic dependencies
# from .field import field_manager, Field


class Operator:

    name = 'Operator'
    n_args = None

    def __init__(self, *args, out=None):

        # Inputs
        self.args = list(args)
        self.out = out

        # Store original arguments for resetting
        self._original_args = list(args)

        # Check number of arguments
        if self.n_args is not None:
            if len(args) != self.n_args:
                raise ValueError("Wrong number of arguments.")

    def __repr__(self):

        # Represent as "name(arguments)"
        r_op = self.name
        r_args = [a.__repr__() for a in self.args]

        return r_op + '(' + ', '.join(r_args) + ')'

    def __neg__(self):
        return Negative(self)

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

    def reset(self):

        # Restore original arguments
        for i, a in enumerate(self._original_args):
            self.args[i] = a

    def field_set(self):

        # Recursively collect field arguments
        fields = set()
        for a in self.args:
            if isinstance(a, Field):
                fields.add(a)
            elif isinstance(a, Operator):
                fields.update(a.field_set())

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
                # DEBUG: Might want to reset the operator here to free its field args
                if a_eval is not None:
                    self.args[i] = a_eval
                # Otherwise change argument flag
                else:
                    arg_flag = False

        # Return None if any arguments are not evaluable
        if not arg_flag:
            return None

        # DEBUG:  Check that all field layouts match
        # # Get field layout
        # layout = None
        # for i, a in enumerate(self.arsg):
        #     if isinstance(a, Field):
        #         if layout:
        #             if a.layout is not layout:
        #                 raise ValueError("Operator arguments must have same layout.")
        #         else:
        #             layout = a.layout

        # DEBUG: skip conditions
        # Check layout/space conditions
        # Return None if any condition is not satisfied
        # loop over conditions:
        #     if condition is not satisfied:
        #         return None

        # Allocate out field if necessary
        if self.out is None:
            for a in self.args:
                if isinstance(a, Field):
                    domain = a.domain
                    break
            out = field_manager.get_field(domain)
        else:
            out = self.out

        # Perform operation and return the result
        return self.operation(out)

    def operation(self, out):

        # This method must be implemented in derived classes and should:
        #   - take an output field as its only argument, and return this field
        #   - assume all operator arguments have been evaluated to fields
        #   - not modify arguments

        raise NotImplementedError()


class Negative(Operator):

    name = 'Neg'
    n_args = 1

    def __str__(self):

        # Print as "-arg"
        s_arg = self.args[0].__str__()

        # Parenthesize arithmetic operations
        if isinstance(self.args[0], (Negative, Add, Subtract, Multiply)):
            s_arg = '(' + s_arg + ')'

        return '-' + s_arg

    def operation(self, out):

        out.data[:] = -self.args[0].data

        return out


class Add(Operator):

    name = 'Add'
    n_args = 2

    def __str__(self):

        # Print as "arg1 + arg2"
        s_args = [a.__str__() for a in self.args]

        # Parenthesize arithmetic operations
        for i, a in enumerate(self.args):
            if isinstance(a, (Negative, Add, Subtract, Multiply)):
                s_args[i] = '(' + s_args[i] + ')'

        return ' + '.join(s_args)

    def get_data(self, arg):

        if isinstance(arg, Field):
            return arg.data
        elif np.isscalar(arg):
            return arg
        else:
            raise TypeError("Unsupported type: %s" %type(arg).__name__)

    def operation(self, out):

        out.data[:] = self.get_data(self.args[0]) + self.get_data(self.args[1])

        return out


class Subtract(Operator):

    name = 'Sub'
    n_args = 2

    def __str__(self):

        # Print as "arg1 - arg2"
        s_args = [a.__str__() for a in self.args]

        # Parenthesize arithmetic operations
        for i, a in enumerate(self.args):
            if isinstance(a, (Negative, Add, Subtract, Multiply)):
                s_args[i] = '(' + s_args[i] + ')'

        return ' - '.join(s_args)

    def get_data(self, arg):

        if isinstance(arg, Field):
            return arg.data
        elif np.isscalar(arg):
            return arg
        else:
            raise TypeError("Unsupported type: %s" %type(arg).__name__)

    def operation(self, out):

        out.data[:] = self.get_data(self.args[0]) - self.get_data(self.args[1])

        return out


class Multiply(Operator):

    name = 'Mult'
    n_args = 2

    def __str__(self):

        # Print as "arg1 * arg2"
        s_args = [a.__str__() for a in self.args]

        # Parenthesize arithmetic operations
        for i, a in enumerate(self.args):
            if isinstance(a, (Negative, Add, Subtract, Multiply)):
                s_args[i] = '(' + s_args[i] + ')'

        return ' * '.join(s_args)

    def get_data(self, arg):

        if isinstance(arg, Field):
            return arg.data
        elif np.isscalar(arg):
            return arg
        else:
            raise TypeError("Unsupported type: %s" %type(arg).__name__)

    def operation(self, out):

        out.data[:] = self.get_data(self.args[0]) * self.get_data(self.args[1])

        return out


# Import after definitions to resolve cyclic dependencies
from .field import field_manager, Field

