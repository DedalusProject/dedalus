"""
Classes for future evaluation.

"""

import numpy as np
from functools import partial

from .field import Operand, Field, LockedField
from .domain import Domain
#from .domain import Domain
from ..tools.general import OrderedSet, unify_attributes
from ..tools.cache import CachedAttribute, CachedMethod

import logging
logger = logging.getLogger(__name__.split('.')[-1])

from ..tools.config import config
STORE_OUTPUTS = config['memory'].getboolean('STORE_OUTPUTS')
STORE_LAST_DEFAULT = config['memory'].getboolean('STORE_LAST_DEFAULT')

class Future(Operand):
    """
    Base class for deferred operations on data.

    Parameters
    ----------
    *args : Operands
        Operands. Number must match class attribute `arity`, if present.
    out : data, optional
        Output data object.  If not specified, a new object will be used.

    Notes
    -----
    Operators are stacked (i.e. provided as arguments to other operators) to
    construct trees that represent compound expressions.  Nodes are evaluated
    by first recursively evaluating their subtrees, and then calling the
    `operate` method.

    """

    store_last = STORE_LAST_DEFAULT

    def __init__(self, *args, out=None):
        # # Check output consistency
        # if out is not None:
        #     if out.bases != self.bases:
        #         raise ValueError("Output field has wrong bases.")
        # Attributes
        self.args = list(args)
        self.original_args = tuple(args)
        self.out = out
        self.dist = unify_attributes(args, 'dist', require=False)
        #self.domain = Domain(self.dist, self.bases)
        self._grid_layout = self.dist.grid_layout
        self._coeff_layout = self.dist.coeff_layout
        self.last_id = None
        self.scales = 1 # self.domain.dealias

    # @CachedAttribute
    # def domain(self):
    #     return Domain(self.dist, self.bases)

    def __repr__(self):
        repr_args = map(repr, self.args)
        return '{}({})'.format(self.name, ', '.join(repr_args))

    def __str__(self):
        str_args = map(str, self.args)
        return '{}({})'.format(self.name, ', '.join(str_args))

    # def __eq__(self, other):
    #     # Require same class and arguments
    #     if type(other) is type(self):
    #         return self.args == other.args
    #     else:
    #         return NotImplemented

    # def __ne__(self, other):
    #     # Negate equality test
    #     if type(other) is type(self):
    #         return not self.__eq__(other)
    #     else:
    #         return NotImplemented

    @CachedAttribute
    def bases(self):
        # Subclasses must implement
        raise NotImplementedError()

    # @property
    # def name(self):
    #     return self.base.__name__

    def reset(self):
        """Restore original arguments."""
        self.args = list(self.original_args)

    def atoms(self, *types):
        """Gather all leaf-operands of specified types."""
        atoms = OrderedSet()
        # Recursively collect atoms
        for arg in self.args:
            if isinstance(arg, Operand):
                atoms.update(arg.atoms(*types))
        return atoms

    def has(self, *vars):
        """Determine if tree contains any specified operands/operators."""
        # Check for matching operator
        if any(isinstance(self, var) for var in vars if isinstance(var, type)):
            return True
        # Check arguments
        else:
            return any(arg.has(*vars) for arg in self.args if isinstance(arg, Operand))

    def replace(self, old, new):
        """Replace specified operand/operator."""
        # Check for entire expression match
        if self == old:
            return new
        # Check base and call with replaced arguments
        elif isinstance(old, type) and isinstance(self,old):
            args = [arg.replace(old, new) if isinstance(arg, Operand) else arg for arg in self.args]
            return new(*args)
        # Call with replaced arguments
        else:
            args = [arg.replace(old, new) if isinstance(arg, Operand) else arg for arg in self.args]
            return self.new_operands(*args)

    # def simplify(self, *vars):
    #     """Simplify expression, except subtrees containing specified variables."""
    #     # Simplify arguments if variables are present
    #     if self.has(*vars):
    #         args = [arg.simplify(*vars) if isinstance(arg, Operand) else arg for arg in self.args]
    #         return self.base(*args)
    #     # Otherwise evaluate expression
    #     else:
    #         return self.evaluate()

    def prep_nccs(self, vars):
        for arg in self.args:
            arg.prep_nccs(vars)

    def gather_ncc_coeffs(self):
        for arg in self.args:
            arg.gather_ncc_coeffs()

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
                a.change_scales(a.domain.dealias)
            if isinstance(a, Future):
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
        if force:
            self.enforce_conditions()
        else:
            # Return None if operator conditions are not satisfied
            if not self.check_conditions():
                return None

        # Allocate output field if necessary
        out = self.get_out()
        #out = self.domain.new_data(self.future_type)
        #out = Field(name=str(self), bases=self.bases)

        # Copy metadata
        out.preset_scales(self.domain.dealias)

        # Perform operation
        self.operate(out)

        # Reset to free temporary field arguments
        self.reset()

        # Update storage
        if self.store_last and (id is not None):
            self.last_id = id
            self.last_out = out

        return out

    def get_out(self):
        if self.out:
            return self.out
        else:
            out = self.build_out()
            if STORE_OUTPUTS:
                self.out = out
            return out

    def build_out(self):
        bases = self.domain.bases
        if any(bases):
            return self.future_type(dist=self.dist, bases=bases, tensorsig=self.tensorsig, dtype=self.dtype, name=str(self))
        else:
            return self.future_type(dist=self.dist, tensorsig=self.tensorsig, dtype=self.dtype, name=str(self))

    def attempt(self, id=None):
        """Recursively attempt to evaluate operation."""
        return self.evaluate(id=id, force=False)

    def check_conditions(self):
        """Check that arguments are in a proper layout."""
        # This method must be implemented in derived classes and should return
        # a boolean indicating whether the operation can be computed without
        # changing the layout of any of the field arguments.
        raise NotImplementedError()

    def enforce_conditions(self):
        """Require arguments to be in a proper layout."""
        raise NotImplementedError()

    def operate(self, out):
        """Perform operation."""
        # This method must be implemented in derived classes, take an output
        # field as its only argument, and evaluate the operation into this
        # field without modifying the data of the arguments.
        raise NotImplementedError()

    # def order(self, *ops):
    #     order = max(arg.order(*ops) for arg in self.args)
    #     if type(self) in ops:
    #         order += 1
    #     return order







class FutureField(Future):
    """Class for deferred operations producing a Field."""
    future_type = Field

    def __getitem__(self, layout):
        """Evaluate and return data viewed in specified layout."""
        field = self.evaluate()
        return field[layout]

    @staticmethod
    def parse(string, namespace, dist):
        """Build FutureField from a string expression."""
        expression = eval(string, namespace)
        return FutureField.cast(expression, dist)

    @staticmethod
    def cast(arg, dist):
        """Cast an object to a FutureField."""
        from .operators import FieldCopy
        # Cast to Operand (checks dist)
        arg = Operand.cast(arg, dist)
        # Cast to FutureField
        if isinstance(arg, FutureField):
            return arg
        else:
            return FieldCopy(arg)


class FutureLockedField(Future):
    """Class for deferred operations producing an Array."""
    future_type = LockedField
