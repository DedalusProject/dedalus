"""
Classes for future evaluation.

"""

import numpy as np
import uuid
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

    def __init__(self, *args, out=None, tangent=None, cotangent=None):
        # # Check output consistency
        # if out is not None:
        #     if out.bases != self.bases:
        #         raise ValueError("Output field has wrong bases.")
        # Attributes
        self.args = list(args)
        self.original_args = tuple(args)
        self.out = out
        self.tangent = tangent
        self.cotangent = cotangent
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

    def reset_jvp(self):
        """Restore original arguments."""
        self.args = list(self.original_args)
        self.arg_tangents = [None] * len(self.args)

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

    def evaluate(self, id=None, force=True, tape=None, clean_tangent=False):
        """Recursively evaluate operation."""

        # Default to uuid to cache within evaluation, but not across evaluations
        if id is None:
            id = uuid.uuid4()

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
                a_eval = a.evaluate(id=id, force=force, tape=tape)
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

        # Add to tape
        if tape is not None:
            tape.append(self)

        # Reset to free temporary field arguments
        self.reset()

        # Update storage
        if self.store_last and (id is not None):
            self.last_id = id
            self.last_out = out

        if clean_tangent:
            self.tangent.data.fill(0)

        return out

    def evaluate_jvp(self, tangents, id=None, force=True):
        """Recursively evaluate operation."""

        # Default to uuid to cache within evaluation, but not across evaluations
        if id is None:
            id = uuid.uuid4()

        # Check storage
        if self.store_last and (id is not None):
            if id == self.last_id:
                return self.last_out, self.last_tangent
            else:
                # Clear cache to free output field
                self.last_id = None
                self.last_out = None
                self.last_tangent = NotImplementedError

        # Recursively attempt evaluation of all operator arguments
        # Track evaluation success with flag
        self.arg_tangents = [None] * len(self.args)
        all_eval = True
        for i, a in enumerate(self.args):
            if isinstance(a, Field):
                a.change_scales(a.domain.dealias)
                if a in tangents:
                    tangents[a].change_scales(tangents[a].domain.dealias)
                    self.arg_tangents[i] = tangents[a]
            if isinstance(a, Future):
                a_eval = a.evaluate_jvp(tangents, id=id, force=force)
                # If evaluation succeeds, substitute result
                if a_eval is not None:
                    self.args[i] = a_eval[0]
                    self.arg_tangents[i] = a_eval[1]
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

        # Match tangent layouts to arguments
        for i in range(len(self.args)):
            if self.arg_tangents[i] is not None:
                self.arg_tangents[i].change_layout(self.args[i].layout)

        # Setup output field
        out = self.get_out()
        out.preset_scales(self.domain.dealias)

        # Call operation
        if any(self.arg_tangents):
            # Setup tangent field
            tangent = self.get_tangent()
            tangent.preset_scales(self.domain.dealias)
            self.operate_jvp(out, tangent)
        else:
            tangent = None
            self.operate(out)

        # Reset to free temporary field arguments
        self.reset_jvp()

        # Update storage
        if self.store_last and (id is not None):
            self.last_id = id
            self.last_out = out
            self.last_tangent = tangent

        return out, tangent

    def evaluate_vjp(self, cotangents, id=None, force=False):
        """Recursively evaluate operation."""

        cotangent = cotangents[self]

        # Force store_last
        # TODO: enforce recursively
        self.store_last = True
        if id is None:
            raise ValueError("id must be specified for vjp evaluation.")

        # Check cotangent basis and adjoint status
        if not cotangent.adjoint:
            raise ValueError("Cotangent must be an adjoint field.")
        if cotangent.domain != self.domain:
            raise ValueError("Cotangent must have same domain as operator.")

        # Forward evaluate and save topological sorting
        if id == self.last_id:
            tape = self.last_tape
            out = self.last_out
        else:
            tape = []
            out = self.evaluate(id=id, force=force, tape=tape)
            self.last_tape = tape

        # Clean cotangents
        for op in tape:
            op.get_cotangent()
            op.cotangent.preset_scales(op.domain.dealias)
            op.cotangent.data.fill(0)

        # Copy input cotangent
        self.cotangent.preset_layout(cotangent.layout)
        self.cotangent.data[:] = cotangent.data
        self.cotangent.change_scales(self.domain.dealias)

        # Reverse topological sorting and evaluate adjoint
        for op in tape[::-1]:
            # Replace arguments with operator outputs
            for i in range(len(op.args)):
                if isinstance(op.args[i], Future):
                    op.args[i] = op.args[i].out
            # Enforce conditions to get correct arg layouts
            layout = op.enforce_conditions()
            # Evaluate adoint
            op.operate_vjp(layout, cotangents)
            # Reset arguments
            op.reset()

        return out, cotangents

    def get_out(self):
        if self.out:
            return self.out
        else:
            out = self.build_out()
            if STORE_OUTPUTS:
                self.out = out
            return out

    def get_tangent(self):
        if self.tangent:
            return self.tangent
        else:
            tangent = self.build_out()
            if STORE_OUTPUTS:
                self.tangent = tangent
            return tangent

    def get_cotangent(self):
        if self.cotangent:
            return self.cotangent
        else:
            cotangent = self.build_out()
            cotangent.adjoint = True
            if STORE_OUTPUTS:
                self.cotangent = cotangent
            return cotangent

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
        raise NotImplementedError(f"check_conditions not implemented for {type(self)}")

    def enforce_conditions(self):
        """Require arguments to be in a proper layout."""
        raise NotImplementedError(f"enforce_conditions not implemented for {type(self)}")

    def operate(self, out):
        """Perform operation."""
        # This method must be implemented in derived classes, take an output
        # field as its only argument, and evaluate the operation into this
        # field without modifying the data of the arguments.
        raise NotImplementedError(f"operate not implemented for {type(self)}")

    def operate_jvp(self, out, tangent):
        """Perform operation."""
        raise NotImplementedError(f"operate_jvp not implemented for {type(self)}")

    def operate_vjp(self, cotangents):
        raise NotImplementedError(f"operate_vjp not implemented for {type(self)}")


class ExpressionList:

    def __init__(self, expressions):
        self.expressions = expressions
        self.last_id = None

    def evaluate_vjp(self, cotangents, id=None, force=False):
        if id is None:
            raise ValueError("id must be specified for vjp evaluation.")
        # Run forward evaluation and build joint tape (cached by id)
        if id == self.last_id:
            out = self.last_out
            tape = self.last_tape
        else:
            out = []
            tape = []
            for expr in self.expressions:
                # Force store_last
                # TODO: enforce recursively
                self.store_last = True
                # Forward evaluate and save topological sorting
                out.append(expr.evaluate(id=id, force=force, tape=tape))
            self.last_id = id
            self.last_out = out
            self.last_tape = tape
        # Clean cotangents
        for op in tape:
            op.get_cotangent()
            op.cotangent.preset_scales(op.domain.dealias)
            op.cotangent.data.fill(0)
        # Initialize expresion cotangents
        for expr in self.expressions:
            cotangent = cotangents[expr]
            # Check cotangent basis and adjoint status
            if not cotangent.adjoint:
                raise ValueError("Cotangent must be an adjoint field.")
            if cotangent.domain != expr.domain:
                raise ValueError("Cotangent must have same domain as operator.")
            # Copy input cotangents
            expr.cotangent.preset_layout(cotangent.layout)
            expr.cotangent.data[:] = cotangent.data
            expr.cotangent.change_scales(expr.domain.dealias)
        # Reverse topological sorting and evaluate adjoints
        for op in tape[::-1]:
            # Replace arguments with operator outputs
            for i in range(len(op.args)):
                if isinstance(op.args[i], Future):
                    op.args[i] = op.args[i].out
            # Enforce conditions to get correct arg layouts
            layout = op.enforce_conditions()
            # Evaluate adoint
            op.operate_vjp(layout, cotangents)
            # Reset arguments
            op.reset()
        return out, cotangents


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
