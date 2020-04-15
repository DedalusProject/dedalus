"""
Arithmetic operators.

"""

from functools import reduce
import numpy as np
from scipy import sparse
import itertools
import operator
import numbers

from .domain import Domain
from .field import Operand, Array, Field
from .future import Future, FutureArray, FutureField
from .operators import convert
from ..tools.cache import CachedAttribute
from ..tools.dispatch import MultiClass
from ..tools.exceptions import NonlinearOperatorError
from ..tools.exceptions import SymbolicParsingError
from ..tools.exceptions import SkipDispatchException
from ..tools.general import unify_attributes, DeferredTuple


class Add(Future, metaclass=MultiClass):
    """Addition operator."""

    name = 'Add'

    @classmethod
    def _preprocess_args(cls, *args, **kw):
        # Drop zeros
        args = [arg for arg in args if arg != 0]
        # Flatten additions
        # arg_sets = [arg.args if isinstance(arg, Add) else [arg] for arg in args]
        # args = [arg for arg_set in arg_sets for arg in arg_set]
        # Return single argument
        if len(args) == 1:
            raise SkipDispatchException(output=args[0])
        # Cast all args to Operands, if any present
        if any(isinstance(arg, Operand) for arg in args):
            dist = unify_attributes(args, 'dist', require=False)
            args = [Operand.cast(arg, dist) for arg in args]
            return args, kw
        # Cast all args to Array, if any present
        # elif any(isinstance(arg, (Array, FutureArray)) for arg in args):
        #     raise NotImplementederror()
            # domain = unify_attributes(args, 'domain', require=False)
            # args = [Cast(arg, domain) for arg in args]
            # return args, kw
        # Use python summation
        else:
            raise SkipDispatchException(output=sum(args))

    @classmethod
    def _check_args(cls, *args, **kw):
        return all(isinstance(arg, cls.argtypes) for arg in args)

    def __init__(self, *args, out=None):
        # Convert arguments to output bases
        self._bases = self._build_bases(*args)
        args = [convert(arg, self._bases) for arg in args]
        super().__init__(*args, out=out)

    def __str__(self):
        str_args = map(str, self.args)
        return ' + '.join(str_args)

    def _build_bases(self, *args):
        """Build output bases."""
        bases = []
        for ax_bases in zip(*(arg.domain.full_bases for arg in args)):
            # All constant bases yields constant basis
            if all(basis is None for basis in ax_bases):
                bases.append(None)
            # Combine any constant bases to avoid adding None to None
            elif any(basis is None for basis in ax_bases):
                ax_bases = [basis for basis in ax_bases if basis is not None]
                bases.append(np.sum(ax_bases) + None)
            # Add all bases
            else:
                bases.append(np.sum(ax_bases))
        return tuple(bases)

    @property
    def base(self):
        return Add

    def split(self, *vars):
        """Split into expressions containing and not containing specified operands/operators."""
        # Sum over argument splittings
        split_args = zip(*(arg.split(*vars) for arg in self.args))
        split = tuple(sum(args) for args in split_args)
        return split

    def sym_diff(self, var):
        """Symbolically differentiate with respect to specified operand."""
        # Sum over argument derivatives
        return sum((arg.sym_diff(var) for arg in self.args))

    def expand(self, *vars):
        """Expand expression over specified variables."""
        # Sum over expanded arguments
        if self.has(*vars):
            return sum((arg.expand(*vars) for arg in self.args))
        else:
            return self

    def require_linearity(self, *vars, name=None):
        """Require expression to be linear in specified variables."""
        # Require arguments to be linear
        for arg in self.args:
            arg.require_linearity(*vars, name=name)

    def separability(self, *vars):
        """Determine separable dimensions of expression as a linear operator on specified variables."""
        # Logical and of argument separabilities
        arg_seps = [arg.separability(*vars) for arg in self.args]
        return np.logical_and.reduce(arg_seps)

    # def operator_order(self, operator):
    #     """Determine maximum application order of an operator in the expression."""
    #     # Take maximum order from arguments
    #     return max(arg.operator_order(operator) for arg in self.args)

    def build_ncc_matrices(self, separability, vars, **kw):
        """Precompute non-constant coefficients and build multiplication matrices."""
        # Build argument matrices
        for arg in self.args:
            arg.build_ncc_matrices(separability, vars, **kw)

    def expression_matrices(self, subproblem, vars):
        """Build expression matrices for a specific subproblem and variables."""
        matrices = {}
        # Iteratively add argument expression matrices
        for arg in self.args:
            arg_matrices = arg.expression_matrices(subproblem, vars)
            for var in arg_matrices:
                matrices[var] = matrices.get(var, 0) + arg_matrices[var]
        return matrices


# class AddArrayArray(Add, FutureArray):

#     argtypes = (Array, FutureArray)

#     def check_conditions(self):
#         return True

#     def enforce_conditions(self):
#         pass

#     def operate(self, out):
#         arg0, arg1 = self.args
#         if out.data.size:
#             out.data.fill(0)
#             self.add_subdata(arg0, out)
#             self.add_subdata(arg1, out)


class AddFields(Add, FutureField):
    """Addition operator for fields."""

    argtypes = (Field, FutureField)

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.dist = unify_attributes(self.args, 'dist')
        self.domain = Domain(self.dist, self._bases)
        self.tensorsig = unify_attributes(self.args, 'tensorsig')
        self.dtype = np.result_type(*[arg.dtype for arg in self.args])

    def check_conditions(self):
        """Check that arguments are in a proper layout."""
        # All layouts must match
        layouts = set(arg.layout for arg in self.args)
        return (len(layouts) == 1)

    def enforce_conditions(self):
        """Require arguments to be in a proper layout."""
        # Determine best target layout
        layout = self.choose_layout()
        # Require layout for all args
        for arg in self.args:
            arg.require_layout(layout)

    def choose_layout(self):
        """Determine best target layout."""
        # OPTIMIZE: pick shortest absolute distance?
        # For now arbitrary pick first arg layout
        return self.args[0].layout

    def operate(self, out):
        """Perform operation."""
        args = self.args
        # Set output layout
        out.set_layout(args[0].layout)
        # Add all argument data
        args_data = [arg.data for arg in args]
        # OPTIMIZE: less intermediate arrays?
        np.copyto(out.data, np.sum(args_data, axis=0))


class Multiply(Future, metaclass=MultiClass):
    """Multiplication operator."""

    name = 'Mul'

    @classmethod
    def _preprocess_args(cls, *args, **kw):
        # Drop ones
        args = [arg for arg in args if arg != 1]
        # Flatten multiplications
        #arg_sets = [arg.args if isinstance(arg, Multiply) else [arg] for arg in args]
        #args = [arg for arg_set in arg_sets for arg in arg_set]
        # Return single argument
        if len(args) == 1:
            raise SkipDispatchException(output=args[0])
        # Return zero for any zero arguments
        elif any(arg == 0 for arg in args):
            raise SkipDispatchException(output=0)
        # Cast all args to Field, if any present
        elif any(isinstance(arg, (Field, FutureField)) for arg in args):
            dist = unify_attributes(args, 'dist', require=False)
            #args = [Cast(arg, domain) for arg in args]
            return args, kw
        # Cast all args to Array, if any present
        elif any(isinstance(arg, (Array, FutureArray)) for arg in args):
            raise NotImplementedError()
            # domain = unify_attributes(args, 'domain', require=False)
            # args = [Array.cast(arg, domain) for arg in args]
            # return args, kw
        # Use numpy multiplication
        else:
            raise SkipDispatchException(output=np.prod(args))

    @classmethod
    def _check_args(cls, *args, **kw):
        return all(isinstance(arg, cls.argtypes) for arg in args)

    def __str__(self):
        # Parenthesize addition arguments
        def paren_str(arg):
            if isinstance(arg, Add):
                return '({})'.format(arg)
            else:
                return str(arg)
        str_args = map(paren_str, self.args)
        return '*'.join(str_args)

    def _build_bases(self, *args):
        """Build output bases."""
        bases = []
        for ax_bases in zip(*(arg.domain.full_bases for arg in args)):
            # All constant bases yields constant basis
            if all(basis is None for basis in ax_bases):
                bases.append(None)
            # Combine any constant bases to avoid multiplying None and None
            elif any(basis is None for basis in ax_bases):
                ax_bases = [basis for basis in ax_bases if basis is not None]
                bases.append(np.prod(ax_bases) * None)
            # Multiply all bases
            else:
                bases.append(np.prod(ax_bases))
        return tuple(bases)

    @property
    def base(self):
        return Multiply

    def split(self, *vars):
        """Split into expressions containing and not containing specified operands/operators."""
        # Take cartesian product of argument splittings
        split_args = [arg.split(*vars) if isinstance(arg, Operand) else (0, arg) for arg in self.args]
        split_combos = list(map(np.prod, itertools.product(*split_args)))
        # Last combo is all negative splittings, others contain atleast one positive splitting
        return (sum(split_combos[:-1]), split_combos[-1])

    def sym_diff(self, var):
        """Symbolically differentiate with respect to specified operand."""
        args = self.args
        # Apply product rule to arguments
        partial_diff = lambda i: np.prod([arg.sym_diff(var) if i==j else arg for j,arg in enumerate(args)])
        return sum((partial_diff(i) for i in range(len(args))))

    def expand(self, *vars):
        """Expand expression over specified variables."""
        if self.has(*vars):
            # Expand arguments
            args = [arg.expand(*vars) for arg in self.args]
            # Sum over cartesian product of sums including specified variables
            arg_sets = [arg.args if (isinstance(arg, Add) and arg.has(*vars)) else [arg] for arg in args]
            out = sum(map(np.prod, itertools.product(*arg_sets)))
            return out
        else:
            return self

    # def simplify(self, *vars):
    #     """Simplify expression, except subtrees containing specified variables."""
    #     # Simplify arguments if variables are present
    #     if self.has(*vars):
    #         args = [arg.simplify(*vars) if isinstance(arg, Operand) else arg for arg in self.args]
    #         return self.base(*args)
    #     # Otherwise evaluate expression
    #     else:
    #         return self.evaluate()

    def require_linearity(self, *vars, name=None):
        """Require expression to be linear in specified variables."""
        nccs = [arg for arg in self.args if ((not isinstance(arg, Operand)) or (not arg.has(*vars)))]
        operands = [arg for arg in self.args if (arg not in nccs)]
        #operands = [arg for arg in self.args if arg.has(*vars)]
        # Require exactly one argument to contain vars, for linearity
        if len(operands) != 1:
            raise NonlinearOperatorError("{} is a non-linear product of the specified variables.".format(name if name else str(self)))
        operand = operands[0]
        # Require argument linearity
        operand.require_linearity(*vars, name=name)
        return nccs, operand

    def separability(self, *vars):
        """Determine separable dimensions of expression as a linear operator on specified variables."""
        nccs, operand = self.require_linearity(*vars)
        # NCCs couple along any non-constant dimensions
        ncc_bases = Operand.cast(np.prod(nccs), self.dist).domain.full_bases
        ncc_separability = np.array([(basis is None) for basis in ncc_bases])
        # Combine with operand separability
        return ncc_separability & operand.separability(*vars)

    # def operator_order(self, operator):
    #     """Determine maximum application order of an operator in the expression."""
    #     # Take maximum order from arguments
    #     return max(arg.operator_order(operator) for arg in self.args)

    def build_ncc_matrices(self, separability, vars, **kw):
        """Precompute non-constant coefficients and build multiplication matrices."""
        nccs, operand = self.require_linearity(*vars)
        # Continue NCC matrix construction
        operand.build_ncc_matrices(separability, vars, **kw)
        # Evaluate NCC
        ncc = np.prod(nccs)
        if isinstance(ncc, Future):
            ncc = ncc.evaluate()
        ncc.require_coeff_space()
        # Store NCC matrix, assuming constant-group along operand's constant axes
        # This generalization enables subproblem-agnostic pre-construction
        self._ncc_matrix = self._ncc_matrix_recursion(ncc.data, ncc.bases, operand.bases, separability, **kw)
        self._ncc_operand = operand
        self._ncc_vars = vars
        self._ncc_separability = separability

    @staticmethod
    def _ncc_matrix_recursion(data, ncc_bases, arg_bases, separability, **kw):
        """Build NCC matrix by recursing down through the axes."""
        # Build function for deferred-computation of matrix-valued coefficients
        def build_lower_coeffs(i):
            # Return scalar-valued coefficients at bottom level
            if len(data.shape) == 1:
                return data[i]
            # Otherwise recursively build matrix-valued coefficients
            else:
                args = (data[i], ncc_bases[1:], arg_bases[1:], separability[1:])
                return Multiply._ncc_matrix_recursion(*args, **kw)
        # Build top-level matrix using deferred coeffs
        coeffs = DeferredTuple(build_lower_coeffs, size=data.shape[0])
        ncc_basis = ncc_bases[0]
        arg_basis = arg_bases[0]
        # Kronecker with identities for constant NCC bases
        if ncc_basis is None:
            const = coeffs[0]
            # Trivial Kronecker with [[1]] for constant arg bases
            # This generalization enables problem-agnostic pre-construction
            if arg_basis is None:
                return const
            # Group-size identity for separable dimensions
            if separability[0]:
                I = sparse.identity(arg_basis.space.group_size)
            # Coeff-size identity for non-separable dimensions
            else:
                I = sparse.identity(arg_basis.space.coeff_size)
            # Apply cutoff to scalar coeffs
            if len(const.shape) == 0:
                cutoff = kw.get('cutoff', 1e-6)
                if abs(const) > cutoff:
                    return I * const
                else:
                    return I * 0
            else:
                return sparse.kron(I, const)
        # Call basis method for constructing NCC matrix
        else:
            return ncc_basis.ncc_matrix(arg_basis, coeffs, **kw)

    def expression_matrices(self, subproblem, vars):
        """Build expression matrices for a specific subproblem and variables."""
        # Check arguments vs. NCC matrix construction
        if not hasattr(self, '_ncc_matrix'):
            raise SymbolicParsingError("Must build NCC matrices before expression matrices.")
        if vars != self._ncc_vars:
            raise SymbolicParsingError("Must build NCC matrices with same variables.")
        if tuple(subproblem.problem.separability()) != tuple(self._ncc_separability):
            raise SymbolicParsingError("Must build NCC matrices with same separability.")
        # Build operand matrices
        operand = self._ncc_operand
        operand_mats = operand.expression_matrices(subproblem, vars)
        # Modify NCC matrix for subproblem
        # Build projection matrix dropping constant-groups as necessary
        group_shape = subproblem.group_shape(self.subdomain)
        const_shape = np.maximum(group_shape, 1)
        factors = (sparse.eye(*shape, format='csr') for shape in zip(group_shape, const_shape))
        projection = reduce(sparse.kron, factors, 1).tocsr()
        ncc_mat = projection @ self._ncc_matrix
        # Apply NCC matrix
        return {var: ncc_mat @ operand_mats[var] for var in operand_mats}


# class MultiplyArrayArray(Multiply, FutureArray):

#     argtypes = (Array, FutureArray)

#     def check_conditions(self):
#         return True

#     def enforce_conditions(self):
#         pass

#     def operate(self, out):
#         np.prod([arg.data for arg in self.args], axis=0, out=out.data)


class MultiplyFields(Multiply, FutureField):
    """Multiplication operator for fields."""

    argtypes = (Field, FutureField)

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        # Find required grid axes
        # Require grid space if more than one argument has nonconstant basis
        ax_bases = tuple(zip(*(arg.domain.bases for arg in self.args)))
        nonconst_ax_bases = [[b for b in bases if b is not None] for bases in ax_bases]
        self.required_grid_axes = [len(bases) > 1 for bases in nonconst_ax_bases]
        bases = tuple(reduce(operator.mul, bases) for bases in ax_bases)
        self.dist = unify_attributes(self.args, 'dist')
        self.domain = Domain(self.dist, bases)
        self.tensorsig = sum((arg.tensorsig for arg in self.args), tuple())
        self.dtype = np.result_type(*[arg.dtype for arg in self.args])

    def check_conditions(self):
        """Check that arguments are in a proper layout."""
        # All layouts must match
        layouts = set(arg.layout for arg in self.args)
        if len(layouts) != 1:
            return False
        # Required grid axes must be satisfied
        layout = layouts.pop()
        return all(layout.grid_space[self.required_grid_axes])

    def enforce_conditions(self):
        """Require arguments to be in a proper layout."""
        # Determine best target layout
        layout = self.choose_layout()
        # Require layout for all args
        for arg in self.args:
            arg.require_layout(layout)

    def choose_layout(self):
        """Determine best target layout."""
        # OPTIMIZE: pick shortest absolute distance?
        # For now require full grid space
        return self.domain.dist.grid_layout

    def operate(self, out):
        """Perform operation."""
        args = self.args
        out_order = len(self.tensorsig)
        # Set output layout
        out.set_layout(args[0].layout)
        # Multiply all argument data, reshaped by tensorsig
        args_data = []
        start_index = 0
        for arg in args:
            arg_order = len(arg.tensorsig)
            arg_shape = arg.data.shape
            shape = [1,] * out_order + list(arg_shape[arg_order:])
            shape[start_index: start_index + arg_order] = arg_shape[:arg_order]
            args_data.append(arg.data.reshape(shape))
            start_index += arg_order
        # OPTIMIZE: less intermediate arrays?
        np.copyto(out.data, reduce(np.multiply, args_data))


class MultiplyNumberField(Multiply, FutureField):
    """Multiplication operator for fields."""

    argtypes = ((numbers.Number, (Field, FutureField)),
                ((Field, FutureField), numbers.Number))

    def __init__(self, arg0, arg1, **kw):
        if isinstance(arg1, numbers.Number):
            arg0, arg1 = arg1, arg0
        super().__init__(arg0, arg1, **kw)
        self.domain = arg1.domain
        self.tensorsig = arg1.tensorsig
        self.dtype = np.result_type(type(arg0), arg1.dtype)

    @classmethod
    def _check_args(cls, *args, **kw):
        def check_types(args, types):
            return all(isinstance(arg, type) for arg, type in zip(args, types))
        return any(check_types(args, types) for types in cls.argtypes)

    def check_conditions(self):
        """Check that arguments are in a proper layout."""
        # Any layout
        return True

    def enforce_conditions(self):
        """Require arguments to be in a proper layout."""
        # Any layout
        pass

    def operate(self, out):
        """Perform operation."""
        arg0, arg1 = self.args
        # Set output layout
        out.set_layout(arg1.layout)
        # Multiply all argument data, reshaped by tensorsig
        np.multiply(arg0, arg1.data, out=out.data)

    def expression_matrices(self, subproblem, vars):
        """Build expression matrices for a specific subproblem and variables."""
        arg0, arg1 = self.args
        # Build field matrices
        arg1_mats = arg1.expression_matrices(subproblem, vars)
        # Multiply field matrices
        return {var: arg0 * arg1_mats[var] for var in arg1_mats}

