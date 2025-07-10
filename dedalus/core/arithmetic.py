"""
Arithmetic operators.

"""

import sys
from functools import reduce
import numpy as np
from scipy import sparse
import itertools
import operator
import numbers
import numexpr as ne
from collections import defaultdict
from math import prod

from .domain import Domain
from .field import Operand, Field
from .future import Future, FutureField
from .basis import Basis
from .operators import convert
from ..tools.array import kron
from ..tools.cache import CachedAttribute, CachedMethod
from ..tools.dispatch import MultiClass
from ..tools.exceptions import NonlinearOperatorError
from ..tools.exceptions import SymbolicParsingError
from ..tools.exceptions import SkipDispatchException
from ..tools.general import unify_attributes, DeferredTuple

# Public interface
__all__ = ['Add',
           'Multiply',
           'DotProduct',
           'CrossProduct']

# Aliases
aliases = {}
def alias(*names):
    def register_op(op):
        for name in names:
            aliases[name] = op
        return op
    return register_op


def enum_indices(tensorsig):
    shape = tuple(cs.dim for cs in tensorsig)
    return enumerate(np.ndindex(shape))


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
            tensorsig = unify_attributes(args, 'tensorsig', require=False)
            dtype = unify_attributes(args, 'dtype', require=False)
            args = [Operand.cast(arg, dist, tensorsig, dtype) for arg in args]
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

    def __init__(self, *args, out=None, **kw):
        # Convert arguments to output bases
        self._bases = self._build_bases(*args)
        args = [convert(arg, self._bases) for arg in args]
        super().__init__(*args, out=out)

    def __str__(self):
        str_args = map(str, self.args)
        return ' + '.join(str_args)

    def _build_bases(self, *args):
        """Build output bases."""
        dist = unify_attributes(args, 'dist')
        bases = []
        for coord in args[0].domain.bases_by_coord:
            ax_bases = tuple(arg.domain.bases_by_coord.get(coord, None) for arg in args)
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

    def reinitialize(self, **kw):
        arg0 = self.args[0].reinitialize(**kw)
        arg1 = self.args[1].reinitialize(**kw)
        return self.new_operands(arg0, arg1, **kw)

    def new_operands(self, arg0, arg1, **kw):
        return Add(arg0, arg1, **kw)

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

    def require_linearity(self, *args, **kw):
        """Require expression to be linear in specified variables."""
        for arg in self.args:
            arg.require_linearity(*args, **kw)

    def require_first_order(self, *args, **kw):
        """Require expression to be maximally first order in specified operators."""
        for arg in self.args:
            arg.require_first_order(*args, **kw)

    def matrix_dependence(self, *vars):
        return np.logical_or.reduce([arg.matrix_dependence(*vars) for arg in self.args])

    def matrix_coupling(self, *vars):
        return np.logical_or.reduce([arg.matrix_coupling(*vars) for arg in self.args])

    # def separability(self, *vars):
    #     """Determine separable dimensions of expression as a linear operator on specified variables."""
    #     # Logical and of argument separabilities
    #     arg_seps = [arg.separability(*vars) for arg in self.args]
    #     return np.logical_and.reduce(arg_seps)

    # def operator_order(self, operator):
    #     """Determine maximum application order of an operator in the expression."""
    #     # Take maximum order from arguments
    #     return max(arg.operator_order(operator) for arg in self.args)

    def build_ncc_matrices(self, separability, vars, **kw):
        """Precompute non-constant coefficients and build multiplication matrices."""
        # Build argument matrices
        for arg in self.args:
            arg.build_ncc_matrices(separability, vars, **kw)

    def expression_matrices(self, subproblem, vars, **kw):
        """Build expression matrices for a specific subproblem and variables."""
        # Intercept calls to compute matrices over expressions
        if self in vars:
            size = subproblem.field_size(self)
            matrix = sparse.identity(size, format='csr')
            return {self: matrix}
        matrices = {}
        # Iteratively add argument expression matrices
        for arg in self.args:
            arg_matrices = arg.expression_matrices(subproblem, vars, **kw)
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
            arg.change_layout(layout)

    def choose_layout(self):
        """Determine best target layout."""
        # OPTIMIZE: pick shortest absolute distance?
        # For now arbitrary pick first arg layout
        return self.args[0].layout

    def operate(self, out):
        """Perform operation."""
        arg0, arg1 = self.args
        # Set output layout
        out.preset_layout(arg0.layout)
        np.add(arg0.data, arg1.data, out=out.data)


# used for einsum string manipulation
alphabet = "abcdefghijklmnopqrstuvwxy"


class Product(Future):

    # Want to find some way to include this even though Product is not a MultiClass.  For now hacking into new_operand.
    # @classmethod
    # def _preprocess_args(cls, *args, **kw):
    #     if any(arg == 0 for arg in args):
    #         raise SkipDispatchException(output=0)
    #     else:
    #         return args, kw

    def _build_bases(self, arg0, arg1, ncc=False, ncc_vars=None, **kw):
        """Build output bases."""
        bases = []
        arg0_bases = arg0.domain.bases_by_coord
        arg1_bases = arg1.domain.bases_by_coord
        for coord in arg0_bases:
            b0 = arg0_bases[coord]
            b1 = arg1_bases[coord]
            # All constant bases yields constant basis
            if b0 is None and b1 is None:
                continue
            # Multiply all bases
            elif ncc and arg0.has(*ncc_vars):
                bases.append(b1 @ b0)
            elif ncc and arg1.has(*ncc_vars):
                bases.append(b0 @ b1)
            else:
                bases.append(b0 * b1)
        return tuple(bases)

    def reinitialize(self, **kw):
        arg0 = self.args[0].reinitialize(**kw)
        arg1 = self.args[1].reinitialize(**kw)
        return self.new_operands(arg0, arg1, **kw)

    def new_operands(self, *args, **kw):
        raise NotImplementedError("%s has not implemented new_operands method." %type(self))

    def split(self, *vars):
        """Split into expressions containing and not containing specified operands/operators."""
        # Take cartesian product of argument splittings
        split_args = [arg.split(*vars) if isinstance(arg, Operand) else (0, arg) for arg in self.args]
        split_args = list(itertools.product(*split_args))
        # Drop any terms with a zero since not all product instantiations check for it yet
        drop_last = not (split_args[-1][0] and split_args[-1][1])
        split_args_filt = [(arg0, arg1) for arg0, arg1 in split_args if (arg0 and arg1)]
        # Take product of each term
        split_ops = [self.new_operands(*args) for args in split_args_filt]
        # Append zero in case last combo was dropped
        if drop_last:
            split_ops.append(0)
        # Last combo is all negative splittings, others contain atleast one positive splitting
        return (sum(split_ops[:-1]), split_ops[-1])

    def sym_diff(self, var):
        """Symbolically differentiate with respect to specified operand."""
        args = self.args
        # Apply product rule to arguments
        partial_diff = lambda i: self.new_operands(*[arg.sym_diff(var) if i==j else arg for j,arg in enumerate(args)])
        return sum((partial_diff(i) for i in range(len(args))))

    def expand(self, *vars):
        """Expand expression over specified variables."""
        if self.has(*vars):
            # Expand arguments
            args = [arg.expand(*vars) for arg in self.args]
            # Sum over cartesian product of sums including specified variables
            arg_sets = [arg.args if (isinstance(arg, Add) and arg.has(*vars)) else [arg] for arg in args]
            arg_sets = itertools.product(*arg_sets)
            return sum(self.new_operand(*args) for args in arg_sets)
        else:
            return self

    def require_linearity(self, *vars, allow_affine=False, self_name=None, vars_name=None, error=AssertionError, recurse=True):
        """Require expression to be linear in specified variables."""
        arg0, arg1 = self.args
        op_arg0 = (isinstance(arg0, Operand) and arg0.has(*vars))
        op_arg1 = (isinstance(arg1, Operand) and arg1.has(*vars))
        if op_arg0 and op_arg1:
            if self_name is None:
                self_name = str(self)
            if vars_name is None:
                vars_name = [str(var) for var in vars]
            raise error(f"{self_name} is nonlinear in {vars_name}.")
        elif op_arg0 or op_arg1:
            op_index = int(op_arg1)
            if recurse:
                self.args[op_index].require_linearity(*vars, allow_affine=allow_affine, self_name=self_name, vars_name=vars_name, error=error)
            return op_index
        elif not allow_affine:
            if self_name is None:
                self_name = str(self)
            if vars_name is None:
                vars_name = [str(var) for var in vars]
            raise error(f"{self_name} must be strictly linear in {vars_name}.")

    def require_first_order(self, *args, **kw):
        """Require expression to be maximally first order in specified operators."""
        for arg in self.args:
            if isinstance(arg, Operand):
                arg.require_first_order(*args, **kw)

    def prep_nccs(self, vars):
        """Separate NCC factors."""
        self._ncc_vars = vars
        op_index = self.require_linearity(*vars, recurse=False)
        self.ncc_first = (op_index == 1)
        self.operand = self.args[op_index]
        self.ncc = self.args[1 - op_index]  # Assumes 2 operands
        # Recurse
        self.operand.prep_nccs(vars)

    def gather_ncc_coeffs(self):
        """Communicate NCC coeffs prior to matrix construction."""
        # Recurse
        self.operand.gather_ncc_coeffs()
        # Evaluate NCC
        ncc = self.ncc
        if isinstance(ncc, Future):
            ncc = ncc.evaluate()
        # Allgather NCC coefficients
        if isinstance(ncc, Field):
            ncc.require_coeff_space()
            self._ncc_data = ncc.allgather_data()
        else:
            self._ncc_data = ncc

    def store_ncc_matrices(self, vars, subproblems, **kw):
        self.prep_nccs(vars)
        self.gather_ncc_coeffs()
        self._ncc_matrices = {}
        if 'ncc_cutoff' not in kw:
            kw['ncc_cutoff'] = 1e-6
        if 'max_ncc_terms' not in kw:
            kw['max_ncc_terms'] = None
        for subproblem in subproblems:
            self._ncc_matrices[subproblem] = self.build_ncc_matrix(subproblem, **kw)

    def evaluate_as_ncc(self):
        op = self.operand.evaluate()
        out = self.get_out()
        out['c'] = 0
        for subproblem, ncc_matrix in self._ncc_matrices.items():
            for subsystem in subproblem.subsystems:
                op_ss = op['c'][subsystem.field_slices(op)]
                out_ss = out['c'][subsystem.field_slices(out)]
                out_ss[:] = (ncc_matrix @ op_ss.ravel()).reshape(out_ss.shape)
        return out

    def build_ncc_matrix(self, subproblem, ncc_cutoff, max_ncc_terms):
        # TODO: recurse over output bases
        # ASSUME: NCC is on last axis of last output basis
        if self.dist.single_coordsys and not self.dist.single_coordsys.curvilinear:
            return self.build_cartesian_ncc_matrix(subproblem, ncc_cutoff, max_ncc_terms)
        elif len(self.domain.bases) > 0:
            out_basis = self.domain.bases[-1]
            return out_basis.build_ncc_matrix(self, subproblem, ncc_cutoff, max_ncc_terms)
        else:
            return Basis._last_axis_field_ncc_matrix(self, subproblem, 0, None, None, None, self._ncc_data, ncc_cutoff, max_ncc_terms)

    def build_cartesian_ncc_matrix(self, subproblem, ncc_cutoff, max_ncc_terms):
        # Assumes no intertwiners
        # Assumes basis functions are independent of tensor component
        ncc = self.ncc
        arg = self.operand
        out = self
        # Build Gamma array
        if self.ncc_first:
            Gamma = self.GammaCoord(ncc.tensorsig, arg.tensorsig, out.tensorsig)
            Gamma = Gamma.transpose((2, 1, 0))
        else:
            Gamma = self.GammaCoord(arg.tensorsig, ncc.tensorsig, out.tensorsig)
            Gamma = Gamma.transpose((2, 0, 1))
        # Loop over NCC modes
        shape = (subproblem.field_size(out), subproblem.field_size(arg))
        matrix = sparse.csr_matrix(shape, dtype=self.dtype)
        subproblem_shape = subproblem.coeff_shape(out.domain)
        ncc_rank = len(ncc.tensorsig)
        select_all_comps = tuple(slice(None) for i in range(ncc_rank))
        if np.any(self._ncc_data):
            for ncc_mode in np.ndindex(self._ncc_data.shape[ncc_rank:]):
                ncc_coeffs = self._ncc_data[select_all_comps + ncc_mode]
                if np.max(np.abs(ncc_coeffs)) > ncc_cutoff:
                    mode_matrix = self.cartesian_mode_matrix(subproblem_shape, ncc.domain, arg.domain, out.domain, ncc_mode)
                    mode_matrix = sparse.kron(np.dot(Gamma, ncc_coeffs.ravel()), mode_matrix, format='csr')
                    matrix = matrix + mode_matrix
        return matrix

    @classmethod
    def cartesian_mode_matrix(cls, subproblem_shape, ncc_domain, arg_domain, out_domain, ncc_mode):
        for axis in range(out_domain.dist.dim):
            ncc_basis = ncc_domain.full_bases[axis]
            arg_basis = arg_domain.full_bases[axis]
            out_basis = out_domain.full_bases[axis]
            if ncc_basis is None:
                mode_matrix = sparse.identity(subproblem_shape[axis], format='csr')
            else:
                mode_matrix = ncc_basis.product_matrix(arg_basis, out_basis, ncc_mode[axis])
            if axis == 0:
                matrix = mode_matrix
            else:
                matrix = sparse.kron(matrix, mode_matrix, format='csr')
        return matrix

    # def _ncc_matrix_recursion(self, subproblem, ncc_bases, arg_bases, coeffs, ncc_comp, arg_comp, out_comp, **kw):
    #     #, ncc_ts, ncc_bases, arg_bases, arg_ts, separability, gamma_args, **kw):
    #     """Build NCC matrix by recursing down through the axes."""
    #     # Build function for deferred-computation of matrix-valued coefficients
    #     def build_lower_coeffs(index):
    #         # Return scalar-valued coefficients at bottom level
    #         if coeffs.ndim == 1:
    #             return coeffs[index]
    #         # Otherwise recursively build matrix-valued coefficients
    #         else:
    #             args = (subproblem, ncc_bases[1:], arg_bases[1:], coeffs[index], ncc_comp, arg_comp, out_comp)
    #             return self._ncc_matrix_recursion(*args, **kw)
    #     # Build top-level matrix using deferred coeffs
    #     coeffs = DeferredTuple(build_lower_coeffs, size=data.shape[0])
    #     ncc_basis = ncc_bases[0]
    #     arg_basis = arg_bases[0]
    #     # Kronecker with identities for constant NCC bases
    #     if ncc_basis is None:
    #         const = coeffs[0]
    #         # Trivial Kronecker with [[1]] for constant arg bases
    #         # This generalization enables problem-agnostic pre-construction
    #         if arg_basis is None:
    #             return const
    #         # Group-size identity for separable dimensions
    #         if separability[0]:
    #             I = sparse.identity(arg_basis.space.group_size)
    #         # Coeff-size identity for non-separable dimensions
    #         else:
    #             I = sparse.identity(arg_basis.space.coeff_size)
    #         # Apply cutoff to scalar coeffs
    #         if len(const.shape) == 0:
    #             cutoff = kw.get('cutoff', 1e-6)
    #             if abs(const) > cutoff:
    #                 return I * const
    #             else:
    #                 return I * 0
    #         else:
    #             return sparse.kron(I, const)
    #     # Call basis method for constructing NCC matrix
    #     else:
    #         return getattr(ncc_basis, self.ncc_method)(arg_basis, coeffs, ncc_ts, arg_ts, *gamma_args, **kw)

    def expression_matrices(self, subproblem, vars, **kw):
        """Build expression matrices for a specific subproblem and variables."""
        # Intercept calls to compute matrices over expressions
        if self in vars:
            size = subproblem.field_size(self)
            matrix = sparse.identity(size, format='csr')
            return {self: matrix}
        # Check vars vs. NCC prep
        if vars != self._ncc_vars:
            raise SymbolicParsingError("Must build NCC matrices with same variables.")
        # Apply NCC matrix to operand matrices
        operand_mats = self.operand.expression_matrices(subproblem, vars, **kw)
        ncc_mat = self.build_ncc_matrix(subproblem, **kw)
        return {var: ncc_mat @ operand_mats[var] for var in operand_mats}
        # # Modify NCC matrix for subproblem
        # # Build projection matrix dropping constant-groups as necessary
        # group_shape = subproblem.coeff_shape(self.subdomain)
        # const_shape = np.maximum(group_shape, 1)
        # factors = (sparse.eye(*shape, format='csr') for shape in zip(group_shape, const_shape))
        # projection = reduce(sparse.kron, factors, 1).tocsr()
        # ncc_mat = projection @ self._ncc_matrix

        # # Add factor for components
        # comps = prod([cs.dim for cs in operand.tensorsig])
        # blocks = []
        # for ncc_comp_mat in self._ncc_matrices:
        #     factors = [sparse.identity(comps, format='csr'), ncc_comp_mat]
        #     blocks.append(reduce(sparse.kron, factors, 1).tocsr())
        # ncc_mat = sparse.vstack(blocks, format='csr')

    def matrix_dependence(self, *vars):
        operand = self.operand
        operand_dependence = operand.matrix_dependence(*vars)
        ncc_matrix_dependence = operand.domain.mode_dependence
        return ncc_matrix_dependence | operand_dependence

    def matrix_coupling(self, *vars):
        operand = self.operand
        operand_coupling = operand.matrix_coupling(*vars)
        ncc = self.ncc
        ncc_coupling = ncc.domain.nonconstant
        return ncc_coupling | operand_coupling

    def check_conditions(self):
        layout0 = self.args[0].layout
        layout1 = self.args[1].layout
        # Fields must be in grid layout
        # Just do full grid space for now
        return all(layout0.grid_space) and (layout0 is layout1)

    def enforce_conditions(self):
        # Fields must be in grid layout
        # Just do full grid space for now
        for arg in self.args:
            arg.require_grid_space()

    def Gamma(self, A_tensorsig, B_tensorsig, C_tensorsig, A_group, B_group, C_group, axis):
        """
        Gamma(a,b,c) in components after intertwiners for specified axis.
        Requires mode groups of previous axes, i.e. len(group) = axis
        """
        # Base case
        if axis == 0:
            return self.GammaCoord(A_tensorsig, B_tensorsig, C_tensorsig)
        # Recurse
        if axis > 0:
            G = self.Gamma(A_tensorsig, B_tensorsig, C_tensorsig, A_group, B_group, C_group, axis-1)
        # Apply Q
        cs = self.dist.get_coordsystem(axis)
        cs_axis = self.dist.get_axis(cs)
        subaxis = axis - cs_axis
        QA = cs.backward_intertwiner(subaxis, len(A_tensorsig), A_group[cs_axis:]).T
        QB = cs.backward_intertwiner(subaxis, len(B_tensorsig), B_group[cs_axis:]).T
        QC = cs.forward_intertwiner(subaxis, len(C_tensorsig), C_group[cs_axis:])
        Q = kron(QA, QB, QC)
        G = (Q @ G.ravel()).reshape(G.shape)
        return G

    def GammaCoord(self, A_tensorsig, B_tensorsig, C_tensorsig):
        raise NotImplementedError("%s has not implemented GammaCoord" %type(self))


@alias("dot")
class DotProduct(Product, FutureField):

    name = "Dot"
    ncc_method = "dot_product_ncc"

    def __init__(self, arg0, arg1, indices=(-1,0), out=None, **kw):
        indices = self._check_indices(arg0, arg1, indices)
        super().__init__(arg0, arg1, out=out)
        arg0_ts_reduced = list(arg0.tensorsig)
        arg0_ts_reduced.pop(indices[0])
        arg1_ts_reduced = list(arg1.tensorsig)
        arg1_ts_reduced.pop(indices[1])
        self.indices = indices
        self.gamma_args = [indices]
        # FutureField requirements
        dist = unify_attributes((arg0, arg1), 'dist')
        self.domain = Domain(dist, self._build_bases(arg0, arg1, **kw))
        self.tensorsig = tuple(arg0_ts_reduced + arg1_ts_reduced)
        self.dtype = np.result_type(arg0.dtype, arg1.dtype)
        # Setup ghost broadcasting
        broadcast_dims = np.array(self.domain.nonconstant)
        self.arg0_ghost_broadcaster = GhostBroadcaster(arg0.domain, self.dist.grid_layout, broadcast_dims)
        self.arg1_ghost_broadcaster = GhostBroadcaster(arg1.domain, self.dist.grid_layout, broadcast_dims)
        # Compose eigsum string
        rank0 = len(arg0.tensorsig)
        rank1 = len(arg1.tensorsig)
        arg1_str = alphabet[:rank0]
        arg2_str = alphabet[rank0:rank0+rank1]
        arg1_str = arg1_str.replace(arg1_str[indices[0]], 'z')
        arg2_str = arg2_str.replace(arg2_str[indices[1]], 'z')
        out_str = (arg1_str + arg2_str).replace('z', '')
        self.einsum_str = arg1_str + '...,' + arg2_str + '...->' + out_str + '...'

    def _check_indices(self, arg0, arg1, indices):
        if (not isinstance(arg0, Operand)) or (not isinstance(arg1, Operand)):
            raise ValueError("Both arguments to DotProduct must be Operand")
        arg0_rank = len(arg0.tensorsig)
        arg1_rank = len(arg1.tensorsig)
        indices = list(indices)
        for i,(index,rank) in enumerate(zip(indices,(arg0_rank,arg1_rank))):
            if index > rank or index < -rank:
                raise ValueError("index %i out of range for field with rank %i" %(index,rank))
            if index < 0:
                index += rank
            indices[i] = index
        return tuple(indices)

    def __str__(self):
        # Parenthesize addition arguments
        def paren_str(arg):
            if isinstance(arg, Add):
                return '({})'.format(arg)
            else:
                return str(arg)
        str_args = map(paren_str, self.args)
        return '@'.join(str_args)

    def new_operands(self, arg0, arg1, **kw):
        if arg0 == 0 or arg1 == 0:
            return 0
        return DotProduct(arg0, arg1, indices=self.indices, **kw)

    def GammaCoord(self, A_tensorsig, B_tensorsig, C_tensorsig):
        A_dim = prod([cs.dim for cs in A_tensorsig])
        B_dim = prod([cs.dim for cs in B_tensorsig])
        C_dim = prod([cs.dim for cs in C_tensorsig])
        G = np.zeros((A_dim, B_dim, C_dim), dtype=int)
        for ia, a in enum_indices(A_tensorsig):
            a_other = list(a)
            a_dot = a_other.pop(self.indices[0])
            for ib, b in enum_indices(B_tensorsig):
                b_other = list(b)
                b_dot = b_other.pop(self.indices[1])
                if a_dot == b_dot:
                    for ic, c, in enum_indices(C_tensorsig):
                        if tuple(a_other + b_other) == c:
                            G[ia, ib, ic] = 1
        return G

    def operate(self, out):
        arg0, arg1 = self.args
        out.preset_layout(arg0.layout)
        # Broadcast
        arg0_data = self.arg0_ghost_broadcaster.cast(arg0)
        arg1_data = self.arg1_ghost_broadcaster.cast(arg1)
        # Call einsum
        if out.data.size:
            np.einsum(self.einsum_str, arg0_data, arg1_data, out=out.data, optimize=True)


@alias("cross")
class CrossProduct(Product, FutureField):
    """Cross product on two 3D vector fields."""

    name = "Cross"

    def __init__(self, arg0, arg1, out=None, **kw):
        super().__init__(arg0, arg1, out=out)
        # Check that both fields are rank-1
        if len(arg0.tensorsig) != 1 or len(arg1.tensorsig) != 1:
            raise NotImplementedError("CrossProduct currently only implemented for vector fields.")
        # Check that vector bundles are the same
        if arg0.tensorsig[0] is not arg1.tensorsig[0]:
            raise ValueError("CrossProduct requires identical vector bundles.")
        # Check that vector bundles are 3D
        if arg0.tensorsig[0].dim != 3:
            raise ValueError("CrossProduct requires 3-component vector fields.")
        # FutureField requirements
        self.domain = Domain(arg0.dist, self._build_bases(arg0, arg1, **kw))
        self.tensorsig = arg0.tensorsig
        self.dtype = np.result_type(arg0.dtype, arg1.dtype)
        # Setup ghost broadcasting
        broadcast_dims = np.array(self.domain.nonconstant)
        self.arg0_ghost_broadcaster = GhostBroadcaster(arg0.domain, self.dist.grid_layout, broadcast_dims)
        self.arg1_ghost_broadcaster = GhostBroadcaster(arg1.domain, self.dist.grid_layout, broadcast_dims)
        # Pick operate method based on coordsys handedness
        if self.tensorsig[0].right_handed:
            self.operate = self.operate_right_handed
        else:
            self.operate = self.operate_left_handed

    def operate_right_handed(self, out):
        arg0, arg1 = self.args
        out.preset_layout(arg0.layout)
        arg0_data = self.arg0_ghost_broadcaster.cast(arg0)
        arg1_data = self.arg1_ghost_broadcaster.cast(arg1)
        data00, data01, data02 = arg0_data[0], arg0_data[1], arg0_data[2]
        data10, data11, data12 = arg1_data[0], arg1_data[1], arg1_data[2]
        ne.evaluate("data01*data12 - data02*data11", out=out.data[0])
        ne.evaluate("data02*data10 - data00*data12", out=out.data[1])
        ne.evaluate("data00*data11 - data01*data10", out=out.data[2])

    def operate_left_handed(self, out):
        arg0, arg1 = self.args
        out.preset_layout(arg0.layout)
        arg0_data = self.arg0_ghost_broadcaster.cast(arg0)
        arg1_data = self.arg1_ghost_broadcaster.cast(arg1)
        data00, data01, data02 = arg0_data[0], arg0_data[1], arg0_data[2]
        data10, data11, data12 = arg1_data[0], arg1_data[1], arg1_data[2]
        ne.evaluate("data02*data11 - data01*data12", out=out.data[0])
        ne.evaluate("data00*data12 - data02*data10", out=out.data[1])
        ne.evaluate("data01*data10 - data00*data11", out=out.data[2])

    def new_operands(self, arg0, arg1, **kw):
        if arg0 == 0 or arg1 == 0:
            return 0
        return CrossProduct(arg0, arg1, **kw)

    def GammaCoord(self, A_tensorsig, B_tensorsig, C_tensorsig):
        cs = A_tensorsig[0]
        G = np.zeros((3, 3, 3), dtype=int)
        G[0,1,2] = G[1,2,0] = G[2,0,1] = 1
        G[0,2,1] = G[2,1,0] = G[1,0,2] = -1
        if not cs.right_handed:
            G *= -1
        return G


class Multiply(Product, metaclass=MultiClass):
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
        # elif any(isinstance(arg, (Array, FutureArray)) for arg in args):
        #     raise NotImplementedError()
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

    def new_operands(self, arg0, arg1, **kw):
        return Multiply(arg0, arg1, **kw)

    def GammaCoord(self, A_tensorsig, B_tensorsig, C_tensorsig):
        A_dim = prod([cs.dim for cs in A_tensorsig])
        B_dim = prod([cs.dim for cs in B_tensorsig])
        C_dim = prod([cs.dim for cs in C_tensorsig])
        G = np.zeros((A_dim, B_dim, C_dim), dtype=int)
        for ia, a in enum_indices(A_tensorsig):
            for ib, b in enum_indices(B_tensorsig):
                for ic, c, in enum_indices(C_tensorsig):
                    if a + b == c:
                        G[ia, ib, ic] = 1
        return G

    # def simplify(self, *vars):
    #     """Simplify expression, except subtrees containing specified variables."""
    #     # Simplify arguments if variables are present
    #     if self.has(*vars):
    #         args = [arg.simplify(*vars) if isinstance(arg, Operand) else arg for arg in self.args]
    #         return self.base(*args)
    #     # Otherwise evaluate expression
    #     else:
    #         return self.evaluate()

    # def operator_order(self, operator):
    #     """Determine maximum application order of an operator in the expression."""
    #     # Take maximum order from arguments
    #     return max(arg.operator_order(operator) for arg in self.args)


class MultiplyFields(Multiply, FutureField):
    """Multiplication operator for fields."""

    argtypes = (Field, FutureField)
    ncc_method = "tensor_product_ncc"

    def __init__(self, arg0, arg1, out=None, **kw):
        super().__init__(arg0, arg1, out=out)
        # # Find required grid axes
        # # Require grid space if more than one argument has nonconstant basis
        # ax_bases = tuple(zip(*(arg.domain.full_bases for arg in self.args)))
        # nonconst_ax_bases = [[b for b in bases if b is not None] for bases in ax_bases]
        # self.required_grid_axes = [len(bases) > 1 for bases in nonconst_ax_bases]
        self.dist = unify_attributes((arg0, arg1), 'dist')
        self.domain = Domain(self.dist, self._build_bases(arg0, arg1, **kw))
        self.tensorsig = arg0.tensorsig + arg1.tensorsig
        self.dtype = np.result_type(arg0.dtype, arg1.dtype)
        self.gamma_args = []
        # Setup ghost broadcasting
        broadcast_dims = np.array(self.domain.nonconstant)
        self.arg0_ghost_broadcaster = GhostBroadcaster(arg0.domain, self.dist.grid_layout, broadcast_dims)
        self.arg1_ghost_broadcaster = GhostBroadcaster(arg1.domain, self.dist.grid_layout, broadcast_dims)
        # Compute expanded shapes for broadcasting data
        arg0_gshape = self.dist.grid_layout.local_shape(arg0.domain, arg0.domain.dealias)
        arg0_tshape = tuple(cs.dim for cs in arg0.tensorsig)
        arg0_order = len(arg0.tensorsig)
        arg1_gshape = self.dist.grid_layout.local_shape(arg1.domain, arg1.domain.dealias)
        arg1_tshape = tuple(cs.dim for cs in arg1.tensorsig)
        arg1_order = len(arg1.tensorsig)
        self.arg0_exp_tshape = arg0_tshape + (1,) * arg1_order
        self.arg1_exp_tshape = (1,) * arg0_order + arg1_tshape

    def operate(self, out):
        """Perform operation."""
        arg0, arg1 = self.args
        # Set output layout
        out.preset_layout(arg0.layout)
        # Broadcast
        arg0_data = self.arg0_ghost_broadcaster.cast(arg0)
        arg1_data = self.arg1_ghost_broadcaster.cast(arg1)
        # Reshape arg data to broadcast properly for output tensorsig
        arg0_exp_data = arg0_data.reshape(self.arg0_exp_tshape + arg0_data.shape[len(arg0.tensorsig):])
        arg1_exp_data = arg1_data.reshape(self.arg1_exp_tshape + arg1_data.shape[len(arg1.tensorsig):])
        np.multiply(arg0_exp_data, arg1_exp_data, out=out.data)


class GhostBroadcaster:
    """Copy field data over constant distributed dimensions for arithmetic broadcasting."""

    def __init__(self, domain, layout, broadcast_dims):
        self.domain = domain
        self.layout = layout
        self.broadcast_dims = broadcast_dims
        # Determine deployment dimensions
        deploy_dims_ext = np.array(broadcast_dims) & np.array(domain.constant)
        deploy_dims = deploy_dims_ext[~layout.local]
        # Build subcomm or skip casting
        if any(deploy_dims):
            self.subcomm = domain.dist.comm_cart.Sub(remain_dims=deploy_dims.astype(int))
        else:
            self.cast = self._skip_cast

    @CachedMethod
    def ghost_data(self, shape, dtype):
        # Broadcast root shape
        shape = self.subcomm.bcast(shape, root=0)
        # Make ghost buffers
        if self.subcomm.rank == 0:
            return None
        else:
            return np.empty(dtype=dtype, shape=shape)

    def cast(self, field):
        # Retrieve ghost data on all ranks to prevent deadlocks
        ghost_data = self.ghost_data(field.data.shape, field.dtype)
        if self.subcomm.rank == 0:
            ghost_data = field.data
        # Skip broadcasting on empty subcomms
        if ghost_data.size:
            self.subcomm.Bcast(ghost_data, root=0)
        return ghost_data

    def _skip_cast(self, field):
        return field.data


class MultiplyNumberField(Multiply, FutureField):

    argtypes = ((numbers.Number, (Field, FutureField)),
                ((Field, FutureField), numbers.Number))

    def __init__(self, arg0, arg1, out=None,**kw):
        # Make number come first
        if isinstance(arg1, numbers.Number):
            arg0, arg1 = arg1, arg0
        # Initialization
        super().__init__(arg0, arg1, out=out)
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
        out.preset_layout(arg1.layout)
        # Multiply argument data
        np.multiply(arg0, arg1.data, out=out.data)

    def matrix_dependence(self, *vars):
        return self.args[1].matrix_dependence(*vars)

    def matrix_coupling(self, *vars):
        return self.args[1].matrix_coupling(*vars)

    def expression_matrices(self, subproblem, vars, **kw):
        """Build expression matrices for a specific subproblem and variables."""
        # Intercept calls to compute matrices over expressions
        if self in vars:
            size = subproblem.field_size(self)
            matrix = sparse.identity(size, format='csr')
            return {self: matrix}
        arg0, arg1 = self.args
        # Build field matrices
        arg1_mats = arg1.expression_matrices(subproblem, vars, **kw)
        # Multiply field matrices
        return {var: arg0 * arg1_mats[var] for var in arg1_mats}

    def build_ncc_matrices(self, separability, vars, **kw):
        """Precompute non-constant coefficients and build multiplication matrices."""
        nccs, operand = self.require_linearity(*vars)
        # Continue NCC matrix construction
        operand.build_ncc_matrices(separability, vars, **kw)

    def reinitialize(self, **kw):
        arg0 = self.args[0]
        arg1 = self.args[1].reinitialize(**kw)
        return self.new_operands(arg0, arg1, **kw)

    def sym_diff(self, var):
        """Symbolically differentiate with respect to specified operand."""
        return self.args[0] * self.args[1].sym_diff(var)


# Define aliases
for key, value in aliases.items():
    setattr(sys.modules[__name__], key, value)

# Export aliases
__all__.extend(aliases.keys())

