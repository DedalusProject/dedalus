"""
Arithmetic operators.

"""

from functools import reduce
import numpy as np
from scipy import sparse
import itertools
import operator
import numbers
from collections import defaultdict

from .domain import Domain
from .field import Operand, Array, Field
from .future import Future, FutureArray, FutureField
from .operators import convert
from ..tools.array import kron
from ..tools.cache import CachedAttribute, CachedMethod
from ..tools.dispatch import MultiClass
from ..tools.exceptions import NonlinearOperatorError
from ..tools.exceptions import SymbolicParsingError
from ..tools.exceptions import SkipDispatchException
from ..tools.general import unify_attributes, DeferredTuple


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

    def require_linearity(self, *vars, name=None):
        """Require expression to be linear in specified variables."""
        # Require arguments to be linear
        for arg in self.args:
            arg.require_linearity(*vars, name=name)

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

    def expression_matrices(self, subproblem, vars):
        """Build expression matrices for a specific subproblem and variables."""
        # Intercept calls to compute matrices over expressions
        if self in vars:
            return Field.expression_matrices(self, subproblem, vars)
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


# used for einsum string manipulation
alphabet = "abcdefghijklmnopqrstuvwxy"


class Product(Future):

    def _build_bases(self, arg0, arg1, ncc=False, ncc_vars=None, **kw):
        """Build output bases."""
        bases = []
        for b0, b1 in zip(arg0.domain.full_bases, arg1.domain.full_bases):
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
        bases = {basis.coordsystem: basis for basis in bases}
        return tuple(bases.values())

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
        split_args = itertools.product(*split_args)
        # Take product of each term
        split_ops = [self.new_operands(*args) for args in split_args]
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

    def require_linearity(self, *vars, name=None, recurse=True):
        """Require expression to be linear in specified variables."""
        arg0, arg1 = self.args
        op_arg0 = (isinstance(arg0, Operand) and arg0.has(*vars))
        op_arg1 = (isinstance(arg1, Operand) and arg1.has(*vars))
        # Require exactly one argument to contain vars, for linearity
        if op_arg0 and op_arg1:
            raise NonlinearOperatorError("{} is a non-linear product of the specified variables.".format(name if name else str(self)))
        if not (op_arg0 or op_arg1):
            raise NonlinearOperatorError("{} does not involve the specified variables.".format(name if name else str(self)))
        op_index = int(op_arg1)
        if recurse:
            # Require argument linearity
            self.args[op_index].require_linearity(*vars, name=name)
        return op_index

    def prep_nccs(self, vars):
        """Communicate NCC coeffs prior to matrix construction."""
        self._ncc_vars = vars
        op_index = self.require_linearity(*vars, recurse=False)
        # Prep operand
        self.operand = operand = self.args[op_index]
        operand.prep_nccs(vars)
        # Evaluate NCC
        self.ncc = ncc = self.args[1 - op_index]
        if isinstance(self.ncc, Future):
            ncc = ncc.evaluate()
        # Allgather NCC coefficients
        if isinstance(ncc, Field):
            ncc.require_coeff_space()
            self._ncc_data = ncc.allgather_data()
        else:
            self._ncc_data = ncc

    def store_ncc_matrices(self, subproblems, **kw):
        self._ncc_matrices = {}
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

    def build_ncc_matrix(self, subproblem, **kw):
        """Precompute non-constant coefficients and build multiplication matrices."""
        operand = self.operand
        ncc = self.ncc
        ncc_data = self._ncc_data
        separability = ~ subproblem.problem.matrix_coupling
        #return = self._ncc_matrix_recursion(ncc_data, ncc.tensorsig, ncc.bases, operand.bases, operand.tensorsig, separability, self.gamma_args, **kw)
        ncc_basis = ncc.domain.bases[-1]
        arg_basis = operand.domain.bases[-1]
        ncc_ts = ncc.tensorsig
        ncc_ts_shape = [cs.dim for cs in ncc_ts]
        coeffs = ncc_data.reshape(ncc_ts_shape + [-1])
        arg_ts = operand.tensorsig
        out_ts = self.tensorsig
        gamma_args = self.gamma_args

        # ASSUME NCC IS ALONG LAST AXIS
        axis = self.dist.dim - 1
        group = subproblem.group
        ncc_first = (ncc is self.args[0])
        ncc_group = tuple(0*g if g is not None else None for g in group)
        if ncc_first:
            Gamma = self.Gamma(ncc.tensorsig, operand.tensorsig, self.tensorsig, ncc_group, group, group, axis)
        else:
            Gamma = self.Gamma(operand.tensorsig, ncc.tensorsig, self.tensorsig, group, ncc_group, group, axis)
            Gamma = Gamma.transpose((1,0,2))

        # Loop over input and output components to build matrix blocks
        N = subproblem.coeff_size(operand.domain)
        blocks = []
        for ic, out_comp in enum_indices(self.tensorsig):
            block_row = []
            for ib, arg_comp in enum_indices(operand.tensorsig):
                block = sparse.csr_matrix((N, N))
                # Loop over ncc components
                for ia, ncc_comp in enum_indices(ncc.tensorsig):
                    G = Gamma[ia, ib, ic]
                    if abs(G) > 1e-10:
                        block += G * ncc_basis.multiplication_matrix(subproblem, arg_basis, coeffs[ncc_comp], ncc_comp, arg_comp, out_comp, cutoff=1e-6)
                block_row.append(block)
            blocks.append(block_row)
        return sparse.bmat(blocks, format='csr')
        #return getattr(ncc_basis, self.ncc_method)(arg_basis, coeffs, ncc_ts, arg_ts, out_ts, subproblem, ncc_first, *gamma_args, cutoff=1e-6)
        # tshape = [cs.dim for cs in ncc.tensorsig]
        # self._ncc_matrices = [self._ncc_matrix_recursion(ncc.data[ind], ncc.domain.full_bases, operand.domain.full_bases, separability, **kw) for ind in np.ndindex(*tshape)]

    def _ncc_matrix_recursion(self, data, ncc_ts, ncc_bases, arg_bases, arg_ts, separability, gamma_args, **kw):
        """Build NCC matrix by recursing down through the axes."""
        # Build function for deferred-computation of matrix-valued coefficients
        def build_lower_coeffs(i):
            # Return scalar-valued coefficients at bottom level
            if len(data.shape) - len(ncc_ts) == 1:
                return data[i]
            # Otherwise recursively build matrix-valued coefficients
            else:
                args = (data[i], ncc_ts, ncc_bases[1:], arg_bases[1:], arg_ts, separability[1:], gamma_args)
                return self._ncc_matrix_recursion(*args, **kw)
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
            return getattr(ncc_basis, self.ncc_method)(arg_basis, coeffs, ncc_ts, arg_ts, *gamma_args, **kw)

    def expression_matrices(self, subproblem, vars, **kw):
        """Build expression matrices for a specific subproblem and variables."""
        # Intercept calls to compute matrices over expressions
        if self in vars:
            return Field.expression_matrices(self, subproblem, vars)
        # Check vars vs. NCC prep
        if vars != self._ncc_vars:
            raise SymbolicParsingError("Must build NCC matrices with same variables.")
        # Apply NCC matrix to operand matrices
        operand_mats = self.operand.expression_matrices(subproblem, vars)
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
        # comps = np.prod([cs.dim for cs in operand.tensorsig], dtype=int)
        # blocks = []
        # for ncc_comp_mat in self._ncc_matrices:
        #     factors = [sparse.identity(comps, format='csr'), ncc_comp_mat]
        #     blocks.append(reduce(sparse.kron, factors, 1).tocsr())
        # ncc_mat = sparse.vstack(blocks, format='csr')

    def matrix_dependence(self, *vars):
        coupling = self.matrix_coupling(*vars)
        coupling[1] = True  # HACK HACK HACK for spheres coupling ell
        return coupling

    def matrix_coupling(self, *vars):
        self.prep_nccs(vars)  # HACK: called too much?
        operand = self.operand
        operand_coupling = operand.matrix_coupling(*vars)
        ncc = self.ncc
        ncc_coupling = np.array([basis is not None for basis in ncc.domain.full_bases])
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
        Requires wavenumbers of previous axes, i.e. len(group) = axis
        """
        # Base case
        if axis == 0:
            return self.GammaCoord(A_tensorsig, B_tensorsig, C_tensorsig)
        # Recurse
        if axis > 0:
            G = self.Gamma(A_tensorsig, B_tensorsig, C_tensorsig, A_group, B_group, C_group, axis-1)
        # Apply Q
        cs = self.dist.get_coordsystem(axis)
        QA = cs.backward_intertwiner(axis, len(A_tensorsig), A_group).T
        QB = cs.backward_intertwiner(axis, len(B_tensorsig), B_group).T
        QC = cs.forward_intertwiner(axis, len(C_tensorsig), C_group)
        Q = kron(QA, QB, QC)
        G = (Q @ G.ravel()).reshape(G.shape)
        return G

    def GammaCoord(self, A_tensorsig, B_tensorsig, C_tensorsig):
        raise NotImplementedError("%s has not implemented GammaCoord" %type(self))


class DotProduct(Product, FutureField):

    name = "Dot"
    ncc_method = "dot_product_ncc"

    def __init__(self, arg0, arg1, indices=(-1,0), out=None, **kw):
        indices = self._check_indices(arg0, arg1, indices)
        super().__init__(arg0, arg1, out=out)
        self.arg0_rank = len(arg0.tensorsig)
        self.arg1_rank = len(arg1.tensorsig)
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
        return DotProduct(arg0, arg1, indices=self.indices, **kw)

    def GammaCoord(self, A_tensorsig, B_tensorsig, C_tensorsig):
        A_dim = int(np.prod([cs.dim for cs in A_tensorsig]))
        B_dim = int(np.prod([cs.dim for cs in B_tensorsig]))
        C_dim = int(np.prod([cs.dim for cs in C_tensorsig]))
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

        out.set_layout(arg0.layout)

        # compose eigsum string
        array0_str = alphabet[:self.arg0_rank]
        char0 = array0_str[self.indices[0]]
        array0_str = array0_str.replace(char0,'z')

        array1_str = alphabet[-self.arg1_rank:]
        char1 = array1_str[self.indices[1]]
        array1_str = array1_str.replace(char1,'z')

        array0_str_reduced = array0_str.replace('z','')
        array1_str_reduced = array1_str.replace('z','')
        out_str = array0_str_reduced + array1_str_reduced

        einsum_str = array0_str + '...,' + array1_str + '...->' + out_str + '...'

        np.einsum(einsum_str,arg0.data,arg1.data,out=out.data)


class CrossProduct(Product, FutureField):

    name = "Cross"

    # Should make sure arg0 and arg1 are rank 1
    # and that the cs are the same for arg0 and arg1

    def __init__(self, arg0, arg1, out=None, **kw):
        super().__init__(arg0, arg1, out=out)
        # FutureField requirements
        self.domain = Domain(arg0.dist, self._build_bases(arg0, arg1))
        self.tensorsig = arg0.tensorsig
        self.dtype = np.result_type(arg0.dtype, arg1.dtype)

    @CachedMethod
    def epsilon(self, i, j, k):
        coordsys = self.tensorsig[0]
        return coordsys.epsilon(i, j, k)

    def operate(self, out):
        arg0, arg1 = self.args
        out.set_layout(arg0.layout)
        out.data[0] = self.epsilon(0,1,2)*(arg0.data[1]*arg1.data[2] - arg0.data[2]*arg1.data[1])
        out.data[1] = self.epsilon(1,2,0)*(arg0.data[2]*arg1.data[0] - arg0.data[0]*arg1.data[2])
        out.data[2] = self.epsilon(2,0,1)*(arg0.data[0]*arg1.data[1] - arg0.data[1]*arg1.data[0])


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

    def new_operands(self, arg0, arg1, **kw):
        return Multiply(arg0, arg1, **kw)

    def GammaCoord(self, A_tensorsig, B_tensorsig, C_tensorsig):
        A_dim = int(np.prod([cs.dim for cs in A_tensorsig]))
        B_dim = int(np.prod([cs.dim for cs in B_tensorsig]))
        C_dim = int(np.prod([cs.dim for cs in C_tensorsig]))
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
        dist = unify_attributes((arg0, arg1), 'dist')
        self.domain = Domain(dist, self._build_bases(arg0, arg1, **kw))
        self.tensorsig = arg0.tensorsig + arg1.tensorsig
        self.dtype = np.result_type(arg0.dtype, arg1.dtype)
        self.gamma_args = []

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

    def __init__(self, arg0, arg1, out=None,**kw):
        # Make number come first
        if isinstance(arg1, numbers.Number):
            arg0, arg1 = arg1, arg0
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
        out.set_layout(arg1.layout)
        # Multiply all argument data, reshaped by tensorsig
        np.multiply(arg0, arg1.data, out=out.data)

    def matrix_dependence(self, *vars):
        return self.args[1].matrix_dependence(*vars)

    def matrix_coupling(self, *vars):
        return self.args[1].matrix_coupling(*vars)

    def expression_matrices(self, subproblem, vars):
        """Build expression matrices for a specific subproblem and variables."""
        # Intercept calls to compute matrices over expressions
        if self in vars:
            return Field.expression_matrices(self, subproblem, vars)
        arg0, arg1 = self.args
        # Build field matrices
        arg1_mats = arg1.expression_matrices(subproblem, vars)
        # Multiply field matrices
        return {var: arg0 * arg1_mats[var] for var in arg1_mats}

    def build_ncc_matrices(self, separability, vars, **kw):
        """Precompute non-constant coefficients and build multiplication matrices."""
        nccs, operand = self.require_linearity(*vars)
        # Continue NCC matrix construction
        operand.build_ncc_matrices(separability, vars, **kw)
