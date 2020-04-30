"""
Classes for manipulating pencils.

"""

from functools import partial, reduce
from itertools import product
from collections import defaultdict
import numpy as np
from scipy import sparse
from mpi4py import MPI
import uuid

from .domain import Domain
from ..tools.array import zeros_with_pattern, expand_pattern, sparse_block_diag
from ..tools.cache import CachedAttribute, CachedMethod
from ..tools.general import replace
from ..tools.progress import log_progress

import logging
logger = logging.getLogger(__name__.split('.')[-1])


# for axis in range(domain.dim):
#     if separable[axis]:
#         if len(domain.bases[axis]) > 1:
#             raise ValueError("Cannot have multiple bases along separable dimension")
#         else:
#             basis = domain.bases[axis][0]
#             trans_shape.append(basis.)
#     else:

def build_local_subsystems(problem):
    """Build local subsystem objects."""
    # Check that distributed dimensions are separable
    for axis in range(len(problem.dist.mesh)):
        if problem.matrix_coupling[axis]:
            raise ValueError("Problem is coupled along distributed dimension %i" %axis)
    # Build subsystems indeces as products of basis indeces
    domain = problem.variables[0].domain  # HACK
    basis_indeces = []
    for axis, basis in domain.enumerate_unique_bases():
        if basis is None:
            raise NotImplementedError()
        else:
            basis_coupling = problem.matrix_coupling[basis.first_axis:basis.last_axis+1]
            basis_indeces.append(basis.local_subsystem_indices(basis_coupling))
    local_subsystem_indices = [sum(p, []) for p in product(*basis_indeces)]
    return [Subsystem(index) for index in local_subsystem_indices]


def build_matrices(subproblems, matrices):
    """Build subproblem matrices with progress logger."""
    problem = subproblems[0].problem
    for eq in problem.eqs:
        for matrix in matrices:
            expr = eq[matrix]
            if expr:
                separability = ~problem.matrix_coupling
                expr.build_ncc_matrices(separability, problem.variables)
    for subproblem in log_progress(subproblems, logger, 'info', desc='Building subproblem matrices', iter=np.inf, frac=0.1, dt=10):
        subproblem.build_matrices(matrices)


def enumerate_product(*iterables):
    indices = (range(len(iter)) for iter in iterables)
    return zip(product(*indices), product(*iterables))


class Subsystem:
    """
    Class representing a subset of the global coefficient space.
    I.e. the multidimensional generalization of a pencil.

    Each subsystem is described by a global_index containing a
    group index (for each separable axis) or None (for each coupled
    axis).
    """

    def __init__(self, global_index):
        self.global_index = global_index

    def coeff_slices(self, domain):
        slices = []
        # Loop over bases
        for axis, basis in domain.enumerate_unique_bases():
            if basis is None:
                # Take single mode for constant bases
                ax_index = self.global_index[axis]
                if ax_index is None:
                    slices.append(slice(0, 1))
                else:
                    raise NotImplementedError()
            else:
                # Get slices from basis
                basis_index = self.global_index[basis.first_axis:basis.last_axis+1]
                slices.extend(basis.local_subsystem_slices(basis_index))
        return tuple(slices)

    def coeff_shape(self, domain):
        shape = []
        # Extract shape from slices
        coeff_slices = self.coeff_slices(domain)
        for ax_slice, ax_size in zip(coeff_slices, domain.coeff_shape):
            indices = ax_slice.indices(ax_size)
            shape.append(len(range(*indices)))
        return tuple(shape)

    def coeff_size(self, domain):
        return np.prod(self.coeff_shape(domain))

    def field_slices(self, field):
        comp_slices = (slice(None),) * len(field.tensorsig)
        coeff_slices = self.coeff_slices(field.domain)
        return comp_slices + coeff_slices

    def field_shape(self, field):
        comp_shape = tuple(cs.dim for cs in field.tensorsig)
        coeff_shape = self.coeff_shape(field.domain)
        return comp_shape + coeff_shape

    def field_size(self, field):
        return np.prod(self.field_shape(field))

    def gather(self, fields):
        """Gather and concatenate subsystem data in from multiple fields."""
        # TODO optimize: maybe preallocate here for speed?
        vec = []
        for field in fields:
            if self.field_size(field):
                field_slices = self.field_slices(field)
                field_data = field['c'][field_slices]
                vec.append(field_data.ravel())
        return np.concatenate(vec)

    def scatter(self, data, fields):
        """Scatter concatenated subsystem data out to multiple fields."""
        i0 = 0
        for field in fields:
            field_size = self.field_size(field)
            if field_size:
                i1 = i0 + field_size
                field_slices = self.field_slices(field)
                field_data = field['c'][field_slices]
                vec_data = data[i0:i1].reshape(field_data.shape)
                np.copyto(field_data, vec_data)
                i0 = i1


class Subproblem:
    """
    Object representing one coupled subsystem of a problem.

    Subproblems are identified by their group multi-index, which identifies
    the corresponding group of each separable dimension of the problem.

    This is the generalization of 'pencils' from a problem with exactly one
    coupled dimension.
    """

    def __init__(self, problem, group, index):
        # Remove coupled indices
        group = tuple(replace(group, problem.matrix_coupling, None))
        index = tuple(replace(index, problem.matrix_coupling, None))
        self.problem = problem
        ## HACKs
        self.domain = problem.variables[0].domain
        self.global_index = self.group = group
        self.local_index = index
        #self.first_coupled_axis = np.sum(problem.separable)

    # @CachedAttribute
    # def group_dict(self):
    #     """Group dictionary for evaluation equation conditions."""
    #     group_dict = {}
    #     for axis, group in enumerate(self.group):
    #         if group is not None:
    #             for space in self.domain.spaces[axis]:
    #                 group_dict['n'+space.name] = group
    #     return group_dict



    def inclusion_matrices(self, bases):
        """List of inclusion matrices."""
        matrices = []
        domain = domain.from_bases(self.domain, bases)
        # if any(bases):
        #     domain = domain.from_bases(bases)
        # else:
        #     domain = domain.from_domain(self.domain)
        global_slices = self.global_slices(domain)
        for gs, basis in zip(global_slices, bases):
            if basis is None:
                matrices.append(np.array([[1]])[gs, gs])
            else:
                matrices.append(basis.inclusion_matrix[gs, gs])
        return matrices

    # @CachedMethod
    # def expansion_matrix(self, input_domain, output_domain):
    #     matrices = []
    #     dtype = self.domain.dtype
    #     layout = self.domain.dist.coeff_layout
    #     arg_shape = layout.global_array_shape(input_domain, scales=1)
    #     out_shape = layout.global_array_shape(output_domain, scales=1)
    #     arg_elements = self.global_slices(input_domain)
    #     out_elements = self.global_slices(output_domain)
    #     for axis, (I, J, i, j) in enumerate(zip(out_shape, arg_shape, out_elements, arg_elements)):
    #         matrix = sparse.eye(I, J, dtype=dtype, format='csr')
    #         matrices.append(matrix[i, j])
    #     return reduce(sparse.kron, matrices, 1)

    # def expansion_matrix(self, arg, layout):
    #     matrices = []
    #     dtype = self.domain.dtype
    #     arg_shape = layout.global_array_shape(arg.domain, arg.scales)
    #     out_shape = layout.global_array_shape(self.domain, self.scales)
    #     arg_elements = layout.local_elements(arg.domain, arg.scales)
    #     out_elements = layout.local_elements(self.domain, self.scales)
    #     for axis, (I, J, i, j) in enumerate(zip(out_shape, arg_shape, out_elements, arg_elements)):
    #         if (arg.bases[axis] is None) and layout.grid_space[axis]:
    #             matrix = sparse.csr_matrix(np.ones((I, J), dtype=dtype))
    #         else:
    #             matrix = sparse.eye(I, J, dtype=dtype, format='csr')
    #         # Avoid bug on (1,1) advanced indexing of sparse matrices
    #         if i.size == 1:
    #             i = [[i[0]]]
    #         else:
    #             i = i[:,None]
    #         if j.size == 1:
    #             j = [[j[0]]]
    #         else:
    #             j = j[None,:]
    #         matrices.append(matrix[i, j])
    #     return reduce(sparse.kron, matrices, 1)

    # def expansion_matrix(self, inbases, outbases):
    #     axmats = self.inclusion_matrices(outbases)
    #     for axis, (inbasis, outbasis) in enumerate(zip(inbases, outbases)):
    #         if (inbasis is None) and (outbasis is not None):
    #             axmats[axis] = axmats[axis][:, 0:1]
    #     return reduce(sparse.kron, axmats, 1).tocsr()

    # def local_to_group(self, domain):
    #     """Matrix restricting local data to group data."""
    #     shape = self.domain.dist.coeff_layout.local_array_shape(domain, scales=1)
    #     slices = self.local_slices(domain)
    #     matrices = []
    #     for axis in range(self.domain.dim):
    #         matrix = sparse.identity(shape[axis], format='csr')[slices[axis], :]
    #         matrices.append(matrix)
    #     return reduce(sparse.kron, matrices, 1)

    def group_to_modes(self, field):
        """Matrix restricting group data to nonzero modes."""
        # matrices = []
        # for group, basis in zip(self.group, bases):
        #     if basis is None:
        #         matrices.append(np.array([[1]]))
        #     else:
        #         matrices.append(basis.mode_map(group))
        # return reduce(sparse.kron, matrices, 1).tocsr()
        size = self.subfield_size(field)
        return sparse.identity(size, format='csr')

    # def mode_map(self, basis_sets):
    #     """Restrict group data to nonzero modes."""
    #     var_mats = []
    #     for basis_set in basis_sets:
    #         ax_mats = []
    #         for group, basis in zip(self.group, basis_set):
    #             if basis is None:
    #                 ax_mats.append(np.array([[1]]))
    #             else:
    #                 ax_mats.append(basis.mode_map(group))
    #         var_mat = reduce(sparse.kron, ax_mats, 1).tocsr()
    #         var_mats.append(reduce(sparse.kron, ax_mats, 1).tocsr())
    #     return sparse_block_diag(var_mats).tocsr()



    def build_matrices(self, names):
        """Build problem matrices."""

        eqns = self.problem.equations
        vars = self.problem.variables
        eqn_sizes = [self.subfield_size(eqn['LHS']) for eqn in eqns]
        var_sizes = [self.subfield_size(var) for var in vars]
        I = sum(eqn_sizes)
        J = sum(var_sizes)

        # Construct subsystem matrices
        # Include all equations and group entries
        matrices = {}
        for name in names:
            # Collect entries
            data, rows, cols = [], [], []
            i0 = 0
            for eqn, eqn_size in zip(eqns, eqn_sizes):
                #if eqn.size and eqn.evaluate(self):
                #if eqn_size and eval(eqn['condition_str'], self.group_dict):
                # HACK: skipped conditions
                pass_conditions = True
                if eqn_size and pass_conditions:
                    expr = eqn[name]
                    if expr != 0:
                        eqn_blocks = eqn[name].expression_matrices(subproblem=self, vars=vars)
                        j0 = 0
                        for var, var_size in zip(vars, var_sizes):
                            if var_size and (var in eqn_blocks):
                                block = sparse.coo_matrix(eqn_blocks[var])
                                data.append(block.data)
                                rows.append(i0 + block.row)
                                cols.append(j0 + block.col)
                            j0 += var_size
                i0 += eqn_size
            # Build sparse matrix
            if data:
                data = np.concatenate(data)
                rows = np.concatenate(rows)
                cols = np.concatenate(cols)
            matrices[name] = sparse.coo_matrix((data, (rows, cols)), shape=(I, J)).tocsr()

        # Create maps restricting group data to included modes
        drop_eqn = [self.group_to_modes(eqn['LHS']) for eqn in eqns]
        drop_var = [self.group_to_modes(var) for var in vars]
        # Drop equations that fail condition test
        for n, eqn in enumerate(eqns):
            #if not eval(eqn['condition_str'], self.group_dict):
            if False:  # HACK
                drop_eqn[n] = drop_eqn[n][0:0, :]
        self.drop_eqn = drop_eqn = sparse_block_diag(drop_eqn).tocsr()
        self.drop_var = drop_var = sparse_block_diag(drop_var).tocsr()

        # Check squareness of restricted system
        # if drop_eqn.shape[0] != drop_var.shape[0]:
        #     raise ValueError("Non-square system: group={}, I={}, J={}".format(self.group, drop_eqn.shape[0], drop_var.shape[0]))

        # Restrict matrices to included modes
        for name in matrices:
            matrices[name] = drop_eqn @ matrices[name] @ drop_var.T

        # Eliminate any remaining zeros for maximal sparsity
        for name in matrices:
            matrices[name] = matrices[name].tocsr()
            matrices[name].eliminate_zeros()

        # Store minimal CSR matrices for fast dot products
        for name, matrix in matrices.items():
            setattr(self, '{:}_min'.format(name), matrix.tocsr())

        # Store expanded CSR matrices for fast combination
        self.LHS = zeros_with_pattern(*matrices.values()).tocsr()
        for name, matrix in matrices.items():
            expanded = expand_pattern(matrix, self.LHS)
            setattr(self, '{:}_exp'.format(name), expanded.tocsr())

        # Store RHS conversion matrix
        # F_conv = [self.expansion_matrix(eq['F'].domain, eq['domain']) for eq in eqns]
        # for n, eqn in enumerate(eqns):
        #     if not eval(eqn['raw_condition'], self.group_dict):
        #         F_conv[n] = F_conv[n][0:0, :]
        # self.rhs_map = drop_eqn * sparse_block_diag(F_conv).tocsr()
        self.rhs_map = drop_eqn

