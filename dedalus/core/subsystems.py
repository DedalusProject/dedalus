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

def build_subsystems(problem):
    """Build local subsystem objects."""
    # Check that distributed dimensions are separable
    for axis in range(len(problem.dist.mesh)):
        if problem.matrix_coupling[axis]:
            raise ValueError("Problem is coupled along distributed dimension %i" %axis)
    # Determine local groups for each basis
    domain = problem.variables[0].domain  # HACK
    basis_groups = []
    for axis, basis in domain.enumerate_unique_bases():
        if basis is None:
            raise NotImplementedError()
        else:
            basis_coupling = problem.matrix_coupling[basis.first_axis:basis.last_axis+1]
            basis_groups.append(basis.local_groups(basis_coupling))
    # Build subsystems groups as product of basis groups
    local_groups = [tuple(sum(p, [])) for p in product(*basis_groups)]
    return tuple(Subsystem(problem, group) for group in local_groups)


def build_subproblems(problem, subsystems, matrices):
    """Build subproblem matrices with progress logger."""
    # Setup NCCs
    for eq in problem.eqs:
        for matrix in matrices:
            expr = eq[matrix]
            if expr:
                separability = ~problem.matrix_coupling
                expr.build_ncc_matrices(separability, problem.variables)
    # Get matrix groups
    subproblem_map = defaultdict(list)
    for subsystem in subsystems:
        subproblem_map[subsystem.matrix_group].append(subsystem)
    # Build subproblems
    subproblems = []
    for matrix_group in log_progress(subproblem_map, logger, 'info', desc='Building subproblem matrices', iter=np.inf, frac=0.1, dt=10):
        subsystems = tuple(subproblem_map[matrix_group])
        subproblem = Subproblem(problem, subsystems, matrix_group)
        subproblem.build_matrices(matrices)
        subproblems.append(subproblem)
    return tuple(subproblems)


class Subsystem:
    """
    Class representing a subset of the global coefficient space.
    I.e. the multidimensional generalization of a pencil.

    Each subsystem is described by a "group" tuple containing a
    group index (for each separable axis) or None (for each coupled
    axis).
    """

    def __init__(self, problem, group):
        self.problem = problem
        self.group = group
        # Determine matrix group using problem matrix dependence
        matrix_independence = ~ problem.matrix_dependence
        self.matrix_group = tuple(replace(group, matrix_independence, 0))

    def coeff_slices(self, domain):
        slices = []
        # Loop over bases
        for axis, basis in domain.enumerate_unique_bases():
            if basis is None:
                # Take single mode for constant bases
                ax_group = self.group[axis]
                if ax_group is None:
                    slices.append(slice(0, 1))
                else:
                    raise NotImplementedError()
            else:
                # Get slices from basis
                basis_group = self.group[basis.first_axis:basis.last_axis+1]
                slices.extend(basis.local_group_slices(basis_group))
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

    @CachedMethod
    def field_slices(self, field):
        comp_slices = (slice(None),) * len(field.tensorsig)
        coeff_slices = self.coeff_slices(field.domain)
        return comp_slices + coeff_slices

    @CachedMethod
    def field_shape(self, field):
        comp_shape = tuple(cs.dim for cs in field.tensorsig)
        coeff_shape = self.coeff_shape(field.domain)
        return comp_shape + coeff_shape

    @CachedMethod
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

    def __init__(self, problem, subsystems, group):
        self.problem = problem
        self.subsystems = subsystems
        self.group = group
        self.dist = problem.dist
        self.domain = problem.variables[0].domain  # HACK
        self.dtype = problem.dtype
        # Cross reference from subsystems
        for subsystem in subsystems:
            subsystem.subproblem = self
        # Build group dictionary
        self.group_dict = {}
        for axis, ax_group in enumerate(group):
            if ax_group is not None:
                ax_coord = self.dist.coords[axis]
                self.group_dict['n'+ax_coord.name] = ax_group

    def coeff_slices(self, domain):
        return self.subsystems[0].coeff_slices(domain)

    def coeff_shape(self, domain):
        return self.subsystems[0].coeff_shape(domain)

    def coeff_size(self, domain):
        return self.subsystems[0].coeff_size(domain)

    def field_slices(self, field):
        return self.subsystems[0].field_slices(field)

    def field_shape(self, field):
        return self.subsystems[0].field_shape(field)

    def field_size(self, field):
        return self.subsystems[0].field_size(field)

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
        size = self.field_size(field)
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

    def check_condition(self, eqn):
        return eval(eqn['condition'], self.group_dict)

    def build_matrices(self, names):
        """Build problem matrices."""

        eqns = self.problem.equations
        vars = self.problem.variables
        eqn_conditions = [self.check_condition(eqn) for eqn in eqns]  # HACK
        eqn_sizes = [self.field_size(eqn['LHS']) for eqn in eqns]
        var_sizes = [self.field_size(var) for var in vars]
        I = sum(eqn_sizes)
        J = sum(var_sizes)
        dtype = self.dtype

        # Construct subsystem matrices
        # Include all equations and group entries
        matrices = {}
        for name in names:
            # Collect entries
            data, rows, cols = [], [], []
            i0 = 0
            for eqn, eqn_size, eqn_cond in zip(eqns, eqn_sizes, eqn_conditions):
                if eqn_size and eqn_cond:
                    # Build matrix and append data
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
                else:
                    # Leave empty blocks
                    i0 += eqn_size
            # Build sparse matrix
            if data:
                data = np.concatenate(data)
                rows = np.concatenate(rows)
                cols = np.concatenate(cols)
                # Filter entries
                entry_cutoff = 1e-12  # HACK: this should be passed as a variable somehow
                data[np.abs(data) < entry_cutoff] = 0
            matrices[name] = sparse.coo_matrix((data, (rows, cols)), shape=(I, J), dtype=dtype).tocsr()

        # Create maps restricting group data to included modes
        drop_eqn = [self.group_to_modes(eqn['LHS']) for eqn in eqns]
        #drop_var = [self.group_to_modes(var) for var in vars]
        # Drop equations that fail condition test
        for n, eqn_cond in enumerate(eqn_conditions):
            if not eqn_cond:
                drop_eqn[n] = drop_eqn[n][0:0, :]
        self.drop_eqn = drop_eqn = sparse_block_diag(drop_eqn).tocsr()
        #self.drop_var = drop_var = sparse_block_diag(drop_var).tocsr()

        # Check squareness of restricted system
        if drop_eqn.shape[0] != J:
            raise ValueError("Non-square system: group={}, I={}, J={}".format(self.group, drop_eqn.shape[0], J))

        # Restrict matrices to included modes
        for name in matrices:
            matrices[name] = drop_eqn @ matrices[name]

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

    def expand_matrices(self, matrices):
        matrices = {matrix: getattr(self, '%s_min' %matrix) for matrix in matrices}
        # Store expanded CSR matrices for fast combination
        self.LHS = zeros_with_pattern(*matrices.values()).tocsr()
        for name, matrix in matrices.items():
            expanded = expand_pattern(matrix, self.LHS)
            setattr(self, '{:}_exp'.format(name), expanded.tocsr())


