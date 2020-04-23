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

def build_local_subproblems(problem):
    """Build local subproblem objects."""
    # Check that distributed dimensions are separable
    for axis in range(len(problem.dist.mesh)):
        if problem.matrix_coupling[axis]:
            raise ValueError("Problem is coupled along distributed dimension %i" %axis)
    # Get all local groups
    dist = problem.dist
    domain = problem.variables[0].domain  # HACK
    local_groups = dist.coeff_layout.local_groups(domain, scales=1)
    # Build subsystems from groups of separable domain
    local_subsystems = list(replace(local_groups, problem.matrix_coupling, [0]))
    local_subsystems = enumerate_product(*local_subsystems)
    # Check subsystem validity
    local_subsystems = [ls for ls in local_subsystems if include_subsystem(domain, *ls)]
    return [Subproblem(problem, group, index) for index, group in local_subsystems]


def include_subsystem(domain, index, group):
    # HACK
    # This should throw out invalid pencils in the triangular truncation.
    return True

def build_matrices(subproblems, matrices):
    """Build subproblem matrices with progress logger."""
    problem = subproblems[0].problem
    for eq in problem.eqs:
        for matrix in matrices:
            expr = eq[matrix]
            if expr:
                pass
                #expr.build_ncc_matrices(problem.separability(), problem.variables)
    for subproblem in log_progress(subproblems, logger, 'info', desc='Building subproblem matrices', iter=np.inf, frac=0.1, dt=10):
        subproblem.build_matrices(matrices)


def enumerate_product(*iterables):
    indices = (range(len(iter)) for iter in iterables)
    return zip(product(*indices), product(*iterables))


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

    @CachedMethod
    def subsystem_shape(self, domain):
        """Shape of group coefficients."""
        shape = []
        for axis, (group, basis) in enumerate(zip(self.group, domain.full_bases)):
            if basis is None:
                if group in [0, None]:
                    shape.append(1)
                else:
                    shape.append(0)
            else:
                subaxis = axis - basis.axis
                if group is None:
                    shape.append(basis.shape[subaxis])
                else:
                    shape.append(basis.group_shape[subaxis])
        return tuple(shape)

    @CachedMethod
    def subsystem_size(self, domain):
        """Size of group coefficients."""
        return np.prod(self.subsystem_shape(domain))

    def subfield_size(self, field):
        comps = np.prod([cs.dim for cs in field.tensorsig], dtype=int)
        size = self.subsystem_size(field.domain)
        return comps * size

    @CachedMethod
    def _start(self, domain, group_index):
        """Starting index of group coefficients."""
        start = []
        for axis, (group, index, basis) in enumerate(zip(self.group, group_index, domain.full_bases)):
            if basis is None:
                if group in [0, None]:
                    start.append(0)
                else:
                    start.append(1)
            else:
                if group is None:
                    start.append(0)
                else:
                    subaxis = axis - basis.axis
                    start.append(index * basis.group_shape[subaxis])
        return start

    @CachedMethod
    def _slices(self, domain, group_index):
        """Slices for group coefficients."""
        group_start = self._start(domain, group_index)
        group_shape = self.subsystem_shape(domain)
        slices = []
        for start, shape in zip(group_start, group_shape):
            slices.append(slice(start, start+shape))
        return tuple(slices)

    # def global_start(self, domain):
    #     """Global starting index of group coefficients."""
    #     return self._start(domain, self.global_index)

    # def local_start(self, domain, group_index):
    #     """Local starting index of group coefficients."""
    #     return self._start(subdomian, self.local_index)

    @CachedMethod
    def global_slices(self, domain):
        """Global slices for group coefficients."""
        return self._slices(domain, self.global_index)

    @CachedMethod
    def local_slices(self, domain):
        """Local slices for group coefficients."""
        return self._slices(domain, self.local_index)

    def get_vector(self, vars):
        """Retrieve and concatenate group coefficients from variables."""
        vec = []
        for var in vars:
            if self.group_size(var.domain):
                slices = self.local_slices(var.domain)
                var_data = var['c'][slices]
                vec.append(var_data.ravel())
        return np.concatenate(vec)

    def set_vector(self, vars, data):
        """Assign vectorized group coefficients to variables."""
        i0 = 0
        for var in vars:
            group_size = self.group_size(var.domain)
            if group_size:
                i1 = i0 + group_size
                slices = self.local_slices(var.domain)
                var_data = var['c'][slices]
                vec_data = data[i0:i1].reshape(var_data.shape)
                np.copyto(var_data, vec_data)
                i0 = i1

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
                                block = eqn_blocks[var].tocoo()
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

