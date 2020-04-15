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
    # for axis in range(len(problem.dist.mesh)):
    #     if not problem.separable[axis]:
    #         raise ValueError("Problem is not separable along distributed dimension %i" %axis)
    ## HACKS
    dist = problem.dist
    domain = problem.variables[0].domain
    # Build subproblems on local groups of separable subdomain
    # subdomain = domain.subdomain(problem.separability())
    coeff_layout = dist.coeff_layout
    local_groups = coeff_layout.local_groups(domain, scales=1)
    local_groups = list(replace(local_groups, problem.coupled, [0]))
    return [Subproblem(problem, group, index) for index, group in enumerate_product(*local_groups)]

def build_matrices(subproblems, matrices):
    """Build subproblem matrices with progress logger."""
    problem = subproblems[0].problem
    for eq in problem.eqs:
        for matrix in matrices:
            expr = eq[matrix]
            expr.build_ncc_matrices(problem.separability(), problem.variables)
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
        group = tuple(replace(group, problem.coupled, None))
        index = tuple(replace(index, problem.coupled, None))
        self.problem = problem
        self.domain = problem.domain
        self.global_index = self.group = group
        self.local_index = index
        self.first_coupled_axis = np.sum(problem.separable)

    @CachedAttribute
    def group_dict(self):
        """Group dictionary for evaluation equation conditions."""
        group_dict = {}
        for axis, group in enumerate(self.group):
            if group is not None:
                for space in self.domain.spaces[axis]:
                    group_dict['n'+space.name] = group
        return group_dict

    @CachedMethod
    def group_shape(self, subdomain):
        """Shape of group coefficients."""
        group_shape = []
        for group, space in zip(self.group, subdomain.spaces):
            if space is None:
                if group in [0, None]:
                    group_shape.append(1)
                else:
                    group_shape.append(0)
            else:
                if group is None:
                    group_shape.append(space.coeff_size)
                else:
                    group_shape.append(space.group_size)
        return tuple(group_shape)

    @CachedMethod
    def group_size(self, subdomain):
        """Size of group coefficients."""
        return np.prod(self.group_shape(subdomain))

    @CachedMethod
    def _start(self, subdomain, group_index):
        """Starting index of group coefficients."""
        start = []
        for group, index, space in zip(self.group, group_index, subdomain.spaces):
            if space is None:
                if group in [0, None]:
                    start.append(0)
                else:
                    start.append(1)
            else:
                if group is None:
                    start.append(0)
                else:
                    start.append(index * space.group_size)
        return start

    @CachedMethod
    def _slices(self, subdomain, group_index):
        """Slices for group coefficients."""
        group_start = self._start(subdomain, group_index)
        group_shape = self.group_shape(subdomain)
        slices = []
        for start, shape in zip(group_start, group_shape):
            slices.append(slice(start, start+shape))
        return tuple(slices)

    # def global_start(self, subdomain):
    #     """Global starting index of group coefficients."""
    #     return self._start(subdomain, self.global_index)

    # def local_start(self, subdomain, group_index):
    #     """Local starting index of group coefficients."""
    #     return self._start(subdomian, self.local_index)

    @CachedMethod
    def global_slices(self, subdomain):
        """Global slices for group coefficients."""
        return self._slices(subdomain, self.global_index)

    @CachedMethod
    def local_slices(self, subdomain):
        """Local slices for group coefficients."""
        return self._slices(subdomain, self.local_index)

    def get_vector(self, vars):
        """Retrieve and concatenate group coefficients from variables."""
        vec = []
        for var in vars:
            if self.group_size(var.subdomain):
                slices = self.local_slices(var.subdomain)
                var_data = var['c'][slices]
                vec.append(var_data.ravel())
        return np.concatenate(vec)

    def set_vector(self, vars, data):
        """Assign vectorized group coefficients to variables."""
        i0 = 0
        for var in vars:
            group_size = self.group_size(var.subdomain)
            if group_size:
                i1 = i0 + group_size
                slices = self.local_slices(var.subdomain)
                var_data = var['c'][slices]
                vec_data = data[i0:i1].reshape(var_data.shape)
                np.copyto(var_data, vec_data)
                i0 = i1

    def inclusion_matrices(self, bases):
        """List of inclusion matrices."""
        matrices = []
        subdomain = Subdomain.from_bases(self.domain, bases)
        # if any(bases):
        #     subdomain = Subdomain.from_bases(bases)
        # else:
        #     subdomain = Subdomain.from_domain(self.domain)
        global_slices = self.global_slices(subdomain)
        for gs, basis in zip(global_slices, bases):
            if basis is None:
                matrices.append(np.array([[1]])[gs, gs])
            else:
                matrices.append(basis.inclusion_matrix[gs, gs])
        return matrices

    # @CachedMethod
    # def expansion_matrix(self, input_subdomain, output_subdomain):
    #     matrices = []
    #     dtype = self.domain.dtype
    #     layout = self.domain.dist.coeff_layout
    #     arg_shape = layout.global_array_shape(input_subdomain, scales=1)
    #     out_shape = layout.global_array_shape(output_subdomain, scales=1)
    #     arg_elements = self.global_slices(input_subdomain)
    #     out_elements = self.global_slices(output_subdomain)
    #     for axis, (I, J, i, j) in enumerate(zip(out_shape, arg_shape, out_elements, arg_elements)):
    #         matrix = sparse.eye(I, J, dtype=dtype, format='csr')
    #         matrices.append(matrix[i, j])
    #     return reduce(sparse.kron, matrices, 1)

    # def expansion_matrix(self, arg, layout):
    #     matrices = []
    #     dtype = self.domain.dtype
    #     arg_shape = layout.global_array_shape(arg.subdomain, arg.scales)
    #     out_shape = layout.global_array_shape(self.subdomain, self.scales)
    #     arg_elements = layout.local_elements(arg.subdomain, arg.scales)
    #     out_elements = layout.local_elements(self.subdomain, self.scales)
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

    # def local_to_group(self, subdomain):
    #     """Matrix restricting local data to group data."""
    #     shape = self.domain.dist.coeff_layout.local_array_shape(subdomain, scales=1)
    #     slices = self.local_slices(subdomain)
    #     matrices = []
    #     for axis in range(self.domain.dim):
    #         matrix = sparse.identity(shape[axis], format='csr')[slices[axis], :]
    #         matrices.append(matrix)
    #     return reduce(sparse.kron, matrices, 1)

    def group_to_modes(self, bases):
        """Matrix restricting group data to nonzero modes."""
        matrices = []
        for group, basis in zip(self.group, bases):
            if basis is None:
                matrices.append(np.array([[1]]))
            else:
                matrices.append(basis.mode_map(group))
        return reduce(sparse.kron, matrices, 1).tocsr()

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
        eqn_sizes = [self.group_size(eqn['LHS'].subdomain) for eqn in eqns]
        var_sizes = [self.group_size(var.subdomain) for var in vars]
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
                if eqn_size and eval(eqn['condition_str'], self.group_dict):
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
            data = np.concatenate(data)
            rows = np.concatenate(rows)
            cols = np.concatenate(cols)
            matrices[name] = sparse.coo_matrix((data, (rows, cols)), shape=(I, J)).tocsr()

        # Create maps restricting group data to included modes
        drop_eqn = [self.group_to_modes(eqn['LHS'].bases) for eqn in eqns]
        drop_var = [self.group_to_modes(var.bases) for var in vars]
        # Drop equations that fail condition test
        for n, eqn in enumerate(eqns):
            if not eval(eqn['condition_str'], self.group_dict):
                drop_eqn[n] = drop_eqn[n][0:0, :]
        self.drop_eqn = drop_eqn = sparse_block_diag(drop_eqn).tocsr()
        self.drop_var = drop_var = sparse_block_diag(drop_var).tocsr()

        # Check squareness of restricted system
        if drop_eqn.shape[0] != drop_var.shape[0]:
            raise ValueError("Non-square system: group={}, I={}, J={}".format(self.group, drop_eqn.shape[0], drop_var.shape[0]))

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
        # F_conv = [self.expansion_matrix(eq['F'].subdomain, eq['subdomain']) for eq in eqns]
        # for n, eqn in enumerate(eqns):
        #     if not eval(eqn['raw_condition'], self.group_dict):
        #         F_conv[n] = F_conv[n][0:0, :]
        # self.rhs_map = drop_eqn * sparse_block_diag(F_conv).tocsr()
        self.rhs_map = drop_eqn

