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

from .domain import Subdomain
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
    domain = problem.domain
    # Check that distributed dimensions are separable
    for axis in range(len(domain.dist.mesh)):
        if not problem.separable[axis]:
            raise ValueError("Problem is not separable along distributed dimension %i" %axis)
    # Build subproblems on local groups of separable subdomain
    subdomain = domain.subdomain(problem.separability())
    coeff_layout = domain.dist.coeff_layout
    local_groups = coeff_layout.local_groups(subdomain, scales=1)
    local_groups = list(replace(local_groups, problem.coupled, [0]))
    return [Subproblem(problem, group, index) for index, group in enumerate_product(*local_groups)]

def build_matrices(subproblems, matrices):
    """Build subproblem matrices with progress logger."""
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

    # def inclusion_matrices(self, bases):
    #     """List of inclusion matrices."""
    #     matrices = []
    #     if any(bases):
    #         subdomain = Subdomain.from_bases(bases)
    #     else:
    #         subdomain = Subdomain.from_domain(self.domain)
    #     global_slices = self.global_slices(subdomain)
    #     for gs, basis in zip(global_slices, bases):
    #         if basis is None:
    #             matrices.append(np.array([[1]])[gs, gs])
    #         else:
    #             matrices.append(basis.inclusion_matrix[gs, gs])
    #     return matrices

    @CachedMethod
    def expansion_matrix(self, input_subdomain, output_subdomain):
        matrices = []
        dtype = self.domain.dtype
        layout = self.domain.dist.coeff_layout
        arg_shape = layout.global_array_shape(input_subdomain, scales=1)
        out_shape = layout.global_array_shape(output_subdomain, scales=1)
        arg_elements = self.global_slices(input_subdomain)
        out_elements = self.global_slices(output_subdomain)
        for axis, (I, J, i, j) in enumerate(zip(out_shape, arg_shape, out_elements, arg_elements)):
            matrix = sparse.eye(I, J, dtype=dtype, format='csr')
            matrices.append(matrix[i, j])
        return reduce(sparse.kron, matrices, 1)

    # def expansion_matrix(self, inbases, outbases):
    #     axmats = self.inclusion_matrices(outbases)
    #     for axis, (inbasis, outbasis) in enumerate(zip(inbases, outbases)):
    #         if (inbasis is None) and (outbasis is not None):
    #             axmats[axis] = axmats[axis][:, 0:1]
    #     return reduce(sparse.kron, axmats, 1).tocsr()

    @CachedMethod
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

    def local_to_group(self, subdomain):
        """Matrix restricting local data to group data."""
        shape = self.domain.dist.coeff_layout.local_array_shape(subdomain, scales=1)
        slices = self.local_slices(subdomain)
        matrices = []
        for axis in range(self.domain.dim):
            matrix = sparse.identity(shape[axis], format='csr')[slices[axis], :]
            matrices.append(matrix)
        return reduce(sparse.kron, matrices, 1)

    def build_matrices(self, names):
        """Build problem matrices."""
        # Filter equations by condition and group
        #eqs = [eq for eq in self.problem.eqs if eval(eq['raw_condition'], self.group_dict)]
        eqs = [eq for eq in self.problem.eqs if self.group_size(eq['subdomain'])]
        eq_sizes = [self.group_size(eq['subdomain']) for eq in eqs]
        eq_filters = [self.local_to_group(eq['subdomain']) for eq in eqs]
        I = sum(eq_sizes)

        # Filter variables by group
        vars = [var for var in self.problem.variables if self.group_size(var.subdomain)]
        var_sizes = [self.group_size(var.subdomain) for var in vars]
        var_filters = [self.local_to_group(var.subdomain) for var in vars]
        J = sum(var_sizes)

        # Construct full subsystem matrices
        matrices = {}
        for name in names:
            # Collect subblocks
            data, rows, cols = [], [], []
            i0 = 0
            for eq, eq_size, eq_filter in zip(eqs, eq_sizes, eq_filters):
                op_dict = eq[name+'_op']
                j0 = 0
                for var, var_size, var_filter in zip(vars, var_sizes, var_filters):
                    if var in op_dict:
                        varmat = (eq_filter * op_dict[var] * var_filter.T).tocoo()
                        data.append(varmat.data)
                        rows.append(varmat.row + i0)
                        cols.append(varmat.col + j0)
                    j0 += var_size
                i0 += eq_size
            # Build full matrix
            data = np.concatenate(data)
            rows = np.concatenate(rows)
            cols = np.concatenate(cols)
            matrices[name] = sparse.coo_matrix((data, (rows, cols)), shape=(I,J)).tocsr()

        # # Construct permutation matrix
        # RP = build_permutation([eq['bases'] for eq in eqs])
        # CP = build_permutation([var.bases for var in vars])

        #  F = L.X
        #  RP.F = RP.L.CP*.CP.X
        # (RP.F) = (RP.L.CP*) . (CP.X)

        # Restrict to nonzero modes
        eq_modes = [self.group_to_modes(eq['bases']) for eq in eqs]
        var_modes = [self.group_to_modes(var.bases) for var in vars]
        # Drop equations that fail condition test
        for n, eq in enumerate(eqs):
            if not eval(eq['raw_condition'], self.group_dict):
                eq_modes[n] = eq_modes[n][0:0, :]

        # Store and apply mode maps to matrices
        self.row_map = row_map = sparse_block_diag(eq_modes).tocsr()
        self.col_map = col_map = sparse_block_diag(var_modes).tocsr()
        if row_map.shape[0] != col_map.shape[0]:
            raise ValueError("Non-square system: group={}, I={}, J={}".format(self.group, row_map.shape[0], col_map.shape[0]))
        for name in matrices:
            matrices[name] = row_map * matrices[name] * col_map.T

        # Store minimal CSR matrices for fast dot products
        for name, matrix in matrices.items():
            matrix.eliminate_zeros()
            setattr(self, name, matrix.tocsr())

        # Store expanded CSR matrices for fast combination
        self.LHS = zeros_with_pattern(*matrices.values()).tocsr()
        for name, matrix in matrices.items():
            matrix = expand_pattern(matrix, self.LHS)
            setattr(self, name+'_exp', matrix.tocsr())

        # Store RHS conversion matrix
        F_conv = [self.expansion_matrix(eq['F'].subdomain, eq['subdomain']) for eq in eqs]
        self.rhs_map = row_map * sparse_block_diag(F_conv).tocsr()

