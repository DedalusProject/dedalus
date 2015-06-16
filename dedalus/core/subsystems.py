"""
Classes for manipulating pencils.

"""

from functools import partial, reduce
from collections import defaultdict
import numpy as np
from scipy import sparse
from mpi4py import MPI
import uuid

from ..tools.array import zeros_with_pattern
from ..tools.array import expand_pattern
from ..tools.cache import CachedAttribute
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


# def _check_seperability(self, expr, vars):
#     sep = expr.seperability(vars)
#     distributed_dims = len(self.domain.mesh)
#     if not all(sep[:distributed_dims]):
#         raise ValueError("")


def build_local_subsystems(problem):
    """Build the set of local subsystems for a problem."""
    # Check that distributed dimensions are separable
    for axis in range(len(problem.domain.dist.mesh)):
        if not problem.separable[axis]:
            raise ValueError("Problem not separable along distributed dimension %i" %axis)

    separable_subspace = problem.domain.subspace(problem.separability())
    local_groups = separable_subspace.local_groups()
    return [Subsystem(separable_subspace, group) for group in local_groups]


def build_matrices(subsystems, problem, matrices):
    """Build local subsystem matrices with progress logger."""
    for subsystem in log_progress(subsystems, logger, 'info', desc='Building subsystem matrix', iter=np.inf, frac=0.1, dt=10):
        subsystem.build_matrices(problem, matrices)


class Subsystem:
    """
    Object representing one coupled subsystem of a problem.

    Subsystems are identified by their group multi-index, which identifies
    the corresponding subgroup of each separable dimension of the problem.

    This is the generalization of the pencils in a problem with exactly one
    coupled dimension.
    """

    def __init__(self, subdomain, group):
        self.subdomain = subdomain
        self.domain = subdomain.domain
        self.group = group
        local_start = group - self.domain.dist.coeff_layout.start(subdomain, scales=1)
        self.local_group = group - local_start

    @CachedAttribute
    def group_dict(self):
        """Group dictionary for evaluation equation conditions."""
        group_dict = {}
        for axis, group in enumerate(self.group):
            if group is not None:
                for space in self.domain.spaces[axis]:
                    group_dict['n'+space.name] = group
        return group_dict

    def group_shape(self, subdomain):
        """Coefficient shape for group."""
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

    def group_size(self, subdomain):
        """Coefficient size for group."""
        group_shape = self.group_shape(subdomain)
        return np.prod(group_shape)

    def global_start(self, subdomain):
        """Global starting index of group."""
        global_start = []
        for group, space in zip(self.group, subdomain.spaces):
            if space is None:
                if group in [0, None]:
                    global_start.append(0)
                else:
                    global_start.append(1)
            else:
                if group is None:
                    global_start.append(0)
                else:
                    global_start.append(group * space.group_size)
        return tuple(global_start)

    def local_start(self, subdomain):
        """Local starting index of group."""
        local_start = []
        for group, space in zip(self.local_group, subdomain.spaces):
            if space is None:
                if group in [0, None]:
                    local_start.append(0)
                else:
                    local_start.append(1)
            else:
                if group is None:
                    local_start.append(0)
                else:
                    local_start.append(group * space.group_size)
        return local_start

    def global_slices(self, subdomain):
        """Global slices for computable coefficients."""
        global_start = self.global_start(subdomain)
        group_shape = self.group_shape(subdomain)
        global_slices = []
        for start, shape in zip(global_start, group_shape):
            global_slices.append(slice(start, start+shape))
        return tuple(global_slices)

    def local_slices(self, subdomain):
        """Global slices for computable coefficients."""
        local_start = self.local_start(subdomain)
        group_shape = self.group_shape(subdomain)
        local_slices = []
        for start, shape in zip(local_start, group_shape):
            local_slices.append(slice(start, start+shape))
        return tuple(local_slices)

    def get_vector(self, vars):
        """Retrieve and concatenate group data from variables."""
        vec = []
        for var in vars:
            if self.group_size(var.subdomain):
                slices = self.local_slices(var.subdomain)
                var_data = var['c'][slices]
                vec.append(var_data.ravel())
        return np.concatenate(vec)

    def set_vector(self, vars, data):
        """Assign vectorized group data to variables."""
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

    def inclusion_matrices(self, var):
        matrices = []
        global_slices = self.global_slices(var.subdomain)
        for gs, basis in zip(global_slices, var.bases):
            if basis is None:
                matrices.append(np.array([[1]])[gs, gs])
            else:
                matrices.append(basis.inclusion_matrix[gs, gs])
        return include

    def compute_conversion(self, inbases, outbases):
        axmats = self.include_matrices(outbases)
        for axis, (inbasis, outbasis) in enumerate(zip(inbases, outbases)):
            if (inbasis is None) and (outbasis is not None):
                axmats[axis] = axmats[axis][:, 0:1]
        return reduce(sparse.kron, axmats, 1).tocsr()

    def mode_map(self, basis_sets):
        var_mats = []
        for basis_set in basis_sets:
            ax_mats = []
            for group, basis in zip(self.group, basis_set):
                if basis is None:
                    ax_mats.append(np.array([[1]]))
                else:
                    ax_mats.append(basis.mode_map(group))
            var_mats.append(reduce(sparse.kron, ax_mats, 1).tocsr())
        return sparse.block_diag(var_mats).tocsr()

    def build_matrices(self, problem, names):
        """Build problem matrices."""

        # Filter equations by condition and group
        eqs = [eq for eq in problem.eqs if eval(eq['raw_condition'], self.group_dict)]
        eqs = [eq for eq in eqs if self.size(eq['subdomain'])]
        eq_sizes = [self.size(eq['subdomain']) for eq in eqs]
        I = sum(eq_sizes)

        # Filter variables by group
        vars = [var for var in problem.variables if self.size(var.subdomain)]
        var_sizes = [self.size(var.subdomain) for var in vars]
        J = sum(var_sizes)

        # Construct full subsystem matrices
        matrices = {}
        for name in names:
            # Collect subblocks
            data, rows, cols = [], [], []
            i0 = 0
            for eq, eq_size in zip(eqs, eq_sizes):
                expr = eq[name][0]
                if expr != 0:
                    op_dict = expr.operator_dict(self, vars, **problem.ncc_kw)
                    j0 = 0
                    for var, var_size in zip(vars, var_sizes):
                        if var in op_dict:
                            varmat = op_dict[var].tocoo()
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

        # Store and apply mode maps to matrices
        self.row_map = row_map = self.mode_map([eq['bases'] for eq in eqs])
        self.col_map = col_map = self.mode_map([var.bases for var in vars])
        if row_map.shape[0] != col_map.shape[0]:
            raise ValueError("Non-square system")
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
        F_conv = [self.compute_conversion(eq['F'].bases, eq['bases']) for eq in eqs]
        self.RHS_C = sparse.block_diag(F_conv).tocsr()

