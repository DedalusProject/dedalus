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

    def group_shape(self, bases):
        """Computable-coefficient shape for group."""
        group_shape = []
        for group, basis in zip(self.group, bases):
            if basis is None:
                if group in [0, None]:
                    group_shape.append(1)
                else:
                    group_shape.append(0)
            else:
                if group is None:
                    group_shape.append(basis.n_modes)
                else:
                    group_shape.append(basis.group_size(group))
        return tuple(group_shape)

    def group_size(self, bases):
        """Number of computable coefficients for group."""
        group_shape = self.group_shape(bases)
        return np.prod(group_shape)

    def global_start(self, bases):
        """Global starting index of group."""
        global_start = []
        for group, basis in zip(self.group, bases):
            if basis is None:
                if group in [0, None]:
                    global_start.append(0)
                else:
                    global_start.append(1)
            else:
                if group is None:
                    global_start.append(0)
                else:
                    global_start.append(group * basis.space.group_spacing)
        return tuple(global_start)

    def local_start(self, bases):
        """Local starting index of group."""
        local_start = []
        for group, basis in zip(self.local_group, bases):
            if basis is None:
                if group in [0, None]:
                    local_start.append(0)
                else:
                    local_start.append(1)
            else:
                if group is None:
                    local_start.append(0)
                else:
                    local_start.append(group * basis.space.group_spacing)
        return local_start

    def global_slices(self, bases):
        """Global slices for computable coefficients."""
        global_start = self.global_start(bases)
        group_shape = self.group_shape(bases)
        global_slices = []
        for start, shape in zip(global_start, group_shape):
            global_slices.append(slice(start, start+shape))
        return tuple(global_slices)

    def local_slices(self, bases):
        """Global slices for computable coefficients."""
        local_start = self.local_start(bases)
        group_shape = self.group_shape(bases)
        local_slices = []
        for start, shape in zip(local_start, group_shape):
            local_slices.append(slice(start, start+shape))
        return tuple(local_slices)

    def get_vector(self, vars):
        """Retrieve and concatenate group data from variables."""
        vec = []
        for var in vars:
            if self.group_size(var):
                slices = self.local_slices(var.bases)
                var_data = var['c'][slices]
                vec.append(var_data.ravel())
        return np.concatenate(vec)

    def set_vector(self, vars, data):
        """Assign vectorized group data to variables."""
        i0 = 0
        for var in vars:
            group_size = self.group_size(var):
            if group_size:
                i1 = i0 + group_size
                slices = self.local_slices(var.bases)
                var_data = var['c'][slices]
                vec_data = data[i0:i1].reshape(var_data.shape)
                np.copyto(var_data, vec_data)
                i0 = i1

    def compute_identities(self, bases):
        cid = []
        global_slices = self.global_slices(bases)
        for gs, basis in zip(global_slices, bases):
            if basis is None:
                cid.append(np.array([[1]])[gs, gs])
            else:
                cid.append(basis.compute_identity[gs, gs])
        return cid

    def compute_conversion(self, inbases, outbases):
        axmats = self.compute_identities(outbases)
        for axis, (inbasis, outbasis) in enumerate(zip(inbases, outbases)):
            if (inbasis is None) and (outbasis is not None):
                axmats[axis] = axmats[axis][:, 0:1]
        return reduce(sparse.kron, axmats, 1).tocsr()

    def build_matrices(self, problem, names):
        """Build problem matrices."""

        # Filter equations by condition and group
        eqs = [eq for eq in problem.eqs if eval(eq['raw_condition'], self.group_dict)]
        eqs = [eq for eq in eqs if self.size(eq['bases'])]
        eq_sizes = [self.size(eq['bases']) for eq in eqs]

        # Filter variables by group
        vars = [var for var in problem.variables if self.size(var.bases)]
        var_sizes = [self.size(var.bases) for var in vars]

        # Require squareness
        I = sum(eq_sizes)
        J = sum(var_sizes)
        if I != J:
            raise ValueError("Non-square system.")

        # Construct subsystem matrices
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

