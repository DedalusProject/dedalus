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
    Object holding problem matrices for a given transverse wavevector.

    Parameters
    ----------
    index : tuple of ints
        Transverse indeces for retrieving pencil from system data

    """

    def __init__(self, subdomain, group):
        self.subdomain = subdomain
        self.domain = subdomain.domain
        self.local_start = self.domain.dist.coeff_layout.start(subdomain, scales=1)
        self.group = group

    def global_slices(self, bases):
        slices = []
        for group, basis in zip(self.group, bases):
            if basis is None:
                if group in [0, None]:
                    slices.append(slice(0, 1))
                else:
                    slices.append(slice(1, 1))
            else:
                if group is None:
                    slices.append(slice(None))
                else:
                    start = basis.space.group_size * group
                    end = basis.space.group_size * (group + 1)
                    slices.append(slice(start, end))
        return slices

    def local_slices(self, bases):
        slices = []
        for group, basis, start in zip(self.group, bases, self.local_start):
            if bases is None:
                if group in [0, None]:
                    slices.append(slice(0, 1))
                else:
                    slices.append(slice(1, 1))
            else:
                if group is None:
                    slices.append(slice(None))
                else:
                    start = basis.space.group_size * group - start
                    end = basis.space.group_size * (group + 1) - start
                    slices.append(slice(start, end))
        return slices

    def get_vector(self, vars):
        vec = []
        for var in vars:
            slices = self.local_slices(var.bases)
            var_data = var['c'][slices]
            if var_data.size:
                vec.append(var_data.ravel())
        return np.concatenate(vec)

    def set_vector(self, vars, X):
        i0 = 0
        for var in vars:
            slices = self.local_slices(var.bases)
            var_data = var['c'][slices].ravel()
            if var_data.size:
                i1 = i0 + var_data.size
                np.copyto(var_data.ravel(), X[i0:i1])
                i0 = i1

    def shape(self, bases):
        shape = []
        for group, basis in zip(self.group, bases):
            if basis is None:
                if group in [0, None]:
                    shape.append(1)
                else:
                    shape.append(0)
            else:
                if group is None:
                    shape.append(basis.space.coeff_size)
                else:
                    shape.append(basis.space.group_size)
        return shape

    def size(self, bases):
        shape = self.shape(bases)
        return np.prod(shape)

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

        # Build group dictionary for evaluating equation conditions
        group_dict = {}
        for axis, group in enumerate(self.group):
            if group is not None:
                for space in self.domain.spaces[axis]:
                    group_dict['n'+space.name] = group

        # Filter equations by condition and group
        eqs = [eq for eq in problem.eqs if eval(eq['raw_condition'], group_dict)]
        eqs = [eq for eq in eqs if self.size(eq['bases'])]

        # Filter variables by group
        vars = [var for var in problem.variables if self.size(var.bases)]

        # Build subsystem matrices
        I = sum(self.size(eq['bases']) for eq in eqs)
        J = sum(self.size(var.bases) for var in vars)
        matrices = {name: sparse.lil_matrix((I, J)) for name in names}

        # Add subsystem entries
        for name, matrix in matrices.items():
            rows = []
            cols = []
            data = []

            i0 = i1 = 0
            for eq in eqs:
                expr = eq[name][0]
                if expr != 0:
                    op_dict = expr.operator_dict(self, vars, **problem.ncc_kw)
                    j0 = 0
                    for var in vars:
                        if var in op_dict:
                            varmat = op_dict[var].tocoo()
                            rows.append(varmat.row + i0)
                            cols.append(varmat.col + j0)
                            data.append(varmat.data)
                        j0 += self.size(var.bases)
                i0 += self.size(eq['bases'])

            rows = np.concatenate(rows)
            cols = np.concatenate(cols)
            data = np.concatenate(data)
            matrices[name] = sparse.coo_matrix((data, (rows, cols)), shape=(I,J)).tocsr()

        # Store minimal CSR matrices for fast dot products
        for name, matrix in matrices.items():
            matrix.eliminate_zeros()
            setattr(self, name, matrix.tocsr())

        F_conv = [self.compute_conversion(eq['F'].bases, eq['bases']) for eq in eqs]
        self.RHS_C = sparse.block_diag(F_conv).tocsr()

        # Store expanded CSR matrices for fast combination
        self.LHS = zeros_with_pattern(*matrices.values()).tocsr()
        for name, matrix in matrices.items():
            matrix = expand_pattern(matrix, self.LHS)
            setattr(self, name+'_exp', matrix.tocsr())

        # Store RHS conversion matrix


