"""
Classes for manipulating pencils.

"""

from functools import partial
import numpy as np
from scipy import sparse

from ..tools.array import zeros_with_pattern
from ..tools.array import expand_pattern


def build_pencils(domain):
    """
    Create the set of pencils over a domain.

    Parameters
    ----------
    domain : domain object
        Problem domain

    Returns
    -------
    pencils : list
        Pencil objects

    """

    # Get transverse indeces in fastest sequence
    trans_shape = domain.local_coeff_shape[:-1]
    indices = np.ndindex(*trans_shape)

    # Construct corresponding trans diff consts and build pencils
    pencils = []
    scales = domain.remedy_scales(1)
    start = domain.distributor.coeff_layout.start(scales)
    for index in indices:
        pencils.append(Pencil(domain, index, start+index))

    return pencils


class Pencil:
    """
    Object holding problem matrices for a given transverse wavevector.

    Parameters
    ----------
    index : tuple of ints
        Transverse indeces for retrieving pencil from system data

    """

    def __init__(self, domain, local_index, global_index):
        self.domain = domain
        self.local_index = tuple(local_index)
        self.global_index = tuple(global_index)

    def build_matrices(self, problem):
        """Construct pencil matrices from problem and basis matrices."""

        # Separable operator values
        args = []
        for op_root in problem.op_roots:
            for axis in range(len(problem.axis_names)-1):
                op_name = operators.root_dict(op_root).__name__
                op = getattr(self.domain.bases[axis], op_name)
                args.append(op.scalar_form(self.global_index[axis]))
        # Problem operators
        M_eqn, L_eqn, M_bc, L_bc = problem.num_M_eqn, problem.num_L_eqn, problem.num_M_bc, problem.num_L_bc
        # Selection matrices
        Se, Sb, A, D = problem.selection(args)
        # Basis matrices
        P = basis.Preconditioner
        F = basis.TauFilter
        B = basis.Boundary
        # Precompute some dot products
        A_Se = A * Se
        D_Se = D * Se
        P_F = P * F
        # Use scipy sparse kronecker product with CSR output
        kron = partial(sparse.kron, format='csr')

        # Allocate matrices
        size = problem.nfields * basis.coeff_size
        dtype = basis.coeff_dtype
        M = sparse.csr_matrix((size, size), dtype=dtype)
        L = sparse.csr_matrix((size, size), dtype=dtype)
        # Add equation terms to matrices
        for C, C_eqn in ((M, M_eqn), (L, L_eqn)):
            for Qi, Ci in C_eqn:
                Ci = Ci(*args)
                A_Se_Ci = A_Se * Ci
                if A_Se_Ci.any():
                    C = C + kron(Qi, A_Se_Ci)
                D_Se_Ci = D_Se * Ci
                if D_Se_Ci.any():
                    C = C + kron(P_F*Qi, D_Se_Ci)
        # Add boundary condition terms to matrices
        for C, C_bc in ((M, M_bc), (L, L_bc)):
            for Qi, Ci in C_bc:
                Ci = Ci(*args)
                Sb_Ci = Sb * Ci
                if Sb_Ci.any():
                    C = C + kron(Qi, Sb_Ci)
        # Add match terms to matrices
        if isinstance(basis, Compound):
            L = L + kron(basis.Match, D)

        # Store with expanded sparsity for fast combination during timestepping
        self.LHS = zeros_with_pattern(M, L).tocsr()
        self.M = expand_pattern(M, self.LHS).tocsr()
        self.L = expand_pattern(L, self.LHS).tocsr()

        # Store operators for RHS
        In = sparse.identity(basis.coeff_size, dtype=basis.coeff_dtype)
        self.Ge = kron(In, A_Se) + kron(P_F, D_Se)
        self.Gb = kron(B, Sb)

