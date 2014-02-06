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
    index_list = []
    if domain.dim == 1:
        index_list.append([])
    else:
        trans_shape = domain.distributor.coeff_layout.shape[:-1]
        div = np.arange(np.prod(trans_shape))
        for s in reversed(trans_shape):
            div, mod = divmod(div, s)
            index_list.append(mod)
        index_list = list(zip(*reversed(index_list)))

    # Construct corresponding trans diff consts and build pencils
    pencils = []
    start = domain.distributor.coeff_layout.start
    for index in index_list:
        trans = []
        for i, b in enumerate(domain.bases[:-1]):
            trans.append(b.trans_diff(start[i]+index[i]))
        pencils.append(Pencil(index, trans))

    return pencils


class Pencil:
    """
    Object holding problem matrices for a given transverse wavevector.

    Parameters
    ----------
    index : tuple of ints
        Transverse indeces for retrieving pencil from system data
    trans :  tuple of floats
        Transverse differentiation constants

    """

    def __init__(self, index, trans):

        # Initial attributes
        # Save index as tuple for proper array indexing behavior
        self.index = tuple(index)
        self.trans = trans

    def build_matrices(self, problem, basis):
        """Construct pencil matrices from problem and basis matrices."""

        # References
        size = problem.nfields * basis.coeff_size
        dtype = basis.coeff_dtype

        # Get and unpack problem matrices
        eqn_mat, bc_mat, S_mat = problem.build_matrices(self.trans)
        M0e, M1e, L0e, L1e = eqn_mat
        M0b, M1b, L0b, L1b = bc_mat
        Se, Sl, Sr, Si = S_mat

        # Use scipy sparse kronecker product with CSR output
        kron = partial(sparse.kron, format='csr')

        # Allocate PDE matrices
        Me = sparse.csr_matrix((size, size), dtype=dtype)
        Le = sparse.csr_matrix((size, size), dtype=dtype)

        # Add terms to PDE matrices
        for p in range(problem.order):
            PM  = basis.Pre * basis.Mult(p)
            PMD = basis.Pre * basis.Mult(p) * basis.Diff
            Me = Me + kron(PM,  Se*M0e[p])
            Me = Me + kron(PMD, Se*M1e[p])
            Le = Le + kron(PM,  Se*L0e[p])
            Le = Le + kron(PMD, Se*L1e[p])

        # Allocate BC matrices
        Mb = sparse.csr_matrix((size, size), dtype=dtype)
        Lb = sparse.csr_matrix((size, size), dtype=dtype)

        # Add terms to BC matrices
        if Sl.any():
            for p in range(problem.order):
                LM  = basis.Left * basis.Mult(p)
                LMD = basis.Left * basis.Mult(p) * basis.Diff
                Mb = Mb + kron(LM,  Sl*M0b[p])
                Mb = Mb + kron(LMD, Sl*M1b[p])
                Lb = Lb + kron(LM,  Sl*L0b[p])
                Lb = Lb + kron(LMD, Sl*L1b[p])
        if Sr.any():
            for p in range(problem.order):
                RM  = basis.Right * basis.Mult(p)
                RMD = basis.Right * basis.Mult(p) * basis.Diff
                Mb = Mb + kron(RM,  Sr*M0b[p])
                Mb = Mb + kron(RMD, Sr*M1b[p])
                Lb = Lb + kron(RM,  Sr*L0b[p])
                Lb = Lb + kron(RMD, Sr*L1b[p])
        if Si.any():
            for p in range(problem.order):
                IM  = basis.Int * basis.Mult(p)
                IMD = basis.Int * basis.Mult(p) * basis.Diff
                Mb = Mb + kron(IM,  Si*M0b[p])
                Mb = Mb + kron(IMD, Si*M1b[p])
                Lb = Lb + kron(IM,  Si*L0b[p])
                Lb = Lb + kron(IMD, Si*L1b[p])

        # Build filter matrix to eliminate boundary condition rows
        Mb_rows = Mb.nonzero()[0]
        Lb_rows = Lb.nonzero()[0]
        bc_rows = set(Mb_rows).union(set(Lb_rows))
        F = sparse.eye(size, dtype=dtype, format='dok')
        for i in bc_rows:
            F[i, i] = 0
        F = F.tocsr()

        # Combine filtered PDE matrices with BC matrices
        M = F*Me + Mb
        L = F*Le + Lb

        # Store with expanded sparsity for fast combination during timestepping
        self.LHS = zeros_with_pattern(M, L).tocsr()
        self.M = expand_pattern(M, self.LHS).tocsr()
        self.L = expand_pattern(L, self.LHS).tocsr()

        # Store selection/restriction matrices for RHS
        # Start G_bc with integral term, since the Int matrix is always defined
        self.G_eq = F * kron(basis.Pre, Se)
        self.G_bc = kron(basis.Int, Si)
        if Sl.any():
            self.G_bc = self.G_bc + kron(basis.Left, Sl)
        if Sr.any():
            self.G_bc = self.G_bc + kron(basis.Right, Sr)

