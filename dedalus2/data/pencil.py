"""
Classes for manipulating pencils.

"""

from functools import partial
import numpy as np
from scipy import sparse

from ..tools.array import zeros_with_pattern
from ..tools.array import expand_pattern


class PencilSet:
    """
    Set of pencils in a domain.

    Parameters
    ----------
    domain : domain object
        Problem domain

    Attributes
    ----------
    pencils : list of pencil objects
        Individual pencils

    """

    def __init__(self, domain):

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
        self.pencils = []
        start = domain.distributor.coeff_layout.start
        for index in index_list:
            trans = []
            for i, b in enumerate(domain.bases[:-1]):
                trans.append(b.trans_diff(start[i]+index[i]))
            self.pencils.append(Pencil(index, trans))


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
        for i in range(problem.order):
            if i > 0:
                raise ValueError("Nonconstant coefficients not supported.")
            PM  = basis.Pre * basis.Mult(i)
            PMD = basis.Pre * basis.Mult(i) * basis.Diff
            Me = Me + kron(PM,  Se*M0e)
            Me = Me + kron(PMD, Se*M1e)
            Le = Le + kron(PM,  Se*L0e)
            Le = Le + kron(PMD, Se*L1e)

        # Allocate BC matrices
        Mb = sparse.csr_matrix((size, size), dtype=dtype)
        Lb = sparse.csr_matrix((size, size), dtype=dtype)

        # Add terms to BC matrices
        if Sl.any():
            for i in range(problem.order):
                LM  = basis.Left * basis.Mult(i)
                LMD = basis.Left * basis.Mult(i) * basis.Diff
                Mb = Mb + kron(LM,  Sl*M0b)
                Mb = Mb + kron(LMD, Sl*M1b)
                Lb = Lb + kron(LM,  Sl*L0b)
                Lb = Lb + kron(LMD, Sl*L1b)
        if Sr.any():
            for i in range(problem.order):
                RM  = basis.Right * basis.Mult(i)
                RMD = basis.Right * basis.Mult(i) * basis.Diff
                Mb = Mb + kron(RM,  Sr*M0b)
                Mb = Mb + kron(RMD, Sr*M1b)
                Lb = Lb + kron(RM,  Sr*L0b)
                Lb = Lb + kron(RMD, Sr*L1b)
        if Si.any():
            for i in range(problem.order):
                IM  = basis.Int * basis.Mult(i)
                IMD = basis.Int * basis.Mult(i) * basis.Diff
                Mb = Mb + kron(IM,  Si*M0b)
                Mb = Mb + kron(IMD, Si*M1b)
                Lb = Lb + kron(IM,  Si*L0b)
                Lb = Lb + kron(IMD, Si*L1b)

        # Build filter matrix to eliminate boundary condition rows
        Mb_rows = Mb.nonzero()[0]
        Lb_rows = Lb.nonzero()[0]
        rows = set(Mb_rows).union(set(Lb_rows))
        F = sparse.eye(size, dtype=dtype, format='dok')
        for i in rows:
            F[i, i] = 0.
        F = F.tocsr()

        # Combine filtered PDE matrices with BC matrices
        M = F*Me + Mb
        L = F*Le + Lb

        # Store with expanded sparsity for fast combination during integration
        self.LHS = zeros_with_pattern(M, L).tocsr()
        self.M = expand_pattern(M, self.LHS).tocsr()
        self.L = expand_pattern(L, self.LHS).tocsr()

        # Store selection/restriction operators for RHS
        # Start G_bc with integral term, since that matrix is always defined
        self.G_eq = F * kron(basis.Pre, Se)
        self.G_bc = kron(basis.Int, Si)
        if Sl.any():
            self.G_bc = self.G_bc + kron(L, Sl)
        if Sr.any():
            self.G_bc = self.G_bc + kron(R, Sr)

