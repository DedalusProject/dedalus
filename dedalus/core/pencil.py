"""
Classes for manipulating pencils.

"""

from functools import partial
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
    start = domain.distributor.coeff_layout.start(scales)[:-1]
    for index in indices:
        pencils.append(Pencil(domain, index, start+index))

    return pencils


def build_matrices(pencils, problem, matrices):
    """Build pencil matrices."""
    # Build new cachid for NCC expansions
    cacheid = uuid.uuid4()
    # Build test operator dicts to synchronously expand all NCCs
    for eq in problem.eqs+problem.bcs:
        for matrix in matrices:
            expr, vars = eq[matrix]
            if expr != 0:
                test_index = [0] * problem.domain.dim
                expr.operator_dict(test_index, vars, cacheid=cacheid, **problem.ncc_kw)
    # Build matrices
    for pencil in log_progress(pencils, logger, 'info', desc='Building pencil matrix', iter=np.inf, frac=0.1, dt=10):
        pencil.build_matrices(problem, matrices, cacheid=cacheid)


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
        if domain.bases[-1].coupled:
            self.build_matrices = self._build_coupled_matrices
        else:
            self.build_matrices = self._build_uncoupled_matrices

    def _build_uncoupled_matrices(self, problem, names, cacheid=None):

        matrices = {name: [] for name in (names+['select'])}

        zbasis = self.domain.bases[-1]
        dtype = zbasis.coeff_dtype
        for last_index in range(zbasis.coeff_size):
            submatrices = self._build_uncoupled_submatrices(problem, names, last_index, cacheid=cacheid)
            for name in matrices:
                matrices[name].append(submatrices[name])

        for name in matrices:
            blocks = matrices[name]
            matrix = sparse.block_diag(blocks, format='csr', dtype=dtype)
            matrix.eliminate_zeros()
            matrices[name] = matrix

        # Store minimal CSR matrices for fast dot products
        for name in names:
            matrix = matrices[name]
            # Store full matrix
            matrix.eliminate_zeros()
            setattr(self, name+'_full', matrix.tocsr().copy())
            # Store truncated matrix
            matrix.data[np.abs(matrix.data) < problem.entry_cutoff] = 0
            matrix.eliminate_zeros()
            setattr(self, name, matrix.tocsr().copy())


        # Store expanded CSR matrices for fast combination
        self.LHS = zeros_with_pattern(*[matrices[name] for name in names]).tocsr()
        for name in names:
            matrix = matrices[name]
            matrix = expand_pattern(matrix, self.LHS)
            setattr(self, name+'_exp', matrix.tocsr().copy())

        # Store operators for RHS
        self.G_eq = matrices['select']
        self.G_bc = None

        # no Dirichlet
        self.dirichlet = None

    def _build_uncoupled_submatrices(self, problem, names, last_index, cacheid=None):

        index = list(self.global_index) + [last_index]
        index_dict = {}
        for axis, basis in enumerate(self.domain.bases):
            index_dict['n'+basis.name] = index[axis]

        # Find applicable equations
        selected_eqs = [eq for eq in problem.eqs if eval(eq['raw_condition'], index_dict)]
        # Check selections
        nvars = problem.nvars
        neqs = len(selected_eqs)
        Neqs = len(problem.eqs)
        if neqs != nvars:
            raise ValueError("Pencil {} has {} equations for {} variables.".format(index, neqs, nvars))

        # Build matrices
        dtype = self.domain.bases[-1].coeff_dtype
        matrices = {name: np.zeros((nvars, nvars), dtype=dtype) for name in names}
        matrices['select'] = np.zeros((nvars, Neqs))
        for i, eq in enumerate(selected_eqs):
            j = problem.eqs.index(eq)
            matrices['select'][i,j] = 1
            for name in names:
                expr, vars = eq[name]
                if expr != 0:
                    op_dict = expr.operator_dict(index, vars, cacheid=cacheid, **problem.ncc_kw)
                    matrix = matrices[name]
                    for j in range(nvars):
                        matrix[i,j] = op_dict[vars[j]]

        return matrices

    def _build_coupled_matrices(self, problem, names, cacheid=None):

        index_dict = {}
        for axis, basis in enumerate(self.domain.bases):
            if basis.separable:
                index_dict['n'+basis.name] = self.global_index[axis]

        index = self.global_index
        # Find applicable equations
        selected_eqs = [eq for eq in problem.eqs if eval(eq['raw_condition'], index_dict)]
        selected_bcs = [bc for bc in problem.bcs if eval(bc['raw_condition'], index_dict)]
        ndiff = sum(eq['differential'] for eq in selected_eqs)
        # Check selections
        nvars = problem.nvars
        neqs = len(selected_eqs)
        nbcs = len(selected_bcs)
        if neqs != nvars:
            raise ValueError("Pencil {} has {} equations for {} variables.".format(index, neqs, nvars))
        if nbcs != ndiff:
            raise ValueError("Pencil {} has {} boundary conditions for {} differential equations.".format(index, nbcs, ndiff))
        Neqs = len(problem.eqs)
        Nbcs = len(problem.bcs)

        zbasis = self.domain.bases[-1]
        zsize = zbasis.coeff_size
        zdtype = zbasis.coeff_dtype
        compound = hasattr(zbasis, 'subbases')
        self.dirichlet = dirichlet = any(problem.meta[:][zbasis.name]['dirichlet'])

        # Identity
        Identity = sparse.identity(zsize, dtype=zdtype).tocsr()
        Zero = sparse.csr_matrix((zsize, zsize), dtype=zdtype)

        # Basis matrices
        Ra = Rd = Identity
        if dirichlet:
            Rd = basis.PrefixBoundary
        if ndiff:
            P = basis.Precondition
            Fb = basis.FilterBoundaryRow
            Cb = basis.ConstantToBoundary
            Rd_Fb_P = Rd*Fb*P
            Rd_Cb = Rd*Cb
        if compound:
            Fm = basis.FilterMatchRows
            M = basis.Match
            Ra_Fm = Ra*Fm
        if ndiff and compound:
            Rd_Fm_Fb_P = Rd*Fm*Fb*P

        # Pencil matrices
        G_eq = sparse.csr_matrix((zsize*nvars, zsize*Neqs), dtype=zdtype)
        G_bc = sparse.csr_matrix((zsize*nvars, zsize*Nbcs), dtype=zdtype)
        C = lambda : sparse.csr_matrix((zsize*nvars, zsize*nvars), dtype=zdtype)
        LHS = {name: C() for name in names}

        # Kronecker stencils
        δG_eq = np.zeros([nvars, Neqs])
        δG_bc = np.zeros([nvars, Nbcs])
        δC = np.zeros([nvars, nvars])

        # Use scipy sparse kronecker product with CSR output
        kron = partial(sparse.kron, format='csr')

        # Build matrices
        bc_iter = iter(selected_bcs)
        for i, eq in enumerate(selected_eqs):

            differential = eq['differential']
            if differential:
                bc = next(bc_iter)

            # Build RHS equation process matrix
            if (not differential) and (not compound):
                Gi_eq = Ra
            elif (not differential) and compound:
                Gi_eq = Ra_Fm
            elif differential and (not compound):
                Gi_eq = Rd_Fb_P
            elif differential and compound:
                Gi_eq = Rd_Fm_Fb_P

            # Kronecker into system matrix
            e = problem.eqs.index(eq)
            δG_eq[i,e] = 1
            G_eq = G_eq + kron(Gi_eq, δG_eq)
            δG_eq[i,e] = 0

            if differential:
                # Build RHS BC process matrix
                Gi_bc = Rd_Cb
                # Kronecker into system matrix
                b = problem.bcs.index(bc)
                δG_bc[i,b] = 1
                G_bc = G_bc + kron(Gi_bc, δG_bc)
                δG_bc[i,b] = 0

            # Build LHS matrices
            for name in names:
                C = LHS[name]
                eq_expr, eq_vars = eq[name]
                if eq_expr != 0:
                    Ei = eq_expr.operator_dict(self.global_index, eq_vars, cacheid=cacheid, **problem.ncc_kw)
                else:
                    Ei = defaultdict(int)
                if differential:
                    bc_expr, bc_vars = bc[name]
                    if bc_expr != 0:
                        Bi = bc_expr.operator_dict(self.global_index, bc_vars, cacheid=cacheid, **problem.ncc_kw)
                    else:
                        Bi = defaultdict(int)
                for j in range(nvars):
                    # Build equation terms
                    Eij = Ei[eq_vars[j]]
                    if Eij is 0:
                        Eij = None
                    elif Eij is 1:
                        Eij = Gi_eq
                    else:
                        Eij = Gi_eq*Eij
                    # Build BC terms
                    if differential:
                        Bij = Bi[bc_vars[j]]
                        if Bij is 0:
                            Bij = None
                        elif Bij is 1:
                            Bij = Gi_bc
                        else:
                            Bij = Gi_bc*Bij
                    else:
                        Bij = None
                    # Combine equation and BC
                    if (Eij is None) and (Bij is None):
                        continue
                    elif Eij is None:
                        Cij = Bij
                    elif Bij is None:
                        Cij = Eij
                    else:
                        Cij = Eij + Bij
                    # Kronecker into system
                    δC[i,j] = 1
                    C = C + kron(Cij, δC)
                    δC[i,j] = 0
                LHS[name] = C

        if compound and 'L' in names:
            # Add match terms
            L = LHS['L']
            δM = np.identity(nvars)
            L = L + kron(Ra*M, δM)
            LHS['L'] = L

        if dirichlet:
            # Build right-preconditioner for system
            δD = np.zeros([nvars, nvars])
            D = 0
            for i, var in enumerate(problem.variables):
                if problem.meta[var][zbasis.name]['dirichlet']:
                    Dii = zbasis.Dirichlet
                else:
                    Dii = Identity
                δD[i,i] = 1
                D = D + kron(Dii, δD)
                δD[i,i] = 0
            self.JD = D.tocsr()
            self.JD.eliminate_zeros()

        # Store minimum CSR matrices for fast dot products
        for name, matrix in LHS.items():
            # Store full matrix
            matrix.eliminate_zeros()
            setattr(self, name+'_full', matrix.tocsr().copy())
            # Store truncated matrix
            matrix.data[np.abs(matrix.data) < problem.entry_cutoff] = 0
            matrix.eliminate_zeros()
            setattr(self, name, matrix.tocsr().copy())

        # Apply Dirichlet recombination if applicable
        if dirichlet:
            for name in names:
                LHS[name] = LHS[name] * self.JD

        # Store expanded CSR matrices for fast combination
        self.LHS = zeros_with_pattern(*LHS.values()).tocsr()
        for name, matrix in LHS.items():
            matrix = expand_pattern(matrix, self.LHS)
            setattr(self, name+'_exp', matrix.tocsr().copy())

        # Store operators for RHS
        G_eq.eliminate_zeros()
        self.G_eq = G_eq
        G_bc.eliminate_zeros()
        self.G_bc = G_bc
