"""
Classes for manipulating pencils.

"""

from functools import partial
from collections import defaultdict
import numpy as np
from scipy import sparse
from mpi4py import MPI

from ..tools.array import zeros_with_pattern
from ..tools.array import expand_pattern

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

    def _build_uncoupled_matrices(self, problem, names):

        if problem.bcs:
            raise SymbolicParsingError("Fully periodic domain doesn't support boundary conditions.")

        matrices = {name: [] for name in (names+['select'])}

        zbasis = self.domain.bases[-1]
        dtype = zbasis.coeff_dtype
        for last_index in range(zbasis.coeff_size):
            submatrices = self._build_uncoupled_submatrices(problem, names, last_index)
            for name in matrices:
                matrices[name].append(submatrices[name])

        for name in matrices:
            blocks = matrices[name]
            matrix = sparse.block_diag(blocks, format='csr', dtype=dtype)
            matrix.eliminate_zeros()
            matrices[name] = matrix

        # Store problem matrices with expanded sparsity for fast combination during timestepping
        self.LHS = zeros_with_pattern(*[matrices[name] for name in names]).tocsr()
        for name in names:
            matrix = matrices[name]
            matrix = expand_pattern(matrix, self.LHS).tocsr()
            setattr(self, name, matrix)

        # Store operators for RHS
        self.G_eq = matrices['select']
        self.G_bc = None

    def _build_uncoupled_submatrices(self, problem, names, last_index):

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
        vars = [problem.namespace[var] for var in problem.variables]
        for i, eq in enumerate(selected_eqs):
            j = problem.eqs.index(eq)
            matrices['select'][i,j] = 1
            for name in names:
                if eq[name] != 0:
                    op_dict = eq[name].operator_dict(index, vars, **problem.ncc_kw)
                    matrix = matrices[name]
                    for j in range(nvars):
                        matrix[i,j] = op_dict[vars[j]]

        return matrices

    def _build_coupled_matrices(self, problem, names):

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

        # Identity
        Identity = sparse.identity(zsize, dtype=zdtype).tocsr()
        Zero = sparse.csr_matrix((zsize, zsize), dtype=zdtype)

        # Basis matrices
        R = Identity #basis.Rearrange
        if ndiff:
            P = basis.Precondition
            Fb = basis.FilterBoundaryRow
            Cb = basis.ConstantToBoundary
            R_Fb_P = R*Fb*P
            R_Cb = R*Cb
        if compound:
            Fm = basis.FilterMatchRows
            M = basis.Match
            R_Fm = R*Fm
        if ndiff and compound:
            R_Fm_Fb_P = R_Fm*Fb*P

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

        vars = [problem.namespace[var] for var in problem.variables]

        # Build matrices
        bc_iter = iter(selected_bcs)
        for i, eq in enumerate(selected_eqs):

            differential = eq['differential']
            if differential:
                bc = next(bc_iter)

            # Build RHS equation process matrix
            if (not differential) and (not compound):
                Gi_eq = R
            elif (not differential) and compound:
                Gi_eq = R_Fm
            elif differential and (not compound):
                Gi_eq = R_Fb_P
            elif differential and compound:
                Gi_eq = R_Fm_Fb_P

            # Kronecker into system matrix
            e = problem.eqs.index(eq)
            δG_eq[i,e] = 1
            G_eq = G_eq + kron(Gi_eq, δG_eq)
            δG_eq[i,e] = 0

            if differential:
                # Build RHS BC process matrix
                Gi_bc = R_Cb
                # Kronecker into system matrix
                b = problem.bcs.index(bc)
                δG_bc[i,b] = 1
                G_bc = G_bc + kron(Gi_bc, δG_bc)
                δG_bc[i,b] = 0

            # Build LHS matrices
            for name in names:
                C = LHS[name]
                if eq[name] != 0:
                    Ei = eq[name].operator_dict(self.global_index, vars, **problem.ncc_kw)
                else:
                    Ei = defaultdict(int)
                if differential:
                    if bc[name] != 0:
                        Bi = bc[name].operator_dict(self.global_index, vars, **problem.ncc_kw)
                    else:
                        Bi = defaultdict(int)
                for j in range(nvars):
                    # Build equation terms
                    Eij = Ei[vars[j]]
                    if Eij is 0:
                        Eij = None
                    elif Eij is 1:
                        Eij = Gi_eq
                    else:
                        Eij = Gi_eq*Eij
                    # Build BC terms
                    if differential:
                        Bij = Bi[vars[j]]
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

        if compound:
            # Add match terms
            L = LHS['L']
            δM = np.identity(nvars)
            L = L + kron(R*M, δM)
            LHS['L'] = L

        # Store with expanded sparsity for fast combination during timestepping
        for C in LHS.values():
            C.eliminate_zeros()
        self.LHS = zeros_with_pattern(*LHS.values()).tocsr()
        for name, C in LHS.items():
            C = expand_pattern(C, self.LHS).tocsr()
            setattr(self, name, C)

        # Store operators for RHS
        G_eq.eliminate_zeros()
        self.G_eq = G_eq
        G_bc.eliminate_zeros()
        self.G_bc = G_bc
