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
        if domain.bases[-1].coupled:
            self.build_matrices = self._build_coupled_matrices
        else:
            self.build_matrices = self._build_uncoupled_matrices

    def _build_uncoupled_matrices(self, problem, names):
        raise NotImplementedError()

    def _build_coupled_matrices(self, problem, names):

        index = self.global_index
        # Find applicable equations
        selected_eqs = [eq for eq in problem.eqs if eq['condition'](*index)]
        selected_bcs = [bc for bc in problem.bcs if bc['condition'](*index)]
        ndiff = sum(eq['differential'] for eq in selected_eqs)
        # Check selections
        nvars = problem.nvars
        neqs = len(selected_eqs)
        nbcs = len(selected_bcs)
        if neqs != nvars:
            raise ValueError("Pencil {} has {} equations for {} variables.".format(indices, neqs, nvars))
        if nbcs != ndiff:
            raise ValueError("Pencil {} has {} boudnary conditions for {} differential equations.".format(indices, nbcs, ndiff))

        # Basis matrices
        I = sparse.identity(basis.coeff_size, dtype=basis.coeff_dtype)
        R = basis.Rearrange
        if ndiff:
            P = basis.Precondition
            Fb = basis.FilterBoundaryRow
            Cb = basis.ConstantToBoundary
        if compound:
            Fm = basis.FilterMatchRows
            M = basis.Match

        # Pencil matrices
        G_eq = sparse.csr_matrix((coeffs*nvars, coeffs*Neqs), dtype=basis.dtype)
        G_bc = sparse.csr_matrix((coeffs*nvars, coeffs*Nbcs), dtype=basis.dtype)
        C = lambda : sparse.csr_matrix((coeffs*nvars, coeffs*nvars), dtype=basis.dtype)
        LHS_matrices = {name: C() for name in names}

        # Kronecker stencils
        δG_eq = np.zeros([nvars, len(problem.eqs)])
        δG_bc = np.zeros([nvars, len(problem.bcs)])
        δC = np.zeros([nvars, nvars])

        # Use scipy sparse kronecker product with CSR output
        kron = partial(sparse.kron, format='csr')

        # Build matrices
        bc_iter = iter(seleted_bcs)
        for i, eq in enumerate(selected_eqs):

            differential = eq['differential']
            if differential:
                bc = next(bc_iter)

            eq_atoms = [namespace[atom] for atom in eq['atoms']]
            if differential:
                bc_atoms = [namespace[atom] for atom in bc['atoms']]

            # Build RHS equation process matrix
            if (not differential) and (not compound):
                Gi_eq = R
            elif (not differential) and compound:
                Gi_eq = R*Fm
            elif differential and (not compound):
                Gi_eq = R*Fb*P
            elif differential and compound:
                Gi_eq = R*Fm*Fb*P
            # Kronecker into system matrix
            Gi_eq.eliminate_zeros()
            δG_eq.fill(0); δG_eq[i, problem.eqs.index(eq)] = 1
            G_eq = G_eq + kron(Gi_eq, δG_eq)

            if differential:
                # Build RHS BC process matrix
                Gi_bc = R*Cb
                # Kronecker into system matrix
                Gi_bc.eliminate_zeros()
                δG_bc.fill(0); δG_bc[i, problem.bcs.index(bc)] = 1
                G_bc = G_bc + kron(Gi_bc, δG_bc)

            # Build LHS matrices
            for name, C in LHS.items():
                for j in range(nvars):
                    # Add equation terms
                    Eij = eq[name][j](*eq_atoms)
                    Cij = Gi_eq*Eij
                    if differential:
                        # Add BC terms
                        Bij = bc[name][j](*bc_atoms)
                        Cij = Cij + Gi_bc*Bij
                    # Kronecker into system matrix
                    Cij.eliminate_zeros()
                    if Cij.nnz:
                        δC.fill(0); δC[i, j] = 1
                        C = C + kron(Cij, δC)

        if compound:
            # Add match terms
            L = LHS_matrices['L']
            δM = np.identity(nvars)
            L = L + kron(R*M, δM)

        # Store with expanded sparsity for fast combination during timestepping
        for C in LHS.values():
            C.eliminate_zeros()
        self.LHS = zeros_with_pattern(*LHS.values()).tocsr()
        for name, C in LHS.items():
            setattr(self, name, expand_pattern(C, self.LHS).tocsr())

        # Store operators for RHS
        G_eq.eliminate_zeros()
        self.G_eq = G_eq
        G_bc.eliminate_zeros()
        self.G_bc = G_bc
