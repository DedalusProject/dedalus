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
            raise ValueError("Pencil {} has {} equations for {} variables.".format(index, neqs, nvars))
        if nbcs != ndiff:
            raise ValueError("Pencil {} has {} boudnary conditions for {} differential equations.".format(index, nbcs, ndiff))
        Neqs = len(problem.eqs)
        Nbcs = len(problem.bcs)

        zbasis = self.domain.bases[-1]
        zsize = zbasis.coeff_size
        zdtype = zbasis.coeff_dtype
        compound = hasattr(zbasis, 'subbases')

        # Copy problem namespace
        namespace = problem.coefficient_namespace.copy()
        # Separable basis operators
        for axis, basis in enumerate(problem.domain.bases):
            if basis.separable:
                for op in basis.operators:
                    try:
                        namespace[op.name] = op.scalar_form(index[axis])
                    except AttributeError:
                        pass
        # Identity
        namespace['Identity'] = sparse.identity(zsize, dtype=zdtype).tocsr()

        # Basis matrices
        R = namespace['Identity'] #basis.Rearrange
        if ndiff:
            P = basis.Precondition
            Fb = basis.FilterBoundaryRow
            Cb = basis.ConstantToBoundary
            R_Fb_P = R*Fb*P
            R_Cb = R*Cb
        if compound:
            Fm = basis.FilterMatchRows
            M = basis.Match

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
                Gi_eq = R
            elif (not differential) and compound:
                Gi_eq = R_Fm
            elif differential and (not compound):
                Gi_eq = R_Fb_P
            elif differential and compound:
                Gi_eq = R*Fm*Fb*P
            # Kronecker into system matrix
            Gi_eq.eliminate_zeros()
            δG_eq.fill(0); δG_eq[i, problem.eqs.index(eq)] = 1
            G_eq = G_eq + kron(Gi_eq, δG_eq)

            if differential:
                # Build RHS BC process matrix
                Gi_bc = R_Cb
                # Kronecker into system matrix
                Gi_bc.eliminate_zeros()
                δG_bc.fill(0); δG_bc[i, problem.bcs.index(bc)] = 1
                G_bc = G_bc + kron(Gi_bc, δG_bc)

            # Build LHS matrices
            for name in LHS:
                C = LHS[name]
                for j in range(nvars):
                    # Add equation terms
                    Eij = eval(eq[name][j], namespace)
                    Cij = Gi_eq*Eij
                    if differential:
                        # Add BC terms
                        Bij = eval(bc[name][j], namespace)
                        Cij = Cij + Gi_bc*Bij
                    # Kronecker into system matrix
                    Cij.eliminate_zeros()
                    if Cij.nnz:
                        δC.fill(0); δC[i, j] = 1
                        C = C + kron(Cij, δC)
                LHS[name] = C

        if compound:
            # Add match terms
            L = LHS['L']
            δM = np.identity(nvars)
            L = L + kron(R*M, δM)

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
