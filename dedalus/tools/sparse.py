"""
Tools for working with sparse matrices.
"""

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

from .config import config
STORE_LU = config['linear algebra'].getboolean('store_LU')
PERMC_SPEC = config['linear algebra']['permc_spec']
USE_UMFPACK = config['linear algebra'].getboolean('use_umfpack')


def scipy_sparse_eigs(A, B, N, target, **kw):
    """
    Perform targeted eigenmode search using the scipy/ARPACK sparse solver
    for the reformulated generalized eigenvalue problem

        A.x = λ B.x  ==>  (A - σB)\B.x = (1/(λ-σ)) x

    for eigenvalues λ near the target σ.

    Parameters
    ----------
    A, B : scipy sparse matrices
        Sparse matrices for generalized eigenvalue problem
    N : int
        Number of eigenmodes to return
    target : complex
        Target σ for eigenvalue search

    Other keyword options passed to scipy.sparse.linalg.eigs.

    """
    # Build sparse linear operator representing (A - σB)\B = C\B = D
    C = A - target * B
    if STORE_LU:
        C_LU = spla.splu(C.tocsc(), permc_spec=PERMC_SPEC)
        def matvec(x):
            return C_LU.solve(B.dot(x))
    else:
        def matvec(x):
            return spla.spsolve(C, B.dot(x), use_umfpack=USE_UMFPACK, permc_spec=PERMC_SPEC)
    D = spla.LinearOperator(dtype=A.dtype, shape=A.shape, matvec=matvec)
    # Solve using scipy sparse algorithm
    evals, evecs = spla.eigs(D, k=N, which='LM', sigma=None, **kw)
    # Rectify eigenvalues
    evals = 1 / evals + target
    return evals, evecs
