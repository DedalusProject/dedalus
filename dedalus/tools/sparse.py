"""
Tools for working with sparse matrices.
"""

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

from .config import config
from ..libraries.matsolvers import matsolvers

def scipy_sparse_eigs(A, B, N, target, **kw):
    """
    Perform targeted eigenmode search using the scipy/ARPACK sparse solver
    for the reformulated generalized eigenvalue problem

        A.x = λ B.x  ==>  (A - σB)^I B.x = (1/(λ-σ)) x

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
    matsolver = matsolvers[config['linear algebra']['MATRIX_SOLVER'].lower()]

    # Build sparse linear operator representing (A - σB)^I B = C^I B = D
    C = A - target * B
    solver = matsolver(C)
    def matvec(x):
        return solver.solve(B.dot(x))
    D = spla.LinearOperator(dtype=A.dtype, shape=A.shape, matvec=matvec)
    # Solve using scipy sparse algorithm
    evals, evecs = spla.eigs(D, k=N, which='LM', sigma=None, **kw)
    # Rectify eigenvalues
    evals = 1 / evals + target
    return evals, evecs


def same_dense_block_diag(blocks, format=None, dtype=None):
    """
    Build a block diagonal sparse matrix from identically shaped dense blocks.

    Parameters
    ----------
    blocks : sequence of 2D ndarrays
        Input matrix blocks.
    format : str, optional
        The sparse format of the result (e.g. "csr").  If not given, the matrix
        is returned in "coo" format.
    dtype : dtype specifier, optional
        The data-type of the output matrix.  If not given, the dtype is
        determined from that of `blocks`.

    Returns
    -------
    res : sparse matrix
    """
    N = len(blocks)
    I, J = blocks[0].shape
    # Build coordinate arrays
    nloc = np.arange(N)[:, None, None]
    iloc = np.arange(I)[None, :, None]
    jloc = np.arange(J)[None, None, :]
    rows = (I*nloc + iloc + 0*jloc).ravel()
    cols = (J*nloc + 0*iloc + jloc).ravel()
    # Combine blocks
    data = np.array(blocks).ravel()
    # Build COO matrix
    res = sparse.coo_matrix((data, (rows, cols)), shape=(N*I, N*J), dtype=dtype).asformat(format)
    res.eliminate_zeros()
    return res

