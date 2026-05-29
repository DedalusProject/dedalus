

import numpy as np
import scipy.sparse as sparse


def fast_kron_coo_data(A_shape, A_data, A_row, A_col, B_shape, B_data, B_row, B_col):
    """Kronecker product of two matrices in COO data without scipy types."""
    C_shape = (A_shape[0]*B_shape[0], A_shape[1]*B_shape[1])
    C_data = (A_data[:, None] * B_data).ravel()
    C_row  = (A_row[:, None] * B_shape[0] + B_row).ravel()
    C_col  = (A_col[:, None] * B_shape[1] + B_col).ravel()
    return C_shape, C_data, C_row, C_col


def fast_kron_sparse_sparse(A, B, return_data=False):
    """Kronecker product of two sparse matrices, optionally returning COO data."""
    A = A.tocoo()
    B = B.tocoo()
    C_shape, C_data, C_row, C_col = fast_kron_coo_data(A.shape, A.data, A.row, A.col, B.shape, B.data, B.row, B.col)
    if return_data:
        return C_shape, C_data, C_row, C_col
    else:
        return sparse.coo_matrix((C_data, (C_row, C_col)), shape=C_shape)


def fast_kron_dense_sparse(A, B, return_data=False):
    """Kronecker product of a dense and a sparse matrix, optionally returning COO data."""
    A = np.asarray(A)
    A_data = A.ravel()
    A_row = np.repeat(np.arange(A.shape[0]), A.shape[1])
    A_col = np.tile(np.arange(A.shape[1]), A.shape[0])
    B = B.tocoo()
    C_shape, C_data, C_row, C_col = fast_kron_coo_data(A.shape, A_data, A_row, A_col, B.shape, B.data, B.row, B.col)
    if return_data:
        return C_shape, C_data, C_row, C_col
    else:
        return sparse.coo_matrix((C_data, (C_row, C_col)), shape=C_shape)

