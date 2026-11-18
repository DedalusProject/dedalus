

import numpy as np
from scipy import sparse

from . import jacobi
from .cache import CachedFunction
from .general import DeferredTuple


def scalar_clenshaw(c, A, B, f0):
    """
    Clenshaw algorithm on scalar coefficients, array argument:
        S(x) = sum_n c_n f_n(x)
    """
    N = len(c)
    # Clenshaw
    b0, b1 = 0, 0
    for n in reversed(range(N)):
        b1, b2 = b0, b1
        b0 = c[n] + (A[n] * b1) + (B[n+1] * b2)
    return (b0 * f0)


def matrix_clenshaw(c, A, B, f0, cutoff):
    """
    Clenshaw algorithm on scalar coefficients, matrix argument:
        S(X) = sum_n c_n f_n(X)
    """
    N = len(c)
    I = sparse.identity(f0.shape[0])
    # Clenshaw
    b0, b1 = 0*I, 0*I
    for n in reversed(range(N)):
        b1, b2 = b0, b1
        if abs(c[n]) > cutoff:
            b0 = (c[n] * I) + (A[n] @ b1) + (B[n+1] @ b2)
        else:
            b0 = (A[n] @ b1) + (B[n+1] @ b2)
    return (f0 @ b0)


def kronecker_clenshaw(val_c, norm_c, A, B, f0, cutoff, coeffs_left=True):
    """
    Clenshaw algorithm on matrix coefficients, matrix argument:
        S(X) = sum_n kron(f_n(X), c_n)
    """
    def kron(X, C):
        if coeffs_left:
            return sparse.kron(C, X)
        else:
            return sparse.kron(X, C)
    if np.isscalar(val_c[0]):
        return matrix_clenshaw(val_c, A, B, f0, cutoff)
    N = len(norm_c)
    I0 = sparse.identity(f0.shape[0])
    I1 = sparse.identity(val_c[0].shape[0])
    # Clenshaw
    b0, b1 = 0*kron(I0, val_c[0]), 0*kron(I0, val_c[0])
    for n in reversed(range(N)):
        b1, b2 = b0, b1
        b0 = (kron(A[n], I1) @ b1) + (kron(B[n+1], I1) @ b2)
        if norm_c[n] > cutoff:
            b0 += kron(I0, val_c[n])
    return (kron(f0, I1) @ b0)


def jacobi_recursion(N, a, b, X):
    """
    Build Clenshaw recurrence coefficients for Jacobi polynomials.

    Notes
    -----
    Jacobi matrix recursion:
        J[n,n-1]*f[n-1] + J[n,n]*f[n] + J[n,n+1]*f[n+1] = X*f[n]
        f[n+1] = (X - J[n,n])/J[n,n+1]*f[n] - J[n,n-1]/J[n,n+1]*f[n-1]
    Clenshaw coefficients:
        A[n] = (X - J[n,n])/J[n,n+1]
        B[n] = - J[n,n-1]/J[n,n+1]
    """
    # Jacobi matrix
    J = jacobi.jacobi_matrix(N, a, b)
    JA = J.toarray()
    # Identity element
    if np.isscalar(X):
        I = 1
    else:
        X = X.tocsr()
        I = sparse.identity(X.shape[0], format='csr')
    # Clenshaw coefficients
    def compute_A(n):
        if 0 <= n < (N-1):
            return (X - JA[n,n]*I) / JA[n,n+1]
        else:
            return 0*I
    def compute_B(n):
        if 0 < n < (N-1):
            return (-JA[n,n-1] / JA[n,n+1]) * I
        else:
            return 0*I
    A = DeferredTuple(compute_A, N+1)
    B = DeferredTuple(compute_B, N+1)
    return A, B


@CachedFunction
def jacobi_matrix_bands(N, a, b):
    J = jacobi.jacobi_matrix(N, a, b).todia()
    rows = []
    for offset in [0, 1, -1]:
        if offset in J.offsets:
            rows.append(J.data[np.where(J.offsets == offset)[0][0]])
        else:
            rows.append(np.zeros(N))
    return np.array(rows)


@CachedFunction
def jacobi_matrix_dense(N, a, b):
    J = jacobi.jacobi_matrix(N, a, b).toarray()
    return J


def jacobi_recursion_entry(N, n, a, b, X):
    JB = jacobi_matrix_bands(N, a, b)
    Jnn = JB[0, n]  # J[n,n]
    Jnp = JB[1, n+1]  # J[n,n+1]
    Jnm = JB[-1, n-1]  # J[n,n-1]
    # Identity element
    if np.isscalar(X):
        I = 1
    else:
        X = X.tocsr()
        I = sparse.identity(X.shape[0], format='csr')
    # Clenshaw coefficients
    if 0 <= n < (N-1):
        An = (X - Jnn*I) / Jnp
    else:
        An = 0*I
    if 0 < n < (N-1):
        Bn = (-Jnm / Jnp) * I
    else:
        Bn = 0*I
    return An, Bn

from ..libraries import dedalus_sphere


@CachedFunction
def jacobi_ncc_matrix_csr(a_ncc, b_ncc, n, a_arg, b_arg, N):
    """
    Recursively compute a Jacobi polynomial of a Jacobi matrix.
    Jacobi polynomial: params (a_ncc, b_ncc) and degree n.
    Jacobi matrix: params (a_arg, b_arg) and size N.

    Parameters
    ----------
    a_ncc, b_ncc : float
        Jacobi polynomial parameters
    n : int
        Jacobi polynomial degree
    a_arg, b_arg : float
        Jacobi matrix parameters
    N : int
        Jacobi matrix size

    Notes
    -----
    Computed via forward recursion on Jacobi matrix argument.
        P[n+1] = A[n]*P[n] +B[n]*P[n-1]
    """
    if n == 0:
        I = sparse.identity(N, dtype=np.float64, format='csr')
        f0 = dedalus_sphere.jacobi.polynomials(1, a_ncc, b_ncc, 1)[0]
        return I * f0
    elif n == 1:
        Z = dedalus_sphere.jacobi.operator('Z')(N, a_arg, b_arg).square
        A1, B1 = jacobi_recursion_entry(N, n-1, a_ncc, b_ncc, Z)
        f0 = dedalus_sphere.jacobi.polynomials(1, a_ncc, b_ncc, 1)[0]
        return A1 * f0
    else:
        f1 = jacobi_ncc_matrix_csr(a_ncc, b_ncc, n-1, a_arg, b_arg, N)
        f2 = jacobi_ncc_matrix_csr(a_ncc, b_ncc, n-2, a_arg, b_arg, N)
        Z = dedalus_sphere.jacobi.operator('Z')(N, a_arg, b_arg).square.tocsr()
        A1, B1 = jacobi_recursion_entry(N, n-1, a_ncc, b_ncc, Z)
        return A1 @ f1 + B1 @ f2


@CachedFunction
def jacobi_ncc_matrix_coo(*args):
    return jacobi_ncc_matrix_csr(*args).tocoo()

