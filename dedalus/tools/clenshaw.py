

import numpy as np
from scipy import sparse

from . import jacobi
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


