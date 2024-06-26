
from scipy import sparse
import numpy as np
from . import jacobi

# called from basis of ncc (i.e., r)
# A, B are from basis that the ncc is in
# arg_basis is the basis of the thing we're multiplying by
# i.e., if we are doing u.grad X, then arg_basis is the basis of u

def ncc_matrix(N, a_ncc, b_ncc, a_arg, b_arg, coeffs, cutoff=1e-6):
    """Build NCC matrix via Clenshaw algorithm."""
    # Kronecker Clenshaw on argument Jacobi matrix
    J = jacobi_matrix(N, a_arg, b_arg)
    A, B = jacobi_recursion(N, a_ncc, b_ncc, J)
    f0 = 1/np.sqrt(jacobi.mass(a_ncc,b_ncc)) * sparse.identity(N)
    total = matrix_clenshaw(coeffs, A, B, f0, cutoff=cutoff)
    return total

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
    J = jacobi_matrix(N, a, b)
    JA = J.toarray()
    # Identity element
    if np.isscalar(X):
        I = 1
    else:
        I = sparse.identity(X.shape[0])
    # Clenshaw coefficients
    def compute_A(n):
        if 0 <= n < (N-1):
            return (X - JA[n,n]*I) / JA[n,n+1]
        else:
            return 0*I
    def compute_B(n):
        if 0 < n < (N-1):
            return (-J[n,n-1] / J[n,n+1]) * I
        else:
            return 0*I
    A = DeferredTuple(compute_A, N+1)
    B = DeferredTuple(compute_B, N+1)
    return A, B


def matrix_clenshaw(c, A, B, f0, cutoff):
    """
    Clenshaw algorithm on scalar coefficients, matrix argument:
        S(X) = sum_n c_n f_n(X)
    """
    N = len(c)
    I = sparse.identity(f0.shape[0])
    # Clenshaw
    b0, b1 = 0*I, 0*I
    n_terms = max_term = 0
    for n in reversed(range(N)):
        b1, b2 = b0, b1
        if abs(c[n]) > cutoff:
            b0 = (c[n] * I) + (A[n] @ b1) + (B[n+1] @ b2)
            n_terms += 1
            if max_term == 0 :
                # reversed range, so first term is max_term
                max_term = n
        else:
            b0 = (A[n] @ b1) + (B[n+1] @ b2)
    return n_terms, max_term, (b0 @ f0)

def jacobi_matrix(N, a, b):
    J = jacobi.operator('J',N-1,a,b)
    return J.tocsr().astype(np.float64)


class DeferredTuple:

    def __init__(self, entry_function, size):
        self.entry_function = entry_function
        self.size = size

    def __getitem__(self, key):
        if isinstance(key, int):
            if key < 0:
                key += len(self)
            if key >= len(self):
                raise IndexError("The index (%d) is out of range." %key)
            return self.entry_function(key)
        else:
            raise TypeError("Invalid argument type.")

    def __len__(self):
        return self.size
