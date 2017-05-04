# cython: profile=True

cimport cython
cimport numpy as cnp
import numpy as np
import math

import logging
logger = logging.getLogger(__name__.split('.')[-1])


# Create fused type for double precision real and complex
ctypedef fused double_rc:
    double
    double complex


@cython.boundscheck(False)
def chebyshev_derivative_2d(double_rc[:,::1] A, double_rc[:,::1] B):
    """
    Chebyshev differentiation by recursion on coefficients.
    Operates on last dimension of 2D arrays.

    Parameters
    ----------
    A : 2D array of float64 or complex128
        Input coefficients
    B : 2D array of float64 or complex128
        Output coefficients

    """
    # d_x(T_n) / n = 2 T_(n-1) + d_x(T_(n-2)) / (n-2)
    # Create local copies of loop bounds
    cdef unsigned int I = A.shape[0]
    cdef unsigned int J = A.shape[1]
    # Allocate loop variables
    cdef unsigned int i, j, j1, j2
    # Apply recursive differentiation
    for i in range(I):
        j = J - 1
        B[i, j] = 0
        B[i, j-1] = (2 * j) * A[i, j]
        for j in range(J-3, 0, -1):
            j1 = j + 1
            j2 = j + 2
            B[i, j] = (2 * j1) * A[i, j1] + B[i, j2]
        B[i, 0] = A[i, 1] + 0.5 * B[i, 2]

