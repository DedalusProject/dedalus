"""
Class for representing differential equations.

"""

import numpy as np


class Problem:
    """
    PDE definitions using matrices and operator trees.

    Equations are assumed to take the form (LHS = RHS), where the left-hand side
    contains terms that are linear in the dependent variables (and will be
    represented in coefficient matrices), and the right-hand side contains terms
    that are non-linear (and will be represented by operator trees).  For
    simplicity, the last axis is referred to as the "z" direction.

    Parameters
    ----------
    field_names : list of strs
        Names of required field variables
    order : int
        Number of terms needed to represent non-constant coefficients in the
        linear portion of the equations

    Attributes
    ----------
    size : int
        Number of fields / equations
    parameters : dict
        Parameters used in string representation of right-hand side.

    LHS equation matrices:

    M0 : list of functions returning ndarrays
        Coefficient matrix for (d/dt) terms
    M1 : list of functions returning ndarrays
        Coefficient matrix for (d/dt d/dz) terms
    M0 : list of functions returning ndarrays
        Coefficient matrix for terms with no derivatives
    M1 : list of functions returning ndarrays
        Coefficient matrix for (d/dz) terms

    LHS boundary condition matrices:

    ML : list of functions returning ndarrays
        Coefficient matrix for (d/dt) terms, as evaluated on the left boundary
    MR : list of functions returning ndarrays
        Coefficient matrix for (d/dt) terms, as evaluated on the right boundary
    MI : list of functions returning ndarrays
        Coefficient matrix for (d/dt) terms, as integrated over the z interval
    LL : list of functions returning ndarrays
        Coefficient matrix for linear terms, as evaluated on the left boundary
    LR : list of functions returning ndarrays
        Coefficient matrix for linear terms, as evaluated on the right boundary
    LI : list of functions returning ndarrays
        Coefficient matrix for linear terms, as integrated over the z interval

    Right-hand side terms:

    F : list of strings
        String-representations of the operator trees for the RHS of the equations
    b : function returning ndarrays
        Values of the RHS of the boundary condition equations.
    parameters : dict
        Parameters needed when constructing operator trees from strings


    Notes
    -----
    The (i,j)-th element of a coefficient matrix encodes the dependence of the
    i-th equation/constraint on the j-th field named in the `field_names` list.

    The linear terms are separated into several matrices depending on the
    presence of temporal derivatives and spatial derivatives along the z axis,
    i.e. the last/pencil/implict axis.  Other axes must be represented by
    TransverseBasis objects, and hence their differential operators reduce to
    multiplicative constants for each pencil.

    The attributes defining the problem matrices are lists of functions, one for
    each term in the spectralmexpansion of the non-constant coefficients, that
    take as an argument the list of multiplicative constants for transverse
    differentiation for a given pencil, and return the resulting problem
    coefficients.

    """

    def __init__(self, field_names, order=1):

        # Initial attributes
        self.field_names = field_names
        self.size = len(field_names)
        self.order = order

        # Default problem matrices
        size = self.size
        self.M0 = [lambda d_trans: np.zeros((size, size), dtype=np.complex128) for i in range(order)]
        self.M1 = [lambda d_trans: np.zeros((size, size), dtype=np.complex128) for i in range(order)]
        self.L0 = [lambda d_trans: np.zeros((size, size), dtype=np.complex128) for i in range(order)]
        self.L1 = [lambda d_trans: np.zeros((size, size), dtype=np.complex128) for i in range(order)]

        # Default boundary condition matrices
        self.ML = lambda d_trans: np.zeros((size, size), dtype=np.complex128)
        self.MR = lambda d_trans: np.zeros((size, size), dtype=np.complex128)
        self.MI = lambda d_trans: np.zeros((size, size), dtype=np.complex128)
        self.LL = lambda d_trans: np.zeros((size, size), dtype=np.complex128)
        self.LR = lambda d_trans: np.zeros((size, size), dtype=np.complex128)
        self.LI = lambda d_trans: np.zeros((size, size), dtype=np.complex128)

        # Default RHS operators
        self.parameters = {}
        self.F = [None] * self.size
        self.b = lambda d_trans: np.zeros(size, dtype=np.complex128)
        # self.BL = [None] * self.size
        # self.BR = [None] * self.size
        # self.BI = [None] * self.size

