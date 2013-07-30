

import numpy as np
from scipy import sparse
from itertools import izip


class Pencil(object):
    """Pencil object for viewing one k_trans across system"""

    def __init__(self, slice):

        # Inputs
        self.slice = slice

    def get(self, system):

        # Retrieve slice of all fields
        data = []
        for field in system.fields.itervalues():
            data.append(field['k'][self.slice])
        data = np.hstack(data)

        return data

    def set(self, system, data):

        # Set slice of all fields
        start = 0
        for field in system.fields.itervalues():
            end = start + field.domain.bases[-1].size
            field['k'][self.slice] = data[start:end]
            start = end

    def build_matrices(self, problem, basis):

        # Build PDE matrices starting with constant terms
        M = (sparse.kron(problem.M0[0], basis.Eval) +
             sparse.kron(problem.M1[0], basis.Deriv))
        L = (sparse.kron(problem.L0[0], basis.Eval) +
             sparse.kron(problem.L1[0], basis.Deriv))

        # Convert to easily modifiable structures
        M = M.tolil()
        L = L.tolil()

        # Add higher order terms
        for i in xrange(1, problem.order):
            Ni = basis._build_Mult1(i)
            Eval_i = basis.Eval * Ni
            Deriv_i = basis.Eval * Ni * basis.InvEval * basis.Deriv
            if i < len(problem.M0):
                M += sparse.kron(problem.M0[i], Eval_i)
            if i < len(problem.M1):
                M += sparse.kron(problem.M1[i], Deriv_i)
            if i < len(problem.L0):
                L += sparse.kron(problem.L0[i], Eval_i)
            if i < len(problem.L1):
                L += sparse.kron(problem.L1[i], Deriv_i)

        # Build boundary condition matrices
        M_bc = (sparse.kron(problem.ML, basis.Left) +
                sparse.kron(problem.MR, basis.Right))
        L_bc = (sparse.kron(problem.LL, basis.Left) +
                sparse.kron(problem.LR, basis.Right))

        # Convert to easily iterable structures
        M_bc = M_bc.tocoo()
        L_bc = L_bc.tocoo()

        # Clear boundary condition rows in PDE matrices
        bc_rows = set(M_bc.row).union(set(L_bc.row))
        for i in bc_rows:
            M[i, :] = 0
            L[i, :] = 0

        # Substitute boundary condition terms into PDE matrices
        for i, j, v in izip(M_bc.row, M_bc.col, M_bc.data):
            M[i, j] = v
        for i, j, v in izip(L_bc.row, L_bc.col, L_bc.data):
            L[i, j] = v

        # Convert for efficient manipulation
        self.M = M.tocsr()
        self.L = L.tocsr()

        # BC RHS
        self.b = np.kron(problem.b, basis.last)

