

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

        # Build PDE matrices
        M = (sparse.kron(problem.M0, basis.Eval) +
             sparse.kron(problem.M1, basis.Deriv))
        L = (sparse.kron(problem.L0, basis.Eval) +
             sparse.kron(problem.L1, basis.Deriv))

        # Build boundary condition matrices
        M_bc = (sparse.kron(problem.ML, basis.Left) +
                sparse.kron(problem.MR, basis.Right))
        L_bc = (sparse.kron(problem.LL, basis.Left) +
                sparse.kron(problem.LR, basis.Right))

        # Convert to necessary sparse structures
        M = M.tolil()
        L = L.tolil()
        M_bc = M_bc.tocoo()
        L_bc = L_bc.tocoo()

        # Clear boundary condition rows in PDE matrices
        bc_rows = set(M_bc.row).union(set(L_bc.row))
        for r in bc_rows:
            M[r, :] = 0
            L[r, :] = 0

        # Substitute boundary condition rows into PDE matrices
        for i, j, v in izip(M_bc.row, M_bc.col, M_bc.data):
            M[i, j] = v
        for i, j, v in izip(L_bc.row, L_bc.col, L_bc.data):
            L[i, j] = v

        # Store for efficient manipulation
        self.M = M.tocsr()
        self.L = L.tocsr()

        # BC RHS
        self.b = np.kron(problem.b, basis.last)

