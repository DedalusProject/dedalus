

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

        basis.build_matrices()

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
            # Type-1 mult
            Mult_i = basis._build_Mult1(i)
            Eval_i = basis.Eval * Mult_i
            Deriv_i = basis.Eval * Mult_i * basis.InvEval * basis.Deriv
            # Type-2 mult
            # Mult_i = basis._build_Mult(i)
            # Eval_i = Mult_i * basis.Eval
            # Deriv_i = Mult_i * basis.Deriv

            M += sparse.kron(problem.M0[i], Eval_i)
            M += sparse.kron(problem.M1[i], Deriv_i)
            L += sparse.kron(problem.L0[i], Eval_i)
            L += sparse.kron(problem.L1[i], Deriv_i)

        # Build boundary condition matrices
        Mb = (sparse.kron(problem.ML, basis.Left) +
              sparse.kron(problem.MR, basis.Right))
        Lb = (sparse.kron(problem.LL, basis.Left) +
              sparse.kron(problem.LR, basis.Right))

        # Convert to easily iterable structures
        Mb = Mb.tocoo()
        Lb = Lb.tocoo()

        # Clear boundary condition rows in PDE matrices
        rows = set(Mb.row).union(set(Lb.row))
        for i in rows:
            M[i, :] = 0
            L[i, :] = 0

        # Substitute boundary condition terms into PDE matrices
        for i, j, v in izip(Mb.row, Mb.col, Mb.data):
            M[i, j] = v
        for i, j, v in izip(Lb.row, Lb.col, Lb.data):
            L[i, j] = v

        # Convert for efficient manipulation and store
        self.M = M.tocsr()
        self.L = L.tocsr()

        # BC RHS
        self.b = np.kron(problem.b, basis.last)

