

import numpy as np
from scipy import sparse


class Pencil:
    """Pencil object for viewing one k_trans across system"""

    def __init__(self, slice):

        # Inputs
        self.slice = slice

    def get(self, system):

        # Retrieve slice of all fields
        data = []
        for field in system.fields.values():
            data.append(field['k'][self.slice])
        data = np.hstack(data)

        return data

    def set(self, system, data):

        # Set slice of all fields
        start = 0
        for field in system.fields.values():
            end = start + field.domain.bases[-1].size
            field['k'][self.slice] = data[start:end]
            start = end

    def build_matrices(self, problem, basis):

        # Instruct basis to build its matrices
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
        for i in range(1, problem.order):
            Mult_i = basis._build_Mult(i)
            Eval_i = basis.Eval * Mult_i
            Deriv_i = basis.Eval * Mult_i * basis.InvEval * basis.Deriv

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
        for i, j, v in zip(Mb.row, Mb.col, Mb.data):
            M[i, j] = v
        for i, j, v in zip(Lb.row, Lb.col, Lb.data):
            L[i, j] = v

        # Convert for efficient manipulation and store
        self.M = M.tocsr()
        self.L = L.tocsr()

        # Reference nonlinear expressions
        self.F = problem.F
        self.b = np.kron(problem.b, basis.last)
        self.bc_rows = list(rows)
        self.bc_f = [self.b[r] for r in rows]
        self.parameters = problem.parameters
        self.F_eval = sparse.kron(np.eye(problem.size), basis.Eval)

