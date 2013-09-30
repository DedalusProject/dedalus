

import numpy as np
from scipy import sparse


class Pencil:
    """Pencil object for viewing one k_trans across system"""

    def __init__(self, slice, d_trans):

        # Inputs
        self.slice = slice
        self.d_trans = d_trans

    def get(self, system):

        # Retrieve slice of all fields
        data = []
        for field in system.fields.values():
            data.append(field['K'][self.slice])
        data = np.hstack(data)

        return data

    def set(self, system, data):

        # Set slice of all fields
        start = 0
        for field in system.fields.values():
            end = start + field.domain.bases[-1].coeff_size
            field['K'][self.slice] = data[start:end]
            start = end

    def build_matrices(self, problem, basis):

        # Build PDE matrices starting with constant terms
        Pre_0 = basis.Pre
        Diff_0 = basis.Pre * basis.Diff

        M = (sparse.kron(problem.M0[0](self.d_trans), Pre_0) +
             sparse.kron(problem.M1[0](self.d_trans), Diff_0))
        L = (sparse.kron(problem.L0[0](self.d_trans), Pre_0) +
             sparse.kron(problem.L1[0](self.d_trans), Diff_0))

        # Convert to easily modifiable structures
        M = M.tolil()
        L = L.tolil()

        # Add higher order terms
        for i in range(1, problem.order):
            Pre_i = basis.Pre * basis.Mult[i-1]
            Diff_i = basis.Pre * basis.Mult[i-1] * basis.Diff

            M += sparse.kron(problem.M0[i](self.d_trans), Pre_i)
            M += sparse.kron(problem.M1[i](self.d_trans), Diff_i)
            L += sparse.kron(problem.L0[i](self.d_trans), Pre_i)
            L += sparse.kron(problem.L1[i](self.d_trans), Diff_i)

        # Build boundary condition matrices
        Left = sparse.kron(basis.Left, basis.BC_row)
        Right = sparse.kron(basis.Right, basis.BC_row)
        Int = sparse.kron(basis.Int, basis.BC_row)

        Mb = (sparse.kron(problem.ML(self.d_trans), Left) +
              sparse.kron(problem.MR(self.d_trans), Right) +
              sparse.kron(problem.MI(self.d_trans), Int))
        Lb = (sparse.kron(problem.LL(self.d_trans), Left) +
              sparse.kron(problem.LR(self.d_trans), Right) +
              sparse.kron(problem.LI(self.d_trans), Int))

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
        self.b = np.kron(problem.b(self.d_trans), basis.BC_row[:,0])
        self.bc_rows = list(rows)
        self.bc_f = [self.b[r] for r in rows]
        self.parameters = problem.parameters
        self.F_eval = sparse.kron(np.eye(problem.size), basis.Pre)

