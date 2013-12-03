

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
            data.append(field['K'][self.slice].squeeze())
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

        # Size
        size = problem.size * basis.coeff_size
        dtype = basis.coeff_dtype

        D = self.d_trans

        # Problem matrices
        ML = problem.ML(self.d_trans)
        MR = problem.MR(self.d_trans)
        MI = problem.MI(self.d_trans)
        LL = problem.LL(self.d_trans)
        LR = problem.LR(self.d_trans)
        LI = problem.LI(self.d_trans)

        # Build PDE matrices starting with constant terms
        Pre_0 = basis.Pre
        Diff_0 = basis.Pre * basis.Diff

        M = (sparse.kron(problem.M0[0](D), Pre_0) +
             sparse.kron(problem.M1[0](D), Diff_0))
        L = (sparse.kron(problem.L0[0](D), Pre_0) +
             sparse.kron(problem.L1[0](D), Diff_0))

        # Convert to easily modifiable structures
        M = M.tolil()
        L = L.tolil()

        # Add higher order terms
        for i in range(1, problem.order):
            Pre_i = basis.Pre * basis.Mult(i)
            Diff_i = basis.Pre * basis.Mult(i) * basis.Diff

            M += sparse.kron(problem.M0[i](D), Pre_i)
            M += sparse.kron(problem.M1[i](D), Diff_i)
            L += sparse.kron(problem.L0[i](D), Pre_i)
            L += sparse.kron(problem.L1[i](D), Diff_i)

        # Allocate boundary condition matrices
        Mb = sparse.lil_matrix((size, size), dtype=dtype)
        Lb = sparse.lil_matrix((size, size), dtype=dtype)

        # Add terms to boundary condition matrices
        if np.any(ML):
            Mb += sparse.kron(ML, basis.Left)
        if np.any(MR):
            Mb += sparse.kron(MR, basis.Right)
        if np.any(MI):
            Mb += sparse.kron(MI, basis.Int)
        if np.any(LL):
            Lb += sparse.kron(LL, basis.Left)
        if np.any(LR):
            Lb += sparse.kron(LR, basis.Right)
        if np.any(LI):
            Lb += sparse.kron(LI, basis.Int)

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
        # self.BL = problem.BL
        # self.BR = problem.BR
        # self.BI = problem.BI
        self.b = np.kron(problem.b(D), basis.bc_row[:,0])
        self.bc_rows = list(rows)
        self.bc_f = [self.b[r] for r in rows]
        self.parameters = problem.parameters
        self.F_eval = sparse.kron(np.eye(problem.size), basis.Pre)
        # self.BL_eval = sparse.kron(np.eye(problem.size), basis.Left)
        # self.BR_eval = sparse.kron(np.eye(problem.size), basis.Right)
        # self.BI_eval = sparse.kron(np.eye(problem.size), basis.Int)

