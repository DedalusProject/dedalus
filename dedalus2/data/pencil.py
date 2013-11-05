

import numpy as np
from scipy import sparse


class Pencil:
    """Pencil object for viewing one k_trans across system"""

    def __init__(self, slice, d_trans, dtype, length):

        # Inputs
        self.slice = slice
        self.d_trans = d_trans
        self.dtype = dtype
        self.length = length

        self.get = self._initial_get

    def _initial_get(self, system):

        self.data = np.zeros(system.n_fields * self.length, self.dtype)
        self.get = self._subsequent_get

        return self.get(system)

    def _subsequent_get(self, system):

        # Retrieve slice of all fields
        for i, fn in enumerate(system.field_names):
            start = i * self.length
            end = (i+1) * self.length
            self.data[start:end] = system.fields[fn]['c'][self.slice].squeeze()

        return self.data

    def set(self, system, data):

        # Set slice of all fields
        for i, fn in enumerate(system.field_names):
            start = i * self.length
            end = (i+1) * self.length
            system.fields[fn]['c'][self.slice] = data[start:end]

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
        self.b = np.kron(problem.b(D), basis.bc_vector[:,0])
        self.bc_rows = list(rows)
        self.bc_f = [self.b[r] for r in rows]
        self.parameters = problem.parameters
        self.F_eval = sparse.kron(np.eye(problem.size), basis.Pre)

