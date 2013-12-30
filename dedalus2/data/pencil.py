

import numpy as np
from scipy import sparse

from ..tools.general import rev_enumerate


class PencilSet:
    """Adjascent-memory pencil system for efficient computations."""

    def __init__(self, domain, n_fields):

        # Extend layout shape
        shape = list(domain.distributor.coeff_layout.shape)
        self.stride = shape[-1]
        shape[-1] *= n_fields

        # Allocate data
        dtype = domain.distributor.coeff_layout.dtype
        self.data = np.zeros(shape, dtype=dtype)

        # Build pencils
        self._construct_pencil_info(domain)
        self.pencils = []
        for s, d in zip(self.pencil_slices, self.pencil_dtrans):
            pencil = Pencil(self.data, s, d)
            self.pencils.append(pencil)

    def get_system(self, system):

        for i, field in enumerate(system.fields.values()):
            start = i * self.stride
            end = start + self.stride

            field.require_coeff_space()
            np.copyto(self.data[..., start:end], field.data)

    def set_system(self, system):

        for i, field in enumerate(system.fields.values()):
            start = i * self.stride
            end = start + self.stride

            #field.require_coeff_space()
            field.layout = field.domain.distributor.coeff_layout
            np.copyto(field.data, self.data[..., start:end])

    def _construct_pencil_info(self, domain):

        # Construct pencil slices
        coeff_layout = domain.distributor.coeff_layout
        n_pencils = np.prod(coeff_layout.shape[:-1])
        n = np.arange(n_pencils)
        index_list = []
        dtrans_list = []

        div = n
        start = coeff_layout.start[:-1]
        for i, s in rev_enumerate(coeff_layout.shape[:-1]):
            div, mod = divmod(div, s)
            index_list.append(mod)
            dtrans_list.append(domain.bases[i].trans_diff(start[i]+mod))

        if domain.dim == 1:
            index_list.append([])
            dtrans_list.append([])
        else:
            index_list = list(zip(*index_list))
            dtrans_list = list(zip(*dtrans_list))

        slices = []
        for bl in index_list:
            sli = []
            for i in bl:
                sli.append(slice(i, i+1))
            sli.append(slice(None))
            slices.append(sli)

        self.pencil_slices = slices
        self.pencil_dtrans = dtrans_list


class Pencil:
    """Pencil object for viewing one k_trans across system"""

    def __init__(self, setdata, slice, d_trans):

        # Initial attributes
        self.setdata = setdata
        self.slice = slice
        self.d_trans = d_trans

    @property
    def data(self):

        return self.setdata[self.slice].squeeze()

    @data.setter
    def data(self, data):

        self.setdata[self.slice] = data

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

        # Allocate PDE matrices
        M = sparse.csr_matrix((size, size), dtype=dtype)
        L = sparse.csr_matrix((size, size), dtype=dtype)

        # Add terms to PDE matrices
        for i in range(problem.order):
            Pre_i = basis.Pre * basis.Mult(i)
            Diff_i = basis.Pre * basis.Mult(i) * basis.Diff

            M = M + sparse.kron(problem.M0[i](D), Pre_i, format='csr')
            M = M + sparse.kron(problem.M1[i](D), Diff_i, format='csr')
            L = L + sparse.kron(problem.L0[i](D), Pre_i, format='csr')
            L = L + sparse.kron(problem.L1[i](D), Diff_i, format='csr')

        # Allocate boundary condition matrices
        Mb = sparse.csr_matrix((size, size), dtype=dtype)
        Lb = sparse.csr_matrix((size, size), dtype=dtype)

        # Add terms to boundary condition matrices
        if np.any(ML):
            Mb = Mb + sparse.kron(ML, basis.Left, format='csr')
        if np.any(MR):
            Mb = Mb + sparse.kron(MR, basis.Right, format='csr')
        if np.any(MI):
            Mb = Mb + sparse.kron(MI, basis.Int, format='csr')
        if np.any(LL):
            Lb = Lb + sparse.kron(LL, basis.Left, format='csr')
        if np.any(LR):
            Lb = Lb + sparse.kron(LR, basis.Right, format='csr')
        if np.any(LI):
            Lb = Lb + sparse.kron(LI, basis.Int, format='csr')

        # Get set of boundary condition rows
        Mb_rows = Mb.nonzero()[0]
        Lb_rows = Lb.nonzero()[0]
        rows = set(Mb_rows).union(set(Lb_rows))

        # Clear boundary condition rows in PDE matrices
        clear_bc = sparse.eye(size, dtype=dtype, format='dok')
        for i in rows:
            clear_bc[i, i] = 0.

        clear_bc = clear_bc.tocsr()
        M = M.tocsr()
        L = L.tocsr()

        M = clear_bc * M
        L = clear_bc * L

        # Add boundary condition terms to PDE matrices and store
        self.M = M + Mb
        self.L = L + Lb

        # Reference nonlinear expressions
        self.F = problem.F
        # self.BL = problem.BL
        # self.BR = problem.BR
        # self.BI = problem.BI
        self.b = np.kron(problem.b(D), basis.bc_vector[:,0])
        self.bc_rows = list(rows)
        self.bc_f = [self.b[r] for r in rows]
        self.parameters = problem.parameters
        self.F_eval = sparse.kron(np.eye(problem.size), basis.Pre)
        # self.BL_eval = sparse.kron(np.eye(problem.size), basis.Left)
        # self.BR_eval = sparse.kron(np.eye(problem.size), basis.Right)
        # self.BI_eval = sparse.kron(np.eye(problem.size), basis.Int)

