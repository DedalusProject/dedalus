"""Matrix solver wrappers."""

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla


matsolvers = {}
def add_solver(solver):
    matsolvers[solver.__name__.lower()] = solver
    return solver


class SolverBase:
    """Abstract base class for all solvers."""

    def __init__(self, matrix, solver=None):
        pass

    def solve(self, vector):
        pass


class SparseSolver(SolverBase):
    """Base class for sparse solvers."""
    sparse = True
    banded = False


class BandedSolver(SolverBase):
    """Base class for banded solvers."""
    sparse = False
    banded = True

    @staticmethod
    def sparse_to_banded(matrix, u=None, l=None):
        """Convert sparse matrix to banded format."""
        matrix = sp.dia_matrix(matrix)
        if u is None:
            u = max(0, max(matrix.offsets))
        if l is None:
            l = max(0, max(-matrix.offsets))
        ab = np.zeros((u+l+1, matrix.shape[1]), dtype=matrix.dtype)
        ab[u-matrix.offsets] = matrix.data
        lu = (l, u)
        return lu, ab


class DenseSolver(SolverBase):
    """Base class for dense solvers."""
    sparse = False
    banded = False


@add_solver
class UmfpackSpsolve(SparseSolver):
    """UMFPACK spsolve."""

    def __init__(self, matrix, solver=None):
        from scikits import umfpack
        self.matrix = matrix.copy()

    def solve(self, vector):
        return spla.spsolve(self.matrix, vector, use_umfpack=True)


@add_solver
class SuperluNaturalSpsolve(SparseSolver):
    """SuperLU+NATURAL spsolve."""

    def __init__(self, matrix, solver=None):
        self.matrix = matrix.copy()

    def solve(self, vector):
        return spla.spsolve(self.matrix, vector, permc_spec='NATURAL', use_umfpack=False)


@add_solver
class SuperluColamdSpsolve(SparseSolver):
    """SuperLU+COLAMD spsolve."""

    def __init__(self, matrix, solver=None):
        self.matrix = matrix.copy()

    def solve(self, vector):
        return spla.spsolve(self.matrix, vector, permc_spec='COLAMD', use_umfpack=False)


@add_solver
class UmfpackFactorized(SparseSolver):
    """UMFPACK LU factorized solve."""

    def __init__(self, matrix, solver=None):
        from scikits import umfpack
        self.LU = spla.factorized(matrix.tocsc())

    def solve(self, vector):
        return self.LU(vector)


@add_solver
class SuperluNaturalFactorized(SparseSolver):
    """SuperLU+NATURAL LU factorized solve."""

    def __init__(self, matrix, solver=None):
        self.LU = spla.splu(matrix.tocsc(), permc_spec='NATURAL')

    def solve(self, vector):
        return self.LU.solve(vector)


@add_solver
class SuperluColamdFactorized(SparseSolver):
    """SuperLU+COLAMD LU factorized solve."""

    def __init__(self, matrix, solver=None):
        self.LU = spla.splu(matrix.tocsc(), permc_spec='COLAMD')

    def solve(self, vector):
        return self.LU.solve(vector)


@add_solver
class ScipyBanded(BandedSolver):
    """Scipy banded solve."""

    def __init__(self, matrix, solver=None):
        self.lu, self.ab = self.sparse_to_banded(matrix)

    def solve(self, vector):
        return sla.solve_banded(self.lu, self.ab, vector, check_finite=False)


@add_solver
class SPQR_solve(SparseSolver):
    """SuiteSparse QR solve."""

    def __init__(self, matrix, solver=None):
        import sparseqr
        self.matrix = matrix.copy()

    def solve(self, vector):
        return sparseqr.solve(self.matrix, vector)


@add_solver
class BandedQR(BandedSolver):
    """pybanded QR solve."""

    def __init__(self, matrix, solver=None):
        import pybanded
        matrix = pybanded.BandedMatrix.from_sparse(matrix)
        self.QR = pybanded.BandedQR(matrix)

    def solve(self, vector):
        return self.QR.solve(vector)


@add_solver
class SparseInverse(SparseSolver):
    """Sparse inversion solve."""

    def __init__(self, matrix, solver=None):
        self.matrix_inverse = spla.inv(matrix.tocsc())

    def solve(self, vector):
        return self.matrix_inverse @ vector


@add_solver
class DenseInverse(DenseSolver):
    """Dense inversion solve."""

    def __init__(self, matrix, solver=None):
        self.matrix_inverse = sla.inv(matrix.A)

    def solve(self, vector):
        return self.matrix_inverse @ vector


@add_solver
class BlockInverse(BandedSolver):
    """Block inversion solve."""

    def __init__(self, matrix, solver):
        from dedalus.tools.sparse import same_dense_block_diag
        # Check separability
        if solver.domain.bases[-1].coupled:
            raise ValueError("Block solver requires uncoupled problems.")
        block_size = b = len(solver.problem.variables)
        # Produce inverse
        if block_size == 1:
            # Special-case diagonal matrices
            self.matrix_inv_diagonal = 1 / matrix.todia().data[0]
            self.solve = self._solve_diag
        else:
            # Covert to BSR to extract blocks
            bsr_matrix = matrix.tobsr(blocksize=(b, b))
            # Compute block inverses
            inv_blocks = np.linalg.inv(bsr_matrix.data)
            self.matrix_inverse = same_dense_block_diag(list(inv_blocks), format='csr')
            self.solve = self._solve_block

    def _solve_block(self, vector):
        return self.matrix_inverse @ vector

    def _solve_diag(self, vector):
        return self.matrix_inv_diagonal * vector

