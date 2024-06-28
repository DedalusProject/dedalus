"""Matrix solver wrappers."""

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from functools import partial


matsolvers = {}
def add_solver(solver):
    matsolvers[solver.__name__.lower()] = solver
    return solver


class SolverBase:
    """Abstract base class for all solvers."""

    config = {}

    def __init__(self, matrix, solver=None):
        pass

    def solve(self, vector):
        pass

    def solve_H(self,vector):
        raise NotImplementedError("%s has not implemented 'solve_H' method" % type(self))


@add_solver
class DummySolver(SolverBase):
    """Dummy solver that returns zeros for testing."""

    def solve(self, vector):
        return 0 * vector


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
        out = spla.spsolve(self.matrix, vector, use_umfpack=True)
        # Fix return shape for matrices
        if vector.ndim == 2 and out.ndim == 1:
            out = out[:, None]
        return out


class _SuperluSpsolveBase(SparseSolver):
    """"SuperLU spsolve base class."""

    permc_spec = None

    def __init__(self, matrix, solver=None):
        self.matrix = matrix.copy()

    def solve(self, vector):
        out = spla.spsolve(self.matrix, vector, permc_spec=self.permc_spec, use_umfpack=False)
        # Fix return shape for matrices
        if vector.ndim == 2 and out.ndim == 1:
            out = out[:, None]
        return out


@add_solver
class SuperluNaturalSpsolve(_SuperluSpsolveBase):
    """SuperLU spsolve with 'NATURAL' column permutation."""
    permc_spec = "NATURAL"


@add_solver
class SuperluColamdSpsolve(_SuperluSpsolveBase):
    """SuperLU spsolve with 'COLAMD' column permutation."""
    permc_spec = "COLAMD"


@add_solver
class UmfpackFactorized(SparseSolver):
    """UMFPACK LU factorized solve."""

    def __init__(self, matrix, solver=None):
        from scikits import umfpack
        self.LU = umfpack.splu(matrix.tocsc())

    def solve(self, vector):
        return self.LU.solve(vector)


class _SuperluFactorizedBase(SparseSolver):
    """SuperLU factorized solver base class."""

    permc_spec = None
    diag_pivot_thresh = None
    relax = None
    panel_size = None
    options = {}
    trans = "N"

    def __init__(self, matrix, solver=None):
        if self.trans == "T":
            matrix = matrix.T
        elif self.trans == "H":
            matrix = matrix.conj().T
        self.LU = spla.splu(matrix.tocsc(),
                            permc_spec=self.permc_spec,
                            diag_pivot_thresh=self.diag_pivot_thresh,
                            relax=self.relax,
                            panel_size=self.panel_size,
                            options=self.options)

    def solve(self, vector):
        return self.LU.solve(vector, trans=self.trans)

    def solve_H(self,vector):
        if self.trans == "N":
            return self.LU.solve(vector, trans="H")
        elif self.trans == "H":
            return self.LU.solve(vector)
        elif self.trans == "T":
            return np.conj(self.LU.solve(np.conj(vector)))


@add_solver
class SuperluNaturalFactorized(_SuperluFactorizedBase):
    """SuperLU factorized solve with 'NATURAL' column permutation."""
    permc_spec = "NATURAL"


@add_solver
class SuperluNaturalFactorizedTranspose(_SuperluFactorizedBase):
    """SuperLU factorized solve with 'NATURAL' row permutation."""
    permc_spec = "NATURAL"
    trans = "T"


@add_solver
class SuperluColamdFactorized(_SuperluFactorizedBase):
    """SuperLU factorized solve with 'COLAMD' column permutation."""
    permc_spec = "COLAMD"


@add_solver
class SuperluColamdFactorizedTranspose(_SuperluFactorizedBase):
    """SuperLU factorized solve with 'COLAMD' row permutation."""
    permc_spec = "COLAMD"
    trans = "T"


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
        self.matrix_inverse = sla.inv(matrix.toarray())

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


@add_solver
class ScipyDenseLU(DenseSolver):
    """Scipy dense LU factorized solve."""

    def __init__(self, matrix, solver=None):
        self.LU = sla.lu_factor(matrix.toarray(), check_finite=False)

    def solve(self, vector):
        return sla.lu_solve(self.LU, vector, check_finite=False)


class Woodbury(SparseSolver):
    """Solve top & right bordered matrix using Woodbury formula."""

    config = {'bc_top': True}

    def __init__(self, matrix, subproblem, matsolver):
        self.matrix = matrix
        self.matsolver = matsolver
        self.update_rank = R = subproblem.update_rank
        # Form Woodbury factors
        self.U = U = np.zeros((matrix.shape[0], 2*R), dtype=matrix.dtype)
        self.V = V = np.zeros((2*R, matrix.shape[1]), dtype=matrix.dtype)
        # Remove top border, leaving upper left subblock
        U[:R, :R] = np.identity(R)
        V[:R, R:] = matrix[:R, R:].toarray()
        # Remove right border, leaving upper right and lower right subblocks
        U[R:-R, R:] = matrix[R:-R, -R:].toarray()
        V[-R:, -R:] = np.identity(R)
        self.A = matrix - sp.csr_matrix(U) @ sp.csr_matrix(V)
        # Solve A using specified matsolver
        self.A_matsolver = matsolver(self.A)
        self.Ainv = self.A_matsolver.solve
        self.Ainv_U = self.Ainv(U)
        # Solve S using scipy dense inverse
        S = np.identity(2*R) + V @ self.Ainv_U
        self.Sinv_ = sla.inv(S)
        self.Sinv = lambda Y, Sinv=self.Sinv_: Sinv @ Y

    def solve(self, Y):
        Ainv_Y = self.Ainv(Y)
        return Ainv_Y - self.Ainv_U @ self.Sinv(self.V @ Ainv_Y)


woodbury_matsolvers = {}
for name, matsolver in matsolvers.items():
    woodbury_matsolvers['woodbury' + name] = partial(Woodbury, matsolver=matsolver)
matsolvers.update(woodbury_matsolvers)

