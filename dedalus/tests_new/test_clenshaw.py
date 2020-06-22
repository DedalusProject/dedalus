
import pytest
import numpy as np
from scipy import sparse
from dedalus.core import coords, distributor, basis, field, operators
from dedalus.tools.array import apply_matrix
from dedalus.tools import jacobi
from dedalus.tools import clenshaw

N_range = [8, 12, 16]
regtotal_range = [-1, 0, +1]

@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('regtotal_in', regtotal_range)
@pytest.mark.parametrize('k', [0])
@pytest.mark.parametrize('ell', [2])
@pytest.mark.parametrize('norm', [1, 1/np.sqrt(2)])
def test_clenshaw(N, regtotal_in, k, ell, norm):
    c = coords.SphericalCoordinates('phi', 'theta', 'r')
    d = distributor.Distributor((c,))
    b = basis.BallBasis(c, (N, N, N), radius=1)
    basis_in = basis.BallBasis(c, (N, N, N), k=k, radius=1)
    phi, theta, r = b.local_grids((1, 1, 1))
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    ncc = field.Field(dist=d, bases=(b.radial_basis,), dtype=np.complex128)
    ncc['g'] = 2*r**2-1
    ncc_basis = ncc.domain.bases[0]
    a_ncc = ncc_basis.alpha + ncc_basis.k
    b_ncc = 1/2

    n_size = b.n_size(ell)

    coeffs_filter = ncc['c'][0,0,:n_size]
    J = basis_in.operator_matrix('Z', ell, regtotal_in)
    A, B = clenshaw.jacobi_recursion(n_size, a_ncc, b_ncc, J)
    f0 = 1/np.sqrt(jacobi.mass(a_ncc, b_ncc)) * sparse.identity(n_size)
    matrix = clenshaw.matrix_clenshaw(coeffs_filter, A, B, f0, cutoff=1e-6)

    assert np.allclose(J.todense(), norm*matrix.todense())

