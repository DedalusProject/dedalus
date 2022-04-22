
import pytest
import numpy as np
from scipy import sparse
from dedalus.core import coords, distributor, basis, field, operators
from dedalus.tools.array import apply_matrix
from dedalus.tools import jacobi
from dedalus.tools import clenshaw
from ..libraries import dedalus_sphere

N_range = [8, 12]
regtotal_range = [-1, 0, +1]
regtotal_range = [0]

@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('regtotal_in', regtotal_range)
@pytest.mark.parametrize('k1', [0, 1])
@pytest.mark.parametrize('k2', [0, 1])
@pytest.mark.parametrize('ell', [2, 3])
def test_clenshaw(N, regtotal_in, k1, k2, ell):
    dtype = np.complex128
    c = coords.SphericalCoordinates('phi', 'theta', 'r')
    d = distributor.Distributor((c,))
    b = basis.BallBasis(c, (N, N, N), dtype=dtype, k=k1, radius=1)
    basis_in = basis.BallBasis(c, (N, N, N), dtype=dtype, k=k2, radius=1)
    phi, theta, r = b.local_grids((1, 1, 1))
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    ncc = field.Field(dist=d, bases=(b.radial_basis,), dtype=dtype)
    ncc['g'] = 2*r**2-1
    ncc_basis = ncc.domain.bases[0]
    a_ncc = ncc_basis.alpha + ncc_basis.k
    b_ncc = 1/2
    regtotal_ncc = 0

    n_size = b.n_size(ell)

    coeffs_filter = ncc['c'][0,0,:n_size]
    J = basis_in.operator_matrix('Z', ell, regtotal_in)
    A, B = clenshaw.jacobi_recursion(n_size, a_ncc, b_ncc, J)
    f0 = dedalus_sphere.zernike.polynomials(3, 1, a_ncc, regtotal_ncc, 1)[0].astype(np.float64) * sparse.identity(n_size)
    matrix = clenshaw.matrix_clenshaw(coeffs_filter, A, B, f0, cutoff=1e-6)

    assert np.allclose(J.todense(), matrix.todense())

@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('regtotal_in', regtotal_range)
@pytest.mark.parametrize('k1', [0, 1])
@pytest.mark.parametrize('k2', [0, 1])
@pytest.mark.parametrize('ell', [2, 3])
def test_clenshaw_vector(N, regtotal_in, k1, k2, ell):
    dtype = np.complex128
    c = coords.SphericalCoordinates('phi', 'theta', 'r')
    d = distributor.Distributor((c,))
    b = basis.BallBasis(c, (N, N, N), dtype=dtype, k=k1, radius=1)
    basis_in = basis.BallBasis(c, (N, N, N), dtype=dtype, k=k2, radius=1)
    phi, theta, r = b.local_grids((1, 1, 1))
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    ncc = field.Field(dist=d, bases=(b.radial_basis,), tensorsig=(c,), dtype=dtype)
    ncc['g'][2] = r*(2*r**2-1)
    ncc_basis = ncc.domain.bases[0]
    a_ncc = ncc_basis.alpha + ncc_basis.k
    regtotal_ncc = 1
    b_ncc = 1/2 + regtotal_ncc

    n_size = b.n_size(ell)

    coeffs_filter = ncc['c'][1,0,0,:n_size]
    J = basis_in.operator_matrix('Z', ell, regtotal_in)
    I = basis_in.operator_matrix('Id', ell, regtotal_in)
    A, B = clenshaw.jacobi_recursion(n_size, a_ncc, b_ncc, J)
    f0 = dedalus_sphere.zernike.polynomials(3, 1, a_ncc, regtotal_ncc, 1)[0].astype(np.float64) * sparse.identity(n_size)
    matrix = clenshaw.matrix_clenshaw(coeffs_filter, A, B, f0, cutoff=1e-6)

    assert np.allclose(J.todense(), matrix.todense())

