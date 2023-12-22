"""Test spherical Clenshaw functions."""
# TODO: add tests for regular Jacobi Clenshaw

import pytest
import numpy as np
from scipy import sparse
import dedalus.public as d3
from dedalus.tools import clenshaw
from dedalus.libraries import dedalus_sphere


N_range = [8]
regtotal_range = [-1, 0, +1]
k_range = [0, 1]
ell_range = [0, 1, 2]
dtype_range = [np.float64, np.complex128]


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('regtotal_in', regtotal_range)
@pytest.mark.parametrize('k_ncc', k_range)
@pytest.mark.parametrize('k_arg', k_range)
@pytest.mark.parametrize('ell', ell_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_ball_clenshaw_scalar(N, regtotal_in, k_ncc, k_arg, ell, dtype):
    """Test Clenshaw algorithm in the ball for a scalar NCC."""
    # Bases
    c = d3.SphericalCoordinates('phi', 'theta', 'r')
    d = d3.Distributor(c, dtype=dtype)
    ncc_basis = d3.BallBasis(c, (1, 1, N), dtype=dtype, k=k_ncc, radius=1)
    arg_basis = d3.BallBasis(c, (N, N, N), dtype=dtype, k=k_arg, radius=1)
    phi, theta, r = d.local_grids(arg_basis, scales=(1, 1, 1))
    # Setup NCC to match Z operator
    ncc = d.Field(bases=ncc_basis.radial_basis)  # TODO: cleanup when radial bases are fixed
    ncc['g'] = 2*r**2-1
    # Build NCC matrix
    a_ncc = ncc_basis.alpha + ncc_basis.k
    b_ncc = 1/2
    regtotal_ncc = 0
    n_size = ncc_basis.n_size(ell)
    coeffs_filter = ncc['c'][0, 0, :n_size]
    J = arg_basis.operator_matrix('Z', ell, regtotal_in)
    A, B = clenshaw.jacobi_recursion(n_size, a_ncc, b_ncc, J)
    f0 = dedalus_sphere.zernike.polynomials(3, 1, a_ncc, regtotal_ncc, 1)[0] * sparse.identity(n_size)
    matrix = clenshaw.matrix_clenshaw(coeffs_filter, A, B, f0, cutoff=1e-6)
    assert np.allclose(J.todense(), matrix.todense())


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('regtotal_in', regtotal_range)
@pytest.mark.parametrize('k_ncc', k_range)
@pytest.mark.parametrize('k_arg', k_range)
@pytest.mark.parametrize('ell', ell_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_ball_clenshaw_vector(N, regtotal_in, k_ncc, k_arg, ell, dtype):
    """Test Clenshaw algorithm in the ball for a vector NCC."""
    # Bases
    c = d3.SphericalCoordinates('phi', 'theta', 'r')
    d = d3.Distributor(c, dtype=dtype)
    ncc_basis = d3.BallBasis(c, (1, 1, N), dtype=dtype, k=k_ncc, radius=1)
    arg_basis = d3.BallBasis(c, (N, N, N), dtype=dtype, k=k_arg, radius=1)
    phi, theta, r = d.local_grids(arg_basis, scales=(1, 1, 1))
    # Setup NCC to match Z operator
    ncc = d.VectorField(c, bases=ncc_basis.radial_basis)  # TODO: cleanup when radial bases are fixed
    ncc['g'][2] = r*(2*r**2-1)
    # Build NCC matrix
    regtotal_ncc = 1
    a_ncc = ncc_basis.alpha + ncc_basis.k
    b_ncc = 1/2 + regtotal_ncc
    n_size = ncc_basis.n_size(ell)
    coeffs_filter = ncc['c'][1, 0, 0, :n_size]
    J = arg_basis.operator_matrix('Z', ell, regtotal_in)
    A, B = clenshaw.jacobi_recursion(n_size, a_ncc, b_ncc, J)
    f0 = dedalus_sphere.zernike.polynomials(3, 1, a_ncc, regtotal_ncc, 1)[0] * sparse.identity(n_size)
    matrix = clenshaw.matrix_clenshaw(coeffs_filter, A, B, f0, cutoff=1e-6)
    assert np.allclose(J.todense(), matrix.todense())

