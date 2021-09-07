
import pytest
import numpy as np
from dedalus.core import coords, distributor, basis, field, operators, arithmetic
from dedalus.tools.cache import CachedMethod
from mpi4py import MPI


comm = MPI.COMM_WORLD


Nphi_range = [8]
Ntheta_range = [10]
Nr_range = [8]
radius_range = [1.5]
dealias_range = [1, 1.5]


radius_ball = 1.5
@CachedMethod
def build_ball(Nphi, Ntheta, Nr, dtype, dealias, mesh=None):
    c = coords.SphericalCoordinates('phi', 'theta', 'r')
    d = distributor.Distributor((c,), mesh=mesh)
    b = basis.BallBasis(c, (Nphi, Ntheta, Nr), radius=radius_ball, dtype=dtype, dealias=(dealias, dealias, dealias))
    phi, theta, r = b.local_grids()
    x, y, z = c.cartesian(phi, theta, r)
    return c, d, b, phi, theta, r, x, y, z


radii_shell = (1, 2)
@CachedMethod
def build_shell(Nphi, Ntheta, Nr, dtype, dealias, mesh=None):
    c = coords.SphericalCoordinates('phi', 'theta', 'r')
    d = distributor.Distributor((c,), mesh=mesh)
    b = basis.SphericalShellBasis(c, (Nphi, Ntheta, Nr), radii=radii_shell, dtype=dtype, dealias=(dealias, dealias, dealias))
    phi, theta, r = b.local_grids()
    x, y, z = c.cartesian(phi, theta, r)
    return c, d, b, phi, theta, r, x, y, z


@pytest.mark.mpi(min_size=4)
@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('dealias', dealias_range)
def test_sphere_constant_S2_multiplication(Nphi, Ntheta, Nr, dtype, dealias):
    """Tests ghost broadcasting in colatitude."""
    c, d, b, phi, theta, r, x, y, z = build_ball(Nphi, Ntheta, Nr, dtype, dealias, mesh=(2,2))
    b_S2 = b.S2_basis()
    f = field.Field(dist=d, dtype=dtype)
    g = field.Field(dist=d, bases=(b_S2,), dtype=dtype)
    h = field.Field(dist=d, bases=(b_S2,), dtype=dtype)
    f['g'] = 6
    g['g'] = (5*np.cos(theta)**2 - 1) * np.sin(theta) * np.cos(phi)
    h['g'] = 6 * (5*np.cos(theta)**2 - 1) * np.sin(theta) * np.cos(phi)
    h_op = (f * g).evaluate()
    h_op.change_scales(1)
    assert np.allclose(h_op['g'], h['g'])


@pytest.mark.mpi(min_size=4)
@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('build_basis', [build_shell, build_ball])
def test_sphere_constant_radial_multiplication(Nphi, Ntheta, Nr, dtype, dealias, build_basis):
    """Tests ghost broadcasting in radius."""
    c, d, b, phi, theta, r, x, y, z = build_basis(Nphi, Ntheta, Nr, dtype, dealias, mesh=(2,2))
    b_r = b.radial_basis
    f = field.Field(dist=d, dtype=dtype)
    g = field.Field(dist=d, bases=(b_r,), dtype=dtype)
    h = field.Field(dist=d, bases=(b_r,), dtype=dtype)
    f['g'] = 6
    g['g'] = (r**2 - 0.5*r**4)
    h['g'] = 6 * (r**2 - 0.5*r**4)
    h_op = (f * g).evaluate()
    h_op.change_scales(1)
    assert np.allclose(h_op['g'], h['g'])


@pytest.mark.mpi(min_size=4)
@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('dealias', dealias_range)
def test_shell_S2_radial_multiplication(Nphi, Ntheta, Nr, dtype, dealias):
    """Tests ghost broadcasting in colatitude and radius."""
    c, d, b, phi, theta, r, x, y, z = build_shell(Nphi, Ntheta, Nr, dtype, dealias, mesh=(2,2))
    b_S2 = b.S2_basis()
    b_r = b.radial_basis
    f = field.Field(dist=d, bases=(b_S2,), dtype=dtype)
    g = field.Field(dist=d, bases=(b_r,), dtype=dtype)
    h = field.Field(dist=d, bases=(b,), dtype=dtype)
    f['g'] = (5*np.cos(theta)**2 - 1) * np.sin(theta) * np.cos(phi)
    g['g'] = (r**2 - 0.5*r**3)
    h['g'] = (r**2 - 0.5*r**3) * (5*np.cos(theta)**2 - 1) * np.sin(theta) * np.cos(phi)
    h_op = (f * g).evaluate()
    h_op.change_scales(1)
    assert np.allclose(h_op['g'], h['g'])

