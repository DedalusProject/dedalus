
import pytest
import numpy as np
from dedalus.core import coords, distributor, basis, field, operators, arithmetic
from dedalus.tools.cache import CachedMethod
from mpi4py import MPI

comm = MPI.COMM_WORLD

Nphi_range = [8]
Ntheta_range = [10]
Nr_range = [6]
radius_range = [1.5]
dealias_range = [1, 3/2]

radius_ball = 1.5
@CachedMethod
def build_ball(Nphi, Ntheta, Nr, dealias):
    c = coords.SphericalCoordinates('phi', 'theta', 'r')
    d = distributor.Distributor((c,))
    b = basis.BallBasis(c, (Nphi, Ntheta, Nr), radius=radius_ball, dealias=(dealias, dealias, dealias))
    phi, theta, r = b.local_grids()
    x, y, z = c.cartesian(phi, theta, r)
    return c, d, b, phi, theta, r, x, y, z

radii_shell = (0.5, 3)
@CachedMethod
def build_shell(Nphi, Ntheta, Nr, dealias):
    c = coords.SphericalCoordinates('phi', 'theta', 'r')
    d = distributor.Distributor((c,))
    b = basis.SphericalShellBasis(c, (Nphi, Ntheta, Nr), radii=radii_shell, dealias=(dealias, dealias, dealias))
    phi, theta, r = b.local_grids()
    x, y, z = c.cartesian(phi, theta, r)
    return c, d, b, phi, theta, r, x, y, z

@pytest.mark.mpi
@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('dealias', dealias_range)
def test_S2_Jacobi_scalar_scalar_multiplication(Nphi, Ntheta, Nr, dealias):
    c, d, b, phi, theta, r, x, y, z = build_shell(Nphi, Ntheta, Nr, dealias)
    f0 = field.Field(dist=d, bases=(b,), dtype=np.complex128)
    f0.set_scales(b.domain.dealias)
    phi, theta, r = b.local_grids(b.domain.dealias)
    f0['g'] = (r**2 - 0.5*r**3)*(5*np.cos(theta)**2-1)*np.sin(theta)*np.exp(1j*phi)

    b_S2 = b.S2_basis()
    phi, theta = b_S2.local_grids()
    g = field.Field(dist=d, bases=(b_S2,), dtype=np.complex128)
    g['g'] = (5*np.cos(theta)**2-1)*np.sin(theta)*np.exp(1j*phi)

    b_r = basis.Jacobi(c.radius, size=Nr, a=-1/2, b=-1/2, bounds=radii_shell, dealias=dealias)
    r = b_r.local_grid()
    h = field.Field(dist=d, bases=(b_r,), dtype=np.complex128)
    h['g'] = (r**2 - 0.5*r**3)
    f = (g * h).evaluate()
    assert np.allclose(f['g'], f0['g'])

