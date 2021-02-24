import pytest
import numpy as np
from dedalus.core import coords, distributor, basis, field, operators, arithmetic
from dedalus.tools.cache import CachedMethod
from mpi4py import MPI


ufuncs = operators.UnaryGridFunction.supported.values()


@pytest.mark.parametrize('N', [16])
@pytest.mark.parametrize('a', [0])
@pytest.mark.parametrize('b', [0])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('dealias', [1])
@pytest.mark.parametrize('func', ufuncs)
def test_jacobi_ufunc(N, a, b, dtype, dealias, func):
    c = coords.Coordinate('x')
    d = distributor.Distributor((c,))
    b = basis.Jacobi(c, size=N, a=a, b=b, bounds=(0, 1), dealias=dealias)
    x = b.local_grid(1)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    if func is np.arccosh:
        f['g'] = 1 + x**2
    else:
        f['g'] = x**2
    g0 = func(f['g'])
    g = func(f).evaluate()
    assert np.allclose(g['g'], g0)


@CachedMethod
def build_ball(Nphi, Ntheta, Nr, dtype, dealias, radius=1):
    c = coords.SphericalCoordinates('phi', 'theta', 'r')
    d = distributor.Distributor((c,))
    b = basis.BallBasis(c, (Nphi, Ntheta, Nr), radius=radius, dtype=dtype, dealias=(dealias, dealias, dealias))
    phi, theta, r = b.local_grids()
    x, y, z = c.cartesian(phi, theta, r)
    return c, d, b, phi, theta, r, x, y, z


@CachedMethod
def build_shell(Nphi, Ntheta, Nr, dtype, dealias, radii=(0.5,1)):
    c = coords.SphericalCoordinates('phi', 'theta', 'r')
    d = distributor.Distributor((c,))
    b = basis.SphericalShellBasis(c, (Nphi, Ntheta, Nr), radii=radii, dtype=dtype, dealias=(dealias, dealias, dealias))
    phi, theta, r = b.local_grids()
    x, y, z = c.cartesian(phi, theta, r)
    return c, d, b, phi, theta, r, x, y, z


@pytest.mark.parametrize('Nphi', [8])
@pytest.mark.parametrize('Ntheta', [16])
@pytest.mark.parametrize('Nr', [16])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('dealias', [1])
@pytest.mark.parametrize('build_basis', [build_shell, build_ball])
@pytest.mark.parametrize('func', ufuncs)
def test_spherical_field_ufunc(Nphi, Ntheta, Nr, dtype, dealias, build_basis, func):
    c, d, b, phi, theta, r, x, y, z = build_basis(Nphi, Ntheta, Nr, dtype, dealias)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    if func is np.arccosh:
        f['g'] = 1 + r**2
    else:
        f['g'] = r**2
    g0 = func(f['g'])
    g = func(f).evaluate()
    assert np.allclose(g['g'], g0)


@pytest.mark.parametrize('Nphi', [8])
@pytest.mark.parametrize('Ntheta', [16])
@pytest.mark.parametrize('Nr', [16])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('dealias', [1])
@pytest.mark.parametrize('build_basis', [build_shell, build_ball])
@pytest.mark.parametrize('func', ufuncs)
def test_spherical_operator_ufunc(Nphi, Ntheta, Nr, dtype, dealias, build_basis, func):
    c, d, b, phi, theta, r, x, y, z = build_basis(Nphi, Ntheta, Nr, dtype, dealias)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    if func is np.arccosh:
        f['g'] = 1 + r**2
        a = 2
    else:
        f['g'] = r**2
        a = 0.5
    g0 = func(a*f['g'])
    g = func(a*f).evaluate()
    assert np.allclose(g['g'], g0)

