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


@pytest.mark.parametrize('N', [16])
@pytest.mark.parametrize('a', [0])
@pytest.mark.parametrize('b', [0])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('dealias', [1, 3/2])
def test_jacobi_GeneralFunction(N, a, b, dtype, dealias):
    c = coords.Coordinate('x')
    d = distributor.Distributor((c,))
    b = basis.Jacobi(c, size=N, a=a, b=b, bounds=(0, 1), dealias=dealias)
    x = b.local_grid(dealias)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    def initialize_scalar(*args):
        x = args[0]
        return np.sin(x)
    def F(*args):
        return operators.GeneralFunction(d, f.domain, layout='g', tensorsig=f.tensorsig, dtype=dtype, func=initialize_scalar, args=args)
    f = F(x).evaluate()
    assert np.allclose(f['g'], initialize_scalar(x))


@pytest.mark.parametrize('N', [16])
@pytest.mark.parametrize('a', [0])
@pytest.mark.parametrize('b', [0])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('dealias', [1, 3/2])
def test_jacobi_GeneralFunction_field(N, a, b, dtype, dealias):
    c = coords.Coordinate('x')
    d = distributor.Distributor((c,))
    b = basis.Jacobi(c, size=N, a=a, b=b, bounds=(0, 1), dealias=dealias)
    x = b.local_grid(dealias)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    f.preset_scales(dealias)
    f['g'] = np.cos(x)
    def initialize_scalar(x, f):
        return 2*f['g'] + np.sin(x)
    def F(*args):
        return operators.GeneralFunction(d, f.domain, layout='g', tensorsig=f.tensorsig, dtype=dtype, func=initialize_scalar, args=args)
    g = F(x, f).evaluate()
    assert np.allclose(g['g'], 2*np.cos(x)+np.sin(x))


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
    b = basis.ShellBasis(c, (Nphi, Ntheta, Nr), radii=radii, dtype=dtype, dealias=(dealias, dealias, dealias))
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


@pytest.mark.parametrize('Nphi', [8])
@pytest.mark.parametrize('Ntheta', [16])
@pytest.mark.parametrize('Nr', [16])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('dealias', [1])
@pytest.mark.parametrize('build_basis', [build_shell, build_ball])
def test_spherical_vector_GeneralFunction(Nphi, Ntheta, Nr, dtype, dealias, build_basis):
    c, d, b, phi, theta, r, x, y, z = build_basis(Nphi, Ntheta, Nr, dtype, dealias)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    f['g'] = x*y*z
    u = operators.Gradient(f).evaluate()
    def initialize_vector(*args):
        r, th, ph = args
        ur = 3*r**2*np.cos(th)*np.sin(th)**2*np.cos(ph)*np.sin(ph)
        uth = 1/4*r**2*(1+3*np.cos(2*th))*np.sin(th)*np.sin(2*ph)
        uph = r**2*np.cos(th)*np.sin(th)*np.cos(2*ph)
        return np.array([uph, uth, ur])
    def F(*args):
        return operators.GeneralFunction(d, u.domain, layout='g', tensorsig=u.tensorsig, dtype=dtype, func=initialize_vector, args=args)
    v = F(r, theta, phi).evaluate()
    assert np.allclose(u['g'], v['g'])

