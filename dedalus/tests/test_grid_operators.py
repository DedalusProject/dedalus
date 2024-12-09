"""Test ufuncs and GeneralFunctions."""

import pytest
import numpy as np
import dedalus.public as d3
from dedalus.tools.cache import CachedMethod


N_range = [16]
dealias_range = [1]
dtype_range = [np.float64, np.complex128]
ufuncs = d3.UnaryGridFunction.ufunc_derivatives.keys()


@CachedMethod
def build_ball(N, dealias, dtype, radius=1):
    c = d3.SphericalCoordinates('phi', 'theta', 'r')
    d = d3.Distributor(c, dtype=dtype)
    b = d3.BallBasis(c, (2*N, N, N), radius=radius, dtype=dtype, dealias=dealias)
    phi, theta, r = d.local_grids(b)
    x, y, z = c.cartesian(phi, theta, r)
    return c, d, b, phi, theta, r, x, y, z


@CachedMethod
def build_shell(N, dealias, dtype, radii=(0.5,1)):
    c = d3.SphericalCoordinates('phi', 'theta', 'r')
    d = d3.Distributor(c, dtype=dtype)
    b = d3.ShellBasis(c, (2*N, N, N), radii=radii, dtype=dtype, dealias=dealias)
    phi, theta, r = d.local_grids(b)
    x, y, z = c.cartesian(phi, theta, r)
    return c, d, b, phi, theta, r, x, y, z


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('a, b', [(-1/2, -1/2), (0, 0)])
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('func', ufuncs)
def test_jacobi_ufunc_field(N, a, b, dealias, dtype, func):
    """Test ufuncs on field arguments with Jacobi basis."""
    c = d3.Coordinate('x')
    d = d3.Distributor(c, dtype=dtype)
    b = d3.Jacobi(c, size=N, a=a, b=b, bounds=(0, 1), dealias=dealias)
    x = d.local_grid(b, scale=1)
    f = d.Field(bases=b)
    if func is np.arccosh:
        f['g'] = 1 + x**2
    else:
        f['g'] = x**2
    assert np.allclose(func(f)['g'], func(f['g']))


@pytest.mark.parametrize('build_basis', [build_shell, build_ball])
@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('func', ufuncs)
def test_spherical_ufunc_field(build_basis, N, dealias, dtype, func):
    """Test ufuncs on field arguments with spherical bases."""
    c, d, b, phi, theta, r, x, y, z = build_basis(N, dealias, dtype)
    f = d.Field(bases=b)
    if func is np.arccosh:
        f['g'] = 1 + r**2
    else:
        f['g'] = r**2
    assert np.allclose(func(f)['g'], func(f['g']))


@pytest.mark.parametrize('build_basis', [build_shell, build_ball])
@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('func', ufuncs)
def test_spherical_ufunc_operator(build_basis, N, dealias, dtype, func):
    """Test ufuncs on operator arguments with spherical bases."""
    c, d, b, phi, theta, r, x, y, z = build_basis(N, dealias, dtype)
    f = d.Field(bases=b)
    if func is np.arccosh:
        f['g'] = 1 + r**2
        a = 2
    else:
        f['g'] = r**2
        a = 0.5
    assert np.allclose(func(a*f)['g'], func(a*f['g']))


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('a, b', [(-1/2, -1/2), (0, 0)])
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_jacobi_GeneralFunction_coord(N, a, b, dealias, dtype):
    """Test GeneralFunction on coordinate arguments with Jacobi basis."""
    c = d3.Coordinate('x')
    d = d3.Distributor(c, dtype=dtype)
    b = d3.Jacobi(c, size=N, a=a, b=b, bounds=(0, 1), dealias=dealias)
    x = d.local_grid(b, dealias)
    f = d.Field(bases=b)
    def F(x):
        return np.sin(x)
    def F_op(*args):
        return d3.GeneralFunction(d, f.domain, layout='g', tensorsig=f.tensorsig, dtype=dtype, func=F, args=args)
    assert np.allclose(F_op(x)['g'], F(x))


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('a, b', [(-1/2, -1/2), (0, 0)])
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_jacobi_GeneralFunction_field(N, a, b, dealias, dtype):
    """Test GeneralFunction on field arguments with Jacobi basis."""
    c = d3.Coordinate('x')
    d = d3.Distributor(c, dtype=dtype)
    b = d3.Jacobi(c, size=N, a=a, b=b, bounds=(0, 1), dealias=dealias)
    x = d.local_grid(b, dealias)
    f = d.Field(bases=b)
    f.preset_scales(dealias)
    f['g'] = np.cos(x)
    def F(x, f):
        return 2*f['g'] + np.sin(x)
    def F_op(*args):
        return d3.GeneralFunction(d, f.domain, layout='g', tensorsig=f.tensorsig, dtype=dtype, func=F, args=args)
    assert np.allclose(F_op(x, f)['g'], F(x, f))


@pytest.mark.parametrize('build_basis', [build_shell, build_ball])
@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_spherical_GeneralFunction_coord(build_basis, N, dealias, dtype):
    """Test GeneralFunction on coordinate arguments with spherical bases."""
    c, d, b, phi, theta, r, x, y, z = build_basis(N, dealias, dtype)
    u = d.VectorField(c, bases=b)
    def F(ph, th, r):
        ur = 3*r**2*np.cos(th)*np.sin(th)**2*np.cos(ph)*np.sin(ph)
        uth = 1/4*r**2*(1+3*np.cos(2*th))*np.sin(th)*np.sin(2*ph)
        uph = r**2*np.cos(th)*np.sin(th)*np.cos(2*ph)
        return np.array([uph, uth, ur])
    def F_op(*args):
        return d3.GeneralFunction(d, u.domain, layout='g', tensorsig=u.tensorsig, dtype=dtype, func=F, args=args)
    assert np.allclose(F_op(phi, theta, r)['g'], F(phi, theta, r))

