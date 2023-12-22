"""Ball and shell tests for gradient, divergence, curl, laplacian."""

import pytest
import numpy as np
from dedalus.core import coords, distributor, basis, field, operators, arithmetic
from dedalus.tools.cache import CachedFunction


Nphi_range = [16]
Ntheta_range = [8]
Nr_range = [8]
dealias_range = [1, 3/2]
radius_ball = 1.5
radii_shell = (0.5, 3)


@CachedFunction
def build_ball(Nphi, Ntheta, Nr, dealias, dtype):
    c = coords.SphericalCoordinates('phi', 'theta', 'r')
    d = distributor.Distributor((c,))
    b = basis.BallBasis(c, (Nphi, Ntheta, Nr), radius=radius_ball, dealias=(dealias, dealias, dealias), dtype=dtype)
    phi, theta, r = d.local_grids(b, scales=dealias)
    x, y, z = c.cartesian(phi, theta, r)
    return c, d, b, phi, theta, r, x, y, z


@CachedFunction
def build_shell(Nphi, Ntheta, Nr, dealias, dtype):
    c = coords.SphericalCoordinates('phi', 'theta', 'r')
    d = distributor.Distributor((c,))
    b = basis.ShellBasis(c, (Nphi, Ntheta, Nr), radii=radii_shell, dealias=(dealias, dealias, dealias), dtype=dtype)
    phi, theta, r = d.local_grids(b, scales=dealias)
    x, y, z = c.cartesian(phi, theta, r)
    return c, d, b, phi, theta, r, x, y, z


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_ball, build_shell])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_gradient_scalar(Nphi, Ntheta, Nr, dealias, basis, dtype):
    c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, dealias, dtype)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    f.preset_scales(dealias)
    f['g'] = fg = 3*x**2 + 2*y*z
    u = operators.Gradient(f, c).evaluate()
    ug = 0 * u['g']
    ug[2] = (6*x**2+4*y*z)/r
    ug[1] = -2*(y**3+x**2*(y-3*z)-y*z**2)/(r**2*np.sin(theta))
    ug[0] = 2*x*(-3*y+z)/(r*np.sin(theta))
    assert np.allclose(u['g'], ug)


@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_ball, build_shell])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_gradient_radial_scalar(Nr, dealias, basis, dtype):
    Nphi = Ntheta = 1
    c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, dealias, dtype)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    f.preset_scales(dealias)
    f['g'] = fg = r**4/3
    u = operators.Gradient(f, c).evaluate()
    ug = 0 * u['g']
    ug[2] = 4/3*r**3
    assert np.allclose(u['g'], ug)


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_ball, build_shell])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_gradient_vector(Nphi, Ntheta, Nr, dealias, basis, dtype):
    c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, dealias, dtype)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    f.preset_scales(dealias)
    f['g'] = 3*x**2 + 2*y*z
    grad = lambda A: operators.Gradient(A, c)
    T = grad(grad(f)).evaluate()
    Tg = 0 * T['g']
    Tg[2,2] = (6*x**2+4*y*z)/r**2
    Tg[2,1] = Tg[1,2] = -2*(y**3+x**2*(y-3*z)-y*z**2)/(r**3*np.sin(theta))
    Tg[2,0] = Tg[0,2] = 2*x*(z-3*y)/(r**2*np.sin(theta))
    Tg[1,1] = 6*x**2/(r**2*np.sin(theta)**2) - (6*x**2+4*y*z)/r**2
    Tg[1,0] = Tg[0,1] = -2*x*(x**2+y**2+3*y*z)/(r**3*np.sin(theta)**2)
    Tg[0,0] = 6*y**2/(x**2+y**2)
    assert np.allclose(T['g'], Tg)


@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_ball, build_shell])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_gradient_radial_vector(Nr, dealias, basis, dtype):
    Nphi = Ntheta = 1
    c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, dealias, dtype=dtype)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    f.preset_scales(dealias)
    f['g'] = r**4 / 3
    grad = lambda A: operators.Gradient(A, c)
    T = grad(grad(f)).evaluate()
    Tg = 0 * T['g']
    Tg[0,0] = 4 / 3 * r**2
    Tg[1,1] = 4 / 3 * r**2
    Tg[2,2] = 4 * r**2
    assert np.allclose(T['g'], Tg)


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_ball, build_shell])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_divergence_vector(Nphi, Ntheta, Nr, dealias, basis, dtype):
    c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, dealias, dtype)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    f.preset_scales(dealias)
    f['g'] = x**3 + 2*y**3 + 3*z**3
    u = operators.Gradient(f, c)
    h = operators.Divergence(u).evaluate()
    hg = 6*x + 12*y + 18*z
    assert np.allclose(h['g'], hg)


@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_ball, build_shell])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_divergence_radial_vector(Nr, dealias, basis, dtype):
    c, d, b, phi, theta, r, x, y, z = basis(1, 1, Nr, dealias, dtype=dtype)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    f.preset_scales(dealias)
    print(f['g'].shape, r.shape)
    f['g'] = r**4/3
    u = operators.Gradient(f, c)
    h = operators.Divergence(u).evaluate()
    hg = 20/3*r**2
    assert np.allclose(h['g'], hg)


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_ball, build_shell])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_curl_vector(Nphi, Ntheta, Nr, dealias, basis, dtype):
    c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, dealias, dtype)
    u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
    u.preset_scales(dealias)
    ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
    u['g'][2] = r**2*st*(2*ct**2*cp-r*ct**3*sp+r**3*cp**3*st**5*sp**3+r*ct*st**2*(cp**3+sp**3))
    u['g'][1] = r**2*(2*ct**3*cp-r*cp**3*st**4+r**3*ct*cp**3*st**5*sp**3-1/16*r*np.sin(2*theta)**2*(-7*sp+np.sin(3*phi)))
    u['g'][0] = r**2*sp*(-2*ct**2+r*ct*cp*st**2*sp-r**3*cp**2*st**5*sp**3)
    v = operators.Curl(u).evaluate()
    vg = 0 * v['g']
    vg[2] = -r*st*(r*ct**2*cp+r*cp*st**2*sp*(3*cp+sp)+ct*sp*(-4+3*r**3*cp**2*st**3*sp))
    vg[1] = r*(-r*ct**3*cp+4*ct**2*sp+3*r**3*cp**2*st**5*sp**2-r*ct*cp*st**2*sp*(3*cp+sp))
    vg[0] = r*(4*ct*cp+r*ct**2*sp+r*st**2*(-3*cp**3+sp**3))
    assert np.allclose(v['g'], vg)


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_ball, build_shell])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_laplacian_scalar(Nphi, Ntheta, Nr, dealias, basis, dtype):
    c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, dealias, dtype)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    f.preset_scales(dealias)
    f['g'] = x**4 + 2*y**4 + 3*z**4
    h = operators.Laplacian(f, c).evaluate()
    hg = 12*x**2+24*y**2+36*z**2
    assert np.allclose(h['g'], hg)


@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_ball, build_shell])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_laplacian_radial_scalar(Nr, basis, dealias, dtype):
    Nphi = Ntheta = 1
    c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, dealias, dtype=dtype)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    f.preset_scales(dealias)
    f['g'] = r**4 / 3
    h = operators.Laplacian(f, c).evaluate()
    hg = 20/3 * r**2
    assert np.allclose(h['g'], hg)


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_ball, build_shell])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_laplacian_vector(Nphi, Ntheta, Nr, dealias, basis, dtype):
    c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, dealias, dtype)
    u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
    u.preset_scales(dealias)
    ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
    u['g'][2] = r**2*st*(2*ct**2*cp-r*ct**3*sp+r**3*cp**3*st**5*sp**3+r*ct*st**2*(cp**3+sp**3))
    u['g'][1] = r**2*(2*ct**3*cp-r*cp**3*st**4+r**3*ct*cp**3*st**5*sp**3-1/16*r*np.sin(2*theta)**2*(-7*sp+np.sin(3*phi)))
    u['g'][0] = r**2*sp*(-2*ct**2+r*ct*cp*st**2*sp-r**3*cp**2*st**5*sp**3)
    v = operators.Laplacian(u, c).evaluate()
    vg = 0 * v['g']
    vg[2] = 2*(2+3*r*ct)*cp*st+1/2*r**3*st**4*(4*np.sin(2*phi)+np.sin(4*phi))
    vg[1] = 2*r*(-3*cp*st**2+sp)+1/2*ct*(8*cp+r**3*st**3*(4*np.sin(2*phi)+np.sin(4*phi)))
    vg[0] = 2*r*ct*cp+2*sp*(-2-r**3*(2+np.cos(2*phi))*st**3*sp)
    assert np.allclose(v['g'], vg)


@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_ball, build_shell])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_laplacian_radial_vector(Nr, dealias, basis, dtype):
    Nphi = Ntheta = 1
    c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, dealias, dtype=dtype)
    u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
    u.preset_scales(dealias)
    u['g'][2] = 4/3 * r**3
    v = operators.Laplacian(u, c).evaluate()
    vg = 0 * v['g']
    vg[2] = 40/3 * r
    assert np.allclose(v['g'], vg)

