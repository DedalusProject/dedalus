
import pytest
import numpy as np
from dedalus.core import coords, distributor, basis, field, operators, arithmetic
from dedalus.tools.cache import CachedMethod
from mpi4py import MPI

comm = MPI.COMM_WORLD

## Ball
Nphi_range = [8]
Ntheta_range = [10]
Nr_range = [6]
radius_range = [1.5]
dealias_range = [1, 3/2]

def cartesian(phi, theta, r):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

@CachedMethod
def build_ball(Nphi, Ntheta, Nr, radius, dealias):
    c = coords.SphericalCoordinates('phi', 'theta', 'r')
    d = distributor.Distributor((c,))
    b = basis.BallBasis(c, (Nphi, Ntheta, Nr), radius=radius, dealias=(dealias, dealias, dealias))
    phi, theta, r = b.local_grids()
    x, y, z = cartesian(phi, theta, r)
    return c, d, b, phi, theta, r, x, y, z

@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('radius', radius_range)
@pytest.mark.parametrize('dealias', dealias_range)
def test_ball_gradient_scalar(Nphi, Ntheta, Nr, radius, dealias):
    c, d, b, phi, theta, r, x, y, z = build_ball(Nphi, Ntheta, Nr, radius, dealias)
    f = field.Field(dist=d, bases=(b,), dtype=np.complex128)
    f['g'] = fg = 3*x**2 + 2*y*z
    u = operators.Gradient(f, c).evaluate()
    phi, theta, r = b.local_grids(b.domain.dealias)
    x, y, z = cartesian(phi, theta, r)
    ug = np.copy(u['g'])
    ug[2] = (6*x**2+4*y*z)/r
    ug[1] = -2*(y**3+x**2*(y-3*z)-y*z**2)/(r**2*np.sin(theta))
    ug[0] = 2*x*(-3*y+z)/(r*np.sin(theta))
    assert np.allclose(u['g'], ug)

@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('radius', radius_range)
@pytest.mark.parametrize('dealias', dealias_range)
def test_ball_gradient_vector(Nphi, Ntheta, Nr, radius, dealias):
    c, d, b, phi, theta, r, x, y, z = build_ball(Nphi, Ntheta, Nr, radius, dealias)
    f = field.Field(dist=d, bases=(b,), dtype=np.complex128)
    f['g'] = 3*x**2 + 2*y*z
    grad = lambda A: operators.Gradient(A, c)
    T = grad(grad(f)).evaluate()
    phi, theta, r = b.local_grids(b.domain.dealias)
    x, y, z = cartesian(phi, theta, r)
    Tg = np.copy(T['g'])
    Tg[2,2] = (6*x**2+4*y*z)/r**2
    Tg[2,1] = Tg[1,2] = -2*(y**3+x**2*(y-3*z)-y*z**2)/(r**3*np.sin(theta))
    Tg[2,0] = Tg[0,2] = 2*x*(z-3*y)/(r**2*np.sin(theta))
    Tg[1,1] = 6*x**2/(r**2*np.sin(theta)**2) - (6*x**2+4*y*z)/r**2
    Tg[1,0] = Tg[0,1] = -2*x*(x**2+y**2+3*y*z)/(r**3*np.sin(theta)**2)
    Tg[0,0] = 6*y**2/(x**2+y**2)
    assert np.allclose(T['g'], Tg)

@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('radius', radius_range)
@pytest.mark.parametrize('dealias', dealias_range)
def test_ball_divergence_vector(Nphi, Ntheta, Nr, radius, dealias):
    c, d, b, phi, theta, r, x, y, z = build_ball(Nphi, Ntheta, Nr, radius, dealias)
    f = field.Field(dist=d, bases=(b,), dtype=np.complex128)
    f['g'] = x**3 + 2*y**3 + 3*z**3
    u = operators.Gradient(f, c)
    h = operators.Divergence(u).evaluate()
    phi, theta, r = b.local_grids(b.domain.dealias)
    x, y, z = cartesian(phi, theta, r)
    hg = 6*x + 12*y + 18*z
    assert np.allclose(h['g'], hg)

# test function needs higher resolution
@pytest.mark.parametrize('Nphi', [16])
@pytest.mark.parametrize('Ntheta', [16])
@pytest.mark.parametrize('Nr', [8])
@pytest.mark.parametrize('radius', radius_range)
@pytest.mark.parametrize('dealias', dealias_range)
def test_ball_curl_vector(Nphi, Ntheta, Nr, radius, dealias):
    c, d, b, phi, theta, r, x, y, z = build_ball(Nphi, Ntheta, Nr, radius, dealias)
    u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
    ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
    u['g'][2] = r**2*st*(2*ct**2*cp-r*ct**3*sp+r**3*cp**3*st**5*sp**3+r*ct*st**2*(cp**3+sp**3))
    u['g'][1] = r**2*(2*ct**3*cp-r*cp**3*st**4+r**3*ct*cp**3*st**5*sp**3-1/16*r*np.sin(2*theta)**2*(-7*sp+np.sin(3*phi)))
    u['g'][0] = r**2*sp*(-2*ct**2+r*ct*cp*st**2*sp-r**3*cp**2*st**5*sp**3)
    v = operators.Curl(u).evaluate()
    vg = 0*u['g']
    phi, theta, r = b.local_grids(b.domain.dealias)
    ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
    vg[2] = -r*st*(r*ct**2*cp+r*cp*st**2*sp*(3*cp+sp)+ct*sp*(-4+3*r**3*cp**2*st**3*sp))
    vg[1] = r*(-r*ct**3*cp+4*ct**2*sp+3*r**3*cp**2*st**5*sp**2-r*ct*cp*st**2*sp*(3*cp+sp))
    vg[0] = r*(4*ct*cp+r*ct**2*sp+r*st**2*(-3*cp**3+sp**3))
    assert np.allclose(v['g'], vg)

@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('radius', radius_range)
@pytest.mark.parametrize('dealias', dealias_range)
def test_ball_laplacian_scalar(Nphi, Ntheta, Nr, radius, dealias):
    c, d, b, phi, theta, r, x, y, z = build_ball(Nphi, Ntheta, Nr, radius, dealias)
    f = field.Field(dist=d, bases=(b,), dtype=np.complex128)
    f['g'] = x**4 + 2*y**4 + 3*z**4
    h = operators.Laplacian(f, c).evaluate()
    phi, theta, r = b.local_grids(b.domain.dealias)
    x, y, z = cartesian(phi, theta, r)
    hg = 12*x**2+24*y**2+36*z**2
    assert np.allclose(h['g'], hg)

# test function needs higher resolution
@pytest.mark.parametrize('Nphi', [16])
@pytest.mark.parametrize('Ntheta', [16])
@pytest.mark.parametrize('Nr', [8])
@pytest.mark.parametrize('radius', radius_range)
@pytest.mark.parametrize('dealias', dealias_range)
def test_ball_laplacian_vector(Nphi, Ntheta, Nr, radius, dealias):
    c, d, b, phi, theta, r, x, y, z = build_ball(Nphi, Ntheta, Nr, radius, dealias)
    u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
    ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
    u['g'][2] = r**2*st*(2*ct**2*cp-r*ct**3*sp+r**3*cp**3*st**5*sp**3+r*ct*st**2*(cp**3+sp**3))
    u['g'][1] = r**2*(2*ct**3*cp-r*cp**3*st**4+r**3*ct*cp**3*st**5*sp**3-1/16*r*np.sin(2*theta)**2*(-7*sp+np.sin(3*phi)))
    u['g'][0] = r**2*sp*(-2*ct**2+r*ct*cp*st**2*sp-r**3*cp**2*st**5*sp**3)
    v = operators.Laplacian(u, c).evaluate()
    vg = 0*v['g']
    phi, theta, r = b.local_grids(b.domain.dealias)
    ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
    x, y, z = cartesian(phi, theta, r)
    vg[2] = 2*(2+3*r*ct)*cp*st+1/2*r**3*st**4*(4*np.sin(2*phi)+np.sin(4*phi))
    vg[1] = 2*r*(-3*cp*st**2+sp)+1/2*ct*(8*cp+r**3*st**3*(4*np.sin(2*phi)+np.sin(4*phi)))
    vg[0] = 2*r*ct*cp+2*sp*(-2-r**3*(2+np.cos(2*phi))*st**3*sp)
    assert np.allclose(v['g'],vg)

