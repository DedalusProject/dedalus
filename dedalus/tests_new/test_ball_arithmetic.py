
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
def test_ball_cross_product(Nphi, Ntheta, Nr, radius, dealias):
    c, d, b, phi, theta, r, x, y, z = build_ball(Nphi, Ntheta, Nr, radius, dealias)
    x, y, z = cartesian(phi, theta, r)
    f = field.Field(dist=d, bases=(b,), dtype=np.complex128)
    f['g'] = z
    ez = operators.Gradient(f, c).evaluate()
    u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
    u['g'][2] = (6*x**2+4*y*z)/r
    u['g'][1] = -2*(y**3+x**2*(y-3*z)-y*z**2)/(r**2*np.sin(theta))
    u['g'][0] = 2*x*(-3*y+z)/(r*np.sin(theta))
    h = arithmetic.CrossProduct(ez,u).evaluate()
    hg = np.zeros(h['g'].shape, dtype=h['g'].dtype)
    hg[0] = - ez['g'][1]*u['g'][2] + ez['g'][2]*u['g'][1]
    hg[1] = - ez['g'][2]*u['g'][0] + ez['g'][0]*u['g'][2]
    hg[2] = - ez['g'][0]*u['g'][1] + ez['g'][1]*u['g'][0]
    assert np.allclose(h['g'],hg)

@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('radius', radius_range)
@pytest.mark.parametrize('dealias', dealias_range)
def test_ball_dot_product_vector_vector(Nphi, Ntheta, Nr, radius, dealias):
    c, d, b, phi, theta, r, x, y, z = build_ball(Nphi, Ntheta, Nr, radius, dealias)
    x, y, z = cartesian(phi, theta, r)
    f = field.Field(dist=d, bases=(b,), dtype=np.complex128)
    f['g'] = z
    ez = operators.Gradient(f, c).evaluate()
    u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
    u['g'][2] = (6*x**2+4*y*z)/r
    u['g'][1] = -2*(y**3+x**2*(y-3*z)-y*z**2)/(r**2*np.sin(theta))
    u['g'][0] = 2*x*(-3*y+z)/(r*np.sin(theta))
    h = arithmetic.DotProduct(ez,u).evaluate()
    hg = np.sum(ez['g']*u['g'],axis=0)
    assert np.allclose(h['g'],hg)

@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('radius', radius_range)
@pytest.mark.parametrize('dealias', dealias_range)
def test_ball_dot_product_tensor_vector(Nphi, Ntheta, Nr, radius, dealias):
    c, d, b, phi, theta, r, x, y, z = build_ball(Nphi, Ntheta, Nr, radius, dealias)
    x, y, z = cartesian(phi, theta, r)
    u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
    u['g'][2] = (6*x**2+4*y*z)/r
    u['g'][1] = -2*(y**3+x**2*(y-3*z)-y*z**2)/(r**2*np.sin(theta))
    u['g'][0] = 2*x*(-3*y+z)/(r*np.sin(theta))
    T = field.Field(dist=d, bases=(b,), tensorsig=(c,c), dtype=np.complex128)
    T['g'][2,2] = (6*x**2+4*y*z)/r**2
    T['g'][2,1] = T['g'][1,2] = -2*(y**3+x**2*(y-3*z)-y*z**2)/(r**3*np.sin(theta))
    T['g'][2,0] = T['g'][0,2] = 2*x*(z-3*y)/(r**2*np.sin(theta))
    T['g'][1,1] = 6*x**2/(r**2*np.sin(theta)**2) - (6*x**2+4*y*z)/r**2
    T['g'][1,0] = T['g'][0,1] = -2*x*(x**2+y**2+3*y*z)/(r**3*np.sin(theta)**2)
    T['g'][0,0] = 6*y**2/(x**2+y**2)
    v = arithmetic.DotProduct(T,u).evaluate()
    vg = np.sum(T['g']*u['g'][:,None,:,:,:],axis=0)
    assert np.allclose(v['g'], vg)

