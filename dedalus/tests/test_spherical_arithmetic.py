
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
def build_ball(Nphi, Ntheta, Nr, dealias, dtype):
    c = coords.SphericalCoordinates('phi', 'theta', 'r')
    d = distributor.Distributor((c,))
    b = basis.BallBasis(c, (Nphi, Ntheta, Nr), radius=radius_ball, dealias=(dealias, dealias, dealias), dtype=dtype)
    phi, theta, r = d.local_grids(b)
    x, y, z = c.cartesian(phi, theta, r)
    return c, d, b, phi, theta, r, x, y, z

radii_shell = (0.5, 3)
@CachedMethod
def build_shell(Nphi, Ntheta, Nr, dealias, dtype):
    c = coords.SphericalCoordinates('phi', 'theta', 'r')
    d = distributor.Distributor((c,))
    b = basis.ShellBasis(c, (Nphi, Ntheta, Nr), radii=radii_shell, dealias=(dealias, dealias, dealias), dtype=dtype)
    phi, theta, r = d.local_grids(b)
    x, y, z = c.cartesian(phi, theta, r)
    return c, d, b, phi, theta, r, x, y, z

@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('dealias', dealias_range)
def test_S2_radial_scalar_scalar_multiplication(Nphi, Ntheta, Nr, dealias):
    c, d, b, phi, theta, r, x, y, z = build_shell(Nphi, Ntheta, Nr, dealias, np.complex128)
    f0 = field.Field(dist=d, bases=(b,), dtype=np.complex128)
    f0.preset_scales(dealias)
    phi, theta, r = d.local_grids(b, scales=dealias)
    f0['g'] = (r**2 - 0.5*r**3)*(5*np.cos(theta)**2-1)*np.sin(theta)*np.exp(1j*phi)

    b_S2 = b.S2_basis()
    phi, theta = d.local_grids(b_S2)
    g = field.Field(dist=d, bases=(b_S2,), dtype=np.complex128)
    g['g'] = (5*np.cos(theta)**2-1)*np.sin(theta)*np.exp(1j*phi)

    h = field.Field(dist=d, bases=(b.radial_basis,), dtype=np.complex128)
    h.preset_scales(dealias)
    h['g'] = (r**2 - 0.5*r**3)
    f = (g * h).evaluate()
    assert np.allclose(f['g'], f0['g'])

@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('dealias', dealias_range)
def test_S2_radial_vector_scalar_multiplication(Nphi, Ntheta, Nr, dealias):
    c, d, b, phi, theta, r, x, y, z = build_shell(Nphi, Ntheta, Nr, dealias, np.complex128)
    c_S2 = c.S2coordsys
    v0 = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
    v0.preset_scales(dealias)
    phi, theta, r = d.local_grids(b, scales=dealias)
    v0['g'][0] = (r**2 - 0.5*r**3)*(-1j * np.sin(theta)*np.exp(-2j*phi))
    v0['g'][1] = (r**2 - 0.5*r**3)*(np.cos(theta)*np.sin(theta)*np.exp(-2j*phi))
    v0['g'][2] = (r**2 - 0.5*r**3)*(5*np.cos(theta)**2-1)*np.sin(theta)*np.exp(1j*phi)

    b_S2 = b.S2_basis()
    phi, theta = d.local_grids(b_S2)
    u = field.Field(dist=d, bases=(b_S2,), tensorsig=(c,), dtype=np.complex128)
    u['g'][0] = (-1j * np.sin(theta)*np.exp(-2j*phi))
    u['g'][1] = (np.cos(theta)*np.sin(theta)*np.exp(-2j*phi))
    u['g'][2] = (5*np.cos(theta)**2-1)*np.sin(theta)*np.exp(1j*phi)

    h = field.Field(dist=d, bases=(b.radial_basis,), dtype=np.complex128)
    h.preset_scales(dealias)
    h['g'] = (r**2 - 0.5*r**3)
    v = (h * u).evaluate()
    assert np.allclose(v['g'], v0['g'])

@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_ball, build_shell])
def test_cross_product(Nphi, Ntheta, Nr, dealias, basis):
    c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, dealias, np.complex128)
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
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_ball, build_shell])
def test_dot_product_vector_vector(Nphi, Ntheta, Nr, dealias, basis):
    c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, dealias, np.complex128)
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
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_ball, build_shell])
def test_dot_product_tensor_vector(Nphi, Ntheta, Nr, dealias, basis):
    c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, dealias, np.complex128)
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

@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_ball, build_shell])
def test_multiply_number_scalar(Nphi, Ntheta, Nr, dealias, basis):
    c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, dealias, np.complex128)
    f = field.Field(dist=d, bases=(b,), dtype=np.complex128)
    f['g'] = x**3 + 2*y**3 + 3*z**3
    h = (2 * f).evaluate()
    phi, theta, r = d.local_grids(b, scales=dealias)
    x, y, z = c.cartesian(phi, theta, r)
    hg = 2*(x**3 + 2*y**3 + 3*z**3)
    assert np.allclose(h['g'], hg)

@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_ball, build_shell])
def test_multiply_scalar_number(Nphi, Ntheta, Nr, dealias, basis):
    c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, dealias, np.complex128)
    f = field.Field(dist=d, bases=(b,), dtype=np.complex128)
    f['g'] = x**3 + 2*y**3 + 3*z**3
    h = (f * 2).evaluate()
    phi, theta, r = d.local_grids(b, scales=dealias)
    x, y, z = c.cartesian(phi, theta, r)
    hg = 2*(x**3 + 2*y**3 + 3*z**3)
    assert np.allclose(h['g'], hg)

@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_ball, build_shell])
def test_multiply_scalar_scalar(Nphi, Ntheta, Nr, dealias, basis):
    c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, dealias, np.complex128)
    f = field.Field(dist=d, bases=(b,), dtype=np.complex128)
    f['g'] = x**3 + 2*y**3 + 3*z**3
    h = (f * f).evaluate()
    phi, theta, r = d.local_grids(b, scales=dealias)
    x, y, z = c.cartesian(phi, theta, r)
    hg = (x**3 + 2*y**3 + 3*z**3)**2
    assert np.allclose(h['g'], hg)

@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_ball, build_shell])
def test_multiply_scalar_vector(Nphi, Ntheta, Nr, dealias, basis):
    c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, dealias, np.complex128)
    phi, theta, r = d.local_grids(b, scales=dealias)
    x, y, z = c.cartesian(phi, theta, r)
    f = field.Field(dist=d, bases=(b,), dtype=np.complex128)
    f.preset_scales(dealias)
    f['g'] = x**3 + 2*y**3 + 3*z**3
    u = operators.Gradient(f, c).evaluate()
    v = (f * u).evaluate()
    vg = f['g'][None,...]*u['g']
    assert np.allclose(v['g'], vg)

@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_ball, build_shell])
def test_multiply_vector_vector(Nphi, Ntheta, Nr, dealias, basis):
    c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, dealias, np.complex128)
    f = field.Field(dist=d, bases=(b,), dtype=np.complex128)
    f['g'] = x**3 + 2*y**3 + 3*z**3
    u = operators.Gradient(f, c).evaluate()
    T = (u * u).evaluate()
    Tg = u['g'][None,...] * u['g'][:,None,...]
    assert np.allclose(T['g'], Tg)

@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_ball, build_shell])
def test_multiply_vector_tensor(Nphi, Ntheta, Nr, dealias, basis):
    c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, dealias, np.complex128)
    f = field.Field(dist=d, bases=(b,), dtype=np.complex128)
    f['g'] = x**3 + 2*y**3 + 3*z**3
    u = operators.Gradient(f, c).evaluate()
    T = operators.Gradient(u, c).evaluate()
    Q = (u * T).evaluate()
    Qg = u['g'][:,None,None,...] * T['g'][None,...]
    assert np.allclose(Q['g'], Qg)

@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_ball, build_shell])
def test_multiply_tensor_tensor(Nphi, Ntheta, Nr, dealias, basis):
    c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, dealias, np.complex128)
    f = field.Field(dist=d, bases=(b,), dtype=np.complex128)
    f['g'] = x**3 + 2*y**3 + 3*z**3
    u = operators.Gradient(f, c).evaluate()
    T = operators.Gradient(u, c).evaluate()
    Q = (T * T).evaluate()
    Qg = T['g'][:,:,None,None,...] * T['g'][None,None,...]
    assert np.allclose(Q['g'], Qg)

