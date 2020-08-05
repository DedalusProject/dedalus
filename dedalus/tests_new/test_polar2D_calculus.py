
import pytest
import numpy as np
from dedalus.core import coords, distributor, basis, field, operators, arithmetic
from dedalus.tools.cache import CachedMethod
from mpi4py import MPI

comm = MPI.COMM_WORLD

Nphi_range = [16]
Nr_range = [8]
dealias_range = [1, 3/2]
radius_disk = [1.5,]
@CachedMethod
def build_D2(Nphi, Nr, radius, dealias, dtype=np.float64):
    c = coords.PolarCoordinates('phi', 'r')
    d = distributor.Distributor((c,))
    db = basis.DiskBasis(c, (Nphi, Nr), radius=radius, dealias=(dealias, dealias), dtype=dtype)
    phi, r = db.local_grids()
    x, y = c.cartesian(phi, r)
    return c, d, db, phi, r, x, y

#radii_shell = (0.5, 3)
# @CachedMethod
# def build_annulus(Nphi, Ntheta, Nr, dealias, dtype=np.complex128):
#     c = coords.SphericalCoordinates('phi', 'theta', 'r')
#     d = distributor.Distributor((c,))
#     b = basis.SphericalShellBasis(c, (Nphi, Ntheta, Nr), radii=radii_shell, dealias=(dealias, dealias, dealias), dtype=dtype)
#     phi, theta, r = b.local_grids()
#     x, y, z = c.cartesian(phi, theta, r)
#     return c, d, b, phi, theta, r, x, y, z

@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('radius', radius_disk)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_D2])
@pytest.mark.parametrize('dtype', [np.float64])
def test_gradient_scalar(Nphi, Nr, radius, dealias, basis, dtype):
    c, d, b, phi, r, x, y = basis(Nphi, Nr, radius, dealias, dtype)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    f['g'] = fg = 3*x**2 + 2*y
    u = operators.Gradient(f, c).evaluate()
    phi, r = b.local_grids(b.domain.dealias)
    x, y = c.cartesian(phi, r)

    ex = np.array([-np.sin(phi)+0.*r,np.cos(phi)+0.*r])
    ey = np.array([np.cos(phi)+0.*r,np.sin(phi)+0.*r])
    ug = 6*x*ex + 2*ey
    
    assert np.allclose(u['g'], ug)

# @pytest.mark.parametrize('Nr', Nr_range)
# @pytest.mark.parametrize('basis', [build_ball, build_shell])
# @pytest.mark.parametrize('dtype', [np.float64, np.complex128])
# def test_gradient_1D_scalar(Nr, basis, dtype):
#     c, d, b, phi, theta, r, x, y, z = basis(1, 1, Nr, 1, dtype=dtype)
#     f = field.Field(dist=d, bases=(b,), dtype=dtype)
#     f['g'] = fg = r**4/3
#     u = operators.Gradient(f, c).evaluate()
#     print(np.max(np.abs(u['c'])))
#     u.towards_grid_space()
#     print(u.data.shape)
#     print(np.max(np.abs(u.data)))
#     u.towards_grid_space()
#     print(u.data.shape)
#     print(np.max(np.abs(u.data)))
#     u.towards_grid_space()
#     print(u.data.shape)
#     print(np.max(np.abs(u.data)))
#     phi, theta, r = b.local_grids(b.domain.dealias)
#     x, y, z = c.cartesian(phi, theta, r)
#     ug = np.copy(u['g'])
#     ug[2] = 4/3*r**3
#     assert np.allclose(u['g'], ug)

@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('radius', radius_disk)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_D2])
@pytest.mark.parametrize('dtype', [np.float64])
def test_gradient_vector(Nphi, Nr, radius, dealias, basis, dtype):
    c, d, b, phi, r, x, y = basis(Nphi, Nr, radius, dealias, dtype)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    grad = lambda A: operators.Gradient(A, c)

    f['g'] = fg = 3*x**4 + 2*y*x
    T = grad(grad(f)).evaluate()

    phi, r = b.local_grids(b.domain.dealias)
    x, y = c.cartesian(phi, r)

    ex = np.array([-np.sin(phi)+0.*r,np.cos(phi)+0.*r])
    ey = np.array([np.cos(phi)+0.*r,np.sin(phi)+0.*r])
    exex = ex[:,None, ...] * ex[None,...]
    eyex = ey[:,None, ...] * ex[None,...]
    exey = ex[:,None, ...] * ey[None,...]
    eyey = ey[:,None, ...] * ey[None,...]

    Tg = 36*x**2*exex + 2*(exey + eyex)
    
    assert np.allclose(T['g'], Tg)


# @pytest.mark.parametrize('Nr', Nr_range)
# @pytest.mark.parametrize('basis', [build_ball, build_shell])
# @pytest.mark.parametrize('dtype', [np.float64, np.complex128])
# def test_gradient_1D_vector(Nr, basis, dtype):
#     # note: only tests gradient of a radial vector, which is one of three components of ell=0 rank 2 tensor
#     c, d, b, phi, theta, r, x, y, z = basis(1, 1, Nr, 1, dtype=dtype)
#     f = field.Field(dist=d, bases=(b,), dtype=dtype)
#     f['g'] = r**4 / 3
#     grad = lambda A: operators.Gradient(A, c)
#     T = grad(grad(f)).evaluate()
#     phi, theta, r = b.local_grids(b.domain.dealias)
#     Tg = np.copy(T['g'])
#     Tg[2,2] = 12/3 * r**2
#     assert np.allclose(T['g'], Tg)

@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('radius', radius_disk)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_D2])
@pytest.mark.parametrize('dtype', [np.float64])
def test_divergence_vector(Nphi, Nr, radius, dealias, basis, dtype):
    c, d, b, phi, r, x, y = basis(Nphi, Nr, radius, dealias, dtype)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    grad = lambda A: operators.Gradient(A, c)
    div = lambda A: operators.Divergence(A)

    f['g'] = fg = 3*x**4 + 2*y*x
    S = div(grad(f)).evaluate()

    phi, r = b.local_grids(b.domain.dealias)
    x, y = c.cartesian(phi, r)

    Sg = 36*x**2
    
    assert np.allclose(S['g'], Sg)

@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('radius', radius_disk)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_D2])
@pytest.mark.parametrize('dtype', [np.float64])
def test_divergence_tensor(Nphi, Nr, radius, dealias, basis, dtype):
    c, d, b, phi, r, x, y = basis(Nphi, Nr, radius, dealias, dtype)
    v = field.Field(dist=d, tensorsig=(c,), bases=(b,), dtype=dtype)
    grad = lambda A: operators.Gradient(A, c)
    div = lambda A: operators.Divergence(A)

    phi, r = b.local_grids(b.domain.dealias)
    x, y = c.cartesian(phi, r)

    ex = np.array([-np.sin(phi)+0.*r,np.cos(phi)+0.*r])
    ey = np.array([np.cos(phi)+0.*r,np.sin(phi)+0.*r])
    v.set_scales(b.dealias)
    
    v['g'] = 4*x**3*ey + 3*y**2*ey
    U = div(grad(v)).evaluate()

    phi, r = b.local_grids(b.domain.dealias)
    x, y = c.cartesian(phi, r)

    Ug = (24*x + 6)*ey
    
    assert np.allclose(U['g'], Ug)

# @pytest.mark.parametrize('Nr', Nr_range)
# @pytest.mark.parametrize('basis', [build_ball, build_shell])
# @pytest.mark.parametrize('dtype', [np.float64, np.complex128])
# def test_divergence_1D_vector(Nr, basis, dtype):
#     c, d, b, phi, theta, r, x, y, z = basis(1, 1, Nr, 1, dtype=dtype)
#     f = field.Field(dist=d, bases=(b,), dtype=dtype)
#     f['g'] = r**4/3
#     u = operators.Gradient(f, c)
#     h = operators.Divergence(u).evaluate()
#     phi, theta, r = b.local_grids(b.domain.dealias)
#     hg = 20/3*r**2
#     assert np.allclose(h['g'], hg)

@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('radius', radius_disk)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_D2])
@pytest.mark.parametrize('dtype', [np.float64])
def test_curl_vector(Nphi, Nr, radius, dealias, basis, dtype):
    c, d, b, phi, r, x, y = basis(Nphi, Nr, radius, dealias, dtype)
    v = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
    phi, r = b.local_grids(b.domain.dealias)
    x, y = c.cartesian(phi, r)

    ex = np.array([-np.sin(phi)+0.*r,np.cos(phi)+0.*r])
    ey = np.array([np.cos(phi)+0.*r,np.sin(phi)+0.*r])
    v.set_scales(b.dealias)
    
    v['g'] = 4*x**3*ey + 3*y**2*ey

    u = operators.Curl(v).evaluate()

    ug = 12*x**2
    assert np.allclose(u['g'], ug)

@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('radius', radius_disk)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_D2])
@pytest.mark.parametrize('dtype', [np.float64])
def test_laplacian_scalar(Nphi,  Nr, radius, dealias, basis, dtype):
    c, d, b, phi, r, x, y = basis(Nphi, Nr, radius, dealias, dtype)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    f['g'] = x**4 + 2*y**4
    h = operators.Laplacian(f, c).evaluate()
    phi, r = b.local_grids(b.domain.dealias)
    x, y = c.cartesian(phi, r)
    hg = 12*x**2+24*y**2
    assert np.allclose(h['g'], hg)

# @pytest.mark.parametrize('Nr', Nr_range)
# @pytest.mark.parametrize('basis', [build_ball, build_shell])
# @pytest.mark.parametrize('dtype', [np.float64, np.complex128])
# def test_laplacian_1D_scalar(Nr, basis, dtype):
#     c, d, b, phi, theta, r, x, y, z = basis(1, 1, Nr, 1, dtype=dtype)
#     f = field.Field(dist=d, bases=(b,), dtype=dtype)
#     f['g'] = r**4 / 3
#     h = operators.Laplacian(f, c).evaluate()
#     phi, theta, r = b.local_grids(b.domain.dealias)
#     hg = 20/3 * r**2
#     assert np.allclose(h['g'], hg)

@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('radius', radius_disk)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_D2])
@pytest.mark.parametrize('dtype', [np.float64])
def test_laplacian_vector(Nphi,  Nr, radius, dealias, basis, dtype):
    c, d, b, phi, r, x, y = basis(Nphi, Nr, radius, dealias, dtype)
    v = field.Field(dist=d, tensorsig=(c,), bases=(b,), dtype=dtype)

    phi, r = b.local_grids(b.domain.dealias)
    x, y = c.cartesian(phi, r)

    ex = np.array([-np.sin(phi)+0.*r,np.cos(phi)+0.*r])
    ey = np.array([np.cos(phi)+0.*r,np.sin(phi)+0.*r])
    v.set_scales(b.dealias)
    
    v['g'] = 4*x**3*ey + 3*y**2*ey

    U = operators.Laplacian(v,c).evaluate()

    Ug = (24*x + 6)*ey
    assert np.allclose(U['g'], Ug)

# @pytest.mark.parametrize('Nphi', [16])
# @pytest.mark.parametrize('Ntheta', [16])
# @pytest.mark.parametrize('Nr', [8])
# @pytest.mark.parametrize('dealias', dealias_range)
# @pytest.mark.parametrize('basis', [build_ball, build_shell])
# @pytest.mark.parametrize('dtype', [np.float64, np.complex128])
# def test_laplacian_vector(Nphi, Ntheta, Nr, dealias, basis, dtype):
#     c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, dealias, dtype)
#     u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
#     ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
#     u['g'][2] = r**2*st*(2*ct**2*cp-r*ct**3*sp+r**3*cp**3*st**5*sp**3+r*ct*st**2*(cp**3+sp**3))
#     u['g'][1] = r**2*(2*ct**3*cp-r*cp**3*st**4+r**3*ct*cp**3*st**5*sp**3-1/16*r*np.sin(2*theta)**2*(-7*sp+np.sin(3*phi)))
#     u['g'][0] = r**2*sp*(-2*ct**2+r*ct*cp*st**2*sp-r**3*cp**2*st**5*sp**3)
#     v = operators.Laplacian(u, c).evaluate()
#     vg = 0*v['g']
#     phi, theta, r = b.local_grids(b.domain.dealias)
#     ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
#     x, y, z = c.cartesian(phi, theta, r)
#     vg[2] = 2*(2+3*r*ct)*cp*st+1/2*r**3*st**4*(4*np.sin(2*phi)+np.sin(4*phi))
#     vg[1] = 2*r*(-3*cp*st**2+sp)+1/2*ct*(8*cp+r**3*st**3*(4*np.sin(2*phi)+np.sin(4*phi)))
#     vg[0] = 2*r*ct*cp+2*sp*(-2-r**3*(2+np.cos(2*phi))*st**3*sp)
#     assert np.allclose(v['g'],vg)

# @pytest.mark.parametrize('Nr', Nr_range)
# @pytest.mark.parametrize('basis', [build_ball, build_shell])
# @pytest.mark.parametrize('dtype', [np.float64, np.complex128])
# def test_laplacian_1D_vector(Nr, basis, dtype):
#     c, d, b, phi, theta, r, x, y, z = basis(1, 1, Nr, 1, dtype=dtype)
#     u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
#     u['g'][2] = 4/3 * r**3
#     v = operators.Laplacian(u, c).evaluate()
#     vg = 0*v['g']
#     phi, theta, r = b.local_grids(b.domain.dealias)
#     vg[2] = 40/3 * r
#     assert np.allclose(v['g'],vg)


