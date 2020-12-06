
import pytest
import numpy as np
from dedalus.core import coords, distributor, basis, field, operators, arithmetic
from dedalus.tools.cache import CachedMethod
from mpi4py import MPI

comm = MPI.COMM_WORLD

ball_radius = 1.5

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

# @pytest.mark.parametrize('Nphi', Nphi_range)
# @pytest.mark.parametrize('Ntheta', Ntheta_range)
# @pytest.mark.parametrize('Nr', Nr_range)
# @pytest.mark.parametrize('dealias', dealias_range)
# @pytest.mark.parametrize('basis', [build_ball, build_shell])
# def test_spherical_ell_product_scalar(Nphi, Ntheta, Nr, dealias, basis):
#     c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, dealias)
#     f = field.Field(dist=d, bases=(b,), dtype=np.complex128)
#     g = field.Field(dist=d, bases=(b,), dtype=np.complex128)
#     g.set_scales(b.domain.dealias)
#     f['g'] = 3*x**2 + 2*y*z
#     for ell, m_ind, ell_ind in b.ell_maps:
#         g['c'][m_ind, ell_ind, :] = (ell+3)*f['c'][m_ind, ell_ind, :]
#     func = lambda ell: ell+3
#     h = operators.SphericalEllProduct(f, c, func).evaluate()
#     assert np.allclose(h['g'], g['g'])

# @pytest.mark.parametrize('Nphi', Nphi_range)
# @pytest.mark.parametrize('Ntheta', Ntheta_range)
# @pytest.mark.parametrize('Nr', Nr_range)
# @pytest.mark.parametrize('dealias', dealias_range)
# @pytest.mark.parametrize('basis', [build_ball, build_shell])
# def test_spherical_ell_product_vector(Nphi, Ntheta, Nr, dealias, basis):
#     c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, dealias)
#     f = field.Field(dist=d, bases=(b,), dtype=np.complex128)
#     f['g'] = 3*x**2 + 2*y*z
#     u = operators.Gradient(f, c).evaluate()
#     uk0 = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
#     uk0.set_scales(b.domain.dealias)
#     uk0['g'] = u['g']
#     v = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
#     v.set_scales(b.domain.dealias)
#     for ell, m_ind, ell_ind in b.ell_maps:
#         v['c'][0, m_ind, ell_ind, :] = (ell+2)*uk0['c'][0, m_ind, ell_ind, :]
#         v['c'][1, m_ind, ell_ind, :] = (ell+4)*uk0['c'][1, m_ind, ell_ind, :]
#         v['c'][2, m_ind, ell_ind, :] = (ell+3)*uk0['c'][2, m_ind, ell_ind, :]
#     func = lambda ell: ell+3
#     w = operators.SphericalEllProduct(u, c, func).evaluate()
#     assert np.allclose(w['g'], v['g'])
@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('radius', radius_disk)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_D2])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_convert_k2_scalar(Nphi, Nr, radius, dealias, basis, dtype):
    c, d, b, phi, r, x, y = basis(Nphi, Nr, radius, dealias, dtype)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)

    f['g'] = 3*x**2 + 2*y
    h = operators.Laplacian(f,c).evaluate()
    f.require_coeff_space()
    h.require_coeff_space()
    w_op = (f + h)
    w = w_op.evaluate()
    for fld in [f,h,w]:
        fld.require_scales((1,1))
    ans = f['g']+h['g']

    print("f k = {}".format(f.domain.bases[0].k))
    print("h k = {}".format(h.domain.bases[0].k))
    print("w k = {}".format(w.domain.bases[0].k))
    print("w_op = {}".format(w_op))
    print("w_op.args[0,1] = {}, {}".format(repr(w_op.args[0]), repr(w_op.args[1])))
    assert np.allclose(w['g'],ans)



@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('radius', radius_disk)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_D2])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_convert_k2_vector(Nphi, Nr, radius, dealias, basis, dtype):
    c, d, b, phi, r, x, y = basis(Nphi, Nr, radius, dealias, dtype)
    u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)

    phi, r = b.local_grids(b.domain.dealias)
    x, y = c.cartesian(phi, r)

    ex = np.array([-np.sin(phi)+0.*r,np.cos(phi)+0.*r])
    ey = np.array([np.cos(phi)+0.*r,np.sin(phi)+0.*r])
    u.set_scales(b.dealias)
    
    u['g'] = 4*x**3*ey + 3*y**2*ey
    v = operators.Laplacian(u, c).evaluate()
    u.require_coeff_space()
    v.require_coeff_space()
    w = (u + v).evaluate()

    assert np.allclose(w['g'],u['g']+v['g'])

# @pytest.mark.parametrize('Nphi', Nphi_range)
# @pytest.mark.parametrize('Ntheta', Ntheta_range)
# @pytest.mark.parametrize('Nr', Nr_range)
# @pytest.mark.parametrize('dealias', dealias_range)
# @pytest.mark.parametrize('basis', [build_ball, build_shell])
# def test_trace_tensor(Nphi, Ntheta, Nr, dealias, basis):
#     c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, dealias)
#     u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
#     ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
#     u['g'][2] = r**2*st*(2*ct**2*cp-r*ct**3*sp+r**3*cp**3*st**5*sp**3+r*ct*st**2*(cp**3+sp**3))
#     u['g'][1] = r**2*(2*ct**3*cp-r*cp**3*st**4+r**3*ct*cp**3*st**5*sp**3-1/16*r*np.sin(2*theta)**2*(-7*sp+np.sin(3*phi)))
#     u['g'][0] = r**2*sp*(-2*ct**2+r*ct*cp*st**2*sp-r**3*cp**2*st**5*sp**3)
#     T = operators.Gradient(u, c).evaluate()
#     T.require_grid_space()
#     fg = T['g'][0,0] + T['g'][1,1] + T['g'][2,2]
#     T.require_coeff_space()
#     f = operators.Trace(T).evaluate()
#     assert np.allclose(f['g'], fg)

# @pytest.mark.parametrize('Nphi', Nphi_range)
# @pytest.mark.parametrize('Ntheta', Ntheta_range)
# @pytest.mark.parametrize('Nr', Nr_range)
# @pytest.mark.parametrize('dealias', dealias_range)
# @pytest.mark.parametrize('basis', [build_ball, build_shell])
# def test_transpose_grid_tensor(Nphi, Ntheta, Nr, dealias, basis):
#     c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, dealias)
#     u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
#     ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
#     u['g'][2] = r**2*st*(2*ct**2*cp-r*ct**3*sp+r**3*cp**3*st**5*sp**3+r*ct*st**2*(cp**3+sp**3))
#     u['g'][1] = r**2*(2*ct**3*cp-r*cp**3*st**4+r**3*ct*cp**3*st**5*sp**3-1/16*r*np.sin(2*theta)**2*(-7*sp+np.sin(3*phi)))
#     u['g'][0] = r**2*sp*(-2*ct**2+r*ct*cp*st**2*sp-r**3*cp**2*st**5*sp**3)
#     T = operators.Gradient(u, c).evaluate()
#     T.require_grid_space()
#     Tg = np.transpose(np.copy(T['g']),(1,0,2,3,4))
#     T = operators.TransposeComponents(T).evaluate()
#     assert np.allclose(T['g'], Tg)

# @pytest.mark.parametrize('Nphi', Nphi_range)
# @pytest.mark.parametrize('Ntheta', Ntheta_range)
# @pytest.mark.parametrize('Nr', Nr_range)
# @pytest.mark.parametrize('dealias', dealias_range)
# @pytest.mark.parametrize('basis', [build_ball, build_shell])
# def test_transpose_coeff_tensor(Nphi, Ntheta, Nr, dealias, basis):
#     c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, dealias)
#     u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
#     ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
#     u['g'][2] = r**2*st*(2*ct**2*cp-r*ct**3*sp+r**3*cp**3*st**5*sp**3+r*ct*st**2*(cp**3+sp**3))
#     u['g'][1] = r**2*(2*ct**3*cp-r*cp**3*st**4+r**3*ct*cp**3*st**5*sp**3-1/16*r*np.sin(2*theta)**2*(-7*sp+np.sin(3*phi)))
#     u['g'][0] = r**2*sp*(-2*ct**2+r*ct*cp*st**2*sp-r**3*cp**2*st**5*sp**3)
#     T = operators.Gradient(u, c).evaluate()
#     T.require_coeff_space()
#     Tg = np.transpose(np.copy(T['g']),(1,0,2,3,4))
#     T = operators.TransposeComponents(T).evaluate()
#     assert np.allclose(T['g'], Tg)

@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis_radius', [(build_D2, 2.0)])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_interpolation_scalar(Nphi, Nr, dealias, basis_radius, dtype):
    basis, radius = basis_radius
    c, d, b, phi, r, x, y = basis(Nphi, Nr, radius, dealias, dtype)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    f['g'] = x**4 + 2*y**4
    h = operators.interpolate(f,r=radius).evaluate()
    h.require_scales((1,1))

    x0, y0 = c.cartesian(phi, np.array([[radius]]))
    hg = x0**4 + 2*y0**4
    assert np.allclose(h['g'], hg)

# @pytest.mark.parametrize('Nphi', [16])
# @pytest.mark.parametrize('Ntheta', [16])
# @pytest.mark.parametrize('Nr', [8])
# @pytest.mark.parametrize('dealias', dealias_range)
# @pytest.mark.parametrize('basis_radius', basis_radius)
# @pytest.mark.parametrize('k', [0, 1])
# def test_interpolation_vector(Nphi, Ntheta, Nr, dealias, basis_radius, k):
@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis_radius', [(build_D2, 2.0)])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_interpolation_vector(Nphi, Nr, dealias, basis_radius, dtype):
    basis, radius = basis_radius
    c, d, b, phi, r, x, y = basis(Nphi, Nr, radius, dealias, dtype)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    f['g'] = x**4 + 2*y**4

    u = operators.Gradient(f, c).evaluate()
    v = u(r=radius).evaluate()
    x0, y0 = c.cartesian(phi, np.array([[radius]]))
    ex0 = np.array([-np.sin(phi)+0.*np.array([[radius]]),np.cos(phi)+0.*np.array([[radius]])])
    ey0 = np.array([np.cos(phi)+0.*np.array([[radius]]),np.sin(phi)+0.*np.array([[radius]])])

    vg = 4*x0**3*ex0 + 8*y0**3*ey0
    assert np.allclose(v['g'], vg)

@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis_radius', [(build_D2, 2.0)])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_interpolation_tensor(Nphi, Nr, dealias, basis_radius, dtype):
    basis, radius = basis_radius
    c, d, b, phi, r, x, y = basis(Nphi, Nr, radius, dealias, dtype)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    f['g'] = x**4 + 2*y**4

    u = operators.Gradient(f, c).evaluate()
    T = operators.Gradient(u, c).evaluate()
    v = T(r=radius).evaluate()
    x0, y0 = c.cartesian(phi, np.array([[radius]]))
    ex0 = np.array([-np.sin(phi)+0.*np.array([[radius]]),np.cos(phi)+0.*np.array([[radius]])])
    ey0 = np.array([np.cos(phi)+0.*np.array([[radius]]),np.sin(phi)+0.*np.array([[radius]])])
    exex0 = ex0[:,None, ...] * ex0[None,...]
    eyey0 = ey0[:,None, ...] * ey0[None,...]
    vg = 12*x0**2*exex0 + 24*y0**2*eyey0
    assert np.allclose(v['g'], vg)

# @pytest.mark.parametrize('Nphi', Nphi_range)
# @pytest.mark.parametrize('Ntheta', Ntheta_range)
# @pytest.mark.parametrize('Nr', Nr_range)
# @pytest.mark.parametrize('dealias', dealias_range)
# @pytest.mark.parametrize('basis_radius', basis_radius)
# def test_interpolation_tensor(Nphi, Ntheta, Nr, dealias, basis_radius):
#     # Note: In this test, the boundary restriction of the tensor does not depend on the radius
#     basis, radius = basis_radius
#     c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, dealias)
#     T = field.Field(dist=d, bases=(b,), tensorsig=(c,c), dtype=np.complex128)
#     T['g'][2,2] = (6*x**2+4*y*z)/r**2
#     T['g'][2,1] = T['g'][1,2] = -2*(y**3+x**2*(y-3*z)-y*z**2)/(r**3*np.sin(theta))
#     T['g'][2,0] = T['g'][0,2] = 2*x*(z-3*y)/(r**2*np.sin(theta))
#     T['g'][1,1] = 6*x**2/(r**2*np.sin(theta)**2) - (6*x**2+4*y*z)/r**2
#     T['g'][1,0] = T['g'][0,1] = -2*x*(x**2+y**2+3*y*z)/(r**3*np.sin(theta)**2)
#     T['g'][0,0] = 6*y**2/(x**2+y**2)
#     A = operators.interpolate(T,r=radius).evaluate()
#     Ag = 0*A['g']
#     phi, theta, r = b.local_grids(b.domain.dealias)
#     x, y, z = c.cartesian(phi, theta, r)
#     Ag[2,2] = 2*np.sin(theta)*(3*np.cos(phi)**2*np.sin(theta)+2*np.cos(theta)*np.sin(phi))
#     Ag[2,1] = Ag[1,2] = 6*np.cos(theta)*np.cos(phi)**2*np.sin(theta) + 2*np.cos(2*theta)*np.sin(phi)
#     Ag[2,0] = Ag[0,2] = 2*np.cos(phi)*(np.cos(theta) - 3*np.sin(theta)*np.sin(phi))
#     Ag[1,1] = 2*np.cos(theta)*(3*np.cos(theta)*np.cos(phi)**2 - 2*np.sin(theta)*np.sin(phi))
#     Ag[1,0] = Ag[0,1] = -2*np.cos(phi)*(np.sin(theta) + 3*np.cos(theta)*np.sin(phi))
#     Ag[0,0] = 6*np.sin(phi)**2
#     assert np.allclose(A['g'],Ag)

# # need higher resolution for the test function
# @pytest.mark.parametrize('Nphi', [16])
# @pytest.mark.parametrize('Ntheta', [16])
# @pytest.mark.parametrize('Nr', [8])
# @pytest.mark.parametrize('dealias', dealias_range)
# @pytest.mark.parametrize('basis_radius', basis_radius)
# def test_radial_component_vector(Nphi, Ntheta, Nr, dealias, basis_radius):
#     basis, radius = basis_radius
#     c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, dealias)
#     u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
#     ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
#     u['g'][2] = r**2*st*(2*ct**2*cp-r*ct**3*sp+r**3*cp**3*st**5*sp**3+r*ct*st**2*(cp**3+sp**3))
#     u['g'][1] = r**2*(2*ct**3*cp-r*cp**3*st**4+r**3*ct*cp**3*st**5*sp**3-1/16*r*np.sin(2*theta)**2*(-7*sp+np.sin(3*phi)))
#     u['g'][0] = r**2*sp*(-2*ct**2+r*ct*cp*st**2*sp-r**3*cp**2*st**5*sp**3)
#     v = operators.RadialComponent(operators.interpolate(u,r=radius)).evaluate()
#     phi, theta, r = b.local_grids(b.domain.dealias)
#     ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
#     vg = radius**2*st*(2*ct**2*cp-radius*ct**3*sp+radius**3*cp**3*st**5*sp**3+radius*ct*st**2*(cp**3+sp**3))
#     assert np.allclose(v['g'], vg)

# @pytest.mark.parametrize('Nphi', Nphi_range)
# @pytest.mark.parametrize('Ntheta', Ntheta_range)
# @pytest.mark.parametrize('Nr', Nr_range)
# @pytest.mark.parametrize('dealias', dealias_range)
# @pytest.mark.parametrize('basis_radius', basis_radius)
# def test_radial_component_tensor(Nphi, Ntheta, Nr, dealias, basis_radius):
#     # Note: In this test, the boundary restriction of the tensor does not depend on the radius
#     basis, radius = basis_radius
#     c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, dealias)
#     T = field.Field(dist=d, bases=(b,), tensorsig=(c,c), dtype=np.complex128)
#     T['g'][2,2] = (6*x**2+4*y*z)/r**2
#     T['g'][2,1] = T['g'][1,2] = -2*(y**3+x**2*(y-3*z)-y*z**2)/(r**3*np.sin(theta))
#     T['g'][2,0] = T['g'][0,2] = 2*x*(z-3*y)/(r**2*np.sin(theta))
#     T['g'][1,1] = 6*x**2/(r**2*np.sin(theta)**2) - (6*x**2+4*y*z)/r**2
#     T['g'][1,0] = T['g'][0,1] = -2*x*(x**2+y**2+3*y*z)/(r**3*np.sin(theta)**2)
#     T['g'][0,0] = 6*y**2/(x**2+y**2)
#     A = operators.RadialComponent(operators.interpolate(T,r=radius)).evaluate()
#     Ag = 0*A['g']
#     phi, theta, r = b.local_grids(b.domain.dealias)
#     x, y, z = c.cartesian(phi, theta, r)
#     Ag[2] = 2*np.sin(theta)*(3*np.cos(phi)**2*np.sin(theta)+2*np.cos(theta)*np.sin(phi))
#     Ag[1] = 6*np.cos(theta)*np.cos(phi)**2*np.sin(theta) + 2*np.cos(2*theta)*np.sin(phi)
#     Ag[0] = 2*np.cos(phi)*(np.cos(theta) - 3*np.sin(theta)*np.sin(phi))
#     assert np.allclose(A['g'],Ag)

# # need higher resolution for the test function
# @pytest.mark.parametrize('Nphi', [16])
# @pytest.mark.parametrize('Ntheta', [16])
# @pytest.mark.parametrize('Nr', [8])
# @pytest.mark.parametrize('dealias', dealias_range)
# @pytest.mark.parametrize('basis_radius', basis_radius)
# def test_angular_component_vector(Nphi, Ntheta, Nr, dealias, basis_radius):
#     basis, radius = basis_radius
#     c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, dealias)
#     u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
#     ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
#     u['g'][2] = r**2*st*(2*ct**2*cp-r*ct**3*sp+r**3*cp**3*st**5*sp**3+r*ct*st**2*(cp**3+sp**3))
#     u['g'][1] = r**2*(2*ct**3*cp-r*cp**3*st**4+r**3*ct*cp**3*st**5*sp**3-1/16*r*np.sin(2*theta)**2*(-7*sp+np.sin(3*phi)))
#     u['g'][0] = r**2*sp*(-2*ct**2+r*ct*cp*st**2*sp-r**3*cp**2*st**5*sp**3)
#     v = operators.AngularComponent(operators.interpolate(u,r=radius)).evaluate()
#     vg = 0*v['g']
#     phi, theta, r = b.local_grids(b.domain.dealias)
#     ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
#     vg[0] = radius**2*sp*(-2*ct**2+radius*ct*cp*st**2*sp-radius**3*cp**2*st**5*sp**3)
#     vg[1] = radius**2*(2*ct**3*cp-radius*cp**3*st**4+radius**3*ct*cp**3*st**5*sp**3-1/16*radius*np.sin(2*theta)**2*(-7*sp+np.sin(3*phi)))
#     assert np.allclose(v['g'], vg)

# @pytest.mark.parametrize('Nphi', Nphi_range)
# @pytest.mark.parametrize('Ntheta', Ntheta_range)
# @pytest.mark.parametrize('Nr', Nr_range)
# @pytest.mark.parametrize('dealias', dealias_range)
# @pytest.mark.parametrize('basis_radius', basis_radius)
# def test_angular_component_tensor(Nphi, Ntheta, Nr, dealias, basis_radius):
#     # Note: In this test, the boundary restriction of the tensor does not depend on the radius
#     basis, radius = basis_radius
#     c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, dealias)
#     T = field.Field(dist=d, bases=(b,), tensorsig=(c,c), dtype=np.complex128)
#     T['g'][2,2] = (6*x**2+4*y*z)/r**2
#     T['g'][2,1] = T['g'][1,2] = -2*(y**3+x**2*(y-3*z)-y*z**2)/(r**3*np.sin(theta))
#     T['g'][2,0] = T['g'][0,2] = 2*x*(z-3*y)/(r**2*np.sin(theta))
#     T['g'][1,1] = 6*x**2/(r**2*np.sin(theta)**2) - (6*x**2+4*y*z)/r**2
#     T['g'][1,0] = T['g'][0,1] = -2*x*(x**2+y**2+3*y*z)/(r**3*np.sin(theta)**2)
#     T['g'][0,0] = 6*y**2/(x**2+y**2)
#     A = operators.AngularComponent(operators.interpolate(T,r=radius),index=1).evaluate()
#     Ag = 0*A['g']
#     phi, theta, r = b.local_grids(b.domain.dealias)
#     x, y, z = c.cartesian(phi, theta, r)
#     Ag[2,1] = 6*np.cos(theta)*np.cos(phi)**2*np.sin(theta) + 2*np.cos(2*theta)*np.sin(phi)
#     Ag[2,0] = 2*np.cos(phi)*(np.cos(theta) - 3*np.sin(theta)*np.sin(phi))
#     Ag[1,1] = 2*np.cos(theta)*(3*np.cos(theta)*np.cos(phi)**2 - 2*np.sin(theta)*np.sin(phi))
#     Ag[1,0] = Ag[0,1] = -2*np.cos(phi)*(np.sin(theta) + 3*np.cos(theta)*np.sin(phi))
#     Ag[0,0] = 6*np.sin(phi)**2
#     assert np.allclose(A['g'],Ag)

