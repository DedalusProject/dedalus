"""Disk and annulus tests for convert, trace, transpose, interpolate."""

import pytest
import numpy as np
from dedalus.core import coords, distributor, basis, field, operators, arithmetic, problems, solvers
from dedalus.tools.cache import CachedFunction
from dedalus.core.basis import DiskBasis


Nphi_range = [8]
Nr_range = [8]
k_range = [0, 1]
dealias_range = [1, 3/2]
radius_disk = 1.5
radii_annulus = (0.5, 3)


@CachedFunction
def build_disk(Nphi, Nr, k, dealias, dtype):
    c = coords.PolarCoordinates('phi', 'r')
    d = distributor.Distributor((c,))
    b = basis.DiskBasis(c, (Nphi, Nr), radius=radius_disk, k=k, dealias=(dealias, dealias), dtype=dtype)
    phi, r = b.local_grids(b.domain.dealias)
    x, y = c.cartesian(phi, r)
    return c, d, b, phi, r, x, y


@CachedFunction
def build_annulus(Nphi, Nr, k, dealias, dtype):
    c = coords.PolarCoordinates('phi', 'r')
    d = distributor.Distributor((c,))
    b = basis.AnnulusBasis(c, (Nphi, Nr), radii=radii_annulus, k=k, dealias=(dealias, dealias), dtype=dtype)
    phi, r = b.local_grids(b.domain.dealias)
    x, y = c.cartesian(phi, r)
    return c, d, b, phi, r, x, y


# @pytest.mark.parametrize('Nphi', Nphi_range)
# @pytest.mark.parametrize('Ntheta', Ntheta_range)
# @pytest.mark.parametrize('Nr', Nr_range)
# @pytest.mark.parametrize('k', k_range)
# @pytest.mark.parametrize('dealias', dealias_range)
# @pytest.mark.parametrize('basis', [build_ball, build_shell])
# @pytest.mark.parametrize('dtype', [np.float64, np.complex128])
# def test_spherical_ell_product_scalar(Nphi, Ntheta, Nr, k, dealias, basis, dtype):
#     c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, k, dealias, dtype)
#     f = field.Field(dist=d, bases=(b,), dtype=dtype)
#     g = field.Field(dist=d, bases=(b,), dtype=dtype)
#     f.set_scales(b.domain.dealias)
#     f['g'] = 3*x**2 + 2*y*z
#     for ell, m_ind, ell_ind in b.ell_maps:
#         g['c'][m_ind, ell_ind, :] = (ell+3)*f['c'][m_ind, ell_ind, :]
#     func = lambda ell: ell+3
#     h = operators.SphericalEllProduct(f, c, func).evaluate()
#     g.set_scales(b.domain.dealias)
#     assert np.allclose(h['g'], g['g'])


# @pytest.mark.parametrize('Nphi', Nphi_range)
# @pytest.mark.parametrize('Ntheta', Ntheta_range)
# @pytest.mark.parametrize('Nr', Nr_range)
# @pytest.mark.parametrize('k', k_range)
# @pytest.mark.parametrize('dealias', dealias_range)
# @pytest.mark.parametrize('basis', [build_ball, build_shell])
# @pytest.mark.parametrize('dtype', [np.float64, np.complex128])
# def test_spherical_ell_product_vector(Nphi, Ntheta, Nr, k, dealias, basis, dtype):
#     c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, k, dealias, dtype)
#     f = field.Field(dist=d, bases=(b,), dtype=dtype)
#     f.set_scales(b.domain.dealias)
#     f['g'] = 3*x**2 + 2*y*z
#     u = operators.Gradient(f, c).evaluate()
#     uk0 = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
#     uk0.set_scales(b.domain.dealias)
#     uk0['g'] = u['g']
#     v = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
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
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_disk, build_annulus])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('layout', ['c', 'g'])
def test_convert_scalar(Nphi, Nr, k, dealias, basis, dtype, layout):
    c, d, b, phi, r, x, y = basis(Nphi, Nr, k, dealias, dtype)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    f.set_scales(b.domain.dealias)
    f['g'] = 3*x**2 + 2*y
    g = operators.Laplacian(f, c).evaluate()
    f.require_layout(layout)
    g.require_layout(layout)
    h = (f + g).evaluate()
    assert np.allclose(h['g'], f['g'] + g['g'])


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_disk, build_annulus])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('layout', ['c', 'g'])
def test_convert_vector(Nphi, Nr, k, dealias, basis, dtype, layout):
    c, d, b, phi, r, x, y = basis(Nphi, Nr, k, dealias, dtype)
    u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
    u.set_scales(b.dealias)
    ex = np.array([-np.sin(phi)+0.*r,np.cos(phi)+0.*r])
    ey = np.array([np.cos(phi)+0.*r,np.sin(phi)+0.*r])
    u['g'] = 4*x**3*ey + 3*y**2*ey
    v = operators.Laplacian(u, c).evaluate()
    u.require_layout(layout)
    v.require_layout(layout)
    w = (u + v).evaluate()
    assert np.allclose(w['g'], u['g'] + v['g'])


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_disk, build_annulus])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('layout', ['c', 'g'])
def test_trace_tensor(Nphi, Nr, k, dealias, basis, dtype, layout):
    c, d, b, phi, r, x, y = basis(Nphi, Nr, k, dealias, dtype)
    u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
    u.set_scales(b.domain.dealias)
    ex = np.array([-np.sin(phi)+0.*r,np.cos(phi)+0.*r])
    ey = np.array([np.cos(phi)+0.*r,np.sin(phi)+0.*r])
    u['g'] = 4*x**3*ey + 3*y**2*ey
    T = operators.Gradient(u, c).evaluate()
    fg = T['g'][0,0] + T['g'][1,1]
    T.require_layout(layout)
    f = operators.Trace(T).evaluate()
    assert np.allclose(f['g'], fg)


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_disk, build_annulus])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('layout', ['c', 'g'])
def test_explicit_transpose_tensor(Nphi, Nr, k, dealias, basis, dtype, layout):
    c, d, b, phi, r, x, y = basis(Nphi, Nr, k, dealias, dtype)
    u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
    u.set_scales(b.domain.dealias)
    ex = np.array([-np.sin(phi)+0.*r,np.cos(phi)+0.*r])
    ey = np.array([np.cos(phi)+0.*r,np.sin(phi)+0.*r])
    u['g'] = 4*x**3*ey + 3*y**2*ey
    T = operators.Gradient(u, c).evaluate()
    Tg = np.transpose(np.copy(T['g']), (1,0,2,3))
    T.require_layout(layout)
    T = operators.TransposeComponents(T).evaluate()
    assert np.allclose(T['g'], Tg)


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_disk, build_annulus])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_implicit_transpose_tensor(Nphi, Nr, k, dealias, basis, dtype):
    c, d, b, phi, r, x, y = basis(Nphi, Nr, k, dealias, dtype)
    u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
    u.set_scales(b.domain.dealias)
    ex = np.array([-np.sin(phi)+0.*r,np.cos(phi)+0.*r])
    ey = np.array([np.cos(phi)+0.*r,np.sin(phi)+0.*r])
    u['g'] = 4*x**3*ey + 3*y**2*ey
    T = operators.Gradient(u, c).evaluate()
    Ttg = np.transpose(np.copy(T['g']), (1,0,2,3))
    Tt = field.Field(dist=d, bases=(b,), tensorsig=(c,c,), dtype=dtype)
    trans = lambda A: operators.TransposeComponents(A)
    problem = problems.LBVP([Tt])
    problem.add_equation((trans(Tt), T))
    # Solver
    solver = solvers.LinearBoundaryValueSolver(problem)
    solver.solve()
    Tt.require_scales(b.domain.dealias)
    assert np.allclose(Tt['g'], Ttg)


@pytest.mark.parametrize('Nphi', [16])
@pytest.mark.parametrize('Nr', [10])
@pytest.mark.parametrize('k', [0, 1, 2, 5])
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_disk, build_annulus])
@pytest.mark.parametrize('dtype', [np.complex128])
@pytest.mark.parametrize('n', [0, 1, 2])
def test_integrate_scalar(Nphi, Nr, k, dealias, dtype, basis, n):
    c, d, b, phi, r, x, y = basis(Nphi, Nr, k, dealias, dtype)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    f.set_scales(b.domain.dealias)
    f['g'] = r**(2*n)
    h = operators.Integrate(f, c).evaluate()
    if isinstance(b, DiskBasis):
        r_inner, r_outer = 0, b.radius
    else:
        r_inner, r_outer = b.radii
    hg = 2 * np.pi * (r_outer**(2 + 2*n) - r_inner**(2 + 2*n)) / (2 + 2*n)
    assert np.allclose(h['g'], hg)


@pytest.mark.parametrize('Nphi', [16])
@pytest.mark.parametrize('Nr', [8])
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_disk, build_annulus])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('phi_interp', [0.5, 1.0, 1.5])
def test_interpolate_azimuth_scalar(Nphi, Nr, k, dealias, basis, dtype, phi_interp):
    c, d, b, phi, r, x, y = basis(Nphi, Nr, k, dealias, dtype)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    f.set_scales(b.domain.dealias)
    f['g'] = x**4 + 2*y**4
    h = operators.interpolate(f, phi=phi_interp).evaluate()
    x, y = c.cartesian(np.array([[phi_interp]]), r)
    hg = x**4 + 2*y**4
    assert np.allclose(h['g'], hg)


@pytest.mark.parametrize('Nphi', [16])
@pytest.mark.parametrize('Nr', [8])
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_disk, build_annulus])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('r_interp', [0.5, 1.0, 1.5])
def test_interpolate_radius_scalar(Nphi, Nr, k, dealias, basis, dtype, r_interp):
    c, d, b, phi, r, x, y = basis(Nphi, Nr, k, dealias, dtype)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    f.set_scales(b.domain.dealias)
    f['g'] = x**4 + 2*y**4
    h = operators.interpolate(f, r=r_interp).evaluate()
    x, y = c.cartesian(phi, np.array([[r_interp]]))
    hg = x**4 + 2*y**4
    assert np.allclose(h['g'], hg)


@pytest.mark.parametrize('Nphi', [16])
@pytest.mark.parametrize('Nr', [8])
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_disk, build_annulus])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('phi_interp', [0.5, 1.0, 1.5])
def test_interpolate_azimuth_vector(Nphi, Nr, k, dealias, basis, dtype, phi_interp):
    c, d, b, phi, r, x, y = basis(Nphi, Nr, k, dealias, dtype)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    f.set_scales(b.domain.dealias)
    f['g'] = x**4 + 2*y**4
    u = operators.Gradient(f, c)
    v = u(phi=phi_interp).evaluate()
    phi = np.array([[phi_interp]])
    x, y = c.cartesian(phi, r)
    ex = np.array([-np.sin(phi), np.cos(phi)])
    ey = np.array([np.cos(phi), np.sin(phi)])
    vg = 4*x**3*ex + 8*y**3*ey
    assert np.allclose(v['g'], vg)


@pytest.mark.parametrize('Nphi', [16])
@pytest.mark.parametrize('Nr', [8])
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_disk, build_annulus])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('r_interp', [0.5, 1.0, 1.5])
def test_interpolate_radius_vector(Nphi, Nr, k, dealias, basis, dtype, r_interp):
    c, d, b, phi, r, x, y = basis(Nphi, Nr, k, dealias, dtype)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    f.set_scales(b.domain.dealias)
    f['g'] = x**4 + 2*y**4
    u = operators.Gradient(f, c)
    v = u(r=r_interp).evaluate()
    r = np.array([[r_interp]])
    x, y = c.cartesian(phi, r)
    ex = np.array([-np.sin(phi), np.cos(phi)])
    ey = np.array([np.cos(phi), np.sin(phi)])
    vg = 4*x**3*ex + 8*y**3*ey
    assert np.allclose(v['g'], vg)


@pytest.mark.parametrize('Nphi', [16])
@pytest.mark.parametrize('Nr', [8])
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_disk, build_annulus])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('phi_interp', [0.5, 1.0, 1.5])
def test_interpolate_azimuth_tensor(Nphi, Nr, k, dealias, basis, dtype, phi_interp):
    c, d, b, phi, r, x, y = basis(Nphi, Nr, k, dealias, dtype)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    f.set_scales(b.domain.dealias)
    f['g'] = x**4 + 2*y**4
    u = operators.Gradient(f, c)
    T = operators.Gradient(u, c)
    v = T(phi=phi_interp).evaluate()
    phi = np.array([[phi_interp]])
    x, y = c.cartesian(phi, r)
    ex = np.array([-np.sin(phi), np.cos(phi)])
    ey = np.array([np.cos(phi), np.sin(phi)])
    exex = ex[:,None, ...] * ex[None,...]
    eyey = ey[:,None, ...] * ey[None,...]
    vg = 12*x**2*exex + 24*y**2*eyey
    assert np.allclose(v['g'], vg)


@pytest.mark.parametrize('Nphi', [16])
@pytest.mark.parametrize('Nr', [8])
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_disk, build_annulus])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('r_interp', [0.5, 1.0, 1.5])
def test_interpolate_radius_tensor(Nphi, Nr, k, dealias, basis, dtype, r_interp):
    c, d, b, phi, r, x, y = basis(Nphi, Nr, k, dealias, dtype)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    f.set_scales(b.domain.dealias)
    f['g'] = x**4 + 2*y**4
    u = operators.Gradient(f, c)
    T = operators.Gradient(u, c)
    v = T(r=r_interp).evaluate()
    r = np.array([[r_interp]])
    x, y = c.cartesian(phi, r)
    ex = np.array([-np.sin(phi), np.cos(phi)])
    ey = np.array([np.cos(phi), np.sin(phi)])
    exex = ex[:,None, ...] * ex[None,...]
    eyey = ey[:,None, ...] * ey[None,...]
    vg = 12*x**2*exex + 24*y**2*eyey
    assert np.allclose(v['g'], vg)


# @pytest.mark.parametrize('Nphi', [16])
# @pytest.mark.parametrize('Ntheta', [8])
# @pytest.mark.parametrize('Nr', Nr_range)
# @pytest.mark.parametrize('k', k_range)
# @pytest.mark.parametrize('dealias', dealias_range)
# @pytest.mark.parametrize('basis', [build_ball, build_shell])
# @pytest.mark.parametrize('dtype', [np.float64, np.complex128])
# @pytest.mark.parametrize('radius', [0.5, 1.0, 1.5])
# def test_radial_component_vector(Nphi, Ntheta, Nr, k, dealias, dtype, basis, radius):
#     c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, k, dealias, dtype)
#     ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
#     u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
#     u.set_scales(b.domain.dealias)
#     u['g'][2] = r**2*st*(2*ct**2*cp-r*ct**3*sp+r**3*cp**3*st**5*sp**3+r*ct*st**2*(cp**3+sp**3))
#     u['g'][1] = r**2*(2*ct**3*cp-r*cp**3*st**4+r**3*ct*cp**3*st**5*sp**3-1/16*r*np.sin(2*theta)**2*(-7*sp+np.sin(3*phi)))
#     u['g'][0] = r**2*sp*(-2*ct**2+r*ct*cp*st**2*sp-r**3*cp**2*st**5*sp**3)
#     v = operators.RadialComponent(operators.interpolate(u, r=radius)).evaluate()
#     vg = radius**2*st*(2*ct**2*cp-radius*ct**3*sp+radius**3*cp**3*st**5*sp**3+radius*ct*st**2*(cp**3+sp**3))
#     assert np.allclose(v['g'], vg)


# @pytest.mark.parametrize('Nphi', [16])
# @pytest.mark.parametrize('Ntheta', [8])
# @pytest.mark.parametrize('Nr', Nr_range)
# @pytest.mark.parametrize('k', k_range)
# @pytest.mark.parametrize('dealias', dealias_range)
# @pytest.mark.parametrize('basis', [build_ball, build_shell])
# @pytest.mark.parametrize('dtype', [np.float64, np.complex128])
# @pytest.mark.parametrize('radius', [0.5, 1.0, 1.5])
# def test_radial_component_tensor(Nphi, Ntheta, Nr, k, dealias, dtype, basis, radius):
#     c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, k, dealias, dtype)
#     T = field.Field(dist=d, bases=(b,), tensorsig=(c,c), dtype=dtype)
#     T.set_scales(b.domain.dealias)
#     T['g'][2,2] = (6*x**2+4*y*z)/r**2
#     T['g'][2,1] = T['g'][1,2] = -2*(y**3+x**2*(y-3*z)-y*z**2)/(r**3*np.sin(theta))
#     T['g'][2,0] = T['g'][0,2] = 2*x*(z-3*y)/(r**2*np.sin(theta))
#     T['g'][1,1] = 6*x**2/(r**2*np.sin(theta)**2) - (6*x**2+4*y*z)/r**2
#     T['g'][1,0] = T['g'][0,1] = -2*x*(x**2+y**2+3*y*z)/(r**3*np.sin(theta)**2)
#     T['g'][0,0] = 6*y**2/(x**2+y**2)
#     A = operators.RadialComponent(operators.interpolate(T, r=radius)).evaluate()
#     Ag = 0 * A['g']
#     Ag[2] = 2*np.sin(theta)*(3*np.cos(phi)**2*np.sin(theta)+2*np.cos(theta)*np.sin(phi))
#     Ag[1] = 6*np.cos(theta)*np.cos(phi)**2*np.sin(theta) + 2*np.cos(2*theta)*np.sin(phi)
#     Ag[0] = 2*np.cos(phi)*(np.cos(theta) - 3*np.sin(theta)*np.sin(phi))
#     assert np.allclose(A['g'], Ag)


# @pytest.mark.parametrize('Nphi', [16])
# @pytest.mark.parametrize('Ntheta', [8])
# @pytest.mark.parametrize('Nr', Nr_range)
# @pytest.mark.parametrize('k', k_range)
# @pytest.mark.parametrize('dealias', dealias_range)
# @pytest.mark.parametrize('basis', [build_ball, build_shell])
# @pytest.mark.parametrize('dtype', [np.float64, np.complex128])
# @pytest.mark.parametrize('radius', [0.5, 1.0, 1.5])
# def test_angular_component_vector(Nphi, Ntheta, Nr, k, dealias, dtype, basis, radius):
#     c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, k, dealias, dtype)
#     ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
#     u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
#     u.set_scales(b.domain.dealias)
#     u['g'][2] = r**2*st*(2*ct**2*cp-r*ct**3*sp+r**3*cp**3*st**5*sp**3+r*ct*st**2*(cp**3+sp**3))
#     u['g'][1] = r**2*(2*ct**3*cp-r*cp**3*st**4+r**3*ct*cp**3*st**5*sp**3-1/16*r*np.sin(2*theta)**2*(-7*sp+np.sin(3*phi)))
#     u['g'][0] = r**2*sp*(-2*ct**2+r*ct*cp*st**2*sp-r**3*cp**2*st**5*sp**3)
#     v = operators.AngularComponent(operators.interpolate(u, r=radius)).evaluate()
#     vg = 0 * v['g']
#     vg[0] = radius**2*sp*(-2*ct**2+radius*ct*cp*st**2*sp-radius**3*cp**2*st**5*sp**3)
#     vg[1] = radius**2*(2*ct**3*cp-radius*cp**3*st**4+radius**3*ct*cp**3*st**5*sp**3-1/16*radius*np.sin(2*theta)**2*(-7*sp+np.sin(3*phi)))
#     assert np.allclose(v['g'], vg)


# @pytest.mark.parametrize('Nphi', [16])
# @pytest.mark.parametrize('Ntheta', [8])
# @pytest.mark.parametrize('Nr', Nr_range)
# @pytest.mark.parametrize('k', k_range)
# @pytest.mark.parametrize('dealias', dealias_range)
# @pytest.mark.parametrize('basis', [build_ball, build_shell])
# @pytest.mark.parametrize('dtype', [np.float64, np.complex128])
# @pytest.mark.parametrize('radius', [0.5, 1.0, 1.5])
# def test_angular_component_tensor(Nphi, Ntheta, Nr, k, dealias, dtype, basis, radius):
#     c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, k, dealias, dtype)
#     T = field.Field(dist=d, bases=(b,), tensorsig=(c,c), dtype=dtype)
#     T.set_scales(b.domain.dealias)
#     T['g'][2,2] = (6*x**2+4*y*z)/r**2
#     T['g'][2,1] = T['g'][1,2] = -2*(y**3+x**2*(y-3*z)-y*z**2)/(r**3*np.sin(theta))
#     T['g'][2,0] = T['g'][0,2] = 2*x*(z-3*y)/(r**2*np.sin(theta))
#     T['g'][1,1] = 6*x**2/(r**2*np.sin(theta)**2) - (6*x**2+4*y*z)/r**2
#     T['g'][1,0] = T['g'][0,1] = -2*x*(x**2+y**2+3*y*z)/(r**3*np.sin(theta)**2)
#     T['g'][0,0] = 6*y**2/(x**2+y**2)
#     A = operators.AngularComponent(operators.interpolate(T, r=radius), index=1).evaluate()
#     Ag = 0 * A['g']
#     Ag[2,1] = 6*np.cos(theta)*np.cos(phi)**2*np.sin(theta) + 2*np.cos(2*theta)*np.sin(phi)
#     Ag[2,0] = 2*np.cos(phi)*(np.cos(theta) - 3*np.sin(theta)*np.sin(phi))
#     Ag[1,1] = 2*np.cos(theta)*(3*np.cos(theta)*np.cos(phi)**2 - 2*np.sin(theta)*np.sin(phi))
#     Ag[1,0] = Ag[0,1] = -2*np.cos(phi)*(np.sin(theta) + 3*np.cos(theta)*np.sin(phi))
#     Ag[0,0] = 6*np.sin(phi)**2
#     assert np.allclose(A['g'], Ag)

