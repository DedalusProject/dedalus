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
dtype_range = [np.float64, np.complex128]
radius_disk = 1.5
radii_annulus = (0.5, 3)


@CachedFunction
def build_disk(Nphi, Nr, k, dealias, dtype):
    c = coords.PolarCoordinates('phi', 'r')
    d = distributor.Distributor((c,), dtype=dtype)
    b = basis.DiskBasis(c, (Nphi, Nr), radius=radius_disk, k=k, dealias=(dealias, dealias), dtype=dtype)
    phi, r = d.local_grids(b, scales=dealias)
    x, y = c.cartesian(phi, r)
    return c, d, b, phi, r, x, y


@CachedFunction
def build_annulus(Nphi, Nr, k, dealias, dtype):
    c = coords.PolarCoordinates('phi', 'r')
    d = distributor.Distributor((c,), dtype=dtype)
    b = basis.AnnulusBasis(c, (Nphi, Nr), radii=radii_annulus, k=k, dealias=(dealias, dealias), dtype=dtype)
    phi, r = d.local_grids(b, scales=dealias)
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
#     f.preset_scales(dealias)
#     f['g'] = 3*x**2 + 2*y*z
#     for ell, m_ind, ell_ind in b.ell_maps:
#         g['c'][m_ind, ell_ind, :] = (ell+3)*f['c'][m_ind, ell_ind, :]
#     func = lambda ell: ell+3
#     h = operators.SphericalEllProduct(f, c, func).evaluate()
#     g.preset_scales(dealias)
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
#     f.preset_scales(dealias)
#     f['g'] = 3*x**2 + 2*y*z
#     u = operators.Gradient(f, c).evaluate()
#     uk0 = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
#     uk0.preset_scales(dealias)
#     uk0['g'] = u['g']
#     v = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
#     v.preset_scales(dealias)
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
def test_convert_constant_scalar(Nphi, Nr, k, dealias, basis, dtype):
    c, d, b, phi, r, x, y = basis(Nphi, Nr, k, dealias, dtype)
    f = field.Field(dist=d, dtype=dtype)
    f['g'] = 1
    g = operators.Convert(f, b).evaluate()
    assert np.allclose(f['g'], g['g'])


@pytest.mark.xfail(reason="Not yet implemented", run=False)
@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_disk, build_annulus])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_convert_constant_tensor(Nphi, Nr, k, dealias, basis, dtype):
    c, d, b, phi, r, x, y = basis(Nphi, Nr, k, dealias, dtype)
    f = field.Field(dist=d, dtype=dtype, tensorsig=(c,c))
    f['g'][0,0] = f['g'][1,1] = 1
    g = operators.Convert(f, b).evaluate()
    assert np.allclose(f['g'], g['g'])


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
    f.preset_scales(dealias)
    f['g'] = 3*x**2 + 2*y
    g = operators.Laplacian(f, c).evaluate()
    f.change_layout(layout)
    g.change_layout(layout)
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
    u.preset_scales(b.dealias)
    ex = np.array([-np.sin(phi)+0.*r,np.cos(phi)+0.*r])
    ey = np.array([np.cos(phi)+0.*r,np.sin(phi)+0.*r])
    u['g'] = 4*x**3*ey + 3*y**2*ey
    v = operators.Laplacian(u, c).evaluate()
    u.change_layout(layout)
    v.change_layout(layout)
    w = (u + v).evaluate()
    assert np.allclose(w['g'], u['g'] + v['g'])


@pytest.mark.parametrize('basis', [build_disk, build_annulus])
@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('layout', ['c', 'g'])
def test_skew_explicit(basis, Nphi, Nr, k, dealias, dtype, layout):
    c, d, b, phi, r, x, y = basis(Nphi, Nr, k, dealias, dtype)
    # Random vector field
    f = d.VectorField(c, bases=b)
    f.fill_random(layout='g')
    f.low_pass_filter(scales=0.75)
    # Evaluate skew
    f.change_layout(layout)
    g = operators.Skew(f).evaluate()
    assert np.allclose(g['g'][0], f['g'][1])
    assert np.allclose(g['g'][1], -f['g'][0])


@pytest.mark.parametrize('basis', [build_disk, build_annulus])
@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_skew_implicit(basis, Nphi, Nr, k, dealias, dtype):
    c, d, b, phi, r, x, y = basis(Nphi, Nr, k, dealias, dtype)
    # Random vector field
    f = d.VectorField(c, bases=b)
    f.fill_random(layout='g')
    f.low_pass_filter(scales=0.75)
    # Skew LBVP
    u = d.VectorField(c, bases=b)
    problem = problems.LBVP([u], namespace=locals())
    problem.add_equation("skew(u) = skew(f)")
    solver = problem.build_solver()
    solver.solve()
    u.change_scales(dealias)
    f.change_scales(dealias)
    assert np.allclose(u['g'], f['g'])


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_disk, build_annulus])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('layout', ['c', 'g'])
def test_explicit_trace_tensor(Nphi, Nr, k, dealias, basis, dtype, layout):
    c, d, b, phi, r, x, y = basis(Nphi, Nr, k, dealias, dtype)
    u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
    u.preset_scales(dealias)
    ex = np.array([-np.sin(phi)+0.*r,np.cos(phi)+0.*r])
    ey = np.array([np.cos(phi)+0.*r,np.sin(phi)+0.*r])
    u['g'] = 4*x**3*ey + 3*y**2*ey
    T = operators.Gradient(u, c).evaluate()
    fg = T['g'][0,0] + T['g'][1,1]
    T.change_layout(layout)
    f = operators.Trace(T).evaluate()
    assert np.allclose(f['g'], fg)


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_disk, build_annulus])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_implicit_trace_tensor(Nphi, Nr, k, dealias, basis, dtype):
    c, d, b, phi, r, x, y = basis(Nphi, Nr, k, dealias, dtype)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    g = field.Field(dist=d, bases=(b,), dtype=dtype)
    g.fill_random('g')
    g.low_pass_filter(scales=0.5)
    I = d.IdentityTensor(c, bases=b.radial_basis)
    problem = problems.LBVP([f])
    problem.add_equation((operators.Trace(I*f), 2*g))
    solver = solvers.LinearBoundaryValueSolver(problem, matrix_coupling=[False, True])
    solver.solve()
    assert np.allclose(f['c'], g['c'])


@pytest.mark.parametrize('basis', [build_disk, build_annulus])
@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('layout', ['c', 'g'])
def test_transpose_explicit(basis, Nphi, Nr, k, dealias, dtype, layout):
    c, d, b, phi, r, x, y = basis(Nphi, Nr, k, dealias, dtype)
    # Random tensor field
    f = d.TensorField((c, c), bases=b)
    f.fill_random(layout='g')
    f.low_pass_filter(scales=0.75)
    # Evaluate transpose
    f.change_layout(layout)
    g = operators.transpose(f).evaluate()
    assert np.allclose(g['g'], np.transpose(f['g'], (1,0,2,3)))


@pytest.mark.parametrize('basis', [build_disk, build_annulus])
@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_transpose_implicit(basis, Nphi, Nr, k, dealias, dtype):
    c, d, b, phi, r, x, y = basis(Nphi, Nr, k, dealias, dtype)
    # Random tensor field
    f = d.TensorField((c, c), bases=b)
    f.fill_random(layout='g')
    f.low_pass_filter(scales=0.75)
    # Transpose LBVP
    u = d.TensorField((c, c), bases=b)
    problem = problems.LBVP([u], namespace=locals())
    problem.add_equation("trans(u) = trans(f)")
    solver = problem.build_solver()
    solver.solve()
    u.change_scales(dealias)
    f.change_scales(dealias)
    assert np.allclose(u['g'], f['g'])


@pytest.mark.parametrize('Nphi', [16])
@pytest.mark.parametrize('Nr', [10])
@pytest.mark.parametrize('k', [0, 1, 2, 5])
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_disk, build_annulus])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_azimuthal_average_scalar(Nphi, Nr, k, dealias, dtype, basis):
    c, d, b, phi, r, x, y = basis(Nphi, Nr, k, dealias, dtype)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    f.preset_scales(dealias)
    f['g'] = r**2 + x
    h = operators.Average(f, c.coords[0]).evaluate()
    hg = r**2
    assert np.allclose(h['g'], hg)


@pytest.mark.parametrize('Nphi', [16])
@pytest.mark.parametrize('Nr', [10])
@pytest.mark.parametrize('k', [0, 1, 2, 5])
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_disk, build_annulus])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('n', [0, 1, 2])
def test_integrate_scalar(Nphi, Nr, k, dealias, dtype, basis, n):
    # Need to test if this fails for alpha != 0?
    c, d, b, phi, r, x, y = basis(Nphi, Nr, k, dealias, dtype)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    f.preset_scales(dealias)
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
    f.preset_scales(dealias)
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
    f.preset_scales(dealias)
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
    f.preset_scales(dealias)
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
    f.preset_scales(dealias)
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
    f.preset_scales(dealias)
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
    f.preset_scales(dealias)
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


@pytest.mark.parametrize('Nphi', [16])
@pytest.mark.parametrize('Nr', [8])
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_disk, build_annulus])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('radius', [0.5, 1.0, 1.5])
def test_radial_component_vector(Nphi, Nr, k, dealias, dtype, basis, radius):
    c, d, b, phi, r, x, y = basis(Nphi, Nr, k, dealias, dtype)
    cp, sp = np.cos(phi), np.sin(phi)
    u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
    u.preset_scales(dealias)
    ex = np.array([-np.sin(phi), np.cos(phi)])
    ey = np.array([np.cos(phi), np.sin(phi)])
    u['g'] = (x**2*y - 2*x*y**5)*ex + (x**2*y + 7*x**3*y**2)*ey
    v = operators.RadialComponent(operators.interpolate(u, r=radius)).evaluate()
    vg = (radius**3*cp**2*sp - 2*radius**6*cp*sp**5)*cp + (radius**3*cp**2*sp + 7*radius**5*cp**3*sp**2)*sp
    assert np.allclose(v['g'], vg)


@pytest.mark.parametrize('Nphi', [16])
@pytest.mark.parametrize('Nr', [8])
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_disk, build_annulus])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('radius', [0.5, 1.0, 1.5])
def test_radial_component_tensor(Nphi, Nr, k, dealias, dtype, basis, radius):
    c, d, b, phi, r, x, y = basis(Nphi, Nr, k, dealias, dtype)
    cp, sp = np.cos(phi), np.sin(phi)
    T = field.Field(dist=d, bases=(b,), tensorsig=(c,c), dtype=dtype)
    T.preset_scales(dealias)
    ex = np.array([-np.sin(phi), np.cos(phi)])
    ey = np.array([np.cos(phi), np.sin(phi)])
    exex = ex[:,None, ...] * ex[None,...]
    exey = ex[:,None, ...] * ey[None,...]
    eyex = ey[:,None, ...] * ex[None,...]
    eyey = ey[:,None, ...] * ey[None,...]
    T['g'] = (3*x**2+y)*exex + y**3*exey + x**2*y**2*eyex + (y**5-2*x*y)*eyey
    A = operators.RadialComponent(operators.interpolate(T, r=radius)).evaluate()
    Ag = (3*radius**2*cp**2 + radius*sp)*cp*ex + radius**3*sp**3*cp*ey + radius**4*cp**2*sp**2*sp*ex + (radius**5*sp**5-2*radius**2*cp*sp)*sp*ey
    assert np.allclose(A['g'], Ag)


@pytest.mark.parametrize('Nphi', [16])
@pytest.mark.parametrize('Nr', [8])
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_disk, build_annulus])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('radius', [0.5, 1.0, 1.5])
def test_azimuthal_component_vector(Nphi, Nr, k, dealias, dtype, basis, radius):
    c, d, b, phi, r, x, y = basis(Nphi, Nr, k, dealias, dtype)
    cp, sp = np.cos(phi), np.sin(phi)
    u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
    u.preset_scales(dealias)
    ex = np.array([-np.sin(phi), np.cos(phi)])
    ey = np.array([np.cos(phi), np.sin(phi)])
    u['g'] = (x**2*y - 2*x*y**5)*ex + (x**2*y + 7*x**3*y**2)*ey
    v = operators.AzimuthalComponent(operators.interpolate(u, r=radius)).evaluate()
    vg = (radius**3*cp**2*sp - 2*radius**6*cp*sp**5)*(-sp) + (radius**3*cp**2*sp + 7*radius**5*cp**3*sp**2)*cp
    assert np.allclose(v['g'], vg)


@pytest.mark.parametrize('Nphi', [16])
@pytest.mark.parametrize('Nr', [8])
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_disk, build_annulus])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('radius', [0.5, 1.0, 1.5])
def test_azimuthal_component_tensor(Nphi, Nr, k, dealias, dtype, basis, radius):
    c, d, b, phi, r, x, y = basis(Nphi, Nr, k, dealias, dtype)
    cp, sp = np.cos(phi), np.sin(phi)
    T = field.Field(dist=d, bases=(b,), tensorsig=(c,c), dtype=dtype)
    T.preset_scales(dealias)
    ex = np.array([-np.sin(phi), np.cos(phi)])
    ey = np.array([np.cos(phi), np.sin(phi)])
    exex = ex[:,None, ...] * ex[None,...]
    exey = ex[:,None, ...] * ey[None,...]
    eyex = ey[:,None, ...] * ex[None,...]
    eyey = ey[:,None, ...] * ey[None,...]
    T['g'] = (3*x**2+y)*exex + y**3*exey + x**2*y**2*eyex + (y**5-2*x*y)*eyey
    A = operators.AzimuthalComponent(operators.interpolate(T, r=radius)).evaluate()
    Ag = (3*radius**2*cp**2 + radius*sp)*(-sp)*ex + radius**3*sp**3*(-sp)*ey + radius**4*cp**2*sp**2*cp*ex + (radius**5*sp**5-2*radius**2*cp*sp)*cp*ey
    assert np.allclose(A['g'], Ag)

