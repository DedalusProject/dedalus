"""Ball and shell tests for spherical_ell_product, convert, trace, transpose, interpolate, radial_component, angular_component."""

import pytest
import numpy as np
from dedalus.core import coords, distributor, basis, field, operators, arithmetic, problems, solvers
from dedalus.tools.cache import CachedFunction
from dedalus.core.basis import BallBasis, ShellBasis


Nphi_range = [8]
Ntheta_range = [4]
Nr_range = [10]
k_range = [0, 1]
dealias_range = [1, 3/2]
radius_ball = 1.5
radii_shell = (0.5, 1.5)


@CachedFunction
def build_ball(Nphi, Ntheta, Nr, k, dealias, dtype):
    c = coords.SphericalCoordinates('phi', 'theta', 'r')
    d = distributor.Distributor((c,))
    b = basis.BallBasis(c, (Nphi, Ntheta, Nr), radius=radius_ball, k=k, dealias=(dealias, dealias, dealias), dtype=dtype)
    phi, theta, r = d.local_grids(b, scales=dealias)
    x, y, z = c.cartesian(phi, theta, r)
    return c, d, b, phi, theta, r, x, y, z


@CachedFunction
def build_shell(Nphi, Ntheta, Nr, k, dealias, dtype):
    c = coords.SphericalCoordinates('phi', 'theta', 'r')
    d = distributor.Distributor((c,))
    b = basis.ShellBasis(c, (Nphi, Ntheta, Nr), radii=radii_shell, k=k, dealias=(dealias, dealias, dealias), dtype=dtype)
    phi, theta, r = d.local_grids(b, scales=dealias)
    x, y, z = c.cartesian(phi, theta, r)
    return c, d, b, phi, theta, r, x, y, z


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_ball, build_shell])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_spherical_ell_product_scalar(Nphi, Ntheta, Nr, k, dealias, basis, dtype):
    c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, k, dealias, dtype)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    g = field.Field(dist=d, bases=(b,), dtype=dtype)
    f.preset_scales(dealias)
    f['g'] = 3*x**2 + 2*y*z
    for ell, m_ind, ell_ind in b.ell_maps(d):
        g['c'][m_ind, ell_ind, :] = (ell+3)*f['c'][m_ind, ell_ind, :]
    func = lambda ell: ell+3
    h = operators.SphericalEllProduct(f, c, func).evaluate()
    g.preset_scales(dealias)
    assert np.allclose(h['g'], g['g'])


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_ball, build_shell])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_spherical_ell_product_vector(Nphi, Ntheta, Nr, k, dealias, basis, dtype):
    c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, k, dealias, dtype)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    f.preset_scales(dealias)
    f['g'] = 3*x**2 + 2*y*z
    u = operators.Gradient(f, c).evaluate()
    uk0 = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
    uk0.preset_scales(dealias)
    uk0['g'] = u['g']
    v = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
    v.preset_scales(dealias)
    for ell, m_ind, ell_ind in b.ell_maps(d):
        v['c'][0, m_ind, ell_ind, :] = (ell+2)*uk0['c'][0, m_ind, ell_ind, :]
        v['c'][1, m_ind, ell_ind, :] = (ell+4)*uk0['c'][1, m_ind, ell_ind, :]
        v['c'][2, m_ind, ell_ind, :] = (ell+3)*uk0['c'][2, m_ind, ell_ind, :]
    func = lambda ell: ell+3
    w = operators.SphericalEllProduct(u, c, func).evaluate()
    assert np.allclose(w['g'], v['g'])


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_ball, build_shell])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_convert_constant_scalar(Nphi, Ntheta, Nr, k, dealias, basis, dtype):
    c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, k, dealias, dtype)
    f = field.Field(dist=d, dtype=dtype)
    f['g'] = 1
    g = operators.Convert(f, b).evaluate()
    assert np.allclose(f['g'], g['g'])


@pytest.mark.xfail(reason="Not yet implemented", run=False)
@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_ball, build_shell])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_convert_constant_tensor(Nphi, Ntheta, Nr, k, dealias, basis, dtype):
    c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, k, dealias, dtype)
    f = field.Field(dist=d, dtype=dtype, tensorsig=(c,c))
    f['g'][0,0] = f['g'][1,1] = f['g'][2,2] = 1
    g = operators.Convert(f, b).evaluate()
    assert np.allclose(f['g'], g['g'])


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_ball, build_shell])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('layout', ['c', 'g'])
def test_convert_scalar(Nphi, Ntheta, Nr, k, dealias, basis, dtype, layout):
    c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, k, dealias, dtype)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    f.preset_scales(dealias)
    f['g'] = 3*x**2 + 2*y*z
    g = operators.Laplacian(f, c).evaluate()
    f.change_layout(layout)
    g.change_layout(layout)
    h = (f + g).evaluate()
    assert np.allclose(h['g'], f['g'] + g['g'])


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_ball, build_shell])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('layout', ['c', 'g'])
def test_convert_vector(Nphi, Ntheta, Nr, k, dealias, basis, dtype, layout):
    c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, k, dealias, dtype)
    ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
    u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
    u.preset_scales(dealias)
    u['g'][2] = r**2*st*(2*ct**2*cp-r*ct**3*sp+r**3*cp**3*st**5*sp**3+r*ct*st**2*(cp**3+sp**3))
    u['g'][1] = r**2*(2*ct**3*cp-r*cp**3*st**4+r**3*ct*cp**3*st**5*sp**3-1/16*r*np.sin(2*theta)**2*(-7*sp+np.sin(3*phi)))
    u['g'][0] = r**2*sp*(-2*ct**2+r*ct*cp*st**2*sp-r**3*cp**2*st**5*sp**3)
    v = operators.Laplacian(u, c).evaluate()
    u.change_layout(layout)
    v.change_layout(layout)
    w = (u + v).evaluate()
    assert np.allclose(w['g'], u['g'] + v['g'])


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_ball, build_shell])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('layout', ['c', 'g'])
def test_explicit_trace_tensor(Nphi, Ntheta, Nr, k, dealias, basis, dtype, layout):
    c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, k, dealias, dtype)
    ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
    u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
    u.preset_scales(dealias)
    u['g'][2] = r**2*st*(2*ct**2*cp-r*ct**3*sp+r**3*cp**3*st**5*sp**3+r*ct*st**2*(cp**3+sp**3))
    u['g'][1] = r**2*(2*ct**3*cp-r*cp**3*st**4+r**3*ct*cp**3*st**5*sp**3-1/16*r*np.sin(2*theta)**2*(-7*sp+np.sin(3*phi)))
    u['g'][0] = r**2*sp*(-2*ct**2+r*ct*cp*st**2*sp-r**3*cp**2*st**5*sp**3)
    T = operators.Gradient(u, c).evaluate()
    fg = T['g'][0,0] + T['g'][1,1] + T['g'][2,2]
    T.change_layout(layout)
    f = operators.Trace(T).evaluate()
    assert np.allclose(f['g'], fg)


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_ball, build_shell])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_implicit_trace_tensor(Nphi, Ntheta, Nr, k, dealias, basis, dtype):
    c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, k, dealias, dtype)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    g = field.Field(dist=d, bases=(b,), dtype=dtype)
    g.preset_scales(g.domain.dealias)
    g['g'] = 3*x**2 + 2*y*z
    I = field.Field(dist=d, tensorsig=(c,c), bases=b.radial_basis, dtype=dtype)
    I['g'][0,0] = I['g'][1,1] = I['g'][2,2] = 1
    trace = lambda A: operators.Trace(A)
    problem = problems.LBVP([f])
    problem.add_equation((trace(I*f), 3*g))
    solver = solvers.LinearBoundaryValueSolver(problem)
    solver.solve()
    assert np.allclose(f['c'], g['c'])


@pytest.mark.xfail(reason="Constant tensors not intertwined", run=False)
@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_ball, build_shell])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_implicit_trace_tensor_constant_I(Nphi, Ntheta, Nr, k, dealias, basis, dtype):
    c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, k, dealias, dtype)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    g = field.Field(dist=d, bases=(b,), dtype=dtype)
    g.preset_scales(g.domain.dealias)
    g['g'] = 3*x**2 + 2*y*z
    I = field.Field(dist=d, tensorsig=(c,c), dtype=dtype)
    I['g'][0,0] = I['g'][1,1] = I['g'][2,2] = 1
    trace = lambda A: operators.Trace(A)
    problem = problems.LBVP([f])
    problem.add_equation((trace(I*f), 3*g))
    solver = solvers.LinearBoundaryValueSolver(problem)
    solver.solve()
    assert np.allclose(f['c'], g['c'])


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_ball, build_shell])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('layout', ['c', 'g'])
def test_explicit_transpose_tensor(Nphi, Ntheta, Nr, k, dealias, dtype, basis, layout):
    c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, k, dealias, dtype)
    ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
    u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
    u.preset_scales(dealias)
    u['g'][2] = r**2*st*(2*ct**2*cp-r*ct**3*sp+r**3*cp**3*st**5*sp**3+r*ct*st**2*(cp**3+sp**3))
    u['g'][1] = r**2*(2*ct**3*cp-r*cp**3*st**4+r**3*ct*cp**3*st**5*sp**3-1/16*r*np.sin(2*theta)**2*(-7*sp+np.sin(3*phi)))
    u['g'][0] = r**2*sp*(-2*ct**2+r*ct*cp*st**2*sp-r**3*cp**2*st**5*sp**3)
    T = operators.Gradient(u, c).evaluate()
    Tg = np.transpose(np.copy(T['g']), (1,0,2,3,4))
    T.change_layout(layout)
    T = operators.TransposeComponents(T).evaluate()
    assert np.allclose(T['g'], Tg)


@pytest.mark.skip(reason="matrices are singular for low ell")
@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_ball, build_shell])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_implicit_transpose_tensor(Nphi, Ntheta, Nr, k, dealias, dtype, basis):
    c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, k, dealias, dtype)
    ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
    u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
    u.preset_scales(dealias)
    u['g'][2] = r**2*st*(2*ct**2*cp-r*ct**3*sp+r**3*cp**3*st**5*sp**3+r*ct*st**2*(cp**3+sp**3))
    u['g'][1] = r**2*(2*ct**3*cp-r*cp**3*st**4+r**3*ct*cp**3*st**5*sp**3-1/16*r*np.sin(2*theta)**2*(-7*sp+np.sin(3*phi)))
    u['g'][0] = r**2*sp*(-2*ct**2+r*ct*cp*st**2*sp-r**3*cp**2*st**5*sp**3)
    T = operators.Gradient(u, c).evaluate()
    Ttg = np.transpose(np.copy(T['g']), (1,0,2,3,4))
    Tt = field.Field(dist=d, bases=(b,), tensorsig=(c,c,), dtype=dtype)
    trans = lambda A: operators.TransposeComponents(A)
    problem = problems.LBVP([Tt])
    problem.add_equation((trans(Tt), T))
    # Solver
    solver = solvers.LinearBoundaryValueSolver(problem)
    solver.solve()
    assert np.allclose(Tt['g'], Ttg)


@pytest.mark.parametrize('Nphi', [16])
@pytest.mark.parametrize('Ntheta', [8])
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('k', [0, 1, 2, 5])
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_ball, build_shell])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_azimuthal_average_scalar(Nphi, Ntheta, Nr, k, dealias, dtype, basis):
    c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, k, dealias, dtype)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    f.preset_scales(dealias)
    f['g'] = r**2 + x + z
    h = operators.Average(f, c.coords[0]).evaluate()
    hg = r**2 + z
    assert np.allclose(h['g'], hg)


@pytest.mark.parametrize('Nphi', [16])
@pytest.mark.parametrize('Ntheta', [8])
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('k', [0, 1, 2, 5])
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_ball, build_shell])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_spherical_average_scalar(Nphi, Ntheta, Nr, k, dealias, dtype, basis):
    c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, k, dealias, dtype)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    f.preset_scales(dealias)
    f['g'] = r**2 + x + z
    h = operators.Average(f, c.S2coordsys).evaluate()
    hg = r**2
    assert np.allclose(h['g'], hg)


@pytest.mark.parametrize('Nphi', [16])
@pytest.mark.parametrize('Ntheta', [8])
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('k', [0, 1, 2, 5])
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_ball, build_shell])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('n', [0, 1, 2])
def test_integrate_scalar(Nphi, Ntheta, Nr, k, dealias, dtype, basis, n):
    c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, k, dealias, dtype)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    f.preset_scales(dealias)
    f['g'] = r**(2*n)
    h = operators.Integrate(f, c).evaluate()
    if isinstance(b, BallBasis):
        r_inner, r_outer = 0, b.radius
    else:
        r_inner, r_outer = b.radii
    hg = 4 * np.pi * (r_outer**(3 + 2*n) - r_inner**(3 + 2*n)) / (3 + 2*n)
    assert np.allclose(h['g'], hg)


@pytest.mark.parametrize('Nphi', [16])
@pytest.mark.parametrize('Ntheta', [8])
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_ball, build_shell])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('phi_interp', [0, 0.1, -0.1, 4.5*np.pi])
def test_interpolate_azimuth_scalar(Nphi, Ntheta, Nr, k, dealias, dtype, basis, phi_interp):
    c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, k, dealias, dtype)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    f.preset_scales(dealias)
    f['g'] = x**4 + 2*y**4 + 3*z**4
    h = operators.interpolate(f, phi=phi_interp).evaluate()
    x, y, z = c.cartesian(np.array([[[phi_interp]]]), theta, r)
    hg = x**4 + 2*y**4 + 3*z**4
    assert np.allclose(h['g'], hg)


@pytest.mark.parametrize('Nphi', [16])
@pytest.mark.parametrize('Ntheta', [8])
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_ball, build_shell])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('theta_interp', [0, np.pi/4, np.pi/2, np.pi])
def test_interpolate_colatitude_scalar(Nphi, Ntheta, Nr, k, dealias, dtype, basis, theta_interp):
    c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, k, dealias, dtype)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    f.preset_scales(dealias)
    f['g'] = x**4 + 2*y**4 + 3*z**4
    h = operators.interpolate(f, theta=theta_interp).evaluate()
    x, y, z = c.cartesian(phi, np.array([[[theta_interp]]]), r)
    hg = x**4 + 2*y**4 + 3*z**4
    assert np.allclose(h['g'], hg)


@pytest.mark.parametrize('Nphi', [16])
@pytest.mark.parametrize('Ntheta', [8])
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_ball, build_shell])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('r_interp', [0.5, 1.0, 1.5])
def test_interpolate_radius_scalar(Nphi, Ntheta, Nr, k, dealias, dtype, basis, r_interp):
    c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, k, dealias, dtype)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    f.preset_scales(dealias)
    f['g'] = x**4 + 2*y**4 + 3*z**4
    h = operators.interpolate(f, r=r_interp).evaluate()
    x, y, z = c.cartesian(phi, theta, np.array([[[r_interp]]]))
    hg = x**4 + 2*y**4 + 3*z**4
    assert np.allclose(h['g'], hg)


@pytest.mark.parametrize('Nphi', [16])
@pytest.mark.parametrize('Ntheta', [8])
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_ball, build_shell])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('phi_interp', [0, 0.1, -0.1, 4.5*np.pi])
def test_interpolate_azimuth_vector(Nphi, Ntheta, Nr, k, dealias, dtype, basis, phi_interp):
    c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, k, dealias, dtype)
    ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
    u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
    u.preset_scales(dealias)
    u['g'][0] = r**2*sp*(-2*ct**2+r*ct*cp*st**2*sp-r**3*cp**2*st**5*sp**3)
    u['g'][1] = r**2*(2*ct**3*cp-r*cp**3*st**4+r**3*ct*cp**3*st**5*sp**3-1/16*r*np.sin(2*theta)**2*(-7*sp+np.sin(3*phi)))
    u['g'][2] = r**2*st*(2*ct**2*cp-r*ct**3*sp+r**3*cp**3*st**5*sp**3+r*ct*st**2*(cp**3+sp**3))
    v = operators.interpolate(u, phi=phi_interp).evaluate()
    vg = 0 * v['g']
    phi = np.array([[[phi_interp]]])
    cp, sp = np.cos(phi), np.sin(phi)
    vg[0] = r**2*sp*(-2*ct**2+r*ct*cp*st**2*sp-r**3*cp**2*st**5*sp**3)
    vg[1] = r**2*(2*ct**3*cp-r*cp**3*st**4+r**3*ct*cp**3*st**5*sp**3-1/16*r*np.sin(2*theta)**2*(-7*sp+np.sin(3*phi)))
    vg[2] = r**2*st*(2*ct**2*cp-r*ct**3*sp+r**3*cp**3*st**5*sp**3+r*ct*st**2*(cp**3+sp**3))
    assert np.allclose(v['g'], vg)


@pytest.mark.parametrize('Nphi', [16])
@pytest.mark.parametrize('Ntheta', [8])
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_ball, build_shell])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('theta_interp', [0, np.pi/4, np.pi/2, np.pi])
def test_interpolate_colatitude_vector(Nphi, Ntheta, Nr, k, dealias, dtype, basis, theta_interp):
    c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, k, dealias, dtype)
    ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
    u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
    u.preset_scales(dealias)
    u['g'][0] = r**2*sp*(-2*ct**2+r*ct*cp*st**2*sp-r**3*cp**2*st**5*sp**3)
    u['g'][1] = r**2*(2*ct**3*cp-r*cp**3*st**4+r**3*ct*cp**3*st**5*sp**3-1/16*r*np.sin(2*theta)**2*(-7*sp+np.sin(3*phi)))
    u['g'][2] = r**2*st*(2*ct**2*cp-r*ct**3*sp+r**3*cp**3*st**5*sp**3+r*ct*st**2*(cp**3+sp**3))
    v = operators.interpolate(u, theta=theta_interp).evaluate()
    vg = 0 * v['g']
    theta = np.array([[[theta_interp]]])
    ct, st = np.cos(theta), np.sin(theta)
    vg[0] = r**2*sp*(-2*ct**2+r*ct*cp*st**2*sp-r**3*cp**2*st**5*sp**3)
    vg[1] = r**2*(2*ct**3*cp-r*cp**3*st**4+r**3*ct*cp**3*st**5*sp**3-1/16*r*np.sin(2*theta)**2*(-7*sp+np.sin(3*phi)))
    vg[2] = r**2*st*(2*ct**2*cp-r*ct**3*sp+r**3*cp**3*st**5*sp**3+r*ct*st**2*(cp**3+sp**3))
    assert np.allclose(v['g'], vg)


@pytest.mark.parametrize('Nphi', [16])
@pytest.mark.parametrize('Ntheta', [8])
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_ball, build_shell])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('r_interp', [0.5, 1.0, 1.5])
def test_interpolate_radius_vector(Nphi, Ntheta, Nr, k, dealias, dtype, basis, r_interp):
    c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, k, dealias, dtype)
    ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
    u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
    u.preset_scales(dealias)
    u['g'][0] = r**2*sp*(-2*ct**2+r*ct*cp*st**2*sp-r**3*cp**2*st**5*sp**3)
    u['g'][1] = r**2*(2*ct**3*cp-r*cp**3*st**4+r**3*ct*cp**3*st**5*sp**3-1/16*r*np.sin(2*theta)**2*(-7*sp+np.sin(3*phi)))
    u['g'][2] = r**2*st*(2*ct**2*cp-r*ct**3*sp+r**3*cp**3*st**5*sp**3+r*ct*st**2*(cp**3+sp**3))
    v = operators.interpolate(u, r=r_interp).evaluate()
    vg = 0 * v['g']
    r = np.array([[[r_interp]]])
    vg[0] = r**2*sp*(-2*ct**2+r*ct*cp*st**2*sp-r**3*cp**2*st**5*sp**3)
    vg[1] = r**2*(2*ct**3*cp-r*cp**3*st**4+r**3*ct*cp**3*st**5*sp**3-1/16*r*np.sin(2*theta)**2*(-7*sp+np.sin(3*phi)))
    vg[2] = r**2*st*(2*ct**2*cp-r*ct**3*sp+r**3*cp**3*st**5*sp**3+r*ct*st**2*(cp**3+sp**3))
    assert np.allclose(v['g'], vg)


@pytest.mark.parametrize('Nphi', [16])
@pytest.mark.parametrize('Ntheta', [8])
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_ball, build_shell])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('phi_interp', [0.5, 1.0, 1.5])
def test_interpolate_azimuth_tensor(Nphi, Ntheta, Nr, k, dealias, dtype, basis, phi_interp):
    c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, k, dealias, dtype)
    T = field.Field(dist=d, bases=(b,), tensorsig=(c,c), dtype=dtype)
    T.preset_scales(dealias)
    T['g'][2,2] = (6*x**2+4*y*z)/r**2
    T['g'][2,1] = T['g'][1,2] = -2*(y**3+x**2*(y-3*z)-y*z**2)/(r**3*np.sin(theta))
    T['g'][2,0] = T['g'][0,2] = 2*x*(z-3*y)/(r**2*np.sin(theta))
    T['g'][1,1] = 6*x**2/(r**2*np.sin(theta)**2) - (6*x**2+4*y*z)/r**2
    T['g'][1,0] = T['g'][0,1] = -2*x*(x**2+y**2+3*y*z)/(r**3*np.sin(theta)**2)
    T['g'][0,0] = 6*y**2/(x**2+y**2)
    A = operators.interpolate(T, phi=phi_interp).evaluate()
    Ag = 0 * A['g']
    phi = np.array([[[phi_interp]]])
    x, y, z = c.cartesian(phi, theta, r)
    Ag[2,2] = (6*x**2+4*y*z)/r**2
    Ag[2,1] = Ag[1,2] = -2*(y**3+x**2*(y-3*z)-y*z**2)/(r**3*np.sin(theta))
    Ag[2,0] = Ag[0,2] = 2*x*(z-3*y)/(r**2*np.sin(theta))
    Ag[1,1] = 6*x**2/(r**2*np.sin(theta)**2) - (6*x**2+4*y*z)/r**2
    Ag[1,0] = Ag[0,1] = -2*x*(x**2+y**2+3*y*z)/(r**3*np.sin(theta)**2)
    Ag[0,0] = 6*y**2/(x**2+y**2)
    assert np.allclose(A['g'], Ag)


@pytest.mark.parametrize('Nphi', [16])
@pytest.mark.parametrize('Ntheta', [8])
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_ball, build_shell])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('theta_interp', [0.5, 1.0, 1.5])
def test_interpolate_colatitude_tensor(Nphi, Ntheta, Nr, k, dealias, dtype, basis, theta_interp):
    c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, k, dealias, dtype)
    T = field.Field(dist=d, bases=(b,), tensorsig=(c,c), dtype=dtype)
    T.preset_scales(dealias)
    T['g'][2,2] = (6*x**2+4*y*z)/r**2
    T['g'][2,1] = T['g'][1,2] = -2*(y**3+x**2*(y-3*z)-y*z**2)/(r**3*np.sin(theta))
    T['g'][2,0] = T['g'][0,2] = 2*x*(z-3*y)/(r**2*np.sin(theta))
    T['g'][1,1] = 6*x**2/(r**2*np.sin(theta)**2) - (6*x**2+4*y*z)/r**2
    T['g'][1,0] = T['g'][0,1] = -2*x*(x**2+y**2+3*y*z)/(r**3*np.sin(theta)**2)
    T['g'][0,0] = 6*y**2/(x**2+y**2)
    A = operators.interpolate(T, theta=theta_interp).evaluate()
    Ag = 0 * A['g']
    theta = np.array([[[theta_interp]]])
    x, y, z = c.cartesian(phi, theta, r)
    Ag[2,2] = (6*x**2+4*y*z)/r**2
    Ag[2,1] = Ag[1,2] = -2*(y**3+x**2*(y-3*z)-y*z**2)/(r**3*np.sin(theta))
    Ag[2,0] = Ag[0,2] = 2*x*(z-3*y)/(r**2*np.sin(theta))
    Ag[1,1] = 6*x**2/(r**2*np.sin(theta)**2) - (6*x**2+4*y*z)/r**2
    Ag[1,0] = Ag[0,1] = -2*x*(x**2+y**2+3*y*z)/(r**3*np.sin(theta)**2)
    Ag[0,0] = 6*y**2/(x**2+y**2)
    assert np.allclose(A['g'], Ag)


@pytest.mark.parametrize('Nphi', [16])
@pytest.mark.parametrize('Ntheta', [8])
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_ball, build_shell])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('r_interp', [0.5, 1.0, 1.5])
def test_interpolate_radius_tensor(Nphi, Ntheta, Nr, k, dealias, dtype, basis, r_interp):
    c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, k, dealias, dtype)
    T = field.Field(dist=d, bases=(b,), tensorsig=(c,c), dtype=dtype)
    T.preset_scales(dealias)
    T['g'][2,2] = (6*x**2+4*y*z)/r**2
    T['g'][2,1] = T['g'][1,2] = -2*(y**3+x**2*(y-3*z)-y*z**2)/(r**3*np.sin(theta))
    T['g'][2,0] = T['g'][0,2] = 2*x*(z-3*y)/(r**2*np.sin(theta))
    T['g'][1,1] = 6*x**2/(r**2*np.sin(theta)**2) - (6*x**2+4*y*z)/r**2
    T['g'][1,0] = T['g'][0,1] = -2*x*(x**2+y**2+3*y*z)/(r**3*np.sin(theta)**2)
    T['g'][0,0] = 6*y**2/(x**2+y**2)
    A = operators.interpolate(T, r=r_interp).evaluate()
    Ag = 0 * A['g']
    r = np.array([[[r_interp]]])
    x, y, z = c.cartesian(phi, theta, r)
    Ag[2,2] = (6*x**2+4*y*z)/r**2
    Ag[2,1] = Ag[1,2] = -2*(y**3+x**2*(y-3*z)-y*z**2)/(r**3*np.sin(theta))
    Ag[2,0] = Ag[0,2] = 2*x*(z-3*y)/(r**2*np.sin(theta))
    Ag[1,1] = 6*x**2/(r**2*np.sin(theta)**2) - (6*x**2+4*y*z)/r**2
    Ag[1,0] = Ag[0,1] = -2*x*(x**2+y**2+3*y*z)/(r**3*np.sin(theta)**2)
    Ag[0,0] = 6*y**2/(x**2+y**2)
    assert np.allclose(A['g'], Ag)


@pytest.mark.parametrize('Nphi', [16])
@pytest.mark.parametrize('Ntheta', [8])
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_ball, build_shell])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('radius', [0.5, 1.0, 1.5])
def test_radial_component_vector(Nphi, Ntheta, Nr, k, dealias, dtype, basis, radius):
    c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, k, dealias, dtype)
    ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
    u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
    u.preset_scales(dealias)
    u['g'][2] = r**2*st*(2*ct**2*cp-r*ct**3*sp+r**3*cp**3*st**5*sp**3+r*ct*st**2*(cp**3+sp**3))
    u['g'][1] = r**2*(2*ct**3*cp-r*cp**3*st**4+r**3*ct*cp**3*st**5*sp**3-1/16*r*np.sin(2*theta)**2*(-7*sp+np.sin(3*phi)))
    u['g'][0] = r**2*sp*(-2*ct**2+r*ct*cp*st**2*sp-r**3*cp**2*st**5*sp**3)
    v = operators.RadialComponent(operators.interpolate(u, r=radius)).evaluate()
    vg = radius**2*st*(2*ct**2*cp-radius*ct**3*sp+radius**3*cp**3*st**5*sp**3+radius*ct*st**2*(cp**3+sp**3))
    assert np.allclose(v['g'], vg)


@pytest.mark.parametrize('Nphi', [16])
@pytest.mark.parametrize('Ntheta', [8])
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_ball, build_shell])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('radius', [0.5, 1.0, 1.5])
def test_radial_component_tensor(Nphi, Ntheta, Nr, k, dealias, dtype, basis, radius):
    c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, k, dealias, dtype)
    T = field.Field(dist=d, bases=(b,), tensorsig=(c,c), dtype=dtype)
    T.preset_scales(dealias)
    T['g'][2,2] = (6*x**2+4*y*z)/r**2
    T['g'][2,1] = T['g'][1,2] = -2*(y**3+x**2*(y-3*z)-y*z**2)/(r**3*np.sin(theta))
    T['g'][2,0] = T['g'][0,2] = 2*x*(z-3*y)/(r**2*np.sin(theta))
    T['g'][1,1] = 6*x**2/(r**2*np.sin(theta)**2) - (6*x**2+4*y*z)/r**2
    T['g'][1,0] = T['g'][0,1] = -2*x*(x**2+y**2+3*y*z)/(r**3*np.sin(theta)**2)
    T['g'][0,0] = 6*y**2/(x**2+y**2)
    A = operators.RadialComponent(operators.interpolate(T, r=radius)).evaluate()
    Ag = 0 * A['g']
    Ag[2] = 2*np.sin(theta)*(3*np.cos(phi)**2*np.sin(theta)+2*np.cos(theta)*np.sin(phi))
    Ag[1] = 6*np.cos(theta)*np.cos(phi)**2*np.sin(theta) + 2*np.cos(2*theta)*np.sin(phi)
    Ag[0] = 2*np.cos(phi)*(np.cos(theta) - 3*np.sin(theta)*np.sin(phi))
    assert np.allclose(A['g'], Ag)


@pytest.mark.parametrize('Nphi', [16])
@pytest.mark.parametrize('Ntheta', [8])
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_ball, build_shell])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('radius', [0.5, 1.0, 1.5])
def test_angular_component_vector(Nphi, Ntheta, Nr, k, dealias, dtype, basis, radius):
    c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, k, dealias, dtype)
    ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
    u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
    u.preset_scales(dealias)
    u['g'][2] = r**2*st*(2*ct**2*cp-r*ct**3*sp+r**3*cp**3*st**5*sp**3+r*ct*st**2*(cp**3+sp**3))
    u['g'][1] = r**2*(2*ct**3*cp-r*cp**3*st**4+r**3*ct*cp**3*st**5*sp**3-1/16*r*np.sin(2*theta)**2*(-7*sp+np.sin(3*phi)))
    u['g'][0] = r**2*sp*(-2*ct**2+r*ct*cp*st**2*sp-r**3*cp**2*st**5*sp**3)
    v = operators.AngularComponent(operators.interpolate(u, r=radius)).evaluate()
    vg = 0 * v['g']
    vg[0] = radius**2*sp*(-2*ct**2+radius*ct*cp*st**2*sp-radius**3*cp**2*st**5*sp**3)
    vg[1] = radius**2*(2*ct**3*cp-radius*cp**3*st**4+radius**3*ct*cp**3*st**5*sp**3-1/16*radius*np.sin(2*theta)**2*(-7*sp+np.sin(3*phi)))
    assert np.allclose(v['g'], vg)


@pytest.mark.parametrize('Nphi', [16])
@pytest.mark.parametrize('Ntheta', [8])
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_ball, build_shell])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('radius', [0.5, 1.0, 1.5])
def test_angular_component_tensor(Nphi, Ntheta, Nr, k, dealias, dtype, basis, radius):
    c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, k, dealias, dtype)
    T = field.Field(dist=d, bases=(b,), tensorsig=(c,c), dtype=dtype)
    T.preset_scales(dealias)
    T['g'][2,2] = (6*x**2+4*y*z)/r**2
    T['g'][2,1] = T['g'][1,2] = -2*(y**3+x**2*(y-3*z)-y*z**2)/(r**3*np.sin(theta))
    T['g'][2,0] = T['g'][0,2] = 2*x*(z-3*y)/(r**2*np.sin(theta))
    T['g'][1,1] = 6*x**2/(r**2*np.sin(theta)**2) - (6*x**2+4*y*z)/r**2
    T['g'][1,0] = T['g'][0,1] = -2*x*(x**2+y**2+3*y*z)/(r**3*np.sin(theta)**2)
    T['g'][0,0] = 6*y**2/(x**2+y**2)
    A = operators.AngularComponent(operators.interpolate(T, r=radius), index=1).evaluate()
    Ag = 0 * A['g']
    Ag[2,1] = 6*np.cos(theta)*np.cos(phi)**2*np.sin(theta) + 2*np.cos(2*theta)*np.sin(phi)
    Ag[2,0] = 2*np.cos(phi)*(np.cos(theta) - 3*np.sin(theta)*np.sin(phi))
    Ag[1,1] = 2*np.cos(theta)*(3*np.cos(theta)*np.cos(phi)**2 - 2*np.sin(theta)*np.sin(phi))
    Ag[1,0] = Ag[0,1] = -2*np.cos(phi)*(np.sin(theta) + 3*np.cos(theta)*np.sin(phi))
    Ag[0,0] = 6*np.sin(phi)**2
    assert np.allclose(A['g'], Ag)

