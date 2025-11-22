"""2-Sphere tests gradient, divergence, curl, laplacian."""

import pytest
import numpy as np
from dedalus.core import coords, distributor, basis, field, operators, arithmetic, problems, solvers
from dedalus.tools.cache import CachedFunction
from scipy.special import sph_harm_y

Nphi_range = [32]
Ntheta_range = [16]
dealias_range = [1, 3/2]
dtype_range = [np.float64, np.complex128]

radius = 1.37

@CachedFunction
def build_sphere(Nphi, Ntheta, dealias, dtype):
    c = coords.S2Coordinates('phi', 'theta')
    d = distributor.Distributor(c, dtype=dtype)
    b = basis.SphereBasis(c, (Nphi, Ntheta), radius=radius, dealias=(dealias, dealias), dtype=dtype)
    phi, theta = d.local_grids(b, scales=dealias)
    return c, d, b, phi, theta


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('layout', ['c', 'g'])
def test_skew_explicit(Nphi, Ntheta, dealias, dtype, layout):
    c, d, b, phi, theta = build_sphere(Nphi, Ntheta, dealias, dtype)
    # Random vector field
    f = d.VectorField(c, bases=b)
    f.fill_random(layout='g')
    f.low_pass_filter(scales=0.75)
    # Evaluate skew
    f.change_layout(layout)
    g = operators.Skew(f).evaluate()
    assert np.allclose(g['g'][0], f['g'][1])
    assert np.allclose(g['g'][1], -f['g'][0])


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_skew_implicit(Nphi,  Ntheta, dealias, dtype):
    c, d, b, phi, theta = build_sphere(Nphi, Ntheta, dealias, dtype)
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
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('layout', ['c', 'g'])
def test_transpose_explicit(Nphi, Ntheta, dealias, dtype, layout):
    c, d, b, phi, theta = build_sphere(Nphi, Ntheta, dealias, dtype)
    # Random tensor field
    f = d.TensorField((c, c), bases=b)
    f.fill_random(layout='g')
    f.low_pass_filter(scales=0.75)
    # Evaluate transpose
    f.change_layout(layout)
    g = operators.transpose(f).evaluate()
    assert np.allclose(g['g'], np.transpose(f['g'], (1,0,2,3)))


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_transpose_implicit(Nphi, Ntheta, dealias, dtype):
    c, d, b, phi, theta = build_sphere(Nphi, Ntheta, dealias, dtype)
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


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_convert_constant_scalar_explicit(Nphi, Ntheta, dealias, dtype):
    c, d, b, phi, theta = build_sphere(Nphi, Ntheta, dealias, dtype)
    f = d.Field()
    if np.iscomplexobj(dtype()):
        fref = 1j
    else:
        fref = 1
    f['g'] = fref
    g = operators.Convert(f, b).evaluate()
    assert np.allclose(g['g'], fref)


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_sphere_average_scalar_explicit(Nphi, Ntheta, dealias, dtype):
    c, d, b, phi, theta = build_sphere(Nphi, Ntheta, dealias, dtype)
    f = d.Field(bases=b)
    f.preset_scales(dealias)
    x = np.sin(theta)*np.cos(phi)
    y = np.sin(theta)*np.sin(phi)
    z = np.cos(theta)
    if np.iscomplexobj(dtype()):
        f['g'] = 1 + x + 1j*z
    else:
        f['g'] = 1 + x + z
    h = operators.Average(f, c).evaluate()
    assert np.allclose(h['g'], 1)


# @pytest.mark.parametrize('Nphi', Nphi_range)
# @pytest.mark.parametrize('Ntheta', Ntheta_range)
# @pytest.mark.parametrize('dealias', dealias_range)
# @pytest.mark.parametrize('dtype', dtype_range)
# def test_convert_constant_tensor(Nphi, Ntheta, dealias, dtype):
#     c, d, b, phi, theta = build_sphere(Nphi, Ntheta, dealias, dtype)
#     f = d.TensorField((c,c))
#     f['g'][0,0] = f['g'][1,1] = 1
#     g = operators.Convert(f, b).evaluate()
#     assert np.allclose(f['g'], g['g'])


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_gradient_scalar_explicit(Nphi, Ntheta, dealias, dtype):
    c, d, b, phi, theta = build_sphere(Nphi, Ntheta, dealias, dtype)
    # Spherical harmonic input
    m, l = 2, 2
    f = d.Field(bases=b)
    f.preset_scales(dealias)
    if np.iscomplexobj(dtype()):
        f['g'] = sph_harm_y(m, l, theta, phi)
    else:
        f['g'] = sph_harm_y(m, l, theta, phi).real
    # Evaluate gradient
    u = operators.Gradient(f).evaluate()
    ug_phi = 1j*np.exp(2j*phi)*np.sqrt(15/(2*np.pi))*np.sin(theta)/2
    ug_theta = np.exp(2j*phi)*np.sqrt(15/(2*np.pi))*np.cos(theta)*np.sin(theta)/2
    ug = np.array([ug_phi, ug_theta]) / radius
    if np.isrealobj(dtype()):
        ug = ug.real
    print(u['g']-ug)
    assert np.allclose(u['g'], ug)


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('rank', [0, 1, 2])
def test_cosine_explicit(Nphi, Ntheta, dealias, dtype, rank):
    c, d, b, phi, theta = build_sphere(Nphi, Ntheta, dealias, dtype)
    # Random input for all components
    f = d.TensorField((c,)*rank, bases=b)
    f.fill_random(layout='g')
    f.low_pass_filter(scales=0.75)
    # Cosine operator
    g = operators.MulCosine(f).evaluate()
    g.change_scales(dealias)
    f.change_scales(dealias)
    assert np.allclose(g['g'], np.cos(theta) * f['g'])


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('rank', [0, 1, 2])
def test_cosine_implicit(Nphi,  Ntheta, dealias, dtype, rank):
    c, d, b, phi, theta = build_sphere(Nphi, Ntheta, dealias, dtype)
    # Random input for all components
    f = d.TensorField((c,)*rank, bases=b)
    f.fill_random(layout='g')
    f.low_pass_filter(scales=0.75)
    # Cosine LBVP
    u = d.TensorField((c,)*rank, bases=b)
    problem = problems.LBVP([u], namespace=locals())
    problem.add_equation("u + MulCosine(u) = f + MulCosine(f)")
    solver = problem.build_solver()
    solver.solve()
    u.change_scales(dealias)
    f.change_scales(dealias)
    assert np.allclose(u['g'], f['g'])


# @pytest.mark.parametrize('Ntheta', Ntheta_range)
# @pytest.mark.parametrize('dealias', dealias_range)
# @pytest.mark.parametrize('basis', [build_disk, build_annulus])
# @pytest.mark.parametrize('dtype', [np.float64, np.complex128])
# def test_gradient_radial_scalar(Ntheta, dealias, basis, dtype):
#     Nphi = 1
#     c, d, b, phi, r, x, y = basis(Nphi, Ntheta, dealias, dtype)
#     f = field.Field(dist=d, bases=(b,), dtype=dtype)
#     f.preset_scales(dealias)
#     f['g'] = r**4
#     u = operators.Gradient(f, c).evaluate()
#     ug = [0*r*phi, 4*r**3 + 0*phi]
#     assert np.allclose(u['g'], ug)


# @pytest.mark.parametrize('Nphi', Nphi_range)
# @pytest.mark.parametrize('Ntheta', Ntheta_range)
# @pytest.mark.parametrize('dealias', dealias_range)
# @pytest.mark.parametrize('basis', [build_disk, build_annulus])
# @pytest.mark.parametrize('dtype', [np.float64, np.complex128])
# def test_gradient_vector(Nphi, Ntheta, dealias, basis, dtype):
#     c, d, b, phi, r, x, y = basis(Nphi, Ntheta, dealias, dtype)
#     f = field.Field(dist=d, bases=(b,), dtype=dtype)
#     f.preset_scales(dealias)
#     f['g'] = 3*x**4 + 2*y*x
#     grad = lambda A: operators.Gradient(A, c)
#     T = grad(grad(f)).evaluate()
#     ex = np.array([-np.sin(phi)+0.*r,np.cos(phi)+0.*r])
#     ey = np.array([np.cos(phi)+0.*r,np.sin(phi)+0.*r])
#     exex = ex[:,None, ...] * ex[None,...]
#     eyex = ey[:,None, ...] * ex[None,...]
#     exey = ex[:,None, ...] * ey[None,...]
#     eyey = ey[:,None, ...] * ey[None,...]
#     Tg = 36*x**2*exex + 2*(exey + eyex)
#     assert np.allclose(T['g'], Tg)


# @pytest.mark.parametrize('Ntheta', Ntheta_range)
# @pytest.mark.parametrize('dealias', dealias_range)
# @pytest.mark.parametrize('basis', [build_disk, build_annulus])
# @pytest.mark.parametrize('dtype', [np.float64, np.complex128])
# def test_gradient_radial_vector(Ntheta, dealias, basis, dtype):
#     Nphi = 1
#     c, d, b, phi, r, x, y = basis(Nphi, Ntheta, dealias, dtype)
#     f = field.Field(dist=d, bases=(b,), dtype=dtype)
#     f.preset_scales(dealias)
#     f['g'] = r**4
#     grad = lambda A: operators.Gradient(A, c)
#     T = grad(grad(f)).evaluate()
#     er = np.array([[[0]], [[1]]])
#     ephi = np.array([[[1]], [[0]]])
#     erer = er[:, None, ...] * er[None, ...]
#     ephiephi = ephi[:, None, ...] * ephi[None, ...]
#     Tg = 12 * r**2 * erer + 4 * r**2 * ephiephi
#     assert np.allclose(T['g'], Tg)


# @pytest.mark.parametrize('Nphi', Nphi_range)
# @pytest.mark.parametrize('Ntheta', Ntheta_range)
# @pytest.mark.parametrize('dealias', dealias_range)
# @pytest.mark.parametrize('basis', [build_disk, build_annulus])
# @pytest.mark.parametrize('dtype', [np.float64, np.complex128])
# def test_divergence_vector(Nphi, Ntheta, dealias, basis, dtype):
#     c, d, b, phi, r, x, y = basis(Nphi, Ntheta, dealias, dtype)
#     f = field.Field(dist=d, bases=(b,), dtype=dtype)
#     f.preset_scales(dealias)
#     f['g'] = 3*x**4 + 2*y*x
#     grad = lambda A: operators.Gradient(A, c)
#     div = lambda A: operators.Divergence(A)
#     S = div(grad(f)).evaluate()
#     Sg = 36*x**2
#     assert np.allclose(S['g'], Sg)


# @pytest.mark.parametrize('Ntheta', Ntheta_range)
# @pytest.mark.parametrize('dealias', dealias_range)
# @pytest.mark.parametrize('basis', [build_disk, build_annulus])
# @pytest.mark.parametrize('dtype', [np.float64, np.complex128])
# def test_divergence_radial_vector(Ntheta, dealias, basis, dtype):
#     Nphi = 1
#     c, d, b, phi, r, x, y = basis(Nphi, Ntheta, dealias, dtype=dtype)
#     f = field.Field(dist=d, bases=(b,), dtype=dtype)
#     f.preset_scales(dealias)
#     f['g'] = r**2
#     grad = lambda A: operators.Gradient(A, c)
#     div = lambda A: operators.Divergence(A)
#     h = div(grad(f)).evaluate()
#     hg = 4
#     assert np.allclose(h['g'], hg)


# @pytest.mark.parametrize('Nphi', Nphi_range)
# @pytest.mark.parametrize('Ntheta', Ntheta_range)
# @pytest.mark.parametrize('dealias', dealias_range)
# @pytest.mark.parametrize('basis', [build_disk, build_annulus])
# @pytest.mark.parametrize('dtype', [np.float64, np.complex128])
# def test_divergence_tensor(Nphi, Ntheta, dealias, basis, dtype):
#     c, d, b, phi, r, x, y = basis(Nphi, Ntheta, dealias, dtype)
#     v = field.Field(dist=d, tensorsig=(c,), bases=(b,), dtype=dtype)
#     v.preset_scales(dealias)
#     ex = np.array([-np.sin(phi)+0.*r,np.cos(phi)+0.*r])
#     ey = np.array([np.cos(phi)+0.*r,np.sin(phi)+0.*r])
#     v['g'] = 4*x**3*ey + 3*y**2*ey
#     grad = lambda A: operators.Gradient(A, c)
#     div = lambda A: operators.Divergence(A)
#     U = div(grad(v)).evaluate()
#     Ug = (24*x + 6)*ey
#     assert np.allclose(U['g'], Ug)


# @pytest.mark.parametrize('Nphi', Nphi_range)
# @pytest.mark.parametrize('Ntheta', Ntheta_range)
# @pytest.mark.parametrize('dealias', dealias_range)
# @pytest.mark.parametrize('basis', [build_disk, build_annulus])
# @pytest.mark.parametrize('dtype', [np.float64, np.complex128])
# def test_curl_vector(Nphi, Ntheta, dealias, basis, dtype):
#     c, d, b, phi, r, x, y = basis(Nphi, Ntheta, dealias, dtype)
#     v = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
#     v.preset_scales(dealias)
#     ex = np.array([-np.sin(phi)+0.*r,np.cos(phi)+0.*r])
#     ey = np.array([np.cos(phi)+0.*r,np.sin(phi)+0.*r])
#     v['g'] = 4*x**3*ey + 3*y**2*ey
#     u = operators.Curl(v).evaluate()
#     ug = 12*x**2
#     assert np.allclose(u['g'], ug)


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_laplacian_scalar_explicit(Nphi,  Ntheta, dealias, dtype):
    c, d, b, phi, theta = build_sphere(Nphi, Ntheta, dealias, dtype)
    # Spherical harmonic input
    m, l = 6, 10
    f = d.Field(bases=b)
    f.preset_scales(dealias)
    if np.iscomplexobj(dtype()):
        f['g'] = sph_harm_y(m, l, theta, phi)
    else:
        f['g'] = sph_harm_y(m, l, theta, phi).real
    # Evaluate Laplacian
    u = operators.Laplacian(f).evaluate()
    assert np.allclose(u['g'], -f['g']*(l*(l+1))/radius**2)


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_laplacian_scalar_implicit(Nphi,  Ntheta, dealias, dtype):
    c, d, b, phi, theta = build_sphere(Nphi, Ntheta, dealias, dtype)
    # Spherical harmonic forcing
    m, l = 5, 10
    f = d.Field(bases=b)
    f.preset_scales(dealias)
    if np.iscomplexobj(dtype()):
        f['g'] = sph_harm_y(m, l, theta, phi)
    else:
        f['g'] = sph_harm_y(m, l, theta, phi).real
    # Poisson LBVP
    u = d.Field(bases=b)
    tau = d.Field()
    problem = problems.LBVP([u, tau], namespace=locals())
    problem.add_equation("lap(u) + tau = f")
    problem.add_equation("ave(u) = 0")
    solver = problem.build_solver()
    solver.solve()
    u.change_scales(1)
    f.change_scales(1)
    assert np.allclose(u['g'], -f['g']/(l*(l+1))*radius**2)


# @pytest.mark.parametrize('Nphi', Nphi_range)
# @pytest.mark.parametrize('Ntheta', Ntheta_range)
# @pytest.mark.parametrize('dealias', dealias_range)
# @pytest.mark.parametrize('dtype', dtype_range)
# def test_implicit_laplacian_vector(Nphi,  Ntheta, dealias, dtype):
#     c, d, b, phi, theta = build_sphere(Nphi, Ntheta, dealias, dtype)
#     u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
#     f = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
#     f.preset_scales(dealias)
#     u.preset_scales(dealias)
#     m0 = 1
#     l0 = 1
#     m1 = 1
#     l1 = 2
#     f['g'][0] = sph_harm_y(m0,l0,theta,phi)
#     f['g'][1] = sph_harm_y(m1,l1,theta,phi)
#     lap = lambda A: operators.Laplacian(A,c)
#     problem = problems.LBVP([u])
#     problem.add_equation((lap(u),f))
#     solver = solvers.LinearBoundaryValueSolver(problem)
#     solver.solve()

#     ug = [np.exp(1j*phi)*np.sqrt(3/(2*np.pi))*(1+2j*np.sqrt(5)*np.tan(theta)**(-2) + np.sin(theta)**-2)*np.sin(theta)/2,
#           -np.exp(1j*phi)*np.sqrt(3/(2*np.pi))*(4j - 7*np.sqrt(5) + 5*np.sqrt(5)*np.cos(2*theta))/(4*np.tan(theta))]
#     assert np.allclose(u['g'], ug)


# @pytest.mark.parametrize('Nphi', Nphi_range)
# @pytest.mark.parametrize('Ntheta', Ntheta_range)
# @pytest.mark.parametrize('dealias', dealias_range)
# @pytest.mark.parametrize('basis', [build_disk, build_annulus])
# @pytest.mark.parametrize('dtype', [np.float64, np.complex128])
# def test_laplacian_vector(Nphi,  Ntheta, dealias, basis, dtype):
#     c, d, b, phi, r, x, y = basis(Nphi, Ntheta, dealias, dtype)
#     v = field.Field(dist=d, tensorsig=(c,), bases=(b,), dtype=dtype)
#     v.preset_scales(dealias)
#     ex = np.array([-np.sin(phi)+0.*r,np.cos(phi)+0.*r])
#     ey = np.array([np.cos(phi)+0.*r,np.sin(phi)+0.*r])
#     v['g'] = 4*x**3*ey + 3*y**2*ey
#     U = operators.Laplacian(v,c).evaluate()
#     Ug = (24*x + 6)*ey
#     assert np.allclose(U['g'], Ug)


# @pytest.mark.parametrize('Ntheta', Ntheta_range)
# @pytest.mark.parametrize('dealias', dealias_range)
# @pytest.mark.parametrize('basis', [build_disk, build_annulus])
# @pytest.mark.parametrize('dtype', [np.float64, np.complex128])
# def test_laplacian_radial_vector(Ntheta, dealias, basis, dtype):
#     Nphi = 1
#     c, d, b, phi, r, x, y = basis(Nphi, Ntheta, dealias, dtype=dtype)
#     u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
#     u.preset_scales(dealias)
#     u['g'][1] = 4 * r**3
#     v = operators.Laplacian(u, c).evaluate()
#     vg = 0 * v['g']
#     vg[1] = 32 * r
#     assert np.allclose(v['g'], vg)


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_divergence_cleaning(Nphi, Ntheta, dealias, dtype):
    c, d, b, phi, theta = build_sphere(Nphi, Ntheta, dealias, dtype)
    # Random vector field
    f = d.Field(bases=b)
    f.fill_random(layout='g')
    f.low_pass_filter(scales=0.75)
    # Build vector field as grad(f) + skew(grad(f))
    g = operators.Gradient(f).evaluate()
    h = operators.Skew(g).evaluate()
    # Divergence cleaning LBVP
    u = d.VectorField(c, bases=b)
    psi = d.Field(bases=b)
    tau = d.Field()
    problem = problems.LBVP([u, psi, tau], namespace=locals())
    problem.add_equation("u + grad(psi) = h + g")
    problem.add_equation("div(u) + tau = 0")
    problem.add_equation("ave(psi) = 0")
    solver = problem.build_solver()
    solver.solve()
    u.change_scales(1)
    h.change_scales(1)
    assert np.allclose(u['g'], h['g'])


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_sphere_ell_product_scalar(Nphi, Ntheta, dealias, dtype):
    c, d, b, phi, theta = build_sphere(Nphi, Ntheta, dealias, dtype)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    g = field.Field(dist=d, bases=(b,), dtype=dtype)
    f.fill_random('g')
    func = lambda ell, r: ell + 3
    for ell, m_ind, ell_ind in b.ell_maps(d):
        g['c'][m_ind, ell_ind] = func(ell, b.radius) * f['c'][m_ind, ell_ind]
    h = operators.SphereEllProduct(f, c, func).evaluate()
    assert np.allclose(g['c'], h['c'])


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_sphere_ell_product_vector(Nphi, Ntheta, dealias, dtype):
    c, d, b, phi, theta = build_sphere(Nphi, Ntheta, dealias, dtype)
    f = field.Field(dist=d, bases=(b,), dtype=dtype, tensorsig=(c,))
    g = field.Field(dist=d, bases=(b,), dtype=dtype, tensorsig=(c,))
    f.fill_random('g')
    func = lambda ell, r: ell + 3
    for ell, m_ind, ell_ind in b.ell_maps(d):
        for i in range(c.dim):
            g['c'][i, m_ind, ell_ind] = func(ell, b.radius) * f['c'][i, m_ind, ell_ind]
    h = operators.SphereEllProduct(f, c, func).evaluate()
    assert np.allclose(g['c'], h['c'])

