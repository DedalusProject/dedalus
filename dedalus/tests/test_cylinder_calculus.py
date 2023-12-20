"""Cylinder tests for gradient, divergence, curl, laplacian."""

import pytest
import numpy as np
from dedalus.core import coords, distributor, basis, field, operators, arithmetic
from dedalus.tools.cache import CachedFunction


length = 1.88
radius_disk = 1.5
radii_annulus = (0.5, 3)

@CachedFunction
def build_periodic_cylinder(Nz, Nphi, Nr, alpha, k, dealias, dtype):
    cz = coords.Coordinate('z')
    cp = coords.PolarCoordinates('phi', 'r')
    c = coords.DirectProduct(cz, cp)
    d = distributor.Distributor(c)
    bz = basis.Fourier(cz, Nz, bounds=(0, length), dealias=dealias, dtype=dtype)
    bp = basis.DiskBasis(cp, (Nphi, Nr), dtype=dtype, radius=radius_disk, alpha=alpha, k=k, dealias=dealias)
    z, phi, r = d.local_grids(bz, bp, scales=dealias)
    x, y = cp.cartesian(phi, r)
    return c, d, (bz, bp), z, phi, r, x, y

@CachedFunction
def build_periodic_cylindrical_annulus(Nz, Nphi, Nr, alpha, k, dealias, dtype):
    cz = coords.Coordinate('z')
    cp = coords.PolarCoordinates('phi', 'r')
    c = coords.DirectProduct(cz, cp)
    d = distributor.Distributor(c)
    bz = basis.Fourier(cz, Nz, bounds=(0, length), dealias=dealias, dtype=dtype)
    bp = basis.AnnulusBasis(cp, (Nphi, Nr), dtype=dtype, radii=radii_annulus, alpha=alpha, k=k, dealias=dealias)
    z, phi, r = d.local_grids(bz, bp, scales=dealias)
    x, y = cp.cartesian(phi, r)
    return c, d, (bz, bp), z, phi, r, x, y


Nz_range = [8]
Nphi_range = [16]
Nr_range = [8]
alpha_range = [0]
k_range = [0]
dealias_range = [1, 3/2]
basis_range = [build_periodic_cylinder, build_periodic_cylindrical_annulus]
dtype_range = [np.float64, np.complex128]


@pytest.mark.parametrize('Nz', Nz_range)
@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('alpha', alpha_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('basis', basis_range)
def test_gradient_scalar(Nz, Nphi, Nr, alpha, k, dealias, dtype, basis):
    c, d, b, z, phi, r, x, y = basis(Nz, Nphi, Nr, alpha, k, dealias, dtype)
    f = field.Field(dist=d, bases=b, dtype=dtype)
    f.preset_scales(dealias)
    kz = 4 * np.pi / length
    f['g'] = 3*x**2 + 2*y + np.sin(kz*z)
    u = operators.Gradient(f, c).evaluate()
    zero_grid = np.zeros((z.size, phi.size, r.size))
    ex = [0, -np.sin(phi), np.cos(phi)]
    ey = [0, np.cos(phi), np.sin(phi)]
    ez = [1, 0, 0]
    ex = np.array([zero_grid+i for i in ex])
    ey = np.array([zero_grid+i for i in ey])
    ez = np.array([zero_grid+i for i in ez])
    ug = 6*x*ex + 2*ey + kz*np.cos(kz*z)*ez
    assert np.allclose(u['g'], ug)


@pytest.mark.parametrize('Nz', Nz_range)
@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('alpha', alpha_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('basis', basis_range)
def test_gradient_vector(Nz, Nphi, Nr, alpha, k, dealias, dtype, basis):
    c, d, b, z, phi, r, x, y = basis(Nz, Nphi, Nr, alpha, k, dealias, dtype)
    f = field.Field(dist=d, bases=b, dtype=dtype)
    f.preset_scales(dealias)
    kz = 4 * np.pi / length
    f['g'] = 3*x**4 + 2*y*x + np.sin(kz*z)*x
    grad = lambda A: operators.Gradient(A, c)
    T = grad(grad(f)).evaluate()
    zero_grid = np.zeros((z.size, phi.size, r.size))
    ex = [0, -np.sin(phi), np.cos(phi)]
    ey = [0, np.cos(phi), np.sin(phi)]
    ez = [1, 0, 0]
    ex = np.array([zero_grid+i for i in ex])
    ey = np.array([zero_grid+i for i in ey])
    ez = np.array([zero_grid+i for i in ez])
    exex = ex[:,None, ...] * ex[None,...]
    exey = ex[:,None, ...] * ey[None,...]
    exez = ex[:,None, ...] * ez[None,...]
    eyex = ey[:,None, ...] * ex[None,...]
    eyey = ey[:,None, ...] * ey[None,...]
    eyez = ey[:,None, ...] * ez[None,...]
    ezex = ez[:,None, ...] * ex[None,...]
    ezey = ez[:,None, ...] * ey[None,...]
    ezez = ez[:,None, ...] * ez[None,...]
    Tg = 36*x**2*exex + 2*(exey + eyex) + kz*np.cos(kz*z)*(exez+ezex) - kz**2*np.sin(kz*z)*x*ezez
    assert np.allclose(T['g'], Tg)


@pytest.mark.parametrize('Nz', Nz_range)
@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('alpha', alpha_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('basis', basis_range)
def test_divergence_vector(Nz, Nphi, Nr, alpha, k, dealias, dtype, basis):
    c, d, b, z, phi, r, x, y = basis(Nz, Nphi, Nr, alpha, k, dealias, dtype)
    f = field.Field(dist=d, bases=b, dtype=dtype)
    f.preset_scales(dealias)
    kz = 4 * np.pi / length
    f['g'] = 3*x**4 + 2*y*x + np.sin(kz*z)*x
    grad = lambda A: operators.Gradient(A, c)
    div = lambda A: operators.Divergence(A)
    S = div(grad(f)).evaluate()
    Sg = 36*x**2 - kz**2*np.sin(kz*z)*x
    assert np.allclose(S['g'], Sg)


@pytest.mark.parametrize('Nz', Nz_range)
@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('alpha', alpha_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('basis', basis_range)
def test_divergence_tensor(Nz, Nphi, Nr, alpha, k, dealias, dtype, basis):
    c, d, b, z, phi, r, x, y = basis(Nz, Nphi, Nr, alpha, k, dealias, dtype)
    v = field.Field(dist=d, tensorsig=(c,), bases=b, dtype=dtype)
    v.preset_scales(dealias)
    zero_grid = np.zeros((z.size, phi.size, r.size))
    ex = [0, -np.sin(phi), np.cos(phi)]
    ey = [0, np.cos(phi), np.sin(phi)]
    ez = [1, 0, 0]
    ex = np.array([zero_grid+i for i in ex])
    ey = np.array([zero_grid+i for i in ey])
    ez = np.array([zero_grid+i for i in ez])
    kz = 4 * np.pi / length
    v['g'] = 4*x**3*ey + 3*y**2*ey + x*y*np.sin(kz*z)*ez
    grad = lambda A: operators.Gradient(A, c)
    div = lambda A: operators.Divergence(A)
    U = div(grad(v)).evaluate()
    Ug = (24*x + 6)*ey - kz**2*x*y*np.sin(kz*z)*ez
    assert np.allclose(U['g'], Ug)


@pytest.mark.parametrize('Nz', Nz_range)
@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('alpha', alpha_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('basis', basis_range)
def test_curl_vector(Nz, Nphi, Nr, alpha, k, dealias, dtype, basis):
    c, d, b, z, phi, r, x, y = basis(Nz, Nphi, Nr, alpha, k, dealias, dtype)
    v = field.Field(dist=d, tensorsig=(c,), bases=b, dtype=dtype)
    v.preset_scales(dealias)
    zero_grid = np.zeros((z.size, phi.size, r.size))
    ex = [0, -np.sin(phi), np.cos(phi)]
    ey = [0, np.cos(phi), np.sin(phi)]
    ez = [1, 0, 0]
    ex = np.array([zero_grid+i for i in ex])
    ey = np.array([zero_grid+i for i in ey])
    ez = np.array([zero_grid+i for i in ez])
    kz = 4 * np.pi / length
    v['g'] = 4*x**3*ey + 3*y**2*ey + x*y*np.sin(kz*z)*ez
    u = operators.Curl(v).evaluate()
    u_true = 12*x**2*ez + x*np.sin(kz*z)*ex - y*np.sin(kz*z)*ey
    assert np.allclose(u['g'], u_true)


@pytest.mark.parametrize('Nz', Nz_range)
@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('alpha', alpha_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('basis', basis_range)
def test_laplacian_scalar(Nz, Nphi, Nr, alpha, k, dealias, dtype, basis):
    c, d, b, z, phi, r, x, y = basis(Nz, Nphi, Nr, alpha, k, dealias, dtype)
    f = field.Field(dist=d, bases=b, dtype=dtype)
    f.preset_scales(dealias)
    kz = 4 * np.pi / length
    f['g'] = x**4 + 2*y**4 + np.sin(kz*z)*x
    h = operators.Laplacian(f, c).evaluate()
    hg = 12*x**2 + 24*y**2 - kz**2*np.sin(kz*z)*x
    assert np.allclose(h['g'], hg)


@pytest.mark.parametrize('Nz', Nz_range)
@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('alpha', alpha_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('basis', basis_range)
def test_laplacian_vector(Nz, Nphi, Nr, alpha, k, dealias, dtype, basis):
    c, d, b, z, phi, r, x, y = basis(Nz, Nphi, Nr, alpha, k, dealias, dtype)
    v = field.Field(dist=d, tensorsig=(c,), bases=b, dtype=dtype)
    v.preset_scales(dealias)
    zero_grid = np.zeros((z.size, phi.size, r.size))
    ex = [0, -np.sin(phi), np.cos(phi)]
    ey = [0, np.cos(phi), np.sin(phi)]
    ez = [1, 0, 0]
    ex = np.array([zero_grid+i for i in ex])
    ey = np.array([zero_grid+i for i in ey])
    ez = np.array([zero_grid+i for i in ez])
    kz = 4 * np.pi / length
    v['g'] = 4*x**3*ey + 3*y**2*ey + np.sin(kz*z)*x*ez
    U = operators.Laplacian(v,c).evaluate()
    Ug = (24*x + 6)*ey - kz**2*np.sin(kz*z)*x*ez
    assert np.allclose(U['g'], Ug)

