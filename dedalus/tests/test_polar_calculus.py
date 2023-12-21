"""Disk and annulus tests for gradient, divergence, curl, laplacian."""

import pytest
import numpy as np
from dedalus.core import coords, distributor, basis, field, operators, arithmetic
from dedalus.tools.cache import CachedFunction


Nphi_range = [16]
Nr_range = [8]
dealias_range = [1, 3/2]
radius_disk = 1.5
radii_annulus = (0.5, 3)


@CachedFunction
def build_disk(Nphi, Nr, dealias, dtype):
    c = coords.PolarCoordinates('phi', 'r')
    d = distributor.Distributor((c,))
    b = basis.DiskBasis(c, (Nphi, Nr), radius=radius_disk, dealias=(dealias, dealias), dtype=dtype)
    phi, r = d.local_grids(b, scales=dealias)
    x, y = c.cartesian(phi, r)
    return c, d, b, phi, r, x, y


@CachedFunction
def build_annulus(Nphi, Nr, dealias, dtype):
    c = coords.PolarCoordinates('phi', 'r')
    d = distributor.Distributor((c,))
    b = basis.AnnulusBasis(c, (Nphi, Nr), radii=radii_annulus, dealias=(dealias, dealias), dtype=dtype)
    phi, r = d.local_grids(b, scales=dealias)
    x, y = c.cartesian(phi, r)
    return c, d, b, phi, r, x, y


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_disk, build_annulus])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_gradient_scalar(Nphi, Nr, dealias, basis, dtype):
    c, d, b, phi, r, x, y = basis(Nphi, Nr, dealias, dtype)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    f.preset_scales(dealias)
    f['g'] = 3*x**2 + 2*y
    u = operators.Gradient(f, c).evaluate()
    ex = np.array([-np.sin(phi)+0.*r,np.cos(phi)+0.*r])
    ey = np.array([np.cos(phi)+0.*r,np.sin(phi)+0.*r])
    ug = 6*x*ex + 2*ey
    assert np.allclose(u['g'], ug)


@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_disk, build_annulus])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_gradient_radial_scalar(Nr, dealias, basis, dtype):
    Nphi = 1
    c, d, b, phi, r, x, y = basis(Nphi, Nr, dealias, dtype)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    f.preset_scales(dealias)
    f['g'] = r**4
    u = operators.Gradient(f, c).evaluate()
    ug = [0*r*phi, 4*r**3 + 0*phi]
    assert np.allclose(u['g'], ug)


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_disk, build_annulus])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_gradient_vector(Nphi, Nr, dealias, basis, dtype):
    c, d, b, phi, r, x, y = basis(Nphi, Nr, dealias, dtype)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    f.preset_scales(dealias)
    f['g'] = 3*x**4 + 2*y*x
    grad = lambda A: operators.Gradient(A, c)
    T = grad(grad(f)).evaluate()
    ex = np.array([-np.sin(phi)+0.*r,np.cos(phi)+0.*r])
    ey = np.array([np.cos(phi)+0.*r,np.sin(phi)+0.*r])
    exex = ex[:,None, ...] * ex[None,...]
    eyex = ey[:,None, ...] * ex[None,...]
    exey = ex[:,None, ...] * ey[None,...]
    eyey = ey[:,None, ...] * ey[None,...]
    Tg = 36*x**2*exex + 2*(exey + eyex)
    assert np.allclose(T['g'], Tg)


@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_disk, build_annulus])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_gradient_radial_vector(Nr, dealias, basis, dtype):
    Nphi = 1
    c, d, b, phi, r, x, y = basis(Nphi, Nr, dealias, dtype)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    f.preset_scales(dealias)
    f['g'] = r**4
    grad = lambda A: operators.Gradient(A, c)
    T = grad(grad(f)).evaluate()
    er = np.array([[[0]], [[1]]])
    ephi = np.array([[[1]], [[0]]])
    erer = er[:, None, ...] * er[None, ...]
    ephiephi = ephi[:, None, ...] * ephi[None, ...]
    Tg = 12 * r**2 * erer + 4 * r**2 * ephiephi
    assert np.allclose(T['g'], Tg)


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_disk, build_annulus])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_divergence_vector(Nphi, Nr, dealias, basis, dtype):
    c, d, b, phi, r, x, y = basis(Nphi, Nr, dealias, dtype)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    f.preset_scales(dealias)
    f['g'] = 3*x**4 + 2*y*x
    grad = lambda A: operators.Gradient(A, c)
    div = lambda A: operators.Divergence(A)
    S = div(grad(f)).evaluate()
    Sg = 36*x**2
    assert np.allclose(S['g'], Sg)


@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_disk, build_annulus])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_divergence_radial_vector(Nr, dealias, basis, dtype):
    Nphi = 1
    c, d, b, phi, r, x, y = basis(Nphi, Nr, dealias, dtype=dtype)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    f.preset_scales(dealias)
    f['g'] = r**2
    grad = lambda A: operators.Gradient(A, c)
    div = lambda A: operators.Divergence(A)
    h = div(grad(f)).evaluate()
    hg = 4
    assert np.allclose(h['g'], hg)


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_disk, build_annulus])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_divergence_tensor(Nphi, Nr, dealias, basis, dtype):
    c, d, b, phi, r, x, y = basis(Nphi, Nr, dealias, dtype)
    v = field.Field(dist=d, tensorsig=(c,), bases=(b,), dtype=dtype)
    v.preset_scales(dealias)
    ex = np.array([-np.sin(phi)+0.*r,np.cos(phi)+0.*r])
    ey = np.array([np.cos(phi)+0.*r,np.sin(phi)+0.*r])
    v['g'] = 4*x**3*ey + 3*y**2*ey
    grad = lambda A: operators.Gradient(A, c)
    div = lambda A: operators.Divergence(A)
    U = div(grad(v)).evaluate()
    Ug = (24*x + 6)*ey
    assert np.allclose(U['g'], Ug)


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_disk, build_annulus])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_curl_vector(Nphi, Nr, dealias, basis, dtype):
    c, d, b, phi, r, x, y = basis(Nphi, Nr, dealias, dtype)
    v = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
    v.preset_scales(dealias)
    ex = np.array([-np.sin(phi)+0.*r,np.cos(phi)+0.*r])
    ey = np.array([np.cos(phi)+0.*r,np.sin(phi)+0.*r])
    v['g'] = 4*x**3*ey + 3*y**2*ey
    curl = lambda A: - operators.Divergence(operators.Skew(A))
    u = curl(v).evaluate()
    ug = 12*x**2
    assert np.allclose(u['g'], ug)


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_disk, build_annulus])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_laplacian_scalar(Nphi,  Nr, dealias, basis, dtype):
    c, d, b, phi, r, x, y = basis(Nphi, Nr, dealias, dtype)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    f.preset_scales(dealias)
    f['g'] = x**4 + 2*y**4
    h = operators.Laplacian(f, c).evaluate()
    hg = 12*x**2+24*y**2
    assert np.allclose(h['g'], hg)


@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_disk, build_annulus])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_laplacian_radial_scalar(Nr, dealias, basis, dtype):
    Nphi = 1
    c, d, b, phi, r, x, y = basis(Nphi, Nr, dealias, dtype=dtype)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    f.preset_scales(dealias)
    f['g'] = r**2
    h = operators.Laplacian(f, c).evaluate()
    hg = 4
    assert np.allclose(h['g'], hg)


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_disk, build_annulus])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_laplacian_vector(Nphi,  Nr, dealias, basis, dtype):
    c, d, b, phi, r, x, y = basis(Nphi, Nr, dealias, dtype)
    v = field.Field(dist=d, tensorsig=(c,), bases=(b,), dtype=dtype)
    v.preset_scales(dealias)
    ex = np.array([-np.sin(phi)+0.*r,np.cos(phi)+0.*r])
    ey = np.array([np.cos(phi)+0.*r,np.sin(phi)+0.*r])
    v['g'] = 4*x**3*ey + 3*y**2*ey
    U = operators.Laplacian(v,c).evaluate()
    Ug = (24*x + 6)*ey
    assert np.allclose(U['g'], Ug)


@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_disk, build_annulus])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_laplacian_radial_vector(Nr, dealias, basis, dtype):
    Nphi = 1
    c, d, b, phi, r, x, y = basis(Nphi, Nr, dealias, dtype=dtype)
    u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
    u.preset_scales(dealias)
    u['g'][1] = 4 * r**3
    v = operators.Laplacian(u, c).evaluate()
    vg = 0 * v['g']
    vg[1] = 32 * r
    assert np.allclose(v['g'], vg)

