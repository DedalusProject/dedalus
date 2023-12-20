"""Cylinder tests for trace, transpose, integrate, average."""

import pytest
import numpy as np
from dedalus.core import coords, distributor, basis, field, operators, arithmetic, problems, solvers
from dedalus.tools.cache import CachedFunction
from dedalus.core.basis import DiskBasis


length = 1.88
radius_disk = 1.5
radii_annulus = (0.5, 3)


@CachedFunction
def build_periodic_cylinder(Nz, Nphi, Nr, alpha, k, dealias, dtype):
    cz = coords.Coordinate('z')
    cp = coords.PolarCoordinates('phi', 'r')
    c = coords.DirectProduct(cz, cp)
    d = distributor.Distributor(c, dtype=dtype)
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
    d = distributor.Distributor(c, dtype=dtype)
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
def test_explicit_trace_tensor(Nz, Nphi, Nr, alpha, k, dealias, dtype, basis):
    c, d, b, z, phi, r, x, y = basis(Nz, Nphi, Nr, alpha, k, dealias, dtype)
    T = d.TensorField((c, c), bases=b)
    T.fill_random('g')
    T.low_pass_filter(scales=0.5)
    f_true = T['g'][0,0] + T['g'][1,1] + T['g'][2,2]
    f = operators.Trace(T).evaluate()
    f.change_scales(1)
    assert np.allclose(f['g'], f_true)


@pytest.mark.parametrize('Nz', Nz_range)
@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('alpha', alpha_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('basis', basis_range)
def test_implicit_trace_tensor(Nz, Nphi, Nr, alpha, k, dealias, dtype, basis):
    c, d, b, z, phi, r, x, y = basis(Nz, Nphi, Nr, alpha, k, dealias, dtype)
    I = d.IdentityTensor(c)
    f = d.Field(bases=b)
    g = d.Field(bases=b)
    g.fill_random('g')
    g.low_pass_filter(scales=0.5)
    problem = problems.LBVP([f])
    problem.add_equation((operators.Trace(I*f), 3*g))
    solver = solvers.LinearBoundaryValueSolver(problem)
    solver.solve()
    assert np.allclose(f['c'], g['c'])


# @pytest.mark.parametrize('basis', [build_disk, build_annulus])
# @pytest.mark.parametrize('Nphi', Nphi_range)
# @pytest.mark.parametrize('Nr', Nr_range)
# @pytest.mark.parametrize('k', k_range)
# @pytest.mark.parametrize('dealias', dealias_range)
# @pytest.mark.parametrize('dtype', [np.float64, np.complex128])
# @pytest.mark.parametrize('layout', ['c', 'g'])
# def test_transpose_explicit(basis, Nphi, Nr, k, dealias, dtype, layout):
#     c, d, b, phi, r, x, y = basis(Nphi, Nr, k, dealias, dtype)
#     # Random tensor field
#     f = d.TensorField((c, c), bases=b)
#     f.fill_random(layout='g')
#     f.low_pass_filter(scales=0.75)
#     # Evaluate transpose
#     f.change_layout(layout)
#     g = operators.transpose(f).evaluate()
#     assert np.allclose(g['g'], np.transpose(f['g'], (1,0,2,3)))


# @pytest.mark.parametrize('basis', [build_disk, build_annulus])
# @pytest.mark.parametrize('Nphi', Nphi_range)
# @pytest.mark.parametrize('Nr', Nr_range)
# @pytest.mark.parametrize('k', k_range)
# @pytest.mark.parametrize('dealias', dealias_range)
# @pytest.mark.parametrize('dtype', [np.float64, np.complex128])
# def test_transpose_implicit(basis, Nphi, Nr, k, dealias, dtype):
#     c, d, b, phi, r, x, y = basis(Nphi, Nr, k, dealias, dtype)
#     # Random tensor field
#     f = d.TensorField((c, c), bases=b)
#     f.fill_random(layout='g')
#     f.low_pass_filter(scales=0.75)
#     # Transpose LBVP
#     u = d.TensorField((c, c), bases=b)
#     problem = problems.LBVP([u], namespace=locals())
#     problem.add_equation("trans(u) = trans(f)")
#     solver = problem.build_solver()
#     solver.solve()
#     u.change_scales(dealias)
#     f.change_scales(dealias)
#     assert np.allclose(u['g'], f['g'])


# @pytest.mark.parametrize('Nphi', [16])
# @pytest.mark.parametrize('Nr', [10])
# @pytest.mark.parametrize('k', [0, 1, 2, 5])
# @pytest.mark.parametrize('dealias', dealias_range)
# @pytest.mark.parametrize('basis', [build_disk, build_annulus])
# @pytest.mark.parametrize('dtype', [np.float64, np.complex128])
# @pytest.mark.parametrize('n', [0, 1, 2])
# def test_integrate_scalar(Nphi, Nr, k, dealias, dtype, basis, n):
#     c, d, b, phi, r, x, y = basis(Nphi, Nr, k, dealias, dtype)
#     f = field.Field(dist=d, bases=(b,), dtype=dtype)
#     f.preset_scales(dealias)
#     f['g'] = r**(2*n)
#     h = operators.Integrate(f, c).evaluate()
#     if isinstance(b, DiskBasis):
#         r_inner, r_outer = 0, b.radius
#     else:
#         r_inner, r_outer = b.radii
#     hg = 2 * np.pi * (r_outer**(2 + 2*n) - r_inner**(2 + 2*n)) / (2 + 2*n)
#     assert np.allclose(h['g'], hg)


# @pytest.mark.parametrize('Nphi', [16])
# @pytest.mark.parametrize('Nr', [10])
# @pytest.mark.parametrize('k', [0, 1, 2, 5])
# @pytest.mark.parametrize('dealias', dealias_range)
# @pytest.mark.parametrize('basis', [build_disk, build_annulus])
# @pytest.mark.parametrize('dtype', [np.float64, np.complex128])
# @pytest.mark.parametrize('n', [0, 1, 2])
# def test_average_scalar(Nphi, Nr, k, dealias, dtype, basis, n):
#     c, d, b, phi, r, x, y = basis(Nphi, Nr, k, dealias, dtype)
#     f = field.Field(dist=d, bases=(b,), dtype=dtype)
#     f.preset_scales(dealias)
#     f['g'] = r**(2*n)
#     h = operators.Integrate(f, c).evaluate()
#     if isinstance(b, DiskBasis):
#         r_inner, r_outer = 0, b.radius
#     else:
#         r_inner, r_outer = b.radii
#     hg = 2 * np.pi * (r_outer**(2 + 2*n) - r_inner**(2 + 2*n)) / (2 + 2*n)
#     assert np.allclose(h['g'], hg)

