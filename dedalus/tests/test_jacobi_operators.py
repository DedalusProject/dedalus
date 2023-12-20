"""Test Jacobi convert, differentiate, interpolate, integrate, average, lift."""

import pytest
import numpy as np
import dedalus.public as d3
from dedalus.tools.cache import CachedMethod


N_range = [8, 9]
ab_range = [(-1/2, -1/2), (0, 0)]
k_range = [0, 1]
dealias_range = [1]
dtype_range = [np.float64, np.complex128]


@CachedMethod
def build_jacobi(N, a, b, k, bounds, dealias, dtype):
    c = d3.Coordinate('x')
    d = d3.Distributor(c, dtype=dtype)
    b = d3.Jacobi(c, size=N, a0=a, b0=b, a=a+k, b=b+k, bounds=bounds, dealias=dealias)
    x = d.local_grid(b, scale=1)
    return c, d, b, x


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('a, b', ab_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('layout', ['g', 'c'])
def test_jacobi_convert_constant(N, a, b, k, dealias, dtype, layout):
    """Test explicit conversion from constant to Jacobi basis."""
    c, d, b, x = build_jacobi(N, a, b, k, (0, 1), dealias, dtype)
    f = d.Field()
    f['g'] = 1
    f.change_layout(layout)
    g = d3.Convert(f, b).evaluate()
    assert np.allclose(g['g'], f['g'])


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('a, b', ab_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dk', [0, 1, 2])
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('layout', ['g', 'c'])
def test_jacobi_convert(N, a, b, k, dk, dealias, dtype, layout):
    """Test explicit conversion between Jacobi bases."""
    c, d, b, x = build_jacobi(N, a, b, k, (0, 1), dealias, dtype)
    f = d.Field(bases=b)
    f.fill_random(layout='g')
    f.low_pass_filter(scales=0.5)
    f.change_layout(layout)
    g = d3.Convert(f, b.derivative_basis(dk)).evaluate()
    assert np.allclose(g['g'], f['g'])


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('a, b', ab_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dk', [0, 1, 2])
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_jacobi_convert_implicit(N, a, b, k, dk, dealias, dtype):
    """Test conversion between Jacobi bases."""
    c, d, b, x = build_jacobi(N, a, b, k, (0, 1), dealias, dtype)
    f = d.Field(bases=b.derivative_basis(dk))
    f.fill_random(layout='g')
    f.low_pass_filter(scales=0.5)
    g = d.Field(bases=b)
    problem = d3.LBVP([g], namespace=locals())
    problem.add_equation("g = f")
    solver = problem.build_solver()
    solver.solve()
    assert np.allclose(g['g'], f['g'])


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('a, b', ab_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_jacobi_differentiate(N, a, b, k, dealias, dtype):
    """Test differentiation in Jacobi basis."""
    c, d, b, x = build_jacobi(N, a, b, k, (0, 1), dealias, dtype)
    f = d.Field(bases=b)
    f['g'] = x**5
    g = d3.Differentiate(f, c).evaluate()
    assert np.allclose(g['g'], 5*x**4)


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('a, b', ab_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_jacobi_interpolate(N, a, b, k, dealias, dtype):
    """Test interpolation in Jacobi basis."""
    c, d, b, x = build_jacobi(N, a, b, k, (0, 1), dealias, dtype)
    f = d.Field(bases=b)
    f['g'] = x**5
    results = []
    for p in [0, 1, np.random.rand()]:
        fp = d3.Interpolate(f, c, p).evaluate()
        results.append(np.allclose(fp['g'], p**5))
    assert all(results)


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('a, b', ab_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_jacobi_integrate(N, a, b, k, dealias, dtype):
    """Test integration in Jacobi basis."""
    c, d, b, x = build_jacobi(N, a, b, k, (0, 3), dealias, dtype)
    f = d.Field(bases=b)
    f['g'] = 6 * x**5
    g = d3.Integrate(f, c).evaluate()
    assert np.allclose(g['g'], 3**6)


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('a, b', ab_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_jacobi_average(N, a, b, k, dealias, dtype):
    """Test averaging in Jacobi basis."""
    c, d, b, x = build_jacobi(N, a, b, k, (0, 3), dealias, dtype)
    f = d.Field(bases=b)
    f['g'] = 6 * x**5
    g = d3.Average(f, c).evaluate()
    assert np.allclose(g['g'], 3**6 / 3)


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('a, b', ab_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('n', [-1, -2])
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_jacobi_lift(N, a, b, k, n, dealias, dtype):
    """Test lifting to Jacobi basis."""
    c, d, b, x = build_jacobi(N, a, b, k, (0, 3), dealias, dtype)
    lift_basis = b.derivative_basis(k)
    f = d.Field(bases=lift_basis)
    f['c'][n] = 2
    tau = d.Field()
    tau['g'] = 2
    g = d3.Lift(tau, lift_basis, n).evaluate()
    assert np.allclose(g['g'], f['g'])

