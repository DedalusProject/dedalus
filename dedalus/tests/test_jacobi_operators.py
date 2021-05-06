"""Test Jacobi conversion, differentiation, interpolation, and integration."""

import pytest
import numpy as np
from dedalus.core import coords, distributor, basis, field, operators, problems, solvers


N_range = [8, 9]
ab_range = [-0.5, 0]
k_range = [0, 1]
dtype_range = [np.float64, np.complex128]


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('a', ab_range)
@pytest.mark.parametrize('b', ab_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('layout', ['g', 'c'])
def test_jacobi_convert_constant(N, a, b, k, dtype, layout):
    c = coords.Coordinate('x')
    d = distributor.Distributor((c,))
    b = basis.Jacobi(c, size=N, a0=a, b0=b, a=a+k, b=b+k, bounds=(0, 1))
    fc = field.Field(dist=d, dtype=dtype)
    fc['g'] = 1
    fc[layout]
    f = operators.convert(fc, (b,)).evaluate()
    assert np.allclose(fc['g'], f['g'])


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('a', ab_range)
@pytest.mark.parametrize('b', ab_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('layout', ['g', 'c'])
def test_jacobi_convert_explicit(N, a, b, k, dtype, layout):
    c = coords.Coordinate('x')
    d = distributor.Distributor((c,))
    b = basis.Jacobi(c, size=N, a0=a, b0=b, a=a+k, b=b+k, bounds=(0, 1))
    x = b.local_grid(1)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    f['g'] = x**5
    fx = operators.Differentiate(f, c)
    f[layout]
    g = operators.convert(f, fx.domain.bases).evaluate()
    assert np.allclose(g['g'], f['g'])


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('a', ab_range)
@pytest.mark.parametrize('b', ab_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_jacobi_convert_implicit(N, a, b, k, dtype):
    c = coords.Coordinate('x')
    d = distributor.Distributor((c,))
    b = basis.Jacobi(c, size=N, a0=a, b0=b, a=a+k, b=b+k, bounds=(0, 1))
    x = b.local_grid(1)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    f['g'] = x**5
    fx = operators.Differentiate(f, c).evaluate()
    g = field.Field(dist=d, bases=(b,), dtype=dtype)
    problem = problems.LBVP([g])
    problem.add_equation((g, fx))
    solver = solvers.LinearBoundaryValueSolver(problem)
    solver.solve()
    assert np.allclose(g['g'], fx['g'])


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('a', ab_range)
@pytest.mark.parametrize('b', ab_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_jacobi_differentiate(N, a, b, k, dtype):
    c = coords.Coordinate('x')
    d = distributor.Distributor((c,))
    b = basis.Jacobi(c, size=N, a0=a, b0=b, a=a+k, b=b+k, bounds=(0, 1))
    x = b.local_grid(1)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    f['g'] = x**5
    fx = operators.Differentiate(f, c).evaluate()
    assert np.allclose(fx['g'], 5*x**4)


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('a', ab_range)
@pytest.mark.parametrize('b', ab_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_jacobi_interpolate(N, a, b, k, dtype):
    c = coords.Coordinate('x')
    d = distributor.Distributor((c,))
    b = basis.Jacobi(c, size=N, a0=a, b0=b, a=a+k, b=b+k, bounds=(0, 1))
    x = b.local_grid(1)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    f['g'] = x**5
    results = []
    for p in [0, 1, np.random.rand()]:
        fp = operators.Interpolate(f, c, p).evaluate()
        results.append(np.allclose(fp['g'], p**5))
    assert all(results)


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('a', ab_range)
@pytest.mark.parametrize('b', ab_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_jacobi_intergrate(N, a, b, k, dtype):
    c = coords.Coordinate('x')
    d = distributor.Distributor((c,))
    b = basis.Jacobi(c, size=N, a0=a, b0=b, a=a+k, b=b+k, bounds=(0, 1))
    x = b.local_grid(1)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    f['g'] = x**5
    fi = operators.Integrate(f, c).evaluate()
    assert np.allclose(fi['g'], 1/6)

