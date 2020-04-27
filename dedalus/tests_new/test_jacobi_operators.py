

import pytest
import numpy as np
from dedalus.core import coords, distributor, basis, field, operators


N_range = [8, 9, 10]
ab_range = [-0.5, 0, 0.5]

@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('a', ab_range)
@pytest.mark.parametrize('b', ab_range)
def test_jacobi_derivatives(N, a, b):
    c = coords.Coordinate('x')
    d = distributor.Distributor((c,))
    b = basis.Jacobi(c, size=N, a=a, b=b, bounds=(0, 1))
    x = b.local_grid(1)
    f = field.Field(dist=d, bases=(b,), dtype=np.complex128)
    f['g'] = x**5
    fx = operators.Differentiate(f, c).evaluate()
    assert np.allclose(fx['g'], 5*x**4)

@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('a', ab_range)
@pytest.mark.parametrize('b', ab_range)
def test_jacobi_conversion(N, a, b):
    c = coords.Coordinate('x')
    d = distributor.Distributor((c,))
    b = basis.Jacobi(c, size=N, a=a, b=b, bounds=(0, 1))
    x = b.local_grid(1)
    f = field.Field(dist=d, bases=(b,), dtype=np.complex128)
    f['g'] = x**5
    fx = operators.Differentiate(f, c).evaluate()
    g = operators.convert(f, fx.domain.bases).evaluate()
    assert np.allclose(g['g'], f['g'])

@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('a', ab_range)
@pytest.mark.parametrize('b', ab_range)
def test_jacobi_interp(N, a, b):
    c = coords.Coordinate('x')
    d = distributor.Distributor((c,))
    b = basis.Jacobi(c, size=N, a=a, b=b, bounds=(0, 1))
    x = b.local_grid(1)
    f = field.Field(dist=d, bases=(b,), dtype=np.complex128)
    f['g'] = x**5
    results = []
    for p in [0, 1, np.random.rand()]:
        fp = operators.Interpolate(f, c, p).evaluate()
        results.append(np.allclose(fp['g'], p**5))
    assert all(results)

@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('a', ab_range)
@pytest.mark.parametrize('b', ab_range)
def test_jacobi_constant(N, a, b):
    c = coords.Coordinate('x')
    d = distributor.Distributor((c,))
    b = basis.Jacobi(c, size=N, a=a, b=b, bounds=(0, 1))
    f = field.Field(dist=d, bases=(b,), dtype=np.complex128)
    f['g'] = 1
    fp = operators.Interpolate(f, c, 0.5)
    f_recon = operators.convert(fp, f.domain.bases).evaluate()
    assert np.allclose(f_recon['g'], f['g'])
