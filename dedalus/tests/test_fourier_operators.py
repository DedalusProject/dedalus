

import pytest
import numpy as np
from dedalus.core import coords, distributor, basis, field, operators


N_range = [8, 10, 12]

@pytest.mark.parametrize('N', N_range)
def test_complex_fourier_derivatives(N):
    c = coords.Coordinate('x')
    d = distributor.Distributor((c,))
    b = basis.ComplexFourier(c, size=N, bounds=(0, 1))
    x = b.local_grid(1)
    f = field.Field(dist=d, bases=(b,), dtype=np.complex128)
    k = 4 * np.pi
    f['g'] = np.sin(k*x)
    fx = operators.Differentiate(f, c).evaluate()
    assert np.allclose(fx['g'], k*np.cos(k*x))

@pytest.mark.parametrize('N', N_range)
def test_complex_fourier_interp(N):
    c = coords.Coordinate('x')
    d = distributor.Distributor((c,))
    b = basis.ComplexFourier(c, size=N, bounds=(0, 1))
    x = b.local_grid(1)
    f = field.Field(dist=d, bases=(b,), dtype=np.complex128)
    k = 4 * np.pi
    f['g'] = np.sin(k*x)
    results = []
    for p in [0, 1, np.random.rand()]:
        fp = operators.Interpolate(f, c, p).evaluate()
        results.append(np.allclose(fp['g'], np.sin(k*p)))
    assert all(results)

@pytest.mark.parametrize('N', N_range)
def test_real_fourier_derivatives(N):
    c = coords.Coordinate('x')
    d = distributor.Distributor((c,))
    b = basis.RealFourier(c, size=N, bounds=(0, 1))
    x = b.local_grid(1)
    f = field.Field(dist=d, bases=(b,), dtype=np.float64)
    k = 4 * np.pi
    f['g'] = np.sin(k*x)
    fx = operators.Differentiate(f, c).evaluate()
    assert np.allclose(fx['g'], k*np.cos(k*x))

@pytest.mark.parametrize('N', N_range)
def test_real_fourier_interp(N):
    c = coords.Coordinate('x')
    d = distributor.Distributor((c,))
    b = basis.RealFourier(c, size=N, bounds=(0, 1))
    x = b.local_grid(1)
    f = field.Field(dist=d, bases=(b,), dtype=np.float64)
    k = 4 * np.pi
    f['g'] = np.sin(k*x)
    results = []
    for p in [0, 1, np.random.rand()]:
        fp = operators.Interpolate(f, c, p).evaluate()
        results.append(np.allclose(fp['g'], np.sin(k*p)))
    assert all(results)

