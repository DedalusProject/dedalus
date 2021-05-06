"""Test Fourier differentiation, interpolation, and integration."""

import pytest
import numpy as np
from dedalus.core import coords, distributor, basis, field, operators


N_range = [8, 10, 12]
dtype_range = [np.float64, np.complex128]


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('layout', ['g', 'c'])
def test_fourier_convert_constant(N, dtype, layout):
    c = coords.Coordinate('x')
    d = distributor.Distributor((c,))
    if dtype == np.float64:
        b = basis.RealFourier(c, size=N, bounds=(0, 1))
    elif dtype == np.complex128:
        b = basis.ComplexFourier(c, size=N, bounds=(0, 1))
    fc = field.Field(dist=d, dtype=dtype)
    fc['g'] = 1
    fc[layout]
    f = operators.convert(fc, (b,)).evaluate()
    assert np.allclose(fc['g'], f['g'])


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_fourier_differentiate(N, dtype):
    c = coords.Coordinate('x')
    d = distributor.Distributor((c,))
    if dtype == np.float64:
        b = basis.RealFourier(c, size=N, bounds=(0, 1))
    elif dtype == np.complex128:
        b = basis.ComplexFourier(c, size=N, bounds=(0, 1))
    x = b.local_grid(1)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    k = 4 * np.pi
    f['g'] = 1 + np.sin(k*x+0.1)
    fx = operators.Differentiate(f, c).evaluate()
    assert np.allclose(fx['g'], k*np.cos(k*x+0.1))


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_fourier_interpolate(N, dtype):
    c = coords.Coordinate('x')
    d = distributor.Distributor((c,))
    if dtype == np.float64:
        b = basis.RealFourier(c, size=N, bounds=(0, 1))
    elif dtype == np.complex128:
        b = basis.ComplexFourier(c, size=N, bounds=(0, 1))
    x = b.local_grid(1)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    k = 4 * np.pi
    f['g'] = 1 + np.sin(k*x+0.1)
    results = []
    for p in [0, 1, np.random.rand()]:
        fp = operators.Interpolate(f, c, p).evaluate()
        results.append(np.allclose(fp['g'], 1 + np.sin(k*p+0.1)))
    assert all(results)


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_fourier_integrate(N, dtype):
    c = coords.Coordinate('x')
    d = distributor.Distributor((c,))
    if dtype == np.float64:
        b = basis.RealFourier(c, size=N, bounds=(0, 1))
    elif dtype == np.complex128:
        b = basis.ComplexFourier(c, size=N, bounds=(0, 1))
    x = b.local_grid(1)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    k = 4 * np.pi
    f['g'] = 1 + np.sin(k*x+0.1)
    fi = operators.Integrate(f, c).evaluate()
    assert np.allclose(fi['g'], 1)

