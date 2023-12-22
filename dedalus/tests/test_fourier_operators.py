"""Test Fourier convert, differentiate, interpolate, integrate, average."""

import pytest
import numpy as np
import dedalus.public as d3
from dedalus.tools.cache import CachedMethod


N_range = [10]
bounds_range = [(0.5, 1.666)]
dealias_range = [1]
dtype_range = [np.float64, np.complex128]


@CachedMethod
def build_fourier(N, bounds, dealias, dtype):
    c = d3.Coordinate('x')
    d = d3.Distributor(c, dtype=dtype)
    b = d3.Fourier(c, size=N, bounds=bounds, dealias=dealias, dtype=dtype)
    x = d.local_grid(b, scale=1)
    return c, d, b, x


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('bounds', bounds_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('layout', ['g', 'c'])
def test_fourier_convert_constant(N, bounds, dealias, dtype, layout):
    """Test conversion from constant to Fourier basis."""
    c, d, b, x = build_fourier(N, bounds, dealias, dtype)
    f = d.Field()
    f['g'] = 1
    f.change_layout(layout)
    g = d3.Convert(f, b).evaluate()
    assert np.allclose(g['g'], f['g'])


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('bounds', bounds_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_fourier_differentiate(N, bounds, dealias, dtype):
    """Test differentiation in Fourier basis."""
    c, d, b, x = build_fourier(N, bounds, dealias, dtype)
    f = d.Field(bases=b)
    k = 4 * np.pi / (bounds[1] - bounds[0])
    f['g'] = 1 + np.sin(k*x+0.1)
    g = d3.Differentiate(f, c).evaluate()
    assert np.allclose(g['g'], k*np.cos(k*x+0.1))


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('bounds', bounds_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_fourier_interpolate(N, bounds, dealias, dtype):
    """Test interpolation in Fourier basis."""
    c, d, b, x = build_fourier(N, bounds, dealias, dtype)
    f = d.Field(bases=b)
    k = 4 * np.pi / (bounds[1] - bounds[0])
    f['g'] = 1 + np.sin(k*x+0.1)
    results = []
    for p in [bounds[0], bounds[1], bounds[0] + (bounds[1] - bounds[0]) * np.random.rand()]:
        g = d3.Interpolate(f, c, p).evaluate()
        results.append(np.allclose(g['g'], 1 + np.sin(k*p+0.1)))
    assert all(results)


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('bounds', bounds_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_fourier_integrate(N, bounds, dealias, dtype):
    """Test integration in Fourier basis."""
    c, d, b, x = build_fourier(N, bounds, dealias, dtype)
    f = d.Field(bases=b)
    k = 4 * np.pi / (bounds[1] - bounds[0])
    f['g'] = 1 + np.sin(k*x+0.1)
    g = d3.Integrate(f, c).evaluate()
    assert np.allclose(g['g'], bounds[1] - bounds[0])


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('bounds', bounds_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_fourier_average(N, bounds, dealias, dtype):
    """Test averaging in Fourier basis."""
    c, d, b, x = build_fourier(N, bounds, dealias, dtype)
    f = d.Field(bases=b)
    k = 4 * np.pi / (bounds[1] - bounds[0])
    f['g'] = 1 + np.sin(k*x+0.1)
    g = d3.Average(f, c).evaluate()
    assert np.allclose(g['g'], 1)

