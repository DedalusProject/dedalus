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


@CachedMethod
def build_parity(N, bounds, dealias, parity, dtype):
    c = d3.Coordinate('x')
    d = d3.Distributor(c, dtype=dtype)
    b = d3.Parity(c, size=N, bounds=bounds, dealias=dealias, parity=parity)
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
@pytest.mark.parametrize('layout', ['g', 'c'])
def test_parity_convert_constant(N, bounds, dealias, dtype, layout):
    """Test conversion from constant to EvenParity basis."""
    c, d, b, x = build_parity(N, bounds, dealias, 1, dtype)
    f = d.Field()
    f['g'] = 1
    f.change_layout(layout)
    g = d3.Convert(f, b).evaluate()
    assert np.allclose(g['g'], f['g'])


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('bounds', bounds_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('order', [1, 2, 3])
def test_fourier_differentiate(N, bounds, dealias, dtype, order):
    """Test differentiation in Fourier basis."""
    c, d, b, x = build_fourier(N, bounds, dealias, dtype)
    f = d.Field(bases=b)
    L = bounds[1] - bounds[0]
    k = 4 * np.pi / L
    f['g'] = 1 + np.sin(k*x+0.1)
    g = d3.Differentiate(f, c, order=order).evaluate()
    if order == 1:
        assert np.allclose(g['g'], k*np.cos(k*x+0.1))
    elif order == 2:
        assert np.allclose(g['g'], -k**2*np.sin(k*x+0.1))
    elif order == 3:
        assert np.allclose(g['g'], -k**3*np.cos(k*x+0.1))


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('bounds', bounds_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('parity', [1, -1])
@pytest.mark.parametrize('order', [1, 2, 3])
def test_parity_differentiate(N, bounds, dealias, dtype, parity, order):
    """Test differentiation in Parity bases."""
    c, d, b, x = build_parity(N, bounds, dealias, parity, dtype)
    f = d.Field(bases=b)
    x0 = bounds[0]
    L = bounds[1] - bounds[0]
    k = 3 * np.pi / L
    if parity == 1:
        f['g'] = 1 + np.cos(k*(x-x0))
        g = d3.Differentiate(f, c, order=order).evaluate()
        if order == 1:
            assert np.allclose(g['g'], -k*np.sin(k*(x-x0)))
        elif order == 2:
            assert np.allclose(g['g'], -k**2*np.cos(k*(x-x0)))
        elif order == 3:
            assert np.allclose(g['g'], k**3*np.sin(k*(x-x0)))
    elif parity == -1:
        f['g'] = np.sin(k*(x-x0))
        g = d3.Differentiate(f, c, order=order).evaluate()
        if order == 1:
            assert np.allclose(g['g'], k*np.cos(k*(x-x0)))
        elif order == 2:
            assert np.allclose(g['g'], -k**2*np.sin(k*(x-x0)))
        elif order == 3:
            assert np.allclose(g['g'], -k**3*np.cos(k*(x-x0)))


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('bounds', bounds_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('order', [1, 1.5, 2])
def test_fourier_riesz(N, bounds, dealias, dtype, order):
    """Test Riesz derivative in Fourier basis."""
    c, d, b, x = build_fourier(N, bounds, dealias, dtype)
    f = d.Field(bases=b)
    L = bounds[1] - bounds[0]
    k = 4 * np.pi / L
    f['g'] = 1 + np.sin(k*x+0.1)
    g = d3.RieszDerivative(f, c, order=order).evaluate()
    assert np.allclose(g['g'], -abs(k)**order*np.sin(k*x+0.1))


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('bounds', bounds_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('parity', [1, -1])
@pytest.mark.parametrize('order', [1, 1.5, 2])
def test_parity_riesz(N, bounds, dealias, dtype, parity, order):
    """Test Riesz derivative in Fourier basis."""
    c, d, b, x = build_parity(N, bounds, dealias, parity, dtype)
    f = d.Field(bases=b)
    x0 = bounds[0]
    L = bounds[1] - bounds[0]
    k = 4 * np.pi / L
    if parity == 1:
        f['g'] = 1 + np.cos(k*(x-x0))
        g = d3.RieszDerivative(f, c, order=order).evaluate()
        assert np.allclose(g['g'], -abs(k)**order*np.cos(k*(x-x0)))
    elif parity == -1:
        f['g'] = np.sin(k*(x-x0))
        g = d3.RieszDerivative(f, c, order=order).evaluate()
        assert np.allclose(g['g'], -abs(k)**order*np.sin(k*(x-x0)))


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('bounds', bounds_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_fourier_hilbert(N, bounds, dealias, dtype):
    """Test Hilbert transform in Fourier basis."""
    c, d, b, x = build_fourier(N, bounds, dealias, dtype)
    f = d.Field(bases=b)
    L = bounds[1] - bounds[0]
    k = 4 * np.pi / L
    f['g'] = 1 + np.sin(k*x+0.1)
    g = d3.HilbertTransform(f, c).evaluate()
    assert np.allclose(g['g'], -np.cos(k*x+0.1))


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('bounds', bounds_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('parity', [1, -1])
def test_parity_hilbert(N, bounds, dealias, dtype, parity):
    """Test Hilbert transform in Parity bases."""
    c, d, b, x = build_parity(N, bounds, dealias, parity, dtype)
    f = d.Field(bases=b)
    x0 = bounds[0]
    L = bounds[1] - bounds[0]
    k = 4 * np.pi / L
    if parity == 1:
        f['g'] = 1 + np.cos(k*(x-x0))
        g = d3.HilbertTransform(f, c).evaluate()
        assert np.allclose(g['g'], np.sin(k*(x-x0)))
    elif parity == -1:
        f['g'] = np.sin(k*(x-x0))
        g = d3.HilbertTransform(f, c).evaluate()
        assert np.allclose(g['g'], -np.cos(k*(x-x0)))


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('bounds', bounds_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_fourier_interpolate(N, bounds, dealias, dtype):
    """Test interpolation in Fourier basis."""
    c, d, b, x = build_fourier(N, bounds, dealias, dtype)
    f = d.Field(bases=b)
    L = bounds[1] - bounds[0]
    k = 4 * np.pi / L
    f0 = lambda x: 1 + np.sin(k*x+0.1)
    f['g'] = f0(x)
    results = []
    for p in [bounds[0], bounds[1], bounds[0] + L*np.random.rand()]:
        g = d3.Interpolate(f, c, p).evaluate()
        results.append(np.allclose(g['g'], f0(p)))
    assert all(results)


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('bounds', bounds_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('parity', [1, -1])
def test_parity_interpolate(N, bounds, dealias, dtype, parity):
    """Test interpolation in Parity bases."""
    c, d, b, x = build_parity(N, bounds, dealias, parity, dtype)
    f = d.Field(bases=b)
    x0 = bounds[0]
    L = bounds[1] - bounds[0]
    k = 3 * np.pi / L
    if parity == 1:
        f0 = lambda x: 1 + np.cos(k*(x-x0))
    elif parity == -1:
        f0 = lambda x: np.sin(k*(x-x0))
    f['g'] = f0(x)
    results = []
    for p in [bounds[0], bounds[1], bounds[0] + (bounds[1] - bounds[0]) * np.random.rand()]:
        g = d3.Interpolate(f, c, p).evaluate()
        results.append(np.allclose(g['g'], f0(p)))
    assert all(results)


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('bounds', bounds_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_fourier_integrate(N, bounds, dealias, dtype):
    """Test integration in Fourier basis."""
    c, d, b, x = build_fourier(N, bounds, dealias, dtype)
    f = d.Field(bases=b)
    L = bounds[1] - bounds[0]
    k = 4 * np.pi / L
    f['g'] = 1 + np.sin(k*x+0.1)
    g = d3.Integrate(f, c).evaluate()
    assert np.allclose(g['g'], L)


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('bounds', bounds_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('parity', [1, -1])
def test_parity_integrate(N, bounds, dealias, dtype, parity):
    """Test integration in Parity bases."""
    c, d, b, x = build_parity(N, bounds, dealias, parity, dtype)
    f = d.Field(bases=b)
    x0 = bounds[0]
    L = bounds[1] - bounds[0]
    k = 3 * np.pi / L
    if parity == 1:
        f['g'] = 1 + np.cos(k*(x-x0))
        g = d3.Integrate(f, c).evaluate()
        assert np.allclose(g['g'], L)
    elif parity == -1:
        f['g'] = np.sin(k*(x-x0))
        g = d3.Integrate(f, c).evaluate()
        assert np.allclose(g['g'], L*2/3/np.pi)


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('bounds', bounds_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_fourier_average(N, bounds, dealias, dtype):
    """Test averaging in Fourier basis."""
    c, d, b, x = build_fourier(N, bounds, dealias, dtype)
    f = d.Field(bases=b)
    L = bounds[1] - bounds[0]
    k = 4 * np.pi / L
    f['g'] = 1 + np.sin(k*x+0.1)
    g = d3.Average(f, c).evaluate()
    assert np.allclose(g['g'], 1)


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('bounds', bounds_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('parity', [1, -1])
def test_parity_average(N, bounds, dealias, dtype, parity):
    """Test averaging in Parity bases."""
    c, d, b, x = build_parity(N, bounds, dealias, parity, dtype)
    f = d.Field(bases=b)
    x0 = bounds[0]
    L = bounds[1] - bounds[0]
    k = 3 * np.pi / L
    if parity == 1:
        f['g'] = 1 + np.cos(k*(x-x0))
        g = d3.Average(f, c).evaluate()
        assert np.allclose(g['g'], 1)
    elif parity == -1:
        f['g'] = np.sin(k*(x-x0))
        g = d3.Average(f, c).evaluate()
        assert np.allclose(g['g'], 2/3/np.pi)
