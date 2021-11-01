"""Test cartesian skew, trace, transpose."""

import pytest
import numpy as np
import dedalus.public as d3
from dedalus.tools.cache import CachedMethod


Nx_range = [8]
Ny_range = [8]
Nz_range = [8]
N_range = [8]
dealias_range = [1, 3/2]
dtype_range = [np.float64, np.complex128]

Lx = 1.3
Ly = 2.4
Lz = 1.9

@CachedMethod
def build_FF(N, dealias, dtype):
    c = d3.CartesianCoordinates('x', 'y')
    d = d3.Distributor(c, dtype=dtype)
    if dtype == np.complex128:
        xb = d3.ComplexFourier(c.coords[0], size=N, bounds=(0, Lx), dealias=dealias)
        yb = d3.ComplexFourier(c.coords[1], size=N, bounds=(0, Ly), dealias=dealias)
    elif dtype == np.float64:
        xb = d3.RealFourier(c.coords[0], size=N, bounds=(0, Lx), dealias=dealias)
        yb = d3.RealFourier(c.coords[1], size=N, bounds=(0, Ly), dealias=dealias)
    b = (xb, yb)
    x = xb.local_grid(1)
    y = yb.local_grid(1)
    r = (x, y)
    return c, d, b, r

@CachedMethod
def build_FC(N, dealias, dtype):
    c = d3.CartesianCoordinates('x', 'y')
    d = d3.Distributor(c, dtype=dtype)
    if dtype == np.complex128:
        xb = d3.ComplexFourier(c.coords[0], size=N, bounds=(0, Lx), dealias=dealias)
    elif dtype == np.float64:
        xb = d3.RealFourier(c.coords[0], size=N, bounds=(0, Lx), dealias=dealias)
    yb = d3.Chebyshev(c.coords[1], size=N, bounds=(0, Ly), dealias=dealias)
    b = (xb, yb)
    x = xb.local_grid(1)
    y = yb.local_grid(1)
    r = (x, y)
    return c, d, b, r

@CachedMethod
def build_FFF(N, dealias, dtype):
    c = d3.CartesianCoordinates('x', 'y', 'z')
    d = d3.Distributor(c, dtype=dtype)
    if dtype == np.complex128:
        xb = d3.ComplexFourier(c.coords[0], size=N, bounds=(0, Lx), dealias=dealias)
        yb = d3.ComplexFourier(c.coords[1], size=N, bounds=(0, Ly), dealias=dealias)
        zb = d3.ComplexFourier(c.coords[2], size=N, bounds=(0, Lz), dealias=dealias)
    elif dtype == np.float64:
        xb = d3.RealFourier(c.coords[0], size=N, bounds=(0, Lx), dealias=dealias)
        yb = d3.RealFourier(c.coords[1], size=N, bounds=(0, Ly), dealias=dealias)
        zb = d3.RealFourier(c.coords[2], size=N, bounds=(0, Lz), dealias=dealias)
    b = (xb, yb, zb)
    x = xb.local_grid(1)
    y = yb.local_grid(1)
    z = zb.local_grid(1)
    r = (x, y, z)
    return c, d, b, r

@CachedMethod
def build_FFC(N, dealias, dtype):
    c = d3.CartesianCoordinates('x', 'y', 'z')
    d = d3.Distributor(c, dtype=dtype)
    if dtype == np.complex128:
        xb = d3.ComplexFourier(c.coords[0], size=N, bounds=(0, Lx), dealias=dealias)
        yb = d3.ComplexFourier(c.coords[1], size=N, bounds=(0, Ly), dealias=dealias)
    elif dtype == np.float64:
        xb = d3.RealFourier(c.coords[0], size=N, bounds=(0, Lx), dealias=dealias)
        yb = d3.RealFourier(c.coords[1], size=N, bounds=(0, Ly), dealias=dealias)
    zb = d3.ChebyshevT(c.coords[2], size=N, bounds=(0, Lz), dealias=dealias)
    b = (xb, yb, zb)
    x = xb.local_grid(1)
    y = yb.local_grid(1)
    z = zb.local_grid(1)
    r = (x, y, z)
    return c, d, b, r


@pytest.mark.parametrize('basis', [build_FF, build_FC])
@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('layout', ['c', 'g'])
def test_skew_explicit(basis, N, dealias, dtype, layout):
    c, d, b, r = basis(N, dealias, dtype)
    # Random vector field
    f = d.VectorField(c, bases=b)
    f.fill_random(layout='g')
    # Evaluate skew
    f.require_layout(layout)
    g = d3.skew(f).evaluate()
    assert np.allclose(g[layout][0], -f[layout][1])
    assert np.allclose(g[layout][1], f[layout][0])


@pytest.mark.parametrize('basis', [build_FF, build_FC])
@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_skew_implicit(basis, N, dealias, dtype):
    c, d, b, r = basis(N, dealias, dtype)
    # Random vector field
    f = d.VectorField(c, bases=b)
    f.fill_random(layout='g')
    # Skew LBVP
    u = d.VectorField(c, bases=b)
    problem = d3.LBVP([u], namespace=locals())
    problem.add_equation("skew(u) = skew(f)")
    solver = problem.build_solver()
    solver.solve()
    assert np.allclose(u['c'], f['c'])


@pytest.mark.parametrize('basis', [build_FF, build_FC, build_FFF, build_FFC])
@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('layout', ['c', 'g'])
def test_trace_explicit(basis, N, dealias, dtype, layout):
    c, d, b, r = basis(N, dealias, dtype)
    # Random tensor field
    f = d.TensorField((c,c), bases=b)
    f.fill_random(layout='g')
    # Evaluate trace
    f.require_layout(layout)
    g = d3.trace(f).evaluate()
    assert np.allclose(g[layout], np.trace(f[layout]))


@pytest.mark.parametrize('basis', [build_FF, build_FC, build_FFF, build_FFC])
@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_trace_implicit(basis, N, dealias, dtype):
    c, d, b, r = basis(N, dealias, dtype)
    # Random scalar field
    f = d.Field(bases=b)
    f.fill_random(layout='g')
    # Trace LBVP
    u = d.Field(bases=b)
    I = d.TensorField((c,c))
    dim = len(r)
    for i in range(dim):
        I['g'][i,i] = 1
    problem = d3.LBVP([u], namespace=locals())
    problem.add_equation("trace(I*u) = dim*f")
    solver = problem.build_solver()
    solver.solve()
    assert np.allclose(u['c'], f['c'])


@pytest.mark.parametrize('basis', [build_FF, build_FC, build_FFF, build_FFC])
@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('layout', ['c', 'g'])
def test_transpose_explicit(basis, N, dealias, dtype, layout):
    c, d, b, r = basis(N, dealias, dtype)
    # Random tensor field
    f = d.TensorField((c,c), bases=b)
    f.fill_random(layout='g')
    # Evaluate transpose
    f.require_layout(layout)
    g = d3.transpose(f).evaluate()
    order = np.arange(2 + len(r))
    order[:2] = [1, 0]
    assert np.allclose(g[layout], np.transpose(f[layout], order))


@pytest.mark.parametrize('basis', [build_FF, build_FC, build_FFF, build_FFC])
@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_transpose_implicit(basis, N, dealias, dtype):
    c, d, b, r = basis(N, dealias, dtype)
    # Random tensor field
    f = d.TensorField((c,c), bases=b)
    f.fill_random(layout='g')
    # Transpose LBVP
    u = d.TensorField((c,c), bases=b)
    problem = d3.LBVP([u], namespace=locals())
    problem.add_equation("transpose(u) = transpose(f)")
    solver = problem.build_solver()
    solver.solve()
    assert np.allclose(u['c'], f['c'])

