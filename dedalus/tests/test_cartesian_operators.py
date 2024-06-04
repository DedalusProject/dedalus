"""Test Cartesian skew, trace, transpose, curl."""
# TODO: add tests for other vector calculus operators, or split off cartesian_calculus.py

import pytest
import numpy as np
import dedalus.public as d3
from dedalus.tools.cache import CachedMethod


N_range = [16]
dealias_range = [1]
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
    x, y = d.local_grids(xb, yb, scales=dealias)
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
    x, y = d.local_grids(xb, yb, scales=dealias)
    r = (x, y)
    return c, d, b, r


@CachedMethod
def build_CC(N, dealias, dtype):
    c = d3.CartesianCoordinates('x', 'y')
    d = d3.Distributor(c, dtype=dtype)
    xb = d3.Chebyshev(c.coords[0], size=N, bounds=(0, Lx), dealias=dealias)
    yb = d3.Chebyshev(c.coords[1], size=N, bounds=(0, Ly), dealias=dealias)
    b = (xb, yb)
    x, y = d.local_grids(xb, yb, scales=dealias)
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
    x, y, z = d.local_grids(xb, yb, zb, scales=dealias)
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
    x, y, z = d.local_grids(xb, yb, zb, scales=dealias)
    r = (x, y, z)
    return c, d, b, r


@pytest.mark.parametrize('basis', [build_FF, build_FC, build_CC])
@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('layout', ['c', 'g'])
def test_skew_explicit(basis, N, dealias, dtype, layout):
    """Test explicit evaluation of skew operator for correctness."""
    c, d, b, r = basis(N, dealias, dtype)
    # Random vector field
    f = d.VectorField(c, bases=b)
    f.fill_random(layout='g')
    # Evaluate skew
    f.change_layout(layout)
    g = d3.skew(f).evaluate()
    assert np.allclose(g[layout][0], -f[layout][1])
    assert np.allclose(g[layout][1], f[layout][0])


@pytest.mark.parametrize('basis', [build_FF, build_FC, build_CC])
@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_skew_implicit(basis, N, dealias, dtype):
    """Test implicit evaluation of skew operator for correctness."""
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


@pytest.mark.parametrize('basis', [build_FF, build_FC, build_CC, build_FFF, build_FFC])
@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('layout', ['c', 'g'])
def test_trace_explicit(basis, N, dealias, dtype, layout):
    """Test explicit evaluation of trace operator for correctness."""
    c, d, b, r = basis(N, dealias, dtype)
    # Random tensor field
    f = d.TensorField((c,c), bases=b)
    f.fill_random(layout='g')
    # Evaluate trace
    f.change_layout(layout)
    g = d3.trace(f).evaluate()
    assert np.allclose(g[layout], np.trace(f[layout]))


@pytest.mark.parametrize('basis', [build_FF, build_FC, build_CC, build_FFF, build_FFC])
@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('layout', ['c', 'g'])
def test_trace_rank3_explicit(basis, N, dealias, dtype, layout):
    """Test explicit evaluation of trace operator for correctness."""
    c, d, b, r = basis(N, dealias, dtype)
    # Random tensor field
    f = d.TensorField((c,c,c), bases=b)
    f.fill_random(layout='g')
    # Evaluate trace
    f.change_layout(layout)
    g = d3.trace(f).evaluate()
    assert np.allclose(g[layout], np.trace(f[layout]))


@pytest.mark.parametrize('basis', [build_FF, build_FC, build_CC, build_FFF, build_FFC])
@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_trace_implicit(basis, N, dealias, dtype):
    """Test implicit evaluation of trace operator for correctness."""
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


@pytest.mark.parametrize('basis', [build_FF, build_FC, build_CC, build_FFF, build_FFC])
@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_trace_rank3_implicit(basis, N, dealias, dtype):
    """Test implicit evaluation of trace operator for correctness."""
    c, d, b, r = basis(N, dealias, dtype)
    # Random scalar field
    f = d.VectorField(c, bases=b)
    f.fill_random(layout='g')
    # Trace LBVP
    u = d.VectorField(c, bases=b)
    I = d.TensorField((c,c))
    dim = len(r)
    for i in range(dim):
        I['g'][i,i] = 1
    problem = d3.LBVP([u], namespace=locals())
    problem.add_equation("trace(I*u) = dim*f")
    solver = problem.build_solver()
    solver.solve()
    assert np.allclose(u['c'], f['c'])


@pytest.mark.parametrize('basis', [build_FF, build_FC, build_CC, build_FFF, build_FFC])
@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('layout', ['c', 'g'])
def test_transpose_explicit(basis, N, dealias, dtype, layout):
    """Test explicit evaluation of transpose operator for correctness."""
    c, d, b, r = basis(N, dealias, dtype)
    # Random tensor field
    f = d.TensorField((c,c), bases=b)
    f.fill_random(layout='g')
    # Evaluate transpose
    f.change_layout(layout)
    g = d3.transpose(f).evaluate()
    order = np.arange(2 + len(r))
    order[:2] = [1, 0]
    assert np.allclose(g[layout], np.transpose(f[layout], order))


@pytest.mark.parametrize('basis', [build_FF, build_FC, build_CC, build_FFF, build_FFC])
@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_transpose_implicit(basis, N, dealias, dtype):
    """Test implicit evaluation of transpose operator for correctness."""
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


@pytest.mark.parametrize('basis', [build_FF, build_FC, build_CC])
@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_2d_curl_explicit_vector(basis, N, dealias, dtype):
    """Test explicit evaluation of 2D vector curl operator for correctness."""
    c, d, b, (x, y) = basis(N, dealias, dtype)
    # Simple vector field
    kx, ky = 2*np.pi/Lx, 2*np.pi/Ly
    f = d.VectorField(c, bases=b)
    f.preset_scales(dealias)
    f['g'][0] = (np.sin(2*kx*x)+np.sin(kx*x))*np.cos(ky*y)
    f['g'][1] = np.sin(kx*x)*np.cos(ky*y)
    # Evaluate curl
    g_op = - d3.div(d3.skew(f)) # z @ curl(f)
    g = d.Field(bases=b)
    g.preset_scales(dealias)
    g['g'] = kx*np.cos(kx*x)*np.cos(ky*y) + ky*(np.sin(2*kx*x)+np.sin(kx*x))*np.sin(ky*y)
    assert np.allclose(g_op.evaluate()['g'], g['g'])


@pytest.mark.parametrize('basis', [build_FF, build_FC, build_CC])
@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_2d_curl_explicit_scalar(basis, N, dealias, dtype):
    """Test explicit evaluation of 2D scalar curl operator for correctness."""
    c, d, b, (x, y) = basis(2*N, dealias, dtype)
    # Simple scalar field
    kx, ky = 2*np.pi/Lx, 2*np.pi/Ly
    f = d.Field(bases=b)
    f.preset_scales(dealias)
    f['g'] = (np.sin(2*kx*x)+np.sin(kx*x))*np.cos(ky*y)
    # Evaluate curl
    g_op = - d3.skew(d3.grad(f)) # curl(f*ez)
    g = d.VectorField(c, bases=b)
    g.preset_scales(dealias)
    g['g'][0] = -ky*(np.sin(2*kx*x)+np.sin(kx*x))*np.sin(ky*y)
    g['g'][1] = -(2*kx*np.cos(2*kx*x)+kx*np.cos(kx*x))*np.cos(ky*y)
    assert np.allclose(g_op.evaluate()['g'], g['g'])


@pytest.mark.parametrize('basis', [build_FFF, build_FFC])
@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_curl_explicit(basis, N, dealias, dtype):
    """Test explicit evaluation of 3D curl operator for correctness."""
    c, d, b, r = basis(N, dealias, dtype)
    # ABC vector field
    k = 2*np.pi*np.array([1/Lx, 1/Ly, 1/Lz])
    f = d.VectorField(c, bases=b)
    f.preset_scales(dealias)
    f['g'][0] = np.sin(k[2]*r[2]) + np.cos(k[1]*r[1])
    f['g'][1] = np.sin(k[0]*r[0]) + np.cos(k[2]*r[2])
    f['g'][2] = np.sin(k[1]*r[1]) + np.cos(k[0]*r[0])
    # Evaluate curl
    g = d.VectorField(c, bases=b)
    g.preset_scales(dealias)
    g['g'][0] = k[2]*np.sin(k[2]*r[2]) + k[1]*np.cos(k[1]*r[1])
    g['g'][1] = k[0]*np.sin(k[0]*r[0]) + k[2]*np.cos(k[2]*r[2])
    g['g'][2] = k[1]*np.sin(k[1]*r[1]) + k[0]*np.cos(k[0]*r[0])
    assert np.allclose(d3.Curl(f).evaluate()['g'], g['g'])


@pytest.mark.parametrize('basis', [build_FFF])
@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_curl_implicit_FFF(basis, N, dealias, dtype):
    """Test implicit evaluation of 3D periodic curl operator for correctness."""
    c, d, b, r = basis(N, dealias, dtype)
    # ABC vector field
    k = 2 * np.pi * np.array([1/Lx, 1/Ly, 1/Lz])
    f = d.VectorField(c, bases=b)
    f.preset_scales(dealias)
    f['g'][0] = np.sin(k[2]*r[2]) + np.cos(k[1]*r[1])
    f['g'][1] = np.sin(k[0]*r[0]) + np.cos(k[2]*r[2])
    f['g'][2] = np.sin(k[1]*r[1]) + np.cos(k[0]*r[0])
    g = d.VectorField(c, bases=b)
    g.preset_scales(dealias)
    g['g'][0] = k[2]*np.sin(k[2]*r[2]) + k[1]*np.cos(k[1]*r[1])
    g['g'][1] = k[0]*np.sin(k[0]*r[0]) + k[2]*np.cos(k[2]*r[2])
    g['g'][2] = k[1]*np.sin(k[1]*r[1]) + k[0]*np.cos(k[0]*r[0])
    # Helmholtz LBVP
    u = d.VectorField(c, name='u', bases=b)
    phi = d.Field(name='phi', bases=b)
    tau1 = d.VectorField(c, name='tau1')
    tau2 = d.Field(name='tau2')
    problem = d3.LBVP([u, phi, tau1, tau2], namespace=locals())
    problem.add_equation("curl(u) + grad(phi) + tau1 = g")
    problem.add_equation("div(u) + tau2 = 0")
    problem.add_equation("integ(phi) = 0")
    problem.add_equation("integ(comp(u,index=0,comp=c['x'])) = 0")
    problem.add_equation("integ(comp(u,index=0,comp=c['y'])) = 0")
    problem.add_equation("integ(comp(u,index=0,comp=c['z'])) = 0")
    solver = problem.build_solver()
    solver.solve()
    assert np.allclose(u['c'], f['c'])


@pytest.mark.parametrize('basis', [build_FFC])
@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_curl_implicit_FFC(basis, N, dealias, dtype):
    """Test implicit evaluation of 3D curl operator for correctness."""
    c, d, b, r = basis(N, dealias, dtype)
    # ABC vector field
    k = 2 * np.pi * np.array([1/Lx, 1/Ly, 1/Lz])
    f = d.VectorField(c, bases=b)
    f.preset_scales(dealias)
    f['g'][0] = np.sin(k[2]*r[2]) + np.cos(k[1]*r[1])
    f['g'][1] = np.sin(k[0]*r[0]) + np.cos(k[2]*r[2])
    f['g'][2] = np.sin(k[1]*r[1]) + np.cos(k[0]*r[0])
    g = d.VectorField(c, bases=b)
    g.preset_scales(dealias)
    g['g'][0] = k[2]*np.sin(k[2]*r[2]) + k[1]*np.cos(k[1]*r[1])
    g['g'][1] = k[0]*np.sin(k[0]*r[0]) + k[2]*np.cos(k[2]*r[2])
    g['g'][2] = k[1]*np.sin(k[1]*r[1]) + k[0]*np.cos(k[0]*r[0])
    # Helmholtz LBVP
    u = d.VectorField(c, name='u', bases=b)
    phi = d.Field(name='phi', bases=b)
    tau1 = d.VectorField(c, name='tau1', bases=b[0:2])
    tau2 = d.Field(name='tau2', bases=b[0:2])
    lift_basis = b[2].derivative_basis(1)
    lift = lambda A, n: d3.Lift(A, lift_basis, n)
    problem = d3.LBVP([u, phi, tau1, tau2], namespace=locals())
    problem.add_equation("curl(u) + grad(phi) + lift(tau1,-1) = g")
    problem.add_equation("div(u) + lift(tau2,-1) = 0")
    problem.add_equation("u(z=0) = f(z=0)")
    problem.add_equation("phi(z=0) = 0")
    solver = problem.build_solver()
    solver.solve()
    assert np.allclose(u['c'], f['c'])

