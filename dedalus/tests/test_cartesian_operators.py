"""Test cartesian skew, trace, transpose."""

import pytest
import numpy as np
import dedalus.public as d3
from dedalus.tools.cache import CachedMethod


Nx_range = [8]
Ny_range = [8]
Nz_range = [8]
N_range = [16]
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
    x = xb.local_grid(dealias)
    y = yb.local_grid(dealias)
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
    x = xb.local_grid(dealias)
    y = yb.local_grid(dealias)
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
    x = xb.local_grid(dealias)
    y = yb.local_grid(dealias)
    z = zb.local_grid(dealias)
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
    x = xb.local_grid(dealias)
    y = yb.local_grid(dealias)
    z = zb.local_grid(dealias)
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
    assert np.allclose(u['c'][:,0], f['c'][:,0])


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


@pytest.mark.parametrize('basis', [build_FFF, build_FFC])
@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_curl_explicit(basis, N, dealias, dtype):
    c, d, b, r = basis(N, dealias, dtype)
    # ABC vector field
    k = 2*np.pi*np.array([1/Lx, 1/Ly, 1/Lz])
    f = d.VectorField(c, bases=b)
    f.set_scales(dealias)
    f['g'][0] = np.sin(k[2]*r[2]) + np.cos(k[1]*r[1])
    f['g'][1] = np.sin(k[0]*r[0]) + np.cos(k[2]*r[2])
    f['g'][2] = np.sin(k[1]*r[1]) + np.cos(k[0]*r[0])
    g = d.VectorField(c, bases=b)
    g.set_scales(dealias)
    g['g'][0] = k[2]*np.sin(k[2]*r[2]) + k[1]*np.cos(k[1]*r[1])
    g['g'][1] = k[0]*np.sin(k[0]*r[0]) + k[2]*np.cos(k[2]*r[2])
    g['g'][2] = k[1]*np.sin(k[1]*r[1]) + k[0]*np.cos(k[0]*r[0])
    assert np.allclose(d3.Curl(f).evaluate()['g'], g['g'])


@pytest.mark.parametrize('basis', [build_FFC])
@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_curl_implicit_FFC(basis, N, dealias, dtype):
    c, d, b, r = basis(N, dealias, dtype)
    Lz = b[2].bounds[1]
    lift_basis = b[2].clone_with(a=1/2, b=1/2) # First derivative basis
    lift = lambda A, n: d3.LiftTau(A, lift_basis, n)
    integ = lambda A: d3.Integrate(d3.Integrate(d3.Integrate(A,c.coords[0]),c.coords[1]),c.coords[2])
    tau1 = d.VectorField(c, name='tau1', bases=b[0:2])
    tau2 = d.Field(name='tau2', bases=b[0:2])
    # ABC vector field
    k = 2*np.pi*np.array([1/Lx, 1/Ly, 1/Lz])
    f = d.VectorField(c, bases=b)
    f.set_scales(dealias)
    f['g'][0] = np.sin(k[2]*r[2]) + np.cos(k[1]*r[1])
    f['g'][1] = np.sin(k[0]*r[0]) + np.cos(k[2]*r[2])
    f['g'][2] = np.sin(k[1]*r[1]) + np.cos(k[0]*r[0])
    g = d.VectorField(c, bases=b)
    g.set_scales(dealias)
    g['g'][0] = k[2]*np.sin(k[2]*r[2]) + k[1]*np.cos(k[1]*r[1])
    g['g'][1] = k[0]*np.sin(k[0]*r[0]) + k[2]*np.cos(k[2]*r[2])
    g['g'][2] = k[1]*np.sin(k[1]*r[1]) + k[0]*np.cos(k[0]*r[0])
    # Skew LBVP
    u = d.VectorField(c, name='u', bases=b)
    phi = d.Field(name='phi', bases=b)
    problem = d3.LBVP([u, phi, tau1, tau2], namespace=locals())
    problem.add_equation("curl(u) + grad(phi) + lift(tau1,-1) = g")
    problem.add_equation("div(u) + lift(tau2,-1) = 0")
    problem.add_equation("u(z=0) = f(z=0)")
    problem.add_equation("phi(z=0) = 0")
    solver = problem.build_solver()
    solver.solve()
    assert np.allclose(u['c'], f['c'])


# dy uz - dz uy + dx phi = gx
# dz ux - dx uz + dy phi = gy
# dx uy - dy ux + dz phi = gz
# dx ux + dy uy + dz uz = 0

# dz uy = -gx
# dz ux = gy
# dz phi = gz
# dz uz = 0


@pytest.mark.parametrize('basis', [build_FFF])
@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_curl_implicit_FFF(basis, N, dealias, dtype):
    c, d, b, r = basis(N, dealias, dtype)
    Lz = b[2].bounds[1]
    integ = lambda A: d3.Integrate(d3.Integrate(d3.Integrate(A,c.coords[0]),c.coords[1]),c.coords[2])
    tau1 = d.VectorField(c, name='tau1')
    tau2 = d.Field(name='tau2')
    # ABC vector field
    k = 2*np.pi*np.array([1/Lx, 1/Ly, 1/Lz])
    f = d.VectorField(c, bases=b)
    f.set_scales(dealias)
    f['g'][0] = np.sin(k[2]*r[2]) + np.cos(k[1]*r[1])
    f['g'][1] = np.sin(k[0]*r[0]) + np.cos(k[2]*r[2])
    f['g'][2] = np.sin(k[1]*r[1]) + np.cos(k[0]*r[0])
    g = d.VectorField(c, bases=b)
    g.set_scales(dealias)
    g['g'][0] = k[2]*np.sin(k[2]*r[2]) + k[1]*np.cos(k[1]*r[1])
    g['g'][1] = k[0]*np.sin(k[0]*r[0]) + k[2]*np.cos(k[2]*r[2])
    g['g'][2] = k[1]*np.sin(k[1]*r[1]) + k[0]*np.cos(k[0]*r[0])
    # Skew LBVP
    u = d.VectorField(c, name='u', bases=b)
    phi = d.Field(name='phi', bases=b)
    problem = d3.LBVP([u, phi, tau1, tau2], namespace=locals())
    problem.add_equation("curl(u) + grad(phi) + tau1 = g")
    problem.add_equation("div(u) + tau2 = 0")
    problem.add_equation("integ(phi) = 0")
    problem.add_equation("integ(comp(u,index=0,coord=c['x'])) = 0")
    problem.add_equation("integ(comp(u,index=0,coord=c['y'])) = 0")
    problem.add_equation("integ(comp(u,index=0,coord=c['z'])) = 0")
    solver = problem.build_solver()
    solver.print_subproblem_ranks()
    solver.solve()
    assert np.allclose(u['c'], f['c'])

    assert np.allclose(Tt['g'], Ttg)


@pytest.mark.parametrize('Nx', Nx_range)
@pytest.mark.parametrize('Ny', Ny_range)
@pytest.mark.parametrize('Nz', Nz_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_3d_box])
@pytest.mark.parametrize('dtype', [np.complex128, np.float64])
def test_curl_3d(Nx, Ny, Nz, dealias, basis, dtype):
    c, d, b, x, y, z = basis(Nx, Ny, Nz, dealias, dtype)
    u = field.Field(dist=d, bases=b, tensorsig=(c,), dtype=dtype)
    u['g'][0] = (np.sin(2*x)+np.sin(x))*np.cos(y)*np.sin(z)
    u['g'][1] = (np.cos(2*x)+np.cos(x))*np.sin(y)*np.sin(z)
    u['g'][2] = np.sin(x)*np.cos(y)*np.cos(z)
    ω = operators.Curl(u).evaluate()
    ω_c = field.Field(dist=d, bases=b, tensorsig=(c,), dtype=dtype)
    ω_c['g'][0] = np.cos(2*x)+np.cos(x))*np.sin(y)*np.cos(z) - (-1*np.sin(x)*np.sin(y)*np.cos(z))
    ω_c['g'][1] = np.cos(x)*np.cos(y)*np.cos(z) - (np.sin(2*x)+np.sin(x))*np.cos(y)*np.cos(z)
    ω_c['g'][2] = -1*(np.sin(2*x)+np.sin(x))*np.sin(y)*np.sin(z) - (-2*np.sin(2*x)-1*np.sin(x))*np.sin(y)*np.sin(z)
    assert np.allclose(ω['g'], ω_c['g'])

@pytest.mark.parametrize('Nx', Nx_range)
@pytest.mark.parametrize('Nz', Nz_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_2d_box])
@pytest.mark.parametrize('dtype', [np.complex128, np.float64])
def test_curl_2d(Nx, Nz, dealias, basis, dtype):
    c, d, b, x, z = basis(Nx, Nz, dealias, dtype)
    u = field.Field(dist=d, bases=b, tensorsig=(c,), dtype=dtype)
    u['g'][0] = (np.sin(2*x)+np.sin(x))*np.cos(y)*np.sin(z)
    u['g'][1] = np.sin(x)*np.cos(y)*np.cos(z) # this is the z-component
    ω = operators.Curl(u).evaluate()
    ω_c = field.Field(dist=d, bases=b, dtype=dtype) # is omega_c just a scalar?  or can we specify the y-component.  Is this 2-D or 2.5D?
    ω_c['g'] = np.cos(x)*np.cos(y)*np.cos(z) - (np.sin(2*x)+np.sin(x))*np.cos(y)*np.cos(z)
    assert np.allclose(ω['g'], ω_c['g'])
