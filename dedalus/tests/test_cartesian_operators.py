import pytest
import numpy as np
from dedalus.core import coords, distributor, basis, field, operators, arithmetic, problems, solvers
from dedalus.tools.cache import CachedMethod
from mpi4py import MPI

comm = MPI.COMM_WORLD

Lx=2
Ly=2
Lz=1
@CachedMethod
def build_3d_box(Nx, Ny, Nz, dealias, dtype, k=0):
    c = coords.CartesianCoordinates('x', 'y', 'z')
    d = distributor.Distributor((c,))
    if dtype == np.complex128:
        xb = basis.ComplexFourier(c.coords[0], size=Nx, bounds=(0, Lx))
        yb = basis.ComplexFourier(c.coords[1], size=Ny, bounds=(0, Ly))
    elif dtype == np.float64:
        xb = basis.RealFourier(c.coords[0], size=Nx, bounds=(0, Lx))
        yb = basis.RealFourier(c.coords[1], size=Ny, bounds=(0, Ly))

    zb = basis.ChebyshevT(c.coords[2], size=Nz, bounds=(0, Lz))
    b = (xb, yb, zb)
    x = xb.local_grid(1)
    y = yb.local_grid(1)
    z = zb.local_grid(1)
    return c, d, b, x, y, z

@CachedMethod
def build_2d_box(Nx, Nz, dealias, dtype, k=0):
    c = coords.CartesianCoordinates('x', 'z')
    d = distributor.Distributor((c,))
    if dtype == np.complex128:
        xb = basis.ComplexFourier(c.coords[0], size=Nx, bounds=(0, Lx))
    elif dtype == np.float64:
        xb = basis.RealFourier(c.coords[0], size=Nx, bounds=(0, Lx))
    zb = basis.ChebyshevT(c.coords[1], size=Nz, bounds=(0, Lz))
    b = (xb, zb)
    x = xb.local_grid(1)
    z = zb.local_grid(1)
    return c, d, b, x, z

Nx_range = [8, 2]
Ny_range = [8]
Nz_range = [8, 2]
dealias_range = [1, 3/2]


@pytest.mark.parametrize('Nx', Nx_range)
@pytest.mark.parametrize('Nz', Nz_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_2d_box])
@pytest.mark.parametrize('dtype', [np.complex128, np.float64])
def test_explicit_trace_2d(Nx, Nz, dealias, basis, dtype):
    c, d, b, x, z = basis(Nx, Nz, dealias, dtype)
    u = field.Field(dist=d, bases=(*b,), tensorsig=(c,), dtype=dtype)
    u['g'][0] = (np.sin(2*x)+np.sin(x))*np.sin(z)
    u['g'][1] = np.sin(x)*np.cos(z)
    T = operators.Gradient(u, c).evaluate()
    f = operators.Trace(T).evaluate()
    T.require_scales(1)
    f.require_scales(1)
    fg = T['g'][0,0] + T['g'][1,1]
    assert np.allclose(f['g'], fg)


@pytest.mark.parametrize('Nx', Nx_range)
@pytest.mark.parametrize('Nz', Nz_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_2d_box])
@pytest.mark.parametrize('dtype', [np.complex128, np.float64])
def test_implicit_trace_2d(Nx, Nz, dealias, basis, dtype):
    c, d, b, x, z = basis(Nx, Nz, dealias, dtype)
    f = field.Field(dist=d, bases=(*b,), dtype=dtype)
    g = field.Field(dist=d, bases=(*b,), dtype=dtype)
    g.require_scales(g.domain.dealias)
    g['g'] = (np.sin(2*x) + np.sin(x)) * np.sin(z)
    I = field.Field(dist=d, tensorsig=(c,c), dtype=dtype)
    I['g'][0,0] = I['g'][1,1] = 1
    trace = lambda A: operators.Trace(A)
    problem = problems.LBVP([f])
    problem.add_equation((trace(I*f), 2*g))
    solver = solvers.LinearBoundaryValueSolver(problem)
    solver.solve()
    assert np.allclose(f['c'], g['c'])


@pytest.mark.parametrize('Nx', Nx_range)
@pytest.mark.parametrize('Ny', Ny_range)
@pytest.mark.parametrize('Nz', Nz_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_3d_box])
@pytest.mark.parametrize('dtype', [np.complex128, np.float64])
def test_explicit_transpose_3d_tensor(Nx, Ny, Nz, dealias, basis, dtype):
    c, d, b, x, y, z = basis(Nx, Ny, Nz, dealias, dtype)
    u = field.Field(dist=d, bases=(*b,), tensorsig=(c,), dtype=dtype)
    u['g'][0] = (np.sin(2*x)+np.sin(x))*np.cos(y)*np.sin(z)
    u['g'][1] = (np.cos(2*x)+np.cos(x))*np.sin(y)*np.sin(z)
    u['g'][2] = np.sin(x)*np.cos(y)*np.cos(z)
    T = operators.Gradient(u, c).evaluate()
    T.require_grid_space()
    Tg = np.transpose(np.copy(T['g']),(1,0,2,3,4))
    T = operators.TransposeComponents(T).evaluate()
    assert np.allclose(T['g'], Tg)

@pytest.mark.parametrize('Nx', Nx_range)
@pytest.mark.parametrize('Nz', Nz_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_2d_box])
@pytest.mark.parametrize('dtype', [np.complex128, np.float64])
def test_explicit_transpose_2d_tensor(Nx, Nz, dealias, basis, dtype):
    c, d, b, x, z = basis(Nx, Nz, dealias, dtype)
    u = field.Field(dist=d, bases=(*b,), tensorsig=(c,), dtype=dtype)
    u['g'][0] = (np.sin(2*x)+np.sin(x))*np.sin(z)
    u['g'][1] = np.sin(x)*np.cos(z)
    T = operators.Gradient(u, c).evaluate()
    T.require_grid_space()
    Tg = np.transpose(np.copy(T['g']),(1,0,2,3))
    T = operators.TransposeComponents(T).evaluate()
    assert np.allclose(T['g'], Tg)

@pytest.mark.parametrize('Nx', Nx_range)
@pytest.mark.parametrize('Ny', Ny_range)
@pytest.mark.parametrize('Nz', Nz_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_3d_box])
@pytest.mark.parametrize('dtype', [np.complex128, np.float64])
def test_implicit_transpose_3d_tensor(Nx, Ny, Nz, dealias, basis, dtype):
    c, d, b, x, y, z = basis(Nx, Ny, Nz, dealias, dtype)
    u = field.Field(dist=d, bases=b, tensorsig=(c,), dtype=dtype)
    u['g'][0] = (np.sin(2*x)+np.sin(x))*np.cos(y)*np.sin(z)
    u['g'][1] = (np.cos(2*x)+np.cos(x))*np.sin(y)*np.sin(z)
    u['g'][2] = np.sin(x)*np.cos(y)*np.cos(z)
    T = operators.Gradient(u, c).evaluate()
    Ttg = np.transpose(np.copy(T['g']),(1,0,2,3,4))
    Tt = field.Field(dist=d, bases=b, tensorsig=(c,c,), dtype=dtype)
    trans = lambda A: operators.TransposeComponents(A)
    problem = problems.LBVP([Tt])
    problem.add_equation((trans(Tt), T))
    # Solver
    solver = solvers.LinearBoundaryValueSolver(problem)
    solver.solve()
    assert np.allclose(Tt['g'], Ttg)

@pytest.mark.parametrize('Nx', Nx_range)
@pytest.mark.parametrize('Nz', Nz_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_2d_box])
@pytest.mark.parametrize('dtype', [np.complex128, np.float64])
def test_implicit_transpose_2d_tensor(Nx, Nz, dealias, basis, dtype):
    c, d, b, x, z = basis(Nx, Nz, dealias, dtype)
    u = field.Field(dist=d, bases=b, tensorsig=(c,), dtype=dtype)
    u['g'][0] = (np.sin(2*x)+np.sin(x))*np.sin(z)
    u['g'][1] = np.sin(x)*np.cos(z)
    T = operators.Gradient(u, c).evaluate()
    T.name = 'T'
    Ttg = np.transpose(np.copy(T['g']),(1,0,2,3))
    Tt = field.Field(name='Tt', dist=d, bases=b, tensorsig=(c,c,), dtype=dtype)
    trans = lambda A: operators.TransposeComponents(A)
    problem = problems.LBVP([Tt])
    problem.add_equation((trans(Tt), T))
    # Solver
    solver = solvers.LinearBoundaryValueSolver(problem)
    solver.solve()
    assert np.allclose(Tt['g'], Ttg)
