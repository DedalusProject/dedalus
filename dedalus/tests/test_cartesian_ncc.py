from dedalus.tools.general import unify
import pytest
import numpy as np
from dedalus.core import coords, distributor, basis, field, operators, arithmetic, problems, solvers
from dedalus.tools.cache import CachedMethod
from mpi4py import MPI

comm = MPI.COMM_WORLD


@pytest.mark.parametrize('N', [16])
@pytest.mark.parametrize('a0', [-1/2])
@pytest.mark.parametrize('b0', [-1/2])
@pytest.mark.parametrize('k_ncc', [0, 1, 2, 3])
@pytest.mark.parametrize('k_arg', [0, 1, 2, 3])
@pytest.mark.parametrize('dealias', [3/2, 2])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_jacobi_ncc_eval(N, a0, b0, k_ncc, k_arg, dealias, dtype):
    """
    This tests for aliasing errors when evaluating the product
        f(x) * g(x)
    as both an NCC operator and with the pseudospectral method.

    With 3/2 dealiasing, the product will generally contain aliasing
    errors in the last 2*max(k_ncc, k_arg) modes. We can eliminate these
    by zeroing the corresponding number of modes of f(x) and/or g(x).
    """
    c = coords.Coordinate('x')
    d = distributor.Distributor((c,))
    b = basis.Jacobi(c, size=N, a=a0, b=b0, bounds=(0, 1), dealias=dealias)
    b_ncc = b._new_a_b(a0+k_ncc, b0+k_ncc)
    b_arg = b._new_a_b(a0+k_arg, b0+k_arg)
    f = field.Field(dist=d, bases=(b_ncc,), dtype=dtype)
    g = field.Field(dist=d, bases=(b_arg,), dtype=dtype)
    f['g'] = np.random.randn(*f['g'].shape)
    g['g'] = np.random.randn(*g['g'].shape)
    kmax = max(k_ncc, k_arg)
    if kmax > 0 and dealias < 2:
        f['c'][-kmax:] = 0
        g['c'][-kmax:] = 0
    vars = [g]
    w0 = f * g
    w1 = w0.reinitialize(ncc=True, ncc_vars=vars)
    problem = problems.LBVP(vars)
    problem.add_equation((w1, 0))
    solver = solvers.LinearBoundaryValueSolver(problem)
    w1.store_ncc_matrices(vars, solver.subproblems)
    w0 = w0.evaluate()
    w1 = w1.evaluate_as_ncc()
    assert np.allclose(w0['c'], w1['c'])


@pytest.mark.parametrize('N', [16])
@pytest.mark.parametrize('a0', [-1/2])
@pytest.mark.parametrize('b0', [-1/2])
@pytest.mark.parametrize('k_ncc', [0, 1, 2, 3])
@pytest.mark.parametrize('k_arg', [0, 1, 2, 3])
@pytest.mark.parametrize('dealias', [3/2, 2])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_jacobi_ncc_solve(N, a0, b0, k_ncc, k_arg, dealias, dtype):
    """
    This tests for aliasing errors when solving the equation
        f(x)*u(x) = f(x)*g(x).

    With 3/2 dealiasing, the RHS product will generally contain aliasing
    errors in the last 2*max(k_ncc, k_arg) modes. We can eliminate these
    by zeroing the corresponding number of modes of f(x) and/or g(x).
    """
    c = coords.Coordinate('x')
    d = distributor.Distributor((c,))
    b = basis.Jacobi(c, size=N, a=a0, b=b0, bounds=(0, 1), dealias=dealias)
    b_ncc = b._new_a_b(a0+k_ncc, b0+k_ncc)
    b_arg = b._new_a_b(a0+k_arg, b0+k_arg)
    f = field.Field(dist=d, bases=(b_ncc,), dtype=dtype)
    g = field.Field(dist=d, bases=(b_arg,), dtype=dtype)
    u = field.Field(dist=d, bases=(b_arg,), dtype=dtype)
    f['g'] = np.random.randn(*f['g'].shape)
    g['g'] = np.random.randn(*g['g'].shape)
    kmax = max(k_ncc, k_arg)
    if kmax > 0 and dealias < 2:
        f['c'][-kmax:] = 0
        g['c'][-kmax:] = 0
    problem = problems.LBVP([u])
    problem.add_equation((f*u, f*g))
    solver = solvers.LinearBoundaryValueSolver(problem)
    solver.solve()
    assert np.allclose(u['c'], g['c'])


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

Nx_range = [8]
Ny_range = [8]
Nz_range = [8]
dealias_range = [1, 3/2]

@pytest.mark.parametrize('Nx', Nx_range)
@pytest.mark.parametrize('Ny', Ny_range)
@pytest.mark.parametrize('Nz', Nz_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_3d_box])
@pytest.mark.parametrize('dtype', [np.complex128, np.float64])
def test_3d_scalar_ncc_scalar(Nx, Ny, Nz, dealias, basis, dtype):
    c, d, b, x, y, z = basis(Nx, Ny, Nz, dealias, dtype)
    u = field.Field(dist=d, bases=b, dtype=dtype)
    ncc = field.Field(name='ncc', dist=d, bases=(b[-1],), dtype=dtype)
    ncc['g'] = 1/(1+z**2)
    problem = problems.LBVP([u])
    problem.add_equation((ncc*u, 1))
    # Solver
    solver = solvers.LinearBoundaryValueSolver(problem)
    solver.solve()
    assert np.allclose(u['g'], 1/ncc['g'])

@pytest.mark.parametrize('Nx', Nx_range)
@pytest.mark.parametrize('Ny', Ny_range)
@pytest.mark.parametrize('Nz', Nz_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_3d_box])
@pytest.mark.parametrize('dtype', [np.complex128, np.float64])
def test_3d_scalar_ncc_scalar_separable(Nx, Ny, Nz, dealias, basis, dtype):
    c, d, b, x, y, z = basis(Nx, Ny, Nz, dealias, dtype)
    u = field.Field(dist=d, bases=(b[0], b[1]), dtype=dtype)
    v = field.Field(dist=d, bases=b, dtype=dtype)
    ncc = field.Field(name='ncc', dist=d, bases=(b[-1],), dtype=dtype)
    ncc['g'] = 1/(1+z**2)
    problem = problems.LBVP([v, u])
    problem.add_equation((v, 0)) # Hack by adding v since problem must have full dimension
    problem.add_equation(((ncc*u)(z=1), ncc(z=1)))
    # Solver
    solver = solvers.LinearBoundaryValueSolver(problem)
    solver.solve()
    assert np.allclose(u['g'], 1)

@pytest.mark.parametrize('Nx', Nx_range)
@pytest.mark.parametrize('Nz', Nz_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_2d_box])
@pytest.mark.parametrize('dtype', [np.complex128, np.float64])
def test_implicit_transpose_2d_tensor(Nx, Nz, dealias, basis, dtype):
    c, d, b, x, z = basis(Nx, Nz, dealias, dtype)
    u = field.Field(dist=d, bases=b, dtype=dtype)
    ncc = field.Field(name='ncc', dist=d, bases=(b[-1],), dtype=dtype)
    ncc['g'] = 1/(1+z**2)
    problem = problems.LBVP([u])
    problem.add_equation((ncc*u, 1))
    # Solver
    solver = solvers.LinearBoundaryValueSolver(problem)
    solver.solve()
    assert np.allclose(u['g'], 1/ncc['g'])
