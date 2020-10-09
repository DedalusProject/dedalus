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

Nx_range = [16,2]
Ny_range = [8]
Nz_range = [8,2]
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
