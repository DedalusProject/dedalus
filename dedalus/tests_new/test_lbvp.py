"""
Test 1D LBVP.
"""
import pytest
import numpy as np
import functools
from dedalus.core import coords, distributor, basis, field, operators, problems, solvers, timesteppers, arithmetic
from dedalus.tools.cache import CachedFunction

@pytest.mark.parametrize('dtype', [np.complex128, np.float64])
@pytest.mark.parametrize('Nx', [32])
def test_heat_1d_periodic(Nx, dtype):
    # Bases
    c = coords.Coordinate('x')
    d = distributor.Distributor((c,))
    if dtype == np.complex128:
        xb = basis.ComplexFourier(c, size=Nx, bounds=(0, 2*np.pi))
    elif dtype == np.float64:
        xb = basis.RealFourier(c, size=Nx, bounds=(0, 2*np.pi))
    x = xb.local_grid(1)
    # Fields
    u = field.Field(name='u', dist=d, bases=(xb,), dtype=dtype)
    F = field.Field(name='F', dist=d, bases=(xb,), dtype=dtype)
    F['g'] = -np.sin(x)
    # Problem
    dx = lambda A: operators.Differentiate(A, c)
    problem = problems.LBVP([u])
    problem.add_equation((dx(dx(u)), F), condition='nx != 0')
    problem.add_equation((u, 0), condition='nx == 0')
    # Solver
    solver = solvers.LinearBoundaryValueSolver(problem)
    solver.solve()
    # Check solution
    u_true = np.sin(x)
    assert np.allclose(u['g'], u_true)

radius_ball = 1
@CachedFunction
def build_ball(Nphi, Ntheta, Nr, dealias, dtype):
    c = coords.SphericalCoordinates('phi', 'theta', 'r')
    d = distributor.Distributor((c,))
    b = basis.BallBasis(c, (Nphi, Ntheta, Nr), radius=radius_ball, dealias=(dealias, dealias, dealias), dtype=dtype)
    phi, theta, r = b.local_grids()
    x, y, z = c.cartesian(phi, theta, r)
    return c, d, b, phi, theta, r, x, y, z


@pytest.mark.parametrize('dtype', [np.complex128, np.float64])
@pytest.mark.parametrize('Nmax', [15])
@pytest.mark.parametrize('Lmax', [3])
@pytest.mark.skip(reason="spherical tau not implemented")
def test_heat_ball(Nmax, Lmax, dtype):
    # Bases
    dealias = 1
    c, d, b, phi, theta, r, x, y, z = build_ball(2*(Lmax+1), Lmax+1, Nmax+1, dealias=dealias, dtype=dtype)
    # Fields
    u = field.Field(name='u', dist=d, bases=(b,), dtype=dtype)
    τu = field.Field(name='u', dist=d, bases=(b.S2_basis(),), dtype=dtype)
    # Problem
    Lap = lambda A: operators.Laplacian(A, c)
    LiftTau = lambda A: operators.LiftTau(A, b, -1)
    problem = problems.LBVP([u, τu])
    problem.add_equation((Lap(u) + LiftTau(τu), 3))
    problem.add_equation((u(r=1), 0))
    # Solver
    solver = solvers.LinearBoundaryValueSolver(problem)
    solver.solve()
    # Check solution
    u_true = np.sin(x)
    assert np.allclose(u['g'], u_true)


@pytest.mark.parametrize('dtype', [np.complex128, np.float64])
@pytest.mark.parametrize('Nmax', [15])
@pytest.mark.parametrize('Lmax', [3])
def test_ncc_ball(Nmax, Lmax, dtype):
    # Bases
    dealias = 1
    c, d, b, phi, theta, r, x, y, z = build_ball(2*(Lmax+1), Lmax+1, Nmax+1, dealias=dealias, dtype=dtype)
    # Fields
    u = field.Field(name='u', dist=d, bases=(b,), dtype=dtype)
    ncc = field.Field(name='ncc', dist=d, bases=(b.radial_basis,), dtype=dtype)
    ncc['g'] = 1+r**2
    F = field.Field(name='F', dist=d, bases=(b,), dtype=dtype)
    F['g'] = 1
    # Problem
    problem = problems.LBVP([u])
    problem.add_equation((ncc*u, F))
    # Solver
    solver = solvers.LinearBoundaryValueSolver(problem)
    solver.solve()
    # Check solution
    u_true = 1/ncc['g']
    assert np.allclose(u['g'], u_true)
