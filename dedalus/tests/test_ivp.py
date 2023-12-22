"""Test simple IVPs with various timesteppers."""

import pytest
import numpy as np
import dedalus.public as d3
from dedalus.core import timesteppers
from dedalus.tools.cache import CachedFunction


@CachedFunction
def build_ball(Nphi, Ntheta, Nr, radius, dealias, dtype):
    c = d3.SphericalCoordinates('phi', 'theta', 'r')
    d = d3.Distributor(c, dtype=dtype)
    b = d3.BallBasis(c, (Nphi, Ntheta, Nr), radius=radius, dealias=dealias, dtype=dtype)
    phi, theta, r = d.local_grids(b)
    x, y, z = c.cartesian(phi, theta, r)
    return c, d, b, phi, theta, r, x, y, z


@pytest.mark.parametrize('basis', [d3.ComplexFourier])
@pytest.mark.parametrize('N', [8])
@pytest.mark.parametrize('dealias', [1])
@pytest.mark.parametrize('dtype', [np.complex128])
@pytest.mark.parametrize('timestepper', list(timesteppers.schemes.keys()))
def test_heat_periodic(basis, N,  dtype, dealias, timestepper):
    """Test 1D heat equation with periodic boundary conditions for correctness."""
    # Bases
    c = d3.Coordinate('x')
    d = d3.Distributor(c, dtype=dtype)
    b = basis(c, size=N, bounds=(0, 2*np.pi), dealias=dealias)
    x = d.local_grid(b, scale=1)
    # Fields
    u = d.Field(bases=b)
    F = d.Field(bases=b)
    F['g'] = np.sin(x)
    # Problem
    dx = lambda A: d3.Differentiate(A, c)
    problem = d3.IVP([u], namespace=locals())
    problem.add_equation("dt(u) - dx(dx(u)) = F")
    # Solver
    solver = problem.build_solver(timestepper)
    dt = 1e-5
    iter = 20
    for i in range(iter):
        solver.step(dt)
    # Check solution
    amp = 1 - np.exp(-solver.sim_time)
    u_true = amp * np.sin(x)
    assert np.allclose(u['g'], u_true)


@pytest.mark.parametrize('N', [8])
@pytest.mark.parametrize('radius', [1])
@pytest.mark.parametrize('dealias', [1])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_vector_diffusion_ball(N, radius, dealias, dtype):
    """Test that vector diffusion in the ball builds and runs."""
    # Bases
    c, d, b, phi, theta, r, x, y, z = build_ball(N, N, N, radius, dealias, dtype)
    # Fields
    A = d.VectorField(c, bases=b)
    phi = d.Field(bases=b)
    tau_A = d.VectorField(c, bases=b.surface)
    Lift = lambda A: d3.Lift(A, b, -1)
    # Problem
    problem = d3.IVP([A, phi, tau_A], namespace=locals())
    problem.add_equation("div(A) = 0")
    problem.add_equation("dt(A) - grad(phi) - lap(A) + Lift(tau_A) = 0")
    problem.add_equation("angular(A(r=1), index=0) = 0")
    problem.add_equation("phi(r=1) = 0")
    # Solver
    solver = problem.build_solver("RK111")
    dt = 1e-5
    # Just check that it runs
    iter = 5
    for i in range(iter):
        solver.step(dt)
    assert True  # TODO: make quantitative test

