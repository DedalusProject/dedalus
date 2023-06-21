"""Test simple IVPs and CFL operators with various timesteppers."""

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


@CachedFunction
def build_shell(Nphi, Ntheta, Nr, radii, dealias, dtype):
    c = d3.SphericalCoordinates('phi', 'theta', 'r')
    d = d3.Distributor(c, dtype=dtype)
    b = d3.ShellBasis(c, (Nphi, Ntheta, Nr), radii=radii, dealias=dealias, dtype=dtype)
    phi, theta, r = d.local_grids(b)
    x, y, z = c.cartesian(phi, theta, r)
    return c, d, b, phi, theta, r, x, y, z


@CachedFunction
def build_disk(Nphi, Nr, radius, dealias, dtype):
    c = d3.PolarCoordinates('phi', 'r')
    d = d3.Distributor(c, dtype=dtype)
    b = d3.DiskBasis(c, (Nphi, Nr), radius=radius, dealias=dealias, dtype=dtype)
    phi, r = d.local_grids(b)
    x, y = c.cartesian(phi, r)
    return c, d, b, phi, r, x, y


@CachedFunction
def build_annulus(Nphi, Nr, radii, dealias, dtype):
    c = d3.PolarCoordinates('phi', 'r')
    d = d3.Distributor(c, dtype=dtype)
    b = d3.AnnulusBasis(c, (Nphi, Nr), radii=radii, dealias=dealias, dtype=dtype)
    phi, r = d.local_grids(b)
    x, y = c.cartesian(phi, r)
    return c, d, b, phi, r, x, y


@CachedFunction
def build_S2(Nphi, Ntheta, radius, dealias, dtype):
    c = d3.S2Coordinates('phi', 'theta')
    d = d3.Distributor(c, dtype=dtype)
    b = d3.SphereBasis(c, (Nphi, Ntheta), radius=radius, dealias=dealias, dtype=dtype)
    phi, theta = d.local_grids(b)
    return c, d, b, phi, theta


@pytest.mark.parametrize('basis', [d3.ComplexFourier])
@pytest.mark.parametrize('N', [16])
@pytest.mark.parametrize('dealias', [1])
@pytest.mark.parametrize('dtype', [np.complex128])
@pytest.mark.parametrize('timestepper', timesteppers.schemes.values())
def test_heat_1d_periodic(basis, N,  dtype, dealias, timestepper):
    """Test 1D heat equation with periodic boundary conditions for correctness."""
    # Bases
    c = d3.Coordinate('x')
    d = d3.Distributor(c, dtype=dtype)
    b = basis(c, size=N, bounds=(0, 2*np.pi), dealias=dealias)
    x = b.local_grid(1)
    # Fields
    u = d.Field(bases=b)
    F = d.Field(bases=b)
    F['g'] = -np.sin(x)
    # Problem
    dx = lambda A: d3.Differentiate(A, c)
    problem = d3.IVP([u], namespace=locals())
    problem.add_equation("-dt(u) + dx(dx(u)) = F")
    # Solver
    solver = problem.build_solver(timestepper)
    dt = 1e-5
    iter = 10
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
    solver = problem.build_solver(timesteppers.SBDF1)
    dt = 1e-5
    # Just check that it runs
    iter = 5
    for i in range(iter):
        solver.step(dt)
    assert True  # TODO: make quantitative test


@pytest.mark.parametrize('Nx', [32])
@pytest.mark.parametrize('Nz', [16])
@pytest.mark.parametrize('dealias', [1, 3/2])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('timestepper', [timesteppers.SBDF1])
@pytest.mark.parametrize('safety', [0.2, 0.4])
@pytest.mark.parametrize('z_velocity_mag', [0, 2])
def test_full_cfl_fourier_chebyshev(Nx, Nz, dealias, dtype, timestepper, safety, z_velocity_mag):
    """Test CFL tool end-to-end with Fourier-Chebyshev domain."""
    # Bases
    Lx = 2
    Lz = 1
    c = d3.CartesianCoordinates('x', 'z')
    d = d3.Distributor(c, dtype=dtype)
    xb = d3.Fourier(c.coords[0], size=Nx, bounds=(0, Lx), dealias=dealias, dtype=dtype)
    zb = d3.Chebyshev(c.coords[1], size=Nz, bounds=(0, Lz), dealias=dealias)
    x, z = d.local_grids(xb, zb)
    # IVP
    u = d.VectorField(c, bases=(xb, zb))
    problem = d3.IVP([u], namespace=locals())
    problem.add_equation("dt(u) = 0")
    solver = problem.build_solver(timestepper)
    # CFL
    cfl = d3.CFL(solver, initial_dt=1, safety=safety, cadence=1)
    cfl.add_velocity(u)
    # Test Fourier CFL
    u.fill_random(layout='g')
    for i in range(2):
        solver.step(1)
    dt_cfl = cfl.compute_timestep()
    cfl_op = d3.AdvectiveCFL(u, c)
    cfl_freq = np.abs(u['g'][0] / cfl_op.cfl_spacing(u)[0])
    cfl_freq += np.abs(u['g'][1] / cfl_op.cfl_spacing(u)[1])
    cfl_freq = np.max(cfl_freq)
    dt_target = safety / cfl_freq
    assert np.allclose(dt_cfl, dt_target)


@pytest.mark.parametrize('N', [32])
@pytest.mark.parametrize('L', [1.44])
@pytest.mark.parametrize('dealias', [1, 3/2])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_cfl_fourier(N, L, dealias, dtype):
    """Test CFL operator with Fourier basis."""
    c = d3.CartesianCoordinates('x')
    d = d3.Distributor(c, dtype=dtype)
    b = d3.Fourier(c.coords[0], size=N, bounds=(0, L), dealias=dealias, dtype=dtype)
    x = b.local_grid(1)
    u = d.VectorField(c, bases=b)
    u.fill_random(layout='g')
    cfl = d3.AdvectiveCFL(u, c)
    cfl_freq = cfl.evaluate()['g']
    target_freq = np.abs(u['g']) / cfl.cfl_spacing(u)[0]
    assert np.allclose(cfl_freq, target_freq)


@pytest.mark.parametrize('N', [32])
@pytest.mark.parametrize('L', [1.44])
@pytest.mark.parametrize('dealias', [1, 3/2])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_cfl_chebyshev(N, L, dealias, dtype):
    """Test CFL operator with Chebyshev basis."""
    c = d3.CartesianCoordinates('x')
    d = d3.Distributor(c, dtype=dtype)
    b = d3.Chebyshev(c.coords[0], size=N, bounds=(0, L), dealias=dealias)
    x = b.local_grid(1)
    u = d.VectorField(c, bases=b)
    u.fill_random(layout='g')
    cfl = d3.AdvectiveCFL(u, c)
    cfl_freq = cfl.evaluate()['g']
    target_freq = np.abs(u['g']) / cfl.cfl_spacing(u)[0]
    assert np.allclose(cfl_freq, target_freq)


@pytest.mark.parametrize('Nx', [32])
@pytest.mark.parametrize('Nz', [16])
@pytest.mark.parametrize('dealias', [1, 3/2])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('z_velocity_mag', [0, 2])
def test_cfl_fourier_chebyshev(Nx, Nz, dealias, dtype, z_velocity_mag):
    """Test CFL operator with Fourier-Chebyshev domain."""
    c = d3.CartesianCoordinates('x', 'z')
    d = d3.Distributor(c, dtype=dtype)
    xb = d3.Fourier(c.coords[0], size=Nx, bounds=(0, 2), dealias=dealias, dtype=dtype)
    zb = d3.ChebyshevT(c.coords[1], size=Nz, bounds=(0, 1), dealias=dealias)
    x, z = d.local_grids(xb, zb)
    u = d.VectorField(c, bases=(xb, zb))
    u.fill_random(layout='g')
    cfl = d3.AdvectiveCFL(u, c)
    cfl_freq = cfl.evaluate()['g']
    target_freq = np.abs(u['g'][0]) / cfl.cfl_spacing(u)[0]
    target_freq += np.abs(u['g'][1]) / cfl.cfl_spacing(u)[1]
    assert np.allclose(cfl_freq, target_freq)


@pytest.mark.parametrize('N', [16])
@pytest.mark.parametrize('dealias', [1, 3/2])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_cfl_sphere(N, dealias, dtype):
    """Test CFL operator with sphere basis."""
    c, d, b, phi, theta = build_S2(2*N, N, dealias=dealias, dtype=dtype, radius=2.5)
    u = d.VectorField(c, bases=b)
    u.fill_random(layout='g')
    cfl = d3.AdvectiveCFL(u, c)
    cfl_freq = cfl.evaluate()['g']
    target_freq = np.sqrt(u['g'][0]**2 + u['g'][1]**2) / cfl.cfl_spacing()[0]
    assert np.allclose(cfl_freq, target_freq)


@pytest.mark.parametrize('N', [16])
@pytest.mark.parametrize('dealias', [1, 3/2])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_cfl_disk(N, dealias, dtype):
    """Test CFL operator with disk basis."""
    c, d, b, phi, r, x, y = build_disk(N, N, radius=2, dealias=dealias, dtype=dtype)
    u = d.VectorField(c, bases=b)
    u.fill_random(layout='g')
    cfl = d3.AdvectiveCFL(u, c)
    cfl_freq = cfl.evaluate()['g']
    target_freq = np.abs(u['g'][0]) / cfl.cfl_spacing()[0]
    target_freq += np.abs(u['g'][1]) / cfl.cfl_spacing()[1]
    assert np.allclose(cfl_freq, target_freq)


@pytest.mark.parametrize('N', [16])
@pytest.mark.parametrize('dealias', [1, 3/2])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_cfl_annulus(N, dealias, dtype):
    """Test CFL operator with annulus basis."""
    c, d, b, phi, r, x, y = build_annulus(N, N, radii=(0.6, 2), dealias=dealias, dtype=dtype)
    u = d.VectorField(c, bases=b)
    u.fill_random(layout='g')
    cfl = d3.AdvectiveCFL(u, c)
    cfl_freq = cfl.evaluate()['g']
    target_freq = np.abs(u['g'][0]) / cfl.cfl_spacing()[0]
    target_freq += np.abs(u['g'][1]) / cfl.cfl_spacing()[1]
    assert np.allclose(cfl_freq, target_freq)


@pytest.mark.parametrize('N', [16])
@pytest.mark.parametrize('dealias', [1, 3/2])
@pytest.mark.parametrize('dtype', [np.complex128, np.float64])
def test_cfl_ball(N, dealias, dtype):
    """Test CFL operator with ball basis."""
    c, d, b, phi, theta, r, x, y, z = build_ball(2*N, N, N, radius=2.5, dealias=dealias, dtype=dtype)
    u = d.VectorField(c, bases=b)
    u.fill_random(layout='g')
    cfl = d3.AdvectiveCFL(u, c)
    cfl_freq = cfl.evaluate()['g']
    target_freq = np.sqrt(u['g'][0]**2 + u['g'][1]**2) / cfl.cfl_spacing()[0]
    target_freq += np.abs(u['g'][2]) / cfl.cfl_spacing()[1]
    assert np.allclose(cfl_freq, target_freq)


@pytest.mark.parametrize('N', [16])
@pytest.mark.parametrize('dealias', [1, 3/2])
@pytest.mark.parametrize('dtype', [np.complex128, np.float64])
def test_cfl_shell(N, dealias, dtype):
    """Test CFL operator with shell basis."""
    c, d, b, phi, theta, r, x, y, z = build_shell(2*N, N, N, radii=(0.4, 2.5), dealias=dealias, dtype=dtype)
    u = d.VectorField(c, bases=b)
    u.fill_random(layout='g')
    cfl = d3.AdvectiveCFL(u, c)
    cfl_freq = cfl.evaluate()['g']
    cfl = d3.AdvectiveCFL(u, c)
    cfl_freq = cfl.evaluate()['g']
    target_freq = np.sqrt(u['g'][0]**2 + u['g'][1]**2) / cfl.cfl_spacing()[0]
    target_freq += np.abs(u['g'][2]) / cfl.cfl_spacing()[1]
    assert np.allclose(cfl_freq, target_freq)

