"""
Test 1D IVP with various timesteppers.
"""

import pytest
import numpy as np
import functools
from dedalus.core import coords, distributor, basis, field, operators, problems, solvers, timesteppers, arithmetic
from dedalus.tools.cache import CachedFunction
from dedalus.extras import flow_tools

@CachedFunction
def build_ball(Nphi, Ntheta, Nr, radius_ball, dealias, dtype, grid_scale=1):
    c = coords.SphericalCoordinates('phi', 'theta', 'r')
    d = distributor.Distributor((c,))
    dealias_tuple = (dealias, dealias, dealias)
    b = basis.BallBasis(c, (Nphi, Ntheta, Nr), radius=radius_ball, dealias=dealias_tuple, dtype=dtype)
    grid_scale_tuple = (grid_scale, grid_scale, grid_scale)
    phi, theta, r = b.local_grids(grid_scale_tuple)
    x, y, z = c.cartesian(phi, theta, r)
    return c, d, b, phi, theta, r, x, y, z

@CachedFunction
def build_shell(Nphi, Ntheta, Nr, radii_shell, dealias, dtype, grid_scale=1):
    c = coords.SphericalCoordinates('phi', 'theta', 'r')
    d = distributor.Distributor((c,))
    dealias_tuple = (dealias, dealias, dealias)
    b = basis.SphericalShellBasis(c, (Nphi, Ntheta, Nr), radii=radii_shell, dealias=dealias_tuple, dtype=dtype)
    grid_scale_tuple = (grid_scale, grid_scale, grid_scale)
    phi, theta, r = b.local_grids(grid_scale_tuple)
    x, y, z = c.cartesian(phi, theta, r)
    return c, d, b, phi, theta, r, x, y, z

@CachedFunction
def build_disk(Nphi, Nr, radius, dealias, dtype=np.float64, grid_scale=1):
    c = coords.PolarCoordinates('phi', 'r')
    d = distributor.Distributor((c,))
    dealias_tuple = (dealias, dealias)
    b = basis.DiskBasis(c, (Nphi, Nr), radius=radius, dealias=dealias_tuple, dtype=dtype)
    grid_scale_tuple = (grid_scale, grid_scale)
    phi, r = b.local_grids(grid_scale_tuple)
    x, y = c.cartesian(phi, r)
    return c, d, b, phi, r, x, y

@CachedFunction
def build_annulus(Nphi, Nr, radii_annulus, dealias, dtype=np.float64, grid_scale=1):
    c = coords.PolarCoordinates('phi', 'r')
    d = distributor.Distributor((c,))
    dealias_tuple = (dealias, dealias)
    b = basis.AnnulusBasis(c, (Nphi, Nr), radii=radii_annulus, dealias=dealias_tuple, dtype=dtype)
    grid_scale_tuple = (grid_scale, grid_scale)
    phi, r = b.local_grids(grid_scale_tuple)
    x, y = c.cartesian(phi, r)
    return c, d, b, phi, r, x, y

@CachedFunction
def build_S2(Nphi, Ntheta, dealias, dtype=np.complex128, grid_scale=1):
    c = coords.S2Coordinates('phi', 'theta')
    d = distributor.Distributor((c,))
    dealias_tuple = (dealias, dealias)
    sb = basis.SpinWeightedSphericalHarmonics(c, (Nphi, Ntheta), radius=1, dealias=dealias_tuple, dtype=dtype)
    grid_scale_tuple = (grid_scale, grid_scale)
    phi, theta = sb.local_grids(grid_scale_tuple)
    return c, d, sb, phi, theta

@pytest.mark.parametrize('dtype', [np.complex128])
@pytest.mark.parametrize('timestepper', timesteppers.schemes.values())
@pytest.mark.parametrize('Nx', [32])
@pytest.mark.parametrize('x_basis_class', [basis.ComplexFourier])
def test_heat_1d_periodic(x_basis_class, Nx, timestepper, dtype):
    # Bases
    c = coords.Coordinate('x')
    d = distributor.Distributor((c,))
    xb = basis.ComplexFourier(c, size=Nx, bounds=(0, 2*np.pi))
    x = xb.local_grid(1)
    # Fields
    u = field.Field(name='u', dist=d, bases=(xb,), dtype=dtype)
    F = field.Field(name='F', dist=d, bases=(xb,), dtype=dtype)
    F['g'] = -np.sin(x)
    # Problem
    dx = lambda A: operators.Differentiate(A, c)
    dt = operators.TimeDerivative
    problem = problems.IVP([u])
    problem.add_equation((-dt(u) + dx(dx(u)), F))
    # Solver
    solver = solvers.InitialValueSolver(problem, timestepper)
    dt = 1e-5
    iter = 10
    for i in range(iter):
        solver.step(dt)
    # Check solution
    amp = 1 - np.exp(-solver.sim_time)
    u_true = amp * np.sin(x)
    assert np.allclose(u['g'], u_true)

@pytest.mark.parametrize('timestepper', [timesteppers.SBDF1])
@pytest.mark.parametrize('Nx', [32])
@pytest.mark.parametrize('Nz', [16])
@pytest.mark.parametrize('dtype,x_basis_class', [(np.complex128, basis.ComplexFourier), (np.float64, basis.RealFourier)])
@pytest.mark.parametrize('safety', [0.2, 0.4])
@pytest.mark.parametrize('z_velocity_mag', [0, 2])
@pytest.mark.parametrize('dealias', [1, 3/2])
def test_flow_tools_cfl(x_basis_class, Nx, Nz, timestepper, dtype, safety, z_velocity_mag, dealias):
    # Bases
    Lx = 2
    Lz = 1
    c = coords.CartesianCoordinates('x', 'z')
    d = distributor.Distributor((c,))
    xb = x_basis_class(c.coords[0], size=Nx, bounds=(0, Lx), dealias=dealias)
    x = xb.local_grid(1)
    zb = basis.ChebyshevT(c.coords[1], size=Nz, bounds=(0, Lz), dealias=dealias)
    z = zb.local_grid(1)
    b = (xb, zb)
    # Fields
    u = field.Field(name='u', dist=d, tensorsig=(c,), bases=b, dtype=dtype)
    # Problem
    ddt = operators.TimeDerivative
    problem = problems.IVP([u])
    problem.add_equation((ddt(u), 0))
    # Solver
    solver = solvers.InitialValueSolver(problem, timestepper)
    # cfl initialization
    dt = 1
    cfl = flow_tools.CFL(solver, dt, safety=safety, cadence=1)
    cfl.add_velocity(u)

    # Test Fourier CFL
    fourier_velocity = lambda x: np.sin(4*np.pi*x/Lx)
    chebyshev_velocity = lambda z: -z_velocity_mag*z
    u['g'][0] = fourier_velocity(x)
    u['g'][1] = chebyshev_velocity(z)
    solver.step(dt)
    solver.step(dt) #need two timesteps to get past stored_dt per compute_timestep logic
    dt = cfl.compute_dt()

    op = operators.AdvectiveCFL(u, c)
    cfl_freq = np.abs(u['g'][0] / op.cfl_spacing(u)[0] )
    cfl_freq += np.abs(u['g'][1] / op.cfl_spacing(u)[1] )
    cfl_freq = np.max(cfl_freq)
    dt_comparison = safety*(cfl_freq)**(-1)
    assert np.allclose(dt, dt_comparison)

@pytest.mark.parametrize('timestepper', [timesteppers.SBDF1])
@pytest.mark.parametrize('Nx', [32])
@pytest.mark.parametrize('dtype,x_basis_class', [(np.complex128, basis.ComplexFourier), (np.float64, basis.RealFourier)])
@pytest.mark.parametrize('dealias', [1, 3/2])
def test_fourier_AdvectiveCFL(x_basis_class, Nx, timestepper, dtype, dealias):
    Lx = 1
    # Bases
    c = coords.CartesianCoordinates('x')
    d = distributor.Distributor((c,))
    xb = x_basis_class(c.coords[0], size=Nx, bounds=(0, Lx), dealias=dealias)
    x = xb.local_grid(1)
    # Fields
    u = field.Field(name='u', dist=d, tensorsig=(c,), bases=(xb,), dtype=dtype)
    velocity = lambda x: np.sin(2*np.pi*x/Lx)
    u['g'][0] = velocity(x)
    # AdvectiveCFL initialization
    cfl = operators.AdvectiveCFL(u, c)
    cfl_freq = cfl.evaluate()['g']
    comparison_freq = np.abs(u['g']) / cfl.cfl_spacing(u)[0]
    assert np.allclose(cfl_freq, comparison_freq)

@pytest.mark.parametrize('timestepper', [timesteppers.SBDF1])
@pytest.mark.parametrize('Nx', [32])
@pytest.mark.parametrize('dtype', [np.complex128, np.float64])
@pytest.mark.parametrize('dealias', [1, 3/2])
def test_chebyshev_AdvectiveCFL(Nx, timestepper, dtype, dealias):
    # Bases
    Lx = 1
    c = coords.CartesianCoordinates('x')
    d = distributor.Distributor((c,))
    xb = basis.ChebyshevT(c.coords[0], size=Nx, bounds=(0, Lx), dealias=dealias)
    x = xb.local_grid(1)
    # Fields
    u = field.Field(name='u', dist=d, tensorsig=(c,), bases=(xb,), dtype=dtype)
    velocity = lambda x: x
    u['g'][0] = velocity(x)
    # AdvectiveCFL initialization
    cfl = operators.AdvectiveCFL(u, c)
    cfl_freq = cfl.evaluate()['g']
    comparison_freq = np.abs(u['g']) / cfl.cfl_spacing(u)[0]
    assert np.allclose(cfl_freq, comparison_freq)

@pytest.mark.parametrize('timestepper', [timesteppers.SBDF1])
@pytest.mark.parametrize('Nx', [32])
@pytest.mark.parametrize('Nz', [16])
@pytest.mark.parametrize('dtype,x_basis_class', [(np.complex128, basis.ComplexFourier), (np.float64, basis.RealFourier)])
@pytest.mark.parametrize('z_velocity_mag', [0, 2])
@pytest.mark.parametrize('dealias', [1, 3/2])
def test_box_AdvectiveCFL(x_basis_class, Nx, Nz, timestepper, dtype, z_velocity_mag, dealias):
    # Bases
    Lx = 2
    Lz = 1
    c = coords.CartesianCoordinates('x', 'z')
    d = distributor.Distributor((c,))
    xb = x_basis_class(c.coords[0], size=Nx, bounds=(0, Lx), dealias=dealias)
    x = xb.local_grid(1)
    zb = basis.ChebyshevT(c.coords[1], size=Nz, bounds=(0, Lz), dealias=dealias)
    z = zb.local_grid(1)
    b = (xb, zb)
    # Fields
    u = field.Field(name='u', dist=d, tensorsig=(c,), bases=b, dtype=dtype)
    print(u.domain.get_basis(c))
    # Test Fourier CFL
    fourier_velocity = lambda x: np.sin(4*np.pi*x/Lx)
    chebyshev_velocity = lambda z: -z_velocity_mag*z
    u['g'][0] = fourier_velocity(x)
    u['g'][1] = chebyshev_velocity(z)
    # AdvectiveCFL initialization
    cfl = operators.AdvectiveCFL(u, c)
    cfl_freq = cfl.evaluate()['g']
    comparison_freq = np.abs(u['g'][0])  / cfl.cfl_spacing(u)[0]
    comparison_freq += np.abs(u['g'][1]) / cfl.cfl_spacing(u)[1]
    assert np.allclose(cfl_freq, comparison_freq)

@pytest.mark.parametrize('timestepper', [timesteppers.SBDF1])
@pytest.mark.parametrize('Lmax', [15])
@pytest.mark.parametrize('dtype', [np.complex128, np.float64])
@pytest.mark.parametrize('dealias', [1, 3/2])
def test_S2_AdvectiveCFL(Lmax, timestepper, dtype, dealias):
    radius=1
    # Bases
    c, d, sb, phi, theta = build_S2(2*(Lmax+1), (Lmax+1), dealias, dtype=dtype, grid_scale=1)
    x = radius*np.sin(theta)*np.cos(phi)
    y = radius*np.sin(theta)*np.sin(phi)
    z = radius*np.cos(theta)
    # Fields
    u = field.Field(name='u', dist=d, tensorsig=(c,), bases=(sb,), dtype=dtype)
    # For a scalar field f = x*z, set velocity as u = grad(f). (S2 grad currently not implemened)
    u['g'][0] = -radius*np.cos(theta)*np.sin(phi)
    u['g'][1] = radius*(np.cos(theta)**2 - np.sin(theta)**2)*np.cos(phi)
    # AdvectiveCFL initialization
    cfl = operators.AdvectiveCFL(u, c)
    cfl_freq = cfl.evaluate()['g']
    comparison_freq = np.sqrt(u['g'][0]**2 + u['g'][1]**2) / cfl.cfl_spacing()[0]
    assert np.allclose(cfl_freq, comparison_freq)

@pytest.mark.parametrize('timestepper', [timesteppers.SBDF1])
@pytest.mark.parametrize('Lmax', [15])
@pytest.mark.parametrize('Nmax', [15])
@pytest.mark.parametrize('dtype', [np.complex128, np.float64])
@pytest.mark.parametrize('dealias', [1, 3/2])
def test_ball_AdvectiveCFL(Lmax, Nmax, timestepper, dtype, dealias):
    radius = 2
    # Bases
    c, d, b, phi, theta, r, x, y, z = build_ball(2*(Lmax+1), (Lmax+1), (Nmax+1), radius, dealias, dtype=dtype, grid_scale=1)
    # Fields
    f = field.Field(name='f', dist=d, bases=(b,), dtype=dtype)
    f['g'] = x*y*z
    u      = operators.Gradient(f, c).evaluate()
    # AdvectiveCFL initialization
    cfl = operators.AdvectiveCFL(u, c)
    cfl_freq = cfl.evaluate()['g']
    comparison_freq = np.sqrt(u['g'][0]**2 + u['g'][1]**2) / cfl.cfl_spacing()[0]
    comparison_freq += np.abs(u['g'][2]) / cfl.cfl_spacing()[1]
    assert np.allclose(cfl_freq, comparison_freq)


@pytest.mark.parametrize('timestepper', [timesteppers.SBDF1])
@pytest.mark.parametrize('Lmax', [15])
@pytest.mark.parametrize('Nmax', [15])
@pytest.mark.parametrize('dtype', [np.complex128, np.float64])
@pytest.mark.parametrize('dealias', [1, 3/2])
def test_spherical_shell_AdvectiveCFL(Lmax, Nmax, timestepper, dtype, dealias):
    radii = (0.5, 2)
    # Bases
    c, d, b, phi, theta, r, x, y, z = build_shell(2*(Lmax+1), (Lmax+1), (Nmax+1), radii, dealias, dtype=dtype, grid_scale=1)
    # Fields
    f = field.Field(name='f', dist=d, bases=(b,), dtype=dtype)
    f['g'] = x*y*z
    u      = operators.Gradient(f, c).evaluate()
    # AdvectiveCFL initialization
    cfl = operators.AdvectiveCFL(u, c)
    cfl_freq = cfl.evaluate()['g']
    comparison_freq = np.sqrt(u['g'][0]**2 + u['g'][1]**2) / cfl.cfl_spacing()[0]
    comparison_freq += np.abs(u['g'][2]) / cfl.cfl_spacing()[1]
    assert np.allclose(cfl_freq, comparison_freq)

@pytest.mark.parametrize('timestepper', [timesteppers.SBDF1])
@pytest.mark.parametrize('Nphi', [16])
@pytest.mark.parametrize('Nr', [15])
@pytest.mark.parametrize('dtype', [np.complex128, np.float64])
@pytest.mark.parametrize('dealias', [1, 3/2])
def test_annulus_AdvectiveCFL(Nr, Nphi, timestepper, dtype, dealias):
    radii = (0.5, 2)
    # Bases
    c, d, db, phi, r, x, y = build_annulus(Nphi, Nr, radii, dealias, dtype=dtype, grid_scale=1)
    # Fields
    f = field.Field(name='f', dist=d, bases=(db,), dtype=dtype)
    f['g'] = x*y
    u = operators.Gradient(f, c).evaluate()
    # AdvectiveCFL initialization
    cfl = operators.AdvectiveCFL(u, c)
    cfl_freq = cfl.evaluate()['g']
    comparison_freq = np.abs(u['g'][0]) / cfl.cfl_spacing()[0]
    comparison_freq += np.abs(u['g'][1]) / cfl.cfl_spacing()[1]
    assert np.allclose(cfl_freq, comparison_freq)

@pytest.mark.parametrize('timestepper', [timesteppers.SBDF1])
@pytest.mark.parametrize('Nphi', [16])
@pytest.mark.parametrize('Nr', [15])
@pytest.mark.parametrize('dtype', [np.complex128, np.float64])
@pytest.mark.parametrize('dealias', [1, 3/2])
def test_disk_AdvectiveCFL(Nr, Nphi, timestepper, dtype, dealias):
    radius = 2
    # Bases
    c, d, db, phi, r, x, y = build_disk(Nphi, Nr, radius, dealias, dtype=dtype, grid_scale=1)
    # Fields
    f = field.Field(name='f', dist=d, bases=(db,), dtype=dtype)
    f['g'] = x*y
    u = operators.Gradient(f, c).evaluate()
    # AdvectiveCFL initialization
    cfl = operators.AdvectiveCFL(u, c)
    cfl_freq = cfl.evaluate()['g']
    comparison_freq = np.abs(u['g'][0]) / cfl.cfl_spacing()[0]
    comparison_freq += np.abs(u['g'][1]) / cfl.cfl_spacing()[1]
    assert np.allclose(cfl_freq, comparison_freq)
