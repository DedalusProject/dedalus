"""Test simple LBVPs."""
# TODO: finish cleanup, add sphere tests, and split off NCC tests

import pytest
import numpy as np
import dedalus.public as d3
from dedalus.tools.cache import CachedFunction


def xfail_param(param, reason, run=True):
    return pytest.param(param, marks=pytest.mark.xfail(reason=reason, run=run))


dtype_range = [np.float64, np.complex128]
disk_azimuth_coupling_range = [False, xfail_param(True, "Azimuthal coupling not implemented for disk basis", run=False)]


@CachedFunction
def build_disk(Nphi, Nr, radius, alpha, dealias, dtype):
    c = d3.PolarCoordinates('phi', 'r')
    d = d3.Distributor(c, dtype=dtype)
    b = d3.DiskBasis(c, (Nphi, Nr), radius=radius, alpha=alpha, dealias=dealias, dtype=dtype)
    phi, r = d.local_grids(b)
    x, y = c.cartesian(phi, r)
    return c, d, b, phi, r, x, y


@CachedFunction
def build_annulus(Nphi, Nr, radii, alpha, dealias, dtype):
    c = d3.PolarCoordinates('phi', 'r')
    d = d3.Distributor(c, dtype=dtype)
    b = d3.AnnulusBasis(c, (Nphi, Nr), radii=radii, alpha=alpha, dealias=dealias, dtype=dtype)
    phi, r = d.local_grids(b)
    x, y = c.cartesian(phi, r)
    return c, d, b, phi, r, x, y


@pytest.mark.parametrize('dtype', dtype_range)
def test_algebraic(dtype):
    """Test algebraic equation."""
    # Bases
    coord = d3.Coordinate('x')
    dist = d3.Distributor(coord, dtype=dtype)
    u = dist.Field(name='u')
    v = dist.Field(name='v')
    F = dist.Field(name='F')
    v['g'] = -1
    F['g'] = -3
    problem = d3.LBVP([u], namespace=locals())
    problem.add_equation("v*u = F")
    solver = problem.build_solver()
    solver.solve()
    # Check solution
    u_true = 3
    assert np.allclose(u['g'], u_true)


@pytest.mark.parametrize('N', [32])
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('matrix_coupling', [False, True])
def test_poisson_fourier(N, dtype, matrix_coupling):
    """Test Poisson equation with Fourier basis."""
    # Bases
    coord = d3.Coordinate('x')
    dist = d3.Distributor(coord, dtype=dtype)
    basis = d3.Fourier(coord, size=N, bounds=(0, 2*np.pi), dtype=dtype)
    x = dist.local_grid(basis)
    # Fields
    u = dist.Field(name='u', bases=basis)
    g = dist.Field(name='c')
    u_true = np.sin(x)
    f = dist.Field(bases=basis)
    f['g'] = -np.sin(x)
    # Problem
    dx = lambda A: d3.Differentiate(A, coord)
    integ = lambda A: d3.Integrate(A, coord)
    problem = d3.LBVP([u, g], namespace=locals())
    problem.add_equation("dx(dx(u)) + g = f")
    problem.add_equation("integ(u) = 0")
    solver = problem.build_solver(matrix_coupling=[matrix_coupling])
    solver.solve()
    assert np.allclose(u['g'], u_true)


@pytest.mark.parametrize('N', [32])
@pytest.mark.parametrize('a, b', [(-1/2, -1/2), (0, 0)])
@pytest.mark.parametrize('dtype', dtype_range)
def test_poisson_jacobi(N, a, b, dtype):
    """Test Poisson equation with Jacobi basis."""
    # Bases
    coord = d3.Coordinate('x')
    dist = d3.Distributor(coord, dtype=dtype)
    basis = d3.Jacobi(coord, size=N, bounds=(0, 2*np.pi), a=a, b=b)
    x = dist.local_grid(basis)
    # Fields
    u = dist.Field(name='u', bases=basis)
    tau1 = dist.Field(name='tau1')
    tau2 = dist.Field(name='tau2')
    u_true = np.sin(x)
    f = dist.Field(bases=basis)
    f['g'] = -np.sin(x)
    # Problem
    dx = lambda A: d3.Differentiate(A, coord)
    lift = lambda A, n: d3.Lift(A, basis.derivative_basis(2), n)
    problem = d3.LBVP([u, tau1, tau2], namespace=locals())
    problem.add_equation("dx(dx(u)) + lift(tau1,-1) + lift(tau2,-2) = f")
    problem.add_equation("u(x='left') = 0")
    problem.add_equation("u(x='right') = 0")
    solver = problem.build_solver()
    solver.solve()
    assert np.allclose(u['g'], u_true)


@pytest.mark.parametrize('Nphi', [1, 8])
@pytest.mark.parametrize('Nr', [8])
@pytest.mark.parametrize('alpha', [0, 1])
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('azimuth_coupling', disk_azimuth_coupling_range)
def test_poisson_annulus(Nr, Nphi, alpha, dtype, azimuth_coupling):
    """Test Poisson equation with annulus basis."""
    if Nphi == 1 and dtype == np.float64:
        pytest.xfail("Nphi=1 not supported for real dtypes.")
    # Bases
    c, d, b, phi, r, x, y = build_annulus(Nphi, Nr, radii=(0.4, 1.5), alpha=alpha, dealias=1, dtype=dtype)
    # Fields
    u = d.Field(bases=b)
    tau1 = d.Field(bases=b.outer_edge)
    tau2 = d.Field(bases=b.outer_edge)
    f = d.Field(bases=b)
    g0 = d.Field(bases=b.inner_edge)
    g1 = d.Field(bases=b.outer_edge)
    if Nphi == 1:
        u_true = r**2 - 1
        f['g'] = 4
        g0['g'] = b.radii[0]**2 - 1
        g1['g'] = b.radii[1]**2 - 1
    else:
        xr0 = b.radii[0] * np.cos(phi)
        yr0 = b.radii[0] * np.sin(phi)
        xr1 = b.radii[1] * np.cos(phi)
        yr1 = b.radii[1] * np.sin(phi)
        u_true = x**3 - y**2
        f['g'] = 6*x - 2
        g0['g'] = xr0**3 - yr0**2
        g1['g'] = xr1**3 - yr1**2
    # Problem
    b2 = b.derivative_basis(2)
    taus = d3.Lift(tau1, b2, -1) + d3.Lift(tau2, b2, -2)
    problem = d3.LBVP([u, tau1, tau2], namespace=locals())
    problem.add_equation("lap(u) + taus = f")
    problem.add_equation("u(r=b.radii[0]) = g0")
    problem.add_equation("u(r=b.radii[1]) = g1")
    solver = problem.build_solver(matrix_coupling=[azimuth_coupling, True])
    solver.solve()
    assert np.allclose(u['g'], u_true)


@pytest.mark.parametrize('Nphi', [1, 8])
@pytest.mark.parametrize('Nr', [8])
@pytest.mark.parametrize('alpha', [0, 1])
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('azimuth_coupling', disk_azimuth_coupling_range)
def test_poisson_disk(Nr, Nphi, alpha, dtype, azimuth_coupling):
    """Test Poisson equation with disk basis."""
    if Nphi == 1 and dtype == np.float64:
        pytest.xfail("Nphi=1 not supported for real dtypes.")
    # Bases
    c, d, b, phi, r, x, y = build_disk(Nphi, Nr, radius=1.5, alpha=alpha, dealias=1, dtype=dtype)
    # Fields
    u = d.Field(bases=b)
    tau = d.Field(bases=b.edge)
    f = d.Field(bases=b)
    g = d.Field(bases=b.edge)
    if Nphi == 1:
        u_true = r**2 - 1
        f['g'] = 4
        g['g'] = b.radius**2 - 1
    else:
        xr = b.radius * np.cos(phi)
        yr = b.radius * np.sin(phi)
        u_true = x**3 - y**2
        f['g'] = 6*x - 2
        g['g'] = xr**3 - yr**2
    # Problem
    lift = lambda A: d3.Lift(A, b, -1)
    problem = d3.LBVP([u, tau], namespace=locals())
    problem.add_equation("lap(u) + lift(tau) = f")
    problem.add_equation("u(r=b.radius) = g")
    solver = problem.build_solver(matrix_coupling=[azimuth_coupling, True])
    solver.solve()
    assert np.allclose(u['g'], u_true)


@pytest.mark.parametrize('Nphi', [1, 12])
@pytest.mark.parametrize('Nr', [6])
@pytest.mark.parametrize('alpha', [0, 1])
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('azimuth_coupling', disk_azimuth_coupling_range)
@pytest.mark.parametrize('component_bcs', [False, True])
def test_vector_poisson_annulus(Nr, Nphi, alpha, dtype, azimuth_coupling, component_bcs):
    """Test vector Poisson equation with annulus basis."""
    if Nphi == 1 and dtype == np.float64:
        pytest.xfail("Nphi=1 not supported for real dtypes.")
    if component_bcs and dtype == np.float64:
        pytest.xfail("Hermitian symmetry not fixed for polar coords")
    # Bases
    c, d, b, phi, r, x, y = build_annulus(Nphi, Nr, radii=(0.4, 1.5), alpha=alpha, dealias=1, dtype=dtype)
    # Fields
    u = d.VectorField(c, bases=b)
    tau1 = d.VectorField(c, bases=b.outer_edge)
    tau2 = d.VectorField(c, bases=b.outer_edge)
    f = d.VectorField(c, bases=b)
    g0 = d.VectorField(c, bases=b.inner_edge)
    g1 = d.VectorField(c, bases=b.outer_edge)
    if Nphi == 1:
        u_true = np.zeros_like(u['g'])
        u_true[1] = r**3
        f['g'][1] = 8 * r
        g0['g'][1] = b.radii[0]**3
        g1['g'][1] = b.radii[1]**3
    else:
        ex = np.array([-np.sin(phi), np.cos(phi)])
        ey = np.array([np.cos(phi), np.sin(phi)])
        xr0 = b.radii[0] * np.cos(phi)
        yr0 = b.radii[0] * np.sin(phi)
        xr1 = b.radii[1] * np.cos(phi)
        yr1 = b.radii[1] * np.sin(phi)
        u_true = (x**2 + y**2)*ex + (x**3 + y)*ey
        f['g'] = 4*ex + 6*x*ey
        g0['g'] = (xr0**2 + yr0**2)*ex + (xr0**3 + yr0)*ey
        g1['g'] = (xr1**2 + yr1**2)*ex + (xr1**3 + yr1)*ey
    # Problem
    b2 = b.derivative_basis(2)
    tau = d3.Lift(tau1, b2, -1) + d3.Lift(tau2, b2, -2)
    problem = d3.LBVP([u, tau1, tau2], namespace=locals())
    problem.add_equation("lap(u) + tau = f")
    if component_bcs:
        problem.add_equation("radial(u(r=b.radii[0])) = radial(g0)")
        problem.add_equation("radial(u(r=b.radii[1])) = radial(g1)")
        problem.add_equation("azimuthal(u(r=b.radii[0])) = azimuthal(g0)")
        problem.add_equation("azimuthal(u(r=b.radii[1])) = azimuthal(g1)")
    else:
        problem.add_equation("u(r=b.radii[0]) = g0")
        problem.add_equation("u(r=b.radii[1]) = g1")
    solver = problem.build_solver(matrix_coupling=[azimuth_coupling, True])
    solver.solve()
    assert np.allclose(u['g'], u_true)


@pytest.mark.parametrize('Nphi', [1, 12])
@pytest.mark.parametrize('Nr', [6])
@pytest.mark.parametrize('alpha', [0, 1])
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('azimuth_coupling', disk_azimuth_coupling_range)
@pytest.mark.parametrize('component_bcs', [False, True])
def test_vector_poisson_disk(Nr, Nphi, alpha, dtype, azimuth_coupling, component_bcs):
    """Test vector Poisson equation with disk basis."""
    if Nphi == 1 and dtype == np.float64:
        pytest.xfail("Nphi=1 not supported for real dtypes.")
    if component_bcs and dtype == np.float64:
        pytest.xfail("Hermitian symmetry not fixed for polar coords")
    # Bases
    c, d, b, phi, r, x, y = build_disk(Nphi, Nr, radius=1.5, alpha=alpha, dealias=1, dtype=dtype)
    # Fields
    u = d.VectorField(c, bases=b)
    tau = d.VectorField(c, bases=b.edge)
    f = d.VectorField(c, bases=b)
    g = d.VectorField(c, bases=b.edge)
    if Nphi == 1:
        u_true = np.zeros_like(u['g'])
        u_true[1] = r**3
        f = d.VectorField(c, bases=b)
        f['g'][1] = 8 * r
        g = d.VectorField(c, bases=b.edge)
        g['g'][1] = b.radius**3
    else:
        ex = np.array([-np.sin(phi), np.cos(phi)])
        ey = np.array([np.cos(phi), np.sin(phi)])
        xr = b.radius * np.cos(phi)
        yr = b.radius * np.sin(phi)
        u_true = (x**2 + y**2)*ex + (x**3 + y)*ey
        f['g'] = 4*ex + 6*x*ey
        g['g'] = (xr**2 + yr**2)*ex + (xr**3 + yr)*ey
    # Problem
    lift = lambda A: d3.Lift(A, b, -1)
    problem = d3.LBVP([u, tau], namespace=locals())
    problem.add_equation("lap(u) + lift(tau) = f")
    if component_bcs:
        problem.add_equation("radial(u(r=b.radius)) = radial(g)")
        problem.add_equation("azimuthal(u(r=b.radius)) = azimuthal(g)")
    else:
        problem.add_equation("u(r=b.radius) = g")
    solver = problem.build_solver(matrix_coupling=[azimuth_coupling, True])
    solver.solve()
    assert np.allclose(u['g'], u_true)


## stopped cleanup


radius_ball = 1


@CachedFunction
def build_ball(Nphi, Ntheta, Nr, dealias, dtype):
    c = d3.SphericalCoordinates('phi', 'theta', 'r')
    d = d3.Distributor((c,), dtype=dtype)
    b = d3.BallBasis(c, (Nphi, Ntheta, Nr), radius=radius_ball, dealias=(dealias, dealias, dealias), dtype=dtype)
    phi, theta, r = d.local_grids(b)
    x, y, z = c.cartesian(phi, theta, r)
    return c, d, b, phi, theta, r, x, y, z


@pytest.mark.parametrize('dtype', [np.complex128, np.float64])
@pytest.mark.parametrize('Nmax', [15])
@pytest.mark.parametrize('Lmax', [3])
def test_heat_ball(Nmax, Lmax, dtype):
    # Bases
    dealias = 1
    c, d, b, phi, theta, r, x, y, z = build_ball(2*(Lmax+1), Lmax+1, Nmax+1, dealias=dealias, dtype=dtype)
    # Fields
    u = d.Field(name='u', bases=(b,), dtype=dtype)
    τu = d.Field(name='u', bases=(b.S2_basis(),), dtype=dtype)
    F = d.Field(name='a', bases=(b,), dtype=dtype)
    F['g'] = 6
    # Problem
    Lap = lambda A: d3.Laplacian(A, c)
    Lift = lambda A: d3.Lift(A, b, -1)
    problem = d3.LBVP([u, τu])
    problem.add_equation((Lap(u) + Lift(τu), F))
    problem.add_equation((u(r=radius_ball), 0))
    # Solver
    solver = problem.build_solver()
    solver.solve()
    # Check solution
    u_true = r**2 - 1
    assert np.allclose(u['g'], u_true)


@pytest.mark.parametrize('dtype', [np.complex128, np.float64])
@pytest.mark.parametrize('Nmax', [7])
@pytest.mark.parametrize('Lmax', [7])
def test_heat_ball_cart(Nmax, Lmax, dtype):
    # Bases
    dealias = 1
    c, d, b, phi, theta, r, x, y, z = build_ball(2*(Lmax+1), Lmax+1, Nmax+1, dealias=dealias, dtype=dtype)
    xr = radius_ball * np.cos(phi) * np.sin(theta)
    yr = radius_ball * np.sin(phi) * np.sin(theta)
    zr = radius_ball * np.cos(theta)
    # Fields
    u = d.Field(name='u', bases=(b,), dtype=dtype)
    τu = d.Field(name='u', bases=(b.S2_basis(),), dtype=dtype)
    f = d.Field(name='a', bases=(b,), dtype=dtype)
    f['g'] = 12*x**2 - 6*y + 2
    g = d.Field(name='a', bases=(b.S2_basis(),), dtype=dtype)
    g['g'] = xr**4 - yr**3 + zr**2
    # Problem
    Lap = lambda A: d3.Laplacian(A, c)
    Lift = lambda A: d3.Lift(A, b, -1)
    problem = d3.LBVP([u, τu])
    problem.add_equation((Lap(u) + Lift(τu), f))
    problem.add_equation((u(r=radius_ball), g))
    # Solver
    solver = problem.build_solver()
    solver.solve()
    # Check solution
    u_true = x**4 - y**3 + z**2
    assert np.allclose(u['g'], u_true)


radii_shell = (1, 2)


@CachedFunction
def build_shell(Nphi, Ntheta, Nr, dealias, dtype):
    c = d3.SphericalCoordinates('phi', 'theta', 'r')
    d = d3.Distributor((c,), dtype=dtype)
    b = d3.ShellBasis(c, (Nphi, Ntheta, Nr), radii=radii_shell, dealias=(dealias, dealias, dealias), dtype=dtype)
    phi, theta, r = d.local_grids(b)
    x, y, z = c.cartesian(phi, theta, r)
    return c, d, b, phi, theta, r, x, y, z


@pytest.mark.parametrize('dtype', [np.complex128, np.float64])
@pytest.mark.parametrize('Nmax', [15])
@pytest.mark.parametrize('Lmax', [3])
def test_heat_shell(Nmax, Lmax, dtype):
    # Bases
    dealias = 1
    c, d, b, phi, theta, r, x, y, z = build_shell(2*(Lmax+1), Lmax+1, Nmax+1, dealias=dealias, dtype=dtype)
    r0, r1 = b.radial_basis.radii
    # Fields
    u = d.Field(name='u', bases=(b,), dtype=dtype)
    τu1 = d.Field(name='τu1', bases=(b.S2_basis(),), dtype=dtype)
    τu2 = d.Field(name='τu2', bases=(b.S2_basis(),), dtype=dtype)
    F = d.Field(name='a', bases=(b,), dtype=dtype)
    F['g'] = 6
    # Problem
    Lap = lambda A: d3.Laplacian(A, c)
    Lift = lambda A, n: d3.Lift(A, b, n)
    problem = d3.LBVP([u, τu1, τu2])
    problem.add_equation((Lap(u) + Lift(τu1,-1) + Lift(τu2,-2), F))
    problem.add_equation((u(r=r0), 0))
    problem.add_equation((u(r=r1), 0))
    # Solver
    solver = problem.build_solver()
    solver.solve()
    # Check solution
    u_true = r**2 + 6 / r - 7
    assert np.allclose(u['g'], u_true)


@pytest.mark.parametrize('dtype', [np.complex128, np.float64])
@pytest.mark.parametrize('Nmax', [15])
@pytest.mark.parametrize('Lmax', [3])
def test_vector_heat_shell(Nmax, Lmax, dtype):
    # Bases
    dealias = 1
    c, d, b, phi, theta, r, x, y, z = build_shell(2*(Lmax+1), Lmax+1, Nmax+1, dealias=dealias, dtype=dtype)
    r0, r1 = b.radial_basis.radii
    # Fields
    u = d.Field(name='u', bases=b, tensorsig=(c,))
    τu1 = d.Field(name='τu1', bases=b.S2_basis(), tensorsig=(c,))
    τu2 = d.Field(name='τu2', bases=b.S2_basis(), tensorsig=(c,))
    ez = d.Field(name='u', bases=b, tensorsig=(c,))
    ez['g'][1] = - np.sin(theta)
    ez['g'][2] = np.cos(theta)
    F = d.Field(name='a', bases=b, tensorsig=(c,))
    F['g'] = 6 * ez['g']
    # Problem
    Lap = lambda A: d3.Laplacian(A, c)
    Lift = lambda A, n: d3.Lift(A, b, n)
    problem = d3.LBVP([u, τu1, τu2])
    problem.add_equation((Lap(u) + Lift(τu1,-1) + Lift(τu2,-2), F))
    problem.add_equation((u(r=r0), 0))
    problem.add_equation((u(r=r1), 0))
    # Solver
    solver = problem.build_solver()
    solver.solve()
    # Check solution
    u_true = (r**2 + 6 / r - 7) * ez['g']
    assert np.allclose(u['g'], u_true)


@pytest.mark.parametrize('dtype', [np.complex128, np.float64])
@pytest.mark.parametrize('Nmax', [15])
@pytest.mark.parametrize('Lmax', [3])
def test_ncc_ball(Nmax, Lmax, dtype):
    # Bases
    dealias = 1
    c, d, b, phi, theta, r, x, y, z = build_ball(2*(Lmax+1), Lmax+1, Nmax+1, dealias=dealias, dtype=dtype)
    # Fields
    u = d.Field(name='u', bases=(b,), dtype=dtype)
    ncc = d.Field(name='ncc', bases=(b.radial_basis,), dtype=dtype)
    ncc['g'] = 1+r**2
    F = d.Field(name='F', bases=(b,), dtype=dtype)
    F['g'] = 1
    # Problem
    problem = d3.LBVP([u])
    problem.add_equation((ncc*u, F))
    # Solver
    solver = problem.build_solver()
    solver.solve()
    # Check solution
    u_true = 1/ncc['g']
    assert np.allclose(u['g'], u_true)


@pytest.mark.parametrize('dtype', [np.complex128, np.float64])
@pytest.mark.parametrize('Nmax', [15])
@pytest.mark.parametrize('Lmax', [3])
@pytest.mark.parametrize('ncc_exponent', [0,1,2,3/2])
@pytest.mark.parametrize('ncc_location', ['LHS', 'RHS'])
@pytest.mark.parametrize('ncc_scale', [1, 3/2])
def test_heat_ncc_shell(Nmax, Lmax, ncc_exponent, ncc_location, ncc_scale, dtype):
    # Bases
    dealias = 1
    c, d, b, phi, theta, r, x, y, z = build_shell(2*(Lmax+1), Lmax+1, Nmax+1, dealias=dealias, dtype=dtype)
    r0, r1 = b.radial_basis.radii
    # Fields
    u = d.Field(name='u', bases=(b,), dtype=dtype)
    τu1 = d.Field(name='τu1', bases=(b.S2_basis(),), dtype=dtype)
    τu2 = d.Field(name='τu2', bases=(b.S2_basis(),), dtype=dtype)
    ncc = d.Field(name='ncc', bases=(b.radial_basis,), dtype=dtype)
    F = d.Field(name='a', bases=(b,), dtype=dtype)
    # Test Parameters
    F_value  = {0:6, 1:2, 2:1, 3/2:3/4}
    analytic = {0:r**2+6/r-7, 1:r+2/r-3, 2:np.log(r)+np.log(4)/r-np.log(4), 3/2:np.sqrt(r)+0.82842712/r-1.82842712}
    if ncc_location=='RHS':
        ncc['g'] = 1
        F['g'] = F_value[ncc_exponent]*r**(-ncc_exponent)
    else:
        ncc['g'] = r**ncc_exponent
        F['g'] = F_value[ncc_exponent]
    u_true = analytic[ncc_exponent]
    ncc.change_scales(ncc_scale)
    ncc['g'] # force transform
    # Problem
    Lap = lambda A: d3.Laplacian(A, c)
    Lift = lambda A, n: d3.Lift(A, b, n)
    problem = d3.LBVP([u, τu1, τu2])
    problem.add_equation((ncc*Lap(u) + Lift(τu1,-1) + Lift(τu2,-2), F))
    problem.add_equation((u(r=r0), 0))
    problem.add_equation((u(r=r1), 0))
    # Solver
    solver = problem.build_solver()
    solver.solve()
    # Check solution
    assert np.allclose(u['g'], u_true)


@pytest.mark.parametrize('dtype', [np.complex128])
@pytest.mark.parametrize('Nmax', [15])
@pytest.mark.parametrize('Lmax', [7])
def test_lap_meridional_ncc_shell(Nmax, Lmax, dtype):
    # Bases
    dealias = 1
    c, d, b, phi, theta, r, x, y, z = build_shell(2*(Lmax+1), Lmax+1, Nmax+1, dealias=dealias, dtype=dtype)
    r0, r1 = b.radial_basis.radii
    # Fields
    u = d.Field(name='u', bases=(b,), dtype=dtype)
    v = d.Field(name='v', bases=(b,), dtype=dtype)
    τu1 = d.Field(name='τu1', bases=(b.S2_basis(),), dtype=dtype)
    τu2 = d.Field(name='τu2', bases=(b.S2_basis(),), dtype=dtype)
    ncc = d.Field(name='ncc', bases=(b.meridional_basis,), dtype=dtype)
    v['g'] = x**2 + z**2
    ncc['g'] = z**2
    # Problem
    Lap = lambda A: d3.Laplacian(A, c)
    Lift = lambda A, n: d3.Lift(A, b, n)
    problem = d3.LBVP([u, τu1, τu2])
    problem.add_equation((ncc*Lap(u) + Lift(τu1,-1) + Lift(τu2,-2), ncc*Lap(v)))
    problem.add_equation((u(r=r0), v(r=r0)))
    problem.add_equation((u(r=r1), v(r=r1)))
    # Solver
    solver = problem.build_solver()
    solver.solve()
    # Check solution
    assert np.allclose(u['g'], v['g'])


@pytest.mark.parametrize('dtype', [np.complex128])
@pytest.mark.parametrize('Nmax', [15])
@pytest.mark.parametrize('Lmax', [7])
def test_lap_meridional_radial_ncc_shell(Nmax, Lmax, dtype):
    # Bases
    dealias = 1
    c, d, b, phi, theta, r, x, y, z = build_shell(2*(Lmax+1), Lmax+1, Nmax+1, dealias=dealias, dtype=dtype)
    r0, r1 = b.radial_basis.radii
    # Fields
    u = d.Field(name='u', bases=(b,), dtype=dtype)
    v = d.Field(name='v', bases=(b,), dtype=dtype)
    τu1 = d.Field(name='τu1', bases=(b.S2_basis(),), dtype=dtype)
    τu2 = d.Field(name='τu2', bases=(b.S2_basis(),), dtype=dtype)
    ncc_m = d.Field(name='ncc', bases=(b.meridional_basis,), dtype=dtype)
    ncc_r = d.Field(name='ncc', bases=(b.radial_basis,), dtype=dtype)
    v['g'] = x**2 + z**2
    ncc_m['g'] = z**2
    ncc_r['g'] = r**2
    # Problem
    Lap = lambda A: d3.Laplacian(A, c)
    Lift = lambda A, n: d3.Lift(A, b, n)
    problem = d3.LBVP([u, τu1, τu2])
    problem.add_equation((ncc_m*Lap(u) + ncc_r*Lap(u) + Lift(τu1,-1) + Lift(τu2,-2), ncc_m*Lap(v) + ncc_r*Lap(v)))
    problem.add_equation((u(r=r0), v(r=r0)))
    problem.add_equation((u(r=r1), v(r=r1)))
    # Solver
    solver = problem.build_solver()
    solver.solve()
    # Check solution
    assert np.allclose(u['g'], v['g'])


@pytest.mark.xfail(reason="Radial NCCs don't work in meridional problems for vectors.")
@pytest.mark.parametrize('dtype', [np.complex128])
@pytest.mark.parametrize('Nmax', [15])
@pytest.mark.parametrize('Lmax', [7])
def test_lap_2dncc_vector_shell(Nmax, Lmax, dtype):
    # Bases
    dealias = 1
    c, d, b, phi, theta, r, x, y, z = build_shell(2*(Lmax+1), Lmax+1, Nmax+1, dealias=dealias, dtype=dtype)
    r0, r1 = b.radial_basis.radii
    # Fields
    u = d.Field(name='u', bases=b, tensorsig=(c,))
    v = d.Field(name='v', bases=b, tensorsig=(c,))
    τu1 = d.Field(name='τu1', bases=b.S2_basis(), tensorsig=(c,))
    τu2 = d.Field(name='τu2', bases=b.S2_basis(), tensorsig=(c,))
    ez = d.Field(name='u', bases=b, tensorsig=(c,))
    ez['g'][1] = - np.sin(theta)
    ez['g'][2] = np.cos(theta)
    ncc_m = d.Field(name='ncc', bases=b.meridional_basis)
    ncc_r = d.Field(name='ncc', bases=b.radial_basis)
    v['g'] = (x**2 + z**2) * ez['g']
    ncc_m['g'] = z**2
    ncc_r['g'] = r**2
    # Problem
    Lap = lambda A: d3.Laplacian(A, c)
    Lift = lambda A, n: d3.Lift(A, b, n)
    problem = d3.LBVP([u, τu1, τu2])
    problem.add_equation((ncc_r*Lap(u) + ncc_m*Lap(u) + Lift(τu1,-1) + Lift(τu2,-2), ncc_r*Lap(v) + ncc_m*Lap(v)))
    problem.add_equation((u(r=r0), v(r=r0)))
    problem.add_equation((u(r=r1), v(r=r1)))
    # Solver
    solver = problem.build_solver()
    solver.solve()
    # Check solution
    assert np.allclose(u['g'], v['g'])


@pytest.mark.parametrize('dtype', [np.complex128, np.float64])
@pytest.mark.parametrize('Nmax', [15])
@pytest.mark.parametrize('Lmax', [3])
@pytest.mark.parametrize('ncc_scale', [1, 3/2])
def test_heat_ncc_cos_ball(Nmax, Lmax, ncc_scale, dtype):
    # Bases
    dealias = 1
    c, d, b, phi, theta, r, x, y, z = build_ball(2*(Lmax+1), Lmax+1, Nmax+1, dealias=dealias, dtype=dtype)
    # Fields
    u = d.Field(name='u', bases=(b,), dtype=dtype)
    τu = d.Field(name='u', bases=(b.S2_basis(),), dtype=dtype)
    ncc = d.Field(name='ncc', bases=(b.radial_basis,), dtype=dtype)
    F = d.Field(name='a', bases=(b,), dtype=dtype)
    # Test Parameters
    R = radius_ball
    u_true = np.cos(np.pi/2*(r/R)**2)
    g = - np.pi/2/R**2 * (4*np.pi/2*(r/R)**2*np.cos(np.pi/2*(r/R)**2) + 6*np.sin(np.pi/2*(r/R)**2))
    ncc['g'] = u_true
    F['g'] = u_true*g
    ncc.change_scales(ncc_scale)
    ncc['g'] # force transform
    # Problem
    Lap = lambda A: d3.Laplacian(A, c)
    Lift = lambda A: d3.Lift(A, b, -1)
    problem = d3.LBVP([u, τu])
    problem.add_equation((ncc*Lap(u) + Lift(τu), F))
    problem.add_equation((u(r=radius_ball), 0))
    # Solver
    solver = problem.build_solver()
    solver.solve()
    # Check solution
    assert np.allclose(u['g'], u_true)
