"""
Test 1D LBVP.
"""
import pytest
import numpy as np
import functools
import dedalus.public as d3
from dedalus.core import coords, distributor, basis, field, operators, problems, solvers, timesteppers, arithmetic
from dedalus.tools.cache import CachedFunction


dtype_range = [np.float64, np.complex128]


@pytest.mark.parametrize('dtype', dtype_range)
def test_algebraic(dtype):
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


@pytest.mark.parametrize('Nx', [32])
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('matrix_coupling', [True, False])
def test_poisson_1d_fourier(Nx, dtype, matrix_coupling):
    # Bases
    coord = d3.Coordinate('x')
    dist = d3.Distributor(coord, dtype=dtype)
    if dtype == np.complex128:
        basis = d3.ComplexFourier(coord, size=Nx, bounds=(0, 2*np.pi))
    elif dtype == np.float64:
        basis = d3.RealFourier(coord, size=Nx, bounds=(0, 2*np.pi))
    x = basis.local_grid(1)
    # Fields
    u = dist.Field(name='u', bases=basis)
    g = dist.Field(name='c')
    # Substitutions
    dx = lambda A: d3.Differentiate(A, coord)
    integ = lambda A: d3.Integrate(A, coord)
    F = dist.Field(bases=basis)
    F['g'] = -np.sin(x)
    # Problem
    problem = d3.LBVP([u, g], namespace=locals())
    problem.add_equation("dx(dx(u)) + g = F")
    problem.add_equation("integ(u) = 0")
    # Solver
    solver = problem.build_solver(matrix_coupling=[matrix_coupling])
    solver.solve()
    # Check solution
    u_true = np.sin(x)
    assert np.allclose(u['g'], u_true)


@pytest.mark.parametrize('Nx', [32])
@pytest.mark.parametrize('a0', [-1/2, 0])
@pytest.mark.parametrize('b0', [-1/2, 0])
@pytest.mark.parametrize('da', [0, 1])
@pytest.mark.parametrize('db', [0, 1])
@pytest.mark.parametrize('dtype', dtype_range)
def test_poisson_1d_jacobi(Nx, a0, b0, da, db, dtype):
    # Bases
    coord = d3.Coordinate('x')
    dist = d3.Distributor(coord, dtype=dtype)
    basis = d3.Jacobi(coord, size=Nx, bounds=(0, 2*np.pi), a=a0+da, b=b0+db, a0=a0, b0=b0)
    x = basis.local_grid(1)
    # Fields
    u = dist.Field(name='u', bases=basis)
    tau1 = dist.Field(name='tau1')
    tau2 = dist.Field(name='tau2')
    # Substitutions
    dx = lambda A: d3.Differentiate(A, coord)
    lift_basis = basis.clone_with(a=a0+da+2, b=b0+db+2)
    lift = lambda A, n: d3.Lift(A, lift_basis, n)
    F = dist.Field(bases=basis)
    F['g'] = -np.sin(x)
    # Problem
    problem = d3.LBVP([u, tau1, tau2], namespace=locals())
    problem.add_equation("dx(dx(u)) + lift(tau1,-1) + lift(tau2,-2) = F")
    problem.add_equation("u(x='left') = 0")
    problem.add_equation("u(x='right') = 0")
    # Solver
    solver = problem.build_solver()
    solver.solve()
    # Check solution
    u_true = np.sin(x)
    assert np.allclose(u['g'], u_true)


@pytest.mark.parametrize('Nx', [32])
@pytest.mark.parametrize('a0', [-1/2, 0])
@pytest.mark.parametrize('b0', [-1/2, 0])
@pytest.mark.parametrize('da', [0, 1])
@pytest.mark.parametrize('db', [0, 1])
@pytest.mark.parametrize('dtype', dtype_range)
def test_poisson_1d_compound(Nx, a0, b0, da, db, dtype):
    # Bases
    coord = d3.Coordinate('x')
    dist = d3.Distributor(coord, dtype=dtype)
    b1 = d3.Jacobi(coord, size=Nx, bounds=(0, np.pi), a=a0+da, b=b0+db, a0=a0, b0=b0)
    b2 = d3.Jacobi(coord, size=Nx, bounds=(np.pi, 2*np.pi), a=a0+da, b=b0+db, a0=a0, b0=b0)
    basis = d3.Compound((b1, b2))
    x = basis.local_grid(1)
    # Fields
    u = dist.Field(name='u', bases=basis)
    tau11 = dist.Field(name='tau11')
    tau12 = dist.Field(name='tau12')
    tau21 = dist.Field(name='tau21')
    tau22 = dist.Field(name='tau22')
    tau1 = (tau11, tau12)
    tau2 = (tau21, tau22)
    # Substitutions
    dx = lambda A: d3.Differentiate(A, coord)
    lift_basis = basis.derivative_basis(2)
    lift = lambda A, n: d3.Lift(A, lift_basis, n)
    F = dist.Field(bases=basis)
    F['g'] = -np.sin(x)
    # Problem
    problem = d3.LBVP([u, *tau1, *tau2], namespace=locals())
    problem.add_equation("dx(dx(u)) + lift(tau1,-1) + lift(tau2,-2) = F")
    problem.add_equation("jump(u, x=np.pi) = 0")
    problem.add_equation("jump(dx(u), x=np.pi) = 0")
    problem.add_equation("u(x='left') = 0")
    problem.add_equation("u(x='right') = 0")
    # Solver
    solver = problem.build_solver()
    solver.solve()
    # Check solution
    u_true = np.sin(x)
    assert np.allclose(u['g'], u_true)


radius_disk = 1


@CachedFunction
def build_disk(Nphi, Nr, dealias, dtype):
    c = coords.PolarCoordinates('phi', 'r')
    d = distributor.Distributor((c,), dtype=dtype)
    b = basis.DiskBasis(c, (Nphi, Nr), radius=radius_disk, dealias=(dealias, dealias), dtype=dtype)
    phi, r = b.local_grids()
    x, y = c.cartesian(phi, r)
    return c, d, b, phi, r, x, y


@pytest.mark.parametrize('dtype', [np.complex128, np.float64])
@pytest.mark.parametrize('Nphi', [4])
@pytest.mark.parametrize('Nr', [8])
def test_scalar_heat_disk_axisymm(Nr, Nphi, dtype):
    # Bases
    dealias = 1
    c, d, b, phi, r, x, y = build_disk(Nphi, Nr, dealias=dealias, dtype=dtype)
    # Fields
    u = field.Field(name='u', dist=d, bases=(b,), dtype=dtype)
    τu = field.Field(name='u', dist=d, bases=(b.S1_basis(),), dtype=dtype)
    F = field.Field(name='a', dist=d, bases=(b,), dtype=dtype)
    F['g'] = 4
    # Problem
    Lap = lambda A: operators.Laplacian(A, c)
    Lift = lambda A: operators.Lift(A, b, -1)
    problem = problems.LBVP([u, τu])
    problem.add_equation((Lap(u) + Lift(τu), F))
    problem.add_equation((u(r=radius_disk), 0))
    # Solver
    solver = solvers.LinearBoundaryValueSolver(problem)
    solver.solve()
    # Check solution
    u_true = r**2 - 1
    assert np.allclose(u['g'], u_true)


@pytest.mark.parametrize('dtype', [np.complex128, np.float64])
@pytest.mark.parametrize('Nphi', [16])
@pytest.mark.parametrize('Nr', [32])
def test_scalar_heat_disk(Nr, Nphi, dtype):
    # Bases
    dealias = 1
    c, d, b, phi, r, x, y = build_disk(Nphi, Nr, dealias=dealias, dtype=dtype)
    xr = radius_disk * np.cos(phi)
    yr = radius_disk * np.sin(phi)
    # Fields
    u = field.Field(name='u', dist=d, bases=(b,), dtype=dtype)
    τu = field.Field(name='u', dist=d, bases=(b.S1_basis(),), dtype=dtype)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    f['g'] = 6*x - 2
    g = field.Field(dist=d, bases=(b.S1_basis(),), dtype=dtype)
    g['g'] = xr**3 - yr**2
    # Problem
    Lap = lambda A: operators.Laplacian(A, c)
    Lift = lambda A: operators.Lift(A, b, -1)
    problem = problems.LBVP([u, τu])
    problem.add_equation((Lap(u) + Lift(τu), f))
    problem.add_equation((u(r=radius_disk), g))
    # Solver
    solver = solvers.LinearBoundaryValueSolver(problem)
    solver.solve()
    # Check solution
    u_true = x**3 - y**2
    assert np.allclose(u['g'], u_true)


@pytest.mark.parametrize('dtype', [np.complex128, np.float64])
@pytest.mark.parametrize('Nphi', [4])
@pytest.mark.parametrize('Nr', [8])
def test_vector_heat_disk_dirichlet(Nr, Nphi, dtype):
    # Bases
    dealias = 1
    c, d, b, phi, r, x, y = build_disk(Nphi, Nr, dealias=dealias, dtype=dtype)
    # Fields
    u = field.Field(name='u', dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
    τu = field.Field(name='u', dist=d, bases=(b.S1_basis(),), tensorsig=(c,), dtype=dtype)
    v = field.Field(name='u', dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
    ex = np.array([-np.sin(phi), np.cos(phi)])
    ey = np.array([np.cos(phi), np.sin(phi)])
    v['g'] = (x+4*y)*ex
    vr = operators.RadialComponent(v(r=radius_disk))
    vph= operators.AzimuthalComponent(v(r=radius_disk))
    # Problem
    Lap = lambda A: operators.Laplacian(A, c)
    Lift = lambda A: operators.Lift(A, b, -1)
    problem = problems.LBVP([u, τu])
    problem.add_equation((Lap(u) + Lift(τu), 0))
    problem.add_equation((u(r=radius_disk), v(r=radius_disk)))
    # Solver
    solver = solvers.LinearBoundaryValueSolver(problem)
    solver.solve()
    assert np.allclose(u['g'], v['g'])


@pytest.mark.parametrize('dtype', [np.complex128, pytest.param(np.float64, marks=pytest.mark.xfail(reason="Hermitian symmetry not fixed for polar coords"))])
@pytest.mark.parametrize('Nphi', [4])
@pytest.mark.parametrize('Nr', [8])
def test_vector_heat_disk_components(Nr, Nphi, dtype):
    # Bases
    dealias = 1
    c, d, b, phi, r, x, y = build_disk(Nphi, Nr, dealias=dealias, dtype=dtype)
    # Fields
    u = field.Field(name='u', dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
    τu = field.Field(name='u', dist=d, bases=(b.S1_basis(),), tensorsig=(c,), dtype=dtype)
    v = field.Field(name='u', dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
    ex = np.array([-np.sin(phi), np.cos(phi)])
    ey = np.array([np.cos(phi), np.sin(phi)])
    v['g'] = (x+4*y)*ex
    vr = operators.RadialComponent(v(r=radius_disk))
    vph= operators.AzimuthalComponent(v(r=radius_disk))
    # Problem
    Lap = lambda A: operators.Laplacian(A, c)
    Lift = lambda A: operators.Lift(A, b, -1)
    problem = problems.LBVP([u, τu])
    problem.add_equation((Lap(u) + Lift(τu), 0))
    problem.add_equation((operators.RadialComponent(u(r=radius_disk)), vr))
    problem.add_equation((operators.AzimuthalComponent(u(r=radius_disk)), vph))
    # Solver
    solver = solvers.LinearBoundaryValueSolver(problem)
    solver.solve()
    assert np.allclose(u['g'], v['g'])



radius_ball = 1


@CachedFunction
def build_ball(Nphi, Ntheta, Nr, dealias, dtype):
    c = coords.SphericalCoordinates('phi', 'theta', 'r')
    d = distributor.Distributor((c,), dtype=dtype)
    b = basis.BallBasis(c, (Nphi, Ntheta, Nr), radius=radius_ball, dealias=(dealias, dealias, dealias), dtype=dtype)
    phi, theta, r = b.local_grids()
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
    u = field.Field(name='u', dist=d, bases=(b,), dtype=dtype)
    τu = field.Field(name='u', dist=d, bases=(b.S2_basis(),), dtype=dtype)
    F = field.Field(name='a', dist=d, bases=(b,), dtype=dtype)
    F['g'] = 6
    # Problem
    Lap = lambda A: operators.Laplacian(A, c)
    Lift = lambda A: operators.Lift(A, b, -1)
    problem = problems.LBVP([u, τu])
    problem.add_equation((Lap(u) + Lift(τu), F))
    problem.add_equation((u(r=radius_ball), 0))
    # Solver
    solver = solvers.LinearBoundaryValueSolver(problem)
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
    u = field.Field(name='u', dist=d, bases=(b,), dtype=dtype)
    τu = field.Field(name='u', dist=d, bases=(b.S2_basis(),), dtype=dtype)
    f = field.Field(name='a', dist=d, bases=(b,), dtype=dtype)
    f['g'] = 12*x**2 - 6*y + 2
    g = field.Field(name='a', dist=d, bases=(b.S2_basis(),), dtype=dtype)
    g['g'] = xr**4 - yr**3 + zr**2
    # Problem
    Lap = lambda A: operators.Laplacian(A, c)
    Lift = lambda A: operators.Lift(A, b, -1)
    problem = problems.LBVP([u, τu])
    problem.add_equation((Lap(u) + Lift(τu), f))
    problem.add_equation((u(r=radius_ball), g))
    # Solver
    solver = solvers.LinearBoundaryValueSolver(problem)
    solver.solve()
    # Check solution
    u_true = x**4 - y**3 + z**2
    assert np.allclose(u['g'], u_true)


radii_shell = (1, 2)


@CachedFunction
def build_shell(Nphi, Ntheta, Nr, dealias, dtype):
    c = coords.SphericalCoordinates('phi', 'theta', 'r')
    d = distributor.Distributor((c,), dtype=dtype)
    b = basis.ShellBasis(c, (Nphi, Ntheta, Nr), radii=radii_shell, dealias=(dealias, dealias, dealias), dtype=dtype)
    phi, theta, r = b.local_grids()
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
    u = field.Field(name='u', dist=d, bases=(b,), dtype=dtype)
    τu1 = field.Field(name='τu1', dist=d, bases=(b.S2_basis(),), dtype=dtype)
    τu2 = field.Field(name='τu2', dist=d, bases=(b.S2_basis(),), dtype=dtype)
    F = field.Field(name='a', dist=d, bases=(b,), dtype=dtype)
    F['g'] = 6
    # Problem
    Lap = lambda A: operators.Laplacian(A, c)
    Lift = lambda A, n: operators.Lift(A, b, n)
    problem = problems.LBVP([u, τu1, τu2])
    problem.add_equation((Lap(u) + Lift(τu1,-1) + Lift(τu2,-2), F))
    problem.add_equation((u(r=r0), 0))
    problem.add_equation((u(r=r1), 0))
    # Solver
    solver = solvers.LinearBoundaryValueSolver(problem)
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
    Lap = lambda A: operators.Laplacian(A, c)
    Lift = lambda A, n: operators.Lift(A, b, n)
    problem = problems.LBVP([u, τu1, τu2])
    problem.add_equation((Lap(u) + Lift(τu1,-1) + Lift(τu2,-2), F))
    problem.add_equation((u(r=r0), 0))
    problem.add_equation((u(r=r1), 0))
    # Solver
    solver = solvers.LinearBoundaryValueSolver(problem)
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
    u = field.Field(name='u', dist=d, bases=(b,), dtype=dtype)
    τu1 = field.Field(name='τu1', dist=d, bases=(b.S2_basis(),), dtype=dtype)
    τu2 = field.Field(name='τu2', dist=d, bases=(b.S2_basis(),), dtype=dtype)
    ncc = field.Field(name='ncc', dist=d, bases=(b.radial_basis,), dtype=dtype)
    F = field.Field(name='a', dist=d, bases=(b,), dtype=dtype)
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
    Lap = lambda A: operators.Laplacian(A, c)
    Lift = lambda A, n: operators.Lift(A, b, n)
    problem = problems.LBVP([u, τu1, τu2])
    problem.add_equation((ncc*Lap(u) + Lift(τu1,-1) + Lift(τu2,-2), F))
    problem.add_equation((u(r=r0), 0))
    problem.add_equation((u(r=r1), 0))
    # Solver
    solver = solvers.LinearBoundaryValueSolver(problem)
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
    u = field.Field(name='u', dist=d, bases=(b,), dtype=dtype)
    v = field.Field(name='v', dist=d, bases=(b,), dtype=dtype)
    τu1 = field.Field(name='τu1', dist=d, bases=(b.S2_basis(),), dtype=dtype)
    τu2 = field.Field(name='τu2', dist=d, bases=(b.S2_basis(),), dtype=dtype)
    ncc = field.Field(name='ncc', dist=d, bases=(b.meridional_basis,), dtype=dtype)
    v['g'] = x**2 + z**2
    ncc['g'] = z**2
    # Problem
    Lap = lambda A: operators.Laplacian(A, c)
    Lift = lambda A, n: operators.Lift(A, b, n)
    problem = problems.LBVP([u, τu1, τu2])
    problem.add_equation((ncc*Lap(u) + Lift(τu1,-1) + Lift(τu2,-2), ncc*Lap(v)))
    problem.add_equation((u(r=r0), v(r=r0)))
    problem.add_equation((u(r=r1), v(r=r1)))
    # Solver
    solver = solvers.LinearBoundaryValueSolver(problem)
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
    u = field.Field(name='u', dist=d, bases=(b,), dtype=dtype)
    v = field.Field(name='v', dist=d, bases=(b,), dtype=dtype)
    τu1 = field.Field(name='τu1', dist=d, bases=(b.S2_basis(),), dtype=dtype)
    τu2 = field.Field(name='τu2', dist=d, bases=(b.S2_basis(),), dtype=dtype)
    ncc_m = field.Field(name='ncc', dist=d, bases=(b.meridional_basis,), dtype=dtype)
    ncc_r = field.Field(name='ncc', dist=d, bases=(b.radial_basis,), dtype=dtype)
    v['g'] = x**2 + z**2
    ncc_m['g'] = z**2
    ncc_r['g'] = r**2
    # Problem
    Lap = lambda A: operators.Laplacian(A, c)
    Lift = lambda A, n: operators.Lift(A, b, n)
    problem = problems.LBVP([u, τu1, τu2])
    problem.add_equation((ncc_m*Lap(u) + ncc_r*Lap(u) + Lift(τu1,-1) + Lift(τu2,-2), ncc_m*Lap(v) + ncc_r*Lap(v)))
    problem.add_equation((u(r=r0), v(r=r0)))
    problem.add_equation((u(r=r1), v(r=r1)))
    # Solver
    solver = solvers.LinearBoundaryValueSolver(problem)
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
    Lap = lambda A: operators.Laplacian(A, c)
    Lift = lambda A, n: operators.Lift(A, b, n)
    problem = problems.LBVP([u, τu1, τu2])
    problem.add_equation((ncc_r*Lap(u) + ncc_m*Lap(u) + Lift(τu1,-1) + Lift(τu2,-2), ncc_r*Lap(v) + ncc_m*Lap(v)))
    problem.add_equation((u(r=r0), v(r=r0)))
    problem.add_equation((u(r=r1), v(r=r1)))
    # Solver
    solver = solvers.LinearBoundaryValueSolver(problem)
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
    u = field.Field(name='u', dist=d, bases=(b,), dtype=dtype)
    τu = field.Field(name='u', dist=d, bases=(b.S2_basis(),), dtype=dtype)
    ncc = field.Field(name='ncc', dist=d, bases=(b.radial_basis,), dtype=dtype)
    F = field.Field(name='a', dist=d, bases=(b,), dtype=dtype)
    # Test Parameters
    R = radius_ball
    u_true = np.cos(np.pi/2*(r/R)**2)
    g = - np.pi/2/R**2 * (4*np.pi/2*(r/R)**2*np.cos(np.pi/2*(r/R)**2) + 6*np.sin(np.pi/2*(r/R)**2))
    ncc['g'] = u_true
    F['g'] = u_true*g
    ncc.change_scales(ncc_scale)
    ncc['g'] # force transform
    # Problem
    Lap = lambda A: operators.Laplacian(A, c)
    Lift = lambda A: operators.Lift(A, b, -1)
    problem = problems.LBVP([u, τu])
    problem.add_equation((ncc*Lap(u) + Lift(τu), F))
    problem.add_equation((u(r=radius_ball), 0))
    # Solver
    solver = solvers.LinearBoundaryValueSolver(problem)
    solver.solve()
    # Check solution
    assert np.allclose(u['g'], u_true)
