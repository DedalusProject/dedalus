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


radius_disk = 1


@CachedFunction
def build_disk(Nphi, Nr, dealias, dtype):
    c = coords.PolarCoordinates('phi', 'r')
    d = distributor.Distributor((c,))
    b = basis.DiskBasis(c, (Nphi, Nr), radius=radius_disk, dealias=(dealias, dealias), dtype=dtype)
    phi, r = b.local_grids()
    x, y = c.cartesian(phi, r)
    return c, d, b, phi, r, x, y


@pytest.mark.parametrize('dtype', [np.complex128, np.float64])
@pytest.mark.parametrize('Nphi', [4])
@pytest.mark.parametrize('Nr', [8])
def test_heat_disk(Nr, Nphi, dtype):
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
    LiftTau = lambda A: operators.LiftTau(A, b, -1)
    problem = problems.LBVP([u, τu])
    problem.add_equation((Lap(u) + LiftTau(τu), F))
    problem.add_equation((u(r=radius_disk), 0))
    # Solver
    solver = solvers.LinearBoundaryValueSolver(problem)
    solver.solve()
    # Check solution
    u_true = r**2 - 1
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
    LiftTau = lambda A: operators.LiftTau(A, b, -1)
    problem = problems.LBVP([u, τu])
    problem.add_equation((Lap(u) + LiftTau(τu), F))
    problem.add_equation((u(r=radius_ball), 0))
    # Solver
    solver = solvers.LinearBoundaryValueSolver(problem)
    solver.solve()
    # Check solution
    u_true = r**2 - 1
    assert np.allclose(u['g'], u_true)


radii_shell = (1, 2)


@CachedFunction
def build_shell(Nphi, Ntheta, Nr, dealias, dtype):
    c = coords.SphericalCoordinates('phi', 'theta', 'r')
    d = distributor.Distributor((c,))
    b = basis.SphericalShellBasis(c, (Nphi, Ntheta, Nr), radii=radii_shell, dealias=(dealias, dealias, dealias), dtype=dtype)
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
    LiftTau = lambda A, n: operators.LiftTau(A, b, n)
    problem = problems.LBVP([u, τu1, τu2])
    problem.add_equation((Lap(u) + LiftTau(τu1,-1) + LiftTau(τu2,-2), F))
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
@pytest.mark.parametrize('ncc_exponent', [0]) #,1,2,3/2])
@pytest.mark.parametrize('ncc_location', ['LHS', 'RHS'])
@pytest.mark.parametrize('ncc_scale', [1, 3/2])
def test_heat_ncc_ball(Nmax, Lmax, ncc_exponent, ncc_location, ncc_scale, dtype):
    # Bases
    dealias = 1
    c, d, b, phi, theta, r, x, y, z = build_ball(2*(Lmax+1), Lmax+1, Nmax+1, dealias=dealias, dtype=dtype)
    # Fields
    u = field.Field(name='u', dist=d, bases=(b,), dtype=dtype)
    τu = field.Field(name='u', dist=d, bases=(b.S2_basis(),), dtype=dtype)
    ncc = field.Field(name='ncc', dist=d, bases=(b.radial_basis,), dtype=dtype)
    F = field.Field(name='a', dist=d, bases=(b,), dtype=dtype)
    # Test Parameters
    F_value  = {0:6, 1:2, 2:1, 3/2:3/4}
    analytic = {0:r**2-1, 1:r-1, 2:np.log(r), 3/2:np.sqrt(r)-1}
    if ncc_location=='RHS':
        ncc['g'] = 1
        F['g'] = F_value[ncc_exponent]*r**(-ncc_exponent)
    else:
        ncc['g'] = r**ncc_exponent
        F['g'] = F_value[ncc_exponent]
    u_true = analytic[ncc_exponent]
    ncc.require_scales(ncc_scale)
    ncc['g'] # force transform
    # Problem
    Lap = lambda A: operators.Laplacian(A, c)
    LiftTau = lambda A: operators.LiftTau(A, b, -1)
    problem = problems.LBVP([u, τu])
    problem.add_equation((ncc*Lap(u) + LiftTau(τu), F))
    problem.add_equation((u(r=radius_ball), 0))
    # Solver
    solver = solvers.LinearBoundaryValueSolver(problem)
    solver.solve()
    # Check solution
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
    ncc.require_scales(ncc_scale)
    ncc['g'] # force transform
    # Problem
    Lap = lambda A: operators.Laplacian(A, c)
    LiftTau = lambda A, n: operators.LiftTau(A, b, n)
    problem = problems.LBVP([u, τu1, τu2])
    problem.add_equation((ncc*Lap(u) + LiftTau(τu1,-1) + LiftTau(τu2,-2), F))
    problem.add_equation((u(r=r0), 0))
    problem.add_equation((u(r=r1), 0))
    # Solver
    solver = solvers.LinearBoundaryValueSolver(problem)
    solver.solve()
    # Check solution
    assert np.allclose(u['g'], u_true)

radius_pi_ball = np.sqrt(np.pi/2)
@CachedFunction
def build_pi_ball(Nphi, Ntheta, Nr, dealias, dtype):
    c = coords.SphericalCoordinates('phi', 'theta', 'r')
    d = distributor.Distributor((c,))
    b = basis.BallBasis(c, (Nphi, Ntheta, Nr), radius=radius_pi_ball, dealias=(dealias, dealias, dealias), dtype=dtype)
    phi, theta, r = b.local_grids()
    x, y, z = c.cartesian(phi, theta, r)
    return c, d, b, phi, theta, r, x, y, z

@pytest.mark.parametrize('dtype', [np.complex128, np.float64])
@pytest.mark.parametrize('Nmax', [15])
@pytest.mark.parametrize('Lmax', [3])
@pytest.mark.parametrize('ncc_test', ['u'])# '1/g'])
@pytest.mark.parametrize('ncc_scale', [1, 3/2])
def test_heat_ncc_cos_ball(Nmax, Lmax, ncc_test, ncc_scale, dtype):
    # Bases
    dealias = 1
    c, d, b, phi, theta, r, x, y, z = build_pi_ball(2*(Lmax+1), Lmax+1, Nmax+1, dealias=dealias, dtype=dtype)
    # Fields
    u = field.Field(name='u', dist=d, bases=(b,), dtype=dtype)
    τu = field.Field(name='u', dist=d, bases=(b.S2_basis(),), dtype=dtype)
    ncc = field.Field(name='ncc', dist=d, bases=(b.radial_basis,), dtype=dtype)
    F = field.Field(name='a', dist=d, bases=(b,), dtype=dtype)
    # Test Parameters
    u_true = np.cos(r**2)
    g = - 4*r**2*np.cos(r**2) -6*np.sin(r**2)
    if ncc_test=='u':
        ncc['g'] = u_true
        F['g'] = u_true*g
    else:
        ncc['g'] = 1/g
        F['g'] = 1
    ncc.require_scales(ncc_scale)
    ncc['g'] # force transform
    # Problem
    Lap = lambda A: operators.Laplacian(A, c)
    LiftTau = lambda A: operators.LiftTau(A, b, -1)
    problem = problems.LBVP([u, τu])
    problem.add_equation((ncc*Lap(u) + LiftTau(τu), F))
    problem.add_equation((u(r=radius_pi_ball), 0))
    # Solver
    solver = solvers.LinearBoundaryValueSolver(problem)
    solver.solve()
    # Check solution
    assert np.allclose(u['g'], u_true)
