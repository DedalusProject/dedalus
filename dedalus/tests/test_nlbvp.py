"""
Test 1D NLBVP.
"""
import pytest
import numpy as np
import functools
from dedalus.core import coords, distributor, basis, field, operators, problems, solvers, timesteppers, arithmetic
from dedalus.tools.cache import CachedFunction

@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('Nx', [32])
@pytest.mark.parametrize('basis_class', [basis.ChebyshevT, basis.Legendre])
def test_sin_nlbvp(basis_class, Nx, dtype, dealias=2):
    # Parameters
    ncc_cutoff = 1e-10
    tolerance = 1e-10
    c = coords.Coordinate('x')
    d = distributor.Distributor((c,))
    xb = basis_class(c, size=Nx, bounds=(0, 1), dealias=dealias)
    x = xb.local_grid(1)
    # Fields
    u = field.Field(name='u', dist=d, bases=(xb,), dtype=dtype)
    # tau
    τ = field.Field(name='τ', dist=d, dtype=dtype)
    xb1 = xb._new_a_b(xb.a+1, xb.b+1)
    P = field.Field(name='P', dist=d, bases=(xb1,), dtype=dtype)
    P['c'][-1] = 1
    # Problem
    dx = lambda A: operators.Differentiate(A, c)
    sqrt = np.sqrt
    problem = problems.NLBVP([u, τ], ncc_cutoff=ncc_cutoff)
    problem.add_equation((dx(u) + τ*P, sqrt(1-u*u)))
    problem.add_equation((u(x=0), 0))
    # Solver
    solver = solvers.NonlinearBoundaryValueSolver(problem)
    u.require_scales(1)
    u['g'] = x
    # Iterations
    def error(perts):
        return np.sum([np.sum(np.abs(pert['c'])) for pert in perts])
    err = np.inf
    while err > tolerance:
        solver.newton_iteration()
        err = error(solver.perturbations)
    # Check solution
    u_true = np.sin(x)
    u.require_scales(1)
    assert np.allclose(u['g'], u_true)

@pytest.mark.parametrize('dtype', [np.complex128,
    pytest.param(np.float64, marks=pytest.mark.xfail(reason="ell = 0 matrices with float are singular?"))])
#@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('Nmax', [7])
def test_heat_ball_nlbvp(Nmax, dtype, tolerance=1e-10):
    radius = 1
    # Bases
    c = coords.SphericalCoordinates('phi', 'theta', 'r')
    d = distributor.Distributor((c,))
    b = basis.BallBasis(c, (1, 1, Nmax+1), radius=radius, dtype=dtype)
    br = b.radial_basis
    phi, theta, r = b.local_grids((1, 1, 1))
    # Fields
    u = field.Field(name='u', dist=d, bases=(br,), dtype=dtype)
    τ = field.Field(name='τ', dist=d, dtype=dtype)
    F = field.Field(name='F', dist=d, bases=(br,), dtype=dtype)
    F['g'] = 6
    # Problem
    Lap = lambda A: operators.Laplacian(A, c)
    LiftTau = lambda A: operators.LiftTau(A, br, -1)
    problem = problems.NLBVP([u, τ])
    problem.add_equation((Lap(u) + LiftTau(τ), F))
    problem.add_equation((u(r=radius), 0))
    # Solver
    solver = solvers.NonlinearBoundaryValueSolver(problem)
    # Initial guess
    u['g'] = 1
    # Iterations
    def error(perts):
        return np.sum([np.sum(np.abs(pert['c'])) for pert in perts])
    err = np.inf
    while err > tolerance:
        solver.newton_iteration()
        err = error(solver.perturbations)
    u_true = r**2 - 1
    assert np.allclose(u['g'], u_true)

@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('Nmax', [63])
def test_lane_emden(Nmax, dtype):
    n = 3.0
    ncc_cutoff = 1e-10
    tolerance = 1e-10
    c = coords.SphericalCoordinates('phi', 'theta', 'r')
    d = distributor.Distributor((c,))
    b = basis.BallBasis(c, (1, 1, Nmax+1), radius=1, dtype=dtype)
    br = b.radial_basis
    phi, theta, r = b.local_grids((1, 1, 1))
    # Fields
    f = field.Field(dist=d, bases=(br,), dtype=dtype, name='f')
    τ = field.Field(dist=d, dtype=dtype, name='τ')
    # Parameters and operators
    lap = lambda A: operators.Laplacian(A, c)
    Pow = lambda A,n: operators.Power(A,n)
    LiftTau = lambda A: operators.LiftTau(A, br, -1)
    problem = problems.NLBVP([f, τ], ncc_cutoff=ncc_cutoff)
    problem.add_equation((lap(f) + LiftTau(τ), -Pow(f,n)))
    problem.add_equation((f(r=1), 0))
    # Solver
    solver = solvers.NonlinearBoundaryValueSolver(problem)
    # Initial guess
    f['g'] = 5 * np.cos(np.pi/2 * r)**2
    # Iterations
    def error(perts):
        return np.sum([np.sum(np.abs(pert['c'])) for pert in perts])
    err = np.inf
    while err > tolerance:
        f0 = f(r=0).evaluate()['g'][0,0,0]
        R = f0 ** ((n - 1) / 2)
        solver.newton_iteration()
        err = error(solver.perturbations)
    # Compare to reference solutions from Boyd
    R_ref = {0.0: np.sqrt(6),
            0.5: 2.752698054065,
            1.0: np.pi,
            1.5: 3.65375373621912608,
            2.0: 4.3528745959461246769735700,
            2.5: 5.355275459010779,
            3.0: 6.896848619376960375454528,
            3.25: 8.018937527,
            3.5: 9.535805344244850444,
            4.0: 14.971546348838095097611066,
            4.5: 31.836463244694285264}
    assert np.allclose(R, R_ref[n])


@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('Nmax', [63])
def test_lane_emden_floating_R(Nmax, dtype):
    n = 3.0
    ncc_cutoff = 1e-10
    tolerance = 1e-10
    c = coords.SphericalCoordinates('phi', 'theta', 'r')
    d = distributor.Distributor((c,))
    b = basis.BallBasis(c, (1, 1, Nmax+1), radius=1, dtype=dtype)
    br = b.radial_basis
    phi, theta, r = b.local_grids((1, 1, 1))
    # Fields
    f = field.Field(dist=d, bases=(br,), dtype=dtype, name='f')
    R = field.Field(dist=d, dtype=dtype, name='R')
    τ = field.Field(dist=d, dtype=dtype, name='τ')
    # Parameters and operators
    lap = lambda A: operators.Laplacian(A, c)
    Pow = lambda A,n: operators.Power(A,n)
    LiftTau = lambda A: operators.LiftTau(A, br, -1)
    problem = problems.NLBVP([f, R, τ], ncc_cutoff=ncc_cutoff)
    problem.add_equation((lap(f) + LiftTau(τ), - R**2 * Pow(f,n)))
    problem.add_equation((f(r=0), 1))
    problem.add_equation((f(r=1), 0))
    # Solver
    solver = solvers.NonlinearBoundaryValueSolver(problem)
    # Initial guess
    f['g'] = np.cos(np.pi/2 * r)**2
    R['g'] = 5
    # Iterations
    def error(perts):
        return np.sum([np.sum(np.abs(pert['c'])) for pert in perts])
    err = np.inf
    while err > tolerance:
        solver.newton_iteration()
        err = error(solver.perturbations)
    # Compare to reference solutions from Boyd
    R_ref = {0.0: np.sqrt(6),
            0.5: 2.752698054065,
            1.0: np.pi,
            1.5: 3.65375373621912608,
            2.0: 4.3528745959461246769735700,
            2.5: 5.355275459010779,
            3.0: 6.896848619376960375454528,
            3.25: 8.018937527,
            3.5: 9.535805344244850444,
            4.0: 14.971546348838095097611066,
            4.5: 31.836463244694285264}
    assert np.allclose(R['g'].ravel(), R_ref[n])

