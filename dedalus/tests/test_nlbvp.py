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

@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('Nmax', [31])
@pytest.mark.parametrize('Lmax', [0,3])
def test_heat_ball_nlbvp(Nmax, Lmax, dtype, tolerance=1e-10):
    # Bases
    c = coords.SphericalCoordinates('phi', 'theta', 'r')
    d = distributor.Distributor((c,))
    if Lmax > 0:
        nm = 2*(Lmax+1)
    else:
        nm = 1
    b = basis.BallBasis(c, (nm,Lmax+1, Nmax+1), radius=1, dtype=dtype)
    phi, theta, r = b.local_grids((1, 1, 1))
    # Fields
    u = field.Field(name='u', dist=d, bases=(b,), dtype=dtype)
    τu = field.Field(name='τu', dist=d, bases=(b.S2_basis(),), dtype=dtype)
    F = field.Field(name='F', dist=d, bases=(b,), dtype=dtype)
    F['g'] = 6
    # Problem
    Lap = lambda A: operators.Laplacian(A, c)
    LiftTau = lambda A: operators.LiftTau(A, b, -1)
    problem = problems.NLBVP([u, τu])
    problem.add_equation((Lap(u) + LiftTau(τu), F))
    problem.add_equation((u(r=1), 0))
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
@pytest.mark.parametrize('Nmax', [31])
@pytest.mark.parametrize('Lmax', [0,3])
def test_lane_emden(Nmax, Lmax, dtype, m=1.5, n_rho=3, radius=1,
                    ncc_cutoff = 1e-10, tolerance = 1e-10):
    c = coords.SphericalCoordinates('phi', 'theta', 'r')
    d = distributor.Distributor((c,))
    if Lmax > 0:
        nm = 2*(Lmax+1)
    else:
        nm = 1
    b = basis.BallBasis(c, (nm,Lmax+1, Nmax+1), radius=radius, dtype=dtype)
    phi, theta, r = b.local_grids((1, 1, 1))
    # Fields
    T = field.Field(dist=d, bases=(b,), dtype=dtype, name='T')
    τ = field.Field(dist=d, bases=(b.S2_basis(),), dtype=dtype, name='τ')
    C = field.Field(dist=d, dtype=dtype, name='C')
    # Parameters and operators
    lap = lambda A: operators.Laplacian(A, c)
    Pow = lambda A,n: operators.Power(A,n)
    LiftTau = lambda A: operators.LiftTau(A, b, -1)
    # from poisson:
    #      lap(phi) = -C1*rho
    # from HS balance:
    #      grad(phi) ~ grad(T)
    # therefore:
    #      lap(T) = -C2*rho = -C3*T**n
    problem = problems.NLBVP([T,τ,C], ncc_cutoff=ncc_cutoff)
    problem.add_equation((lap(T) + LiftTau(τ), -C*Pow(T,m)))
    problem.add_equation((T(r=radius), 0))
    # Solver
    solver = solvers.NonlinearBoundaryValueSolver(problem)
    # Initial guess
    T.require_scales(1)
    T['g'] = np.cos(np.pi/2 * r)*0.9
    C['g'] = 2
    # Iterations
    def error(perts):
        return np.sum([np.sum(np.abs(pert['c'])) for pert in perts])
    err = np.inf
    while err > tolerance:
        solver.newton_iteration()
        err = error(solver.perturbations)
    # TO-DO: implement correct assertion test against literature lane-emden
    # [1]: http://en.wikipedia.org/wiki/Lane–Emden_equation
    # [2]: J. P. Boyd, "Chebyshev spectral methods and the Lane-Emden problem,"
    #     Numerical Mathematics Theory (2011).
    # Note that C here should be R**2 from dedalus2.
    return T
