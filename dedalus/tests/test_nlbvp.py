
import pytest
import numpy as np
import functools
from dedalus.core import coords, distributor, basis, field, operators, problems, solvers, timesteppers, arithmetic
from dedalus.tools.cache import CachedFunction


@pytest.mark.parametrize('Nx', [16])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('dealias', [1, 1.5])
@pytest.mark.parametrize('basis_class', [basis.ChebyshevT, basis.Legendre])
def test_sin_nlbvp(Nx, dtype, dealias, basis_class):
    ncc_cutoff = 1e-10
    tolerance = 1e-10
    # Bases
    c = coords.Coordinate('x')
    d = distributor.Distributor((c,))
    xb = basis_class(c, size=Nx, bounds=(0, 1), dealias=dealias)
    x = xb.local_grid(1)
    # Fields
    u = field.Field(name='u', dist=d, bases=(xb,), dtype=dtype)
    τ = field.Field(name='τ', dist=d, dtype=dtype)
    xb1 = xb.clone_with(a=xb.a+1, b=xb.b+1)
    P = field.Field(name='P', dist=d, bases=(xb1,), dtype=dtype)
    P['c'][-1] = 1
    # Problem
    dx = lambda A: operators.Differentiate(A, c)
    problem = problems.NLBVP([u, τ])
    problem.add_equation((dx(u) + τ*P, np.sqrt(1-u*u)))
    problem.add_equation((u(x=0), 0))
    # Solver
    solver = solvers.NonlinearBoundaryValueSolver(problem, ncc_cutoff=ncc_cutoff)
    # Initial guess
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
    u.change_scales(1)
    assert np.allclose(u['g'], u_true)


@pytest.mark.parametrize('Nr', [16])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('dealias', [1, 1.5])
def test_heat_ball_nlbvp(Nr, dtype, dealias):
    radius = 2
    ncc_cutoff = 1e-10
    tolerance = 1e-10
    # Bases
    c = coords.SphericalCoordinates('phi', 'theta', 'r')
    d = distributor.Distributor((c,))
    b = basis.BallBasis(c, (1, 1, Nr), radius=radius, dtype=dtype, dealias=dealias)
    bs = b.S2_basis(radius=radius)
    phi, theta, r = b.local_grids((1, 1, 1))
    # Fields
    u = field.Field(name='u', dist=d, bases=(b,), dtype=dtype)
    τ = field.Field(name='τ', dist=d, bases=(bs,), dtype=dtype)
    F = field.Field(name='F', dist=d, bases=(b,), dtype=dtype) # todo: make this constant
    F['g'] = 6
    # Problem
    Lap = lambda A: operators.Laplacian(A, c)
    Lift = lambda A: operators.Lift(A, b, -1)
    problem = problems.NLBVP([u, τ])
    problem.add_equation((Lap(u) + Lift(τ), F))
    problem.add_equation((u(r=radius), 0))
    # Solver
    solver = solvers.NonlinearBoundaryValueSolver(problem, ncc_cutoff=ncc_cutoff)
    # Initial guess
    u['g'] = 1
    # Iterations
    def error(perts):
        return np.sum([np.sum(np.abs(pert['c'])) for pert in perts])
    err = np.inf
    while err > tolerance:
        solver.newton_iteration()
        err = error(solver.perturbations)
    u_true = r**2 - radius**2
    u.change_scales(1)
    assert np.allclose(u['g'], u_true)


@pytest.mark.parametrize('Nr', [32])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('dealias', [1, 1.5])
def test_lane_emden_floating_amp(Nr, dtype, dealias):
    n = 3.0
    ncc_cutoff = 1e-10
    tolerance = 1e-10
    # Bases
    c = coords.SphericalCoordinates('phi', 'theta', 'r')
    d = distributor.Distributor((c,))
    b = basis.BallBasis(c, (1, 1, Nr), radius=1, dtype=dtype, dealias=dealias)
    bs = b.S2_basis(radius=1)
    phi, theta, r = b.local_grids((1, 1, 1))
    # Fields
    f = field.Field(dist=d, bases=(b,), dtype=dtype, name='f')
    τ = field.Field(dist=d, bases=(bs,), dtype=dtype, name='τ')
    # Problem
    lap = lambda A: operators.Laplacian(A, c)
    Lift = lambda A: operators.Lift(A, b, -1)
    problem = problems.NLBVP([f, τ])
    problem.add_equation((lap(f) + Lift(τ), -f**n))
    problem.add_equation((f(r=1), 0))
    # Solver
    solver = solvers.NonlinearBoundaryValueSolver(problem, ncc_cutoff=ncc_cutoff)
    # Initial guess
    f['g'] = 5 * np.cos(np.pi/2 * r)**2
    # Iterations
    def error(perts):
        return np.sum([np.sum(np.abs(pert['c'])) for pert in perts])
    err = np.inf
    while err > tolerance and solver.iteration < 10:
        solver.newton_iteration()
        err = error(solver.perturbations)
    f0 = f(r=0).evaluate()['g'].ravel()
    R = f0 ** ((n - 1) / 2)
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


@pytest.mark.xfail(reason="Doesnt work, don't know why", run=False)
@pytest.mark.parametrize('Nr', [32])
@pytest.mark.parametrize('dtype', [np.complex128,
    pytest.param(np.float64, marks=pytest.mark.xfail(reason="floats and constants still dont play nice"))])
#@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('dealias', [1, 1.5])
def test_lane_emden_floating_R(Nr, dtype, dealias):
    n = 3.0
    ncc_cutoff = 1e-10
    tolerance = 1e-10
    # Bases
    c = coords.SphericalCoordinates('phi', 'theta', 'r')
    d = distributor.Distributor((c,))
    b = basis.BallBasis(c, (1, 1, Nr), radius=1, dtype=dtype, dealias=dealias)
    bs = b.S2_basis(radius=1)
    bs0 = b.S2_basis(radius=0)
    phi, theta, r = b.local_grids((1, 1, 1))
    # Fields
    f = field.Field(dist=d, bases=(b,), dtype=dtype, name='f')
    R = field.Field(dist=d, dtype=dtype, name='R')
    τ = field.Field(dist=d, bases=(bs,), dtype=dtype, name='τ')
    one = field.Field(dist=d, bases=(bs0,), dtype=dtype)
    one['g'] = 1
    # Problem
    lap = lambda A: operators.Laplacian(A, c)
    Lift = lambda A: operators.Lift(A, b, -1)
    problem = problems.NLBVP([f, R, τ])
    problem.add_equation((lap(f) + Lift(τ), - R**2 * f**n))
    problem.add_equation((f(r=0), one))
    problem.add_equation((f(r=1), 0))
    # Solver
    solver = solvers.NonlinearBoundaryValueSolver(problem, ncc_cutoff=ncc_cutoff)
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


@pytest.mark.xfail(reason="First order Lane Emden failing for unkown reason.", run=False)
@pytest.mark.parametrize('Nr', [64])
@pytest.mark.parametrize('dtype', [np.complex128,
    pytest.param(np.float64, marks=pytest.mark.xfail(reason="ell = 0 matrices with float are singular?"))])
#@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('dealias', [1, 1.5])
def test_lane_emden_first_order(Nr, dtype, dealias):
    n = 3.0
    ncc_cutoff = 1e-10
    tolerance = 1e-10
    # Bases
    c = coords.SphericalCoordinates('phi', 'theta', 'r')
    d = distributor.Distributor((c,))
    b = basis.BallBasis(c, (1, 1, Nr), radius=1, dtype=dtype, dealias=dealias)
    br = b.radial_basis
    phi, theta, r = b.local_grids((1, 1, 1))
    # Fields
    p = field.Field(dist=d, bases=(br,), dtype=dtype, name='p')
    ρ = field.Field(dist=d, bases=(br,), dtype=dtype, name='ρ')
    φ = field.Field(dist=d, bases=(br,), dtype=dtype, name='φ')
    τ = field.Field(dist=d, dtype=dtype, name='τ')
    τ2 = field.Field(dist=d, dtype=dtype, name='τ2')
    rf = field.Field(dist=d, bases=(br,), tensorsig=(c,), dtype=dtype, name='r')
    rf['g'][2] = r
    # Problem
    lap = lambda A: operators.Laplacian(A, c)
    grad = lambda A: operators.Gradient(A, c)
    div = lambda A: operators.Divergence(A)
    Lift = lambda A: operators.Lift(A, br, -1)
    dot = lambda A, B: arithmetic.DotProduct(A, B)
    rdr = lambda A: dot(rf, grad(A))
    problem = problems.NLBVP([p, ρ, φ, τ, τ2])
    problem.add_equation((p, ρ**(1+1/n)))
    problem.add_equation((lap(φ) + Lift(τ), ρ))
    problem.add_equation((φ(r=1), 0))

    # This works
    # problem.add_equation((-φ, (n+1) * ρ**(1/n)))
    # problem.add_equation((τ2, 0))

    # Also works when near correct solution
    # problem.add_equation((-φ**n, (n+1)**n * ρ))
    # problem.add_equation((τ2, 0))

    # Doesn't work well
    problem.add_equation((rdr(p) + Lift(τ2), -ρ*rdr(φ)))
    problem.add_equation((p(r=1), 0))

    # Also doesn't work well
    # problem.add_equation((lap(p) + Lift(τ2), -div(ρ*grad(φ))))
    # problem.add_equation((p(r=1), 0))

    # Solver
    solver = solvers.NonlinearBoundaryValueSolver(problem, ncc_cutoff=ncc_cutoff)
    # Initial guess
    #φ['g'] = - 55 *  np.cos(np.pi/2 * r)
    #φ['g'] = - 50 *  (1 - r) * (1 + r)
    φ['g'] = np.array([[[-5.49184941e+01-2.10742982e-38j,
         -5.41628923e+01-5.32970546e-38j,
         -5.29461420e+01-5.04522267e-38j,
         -5.13265949e+01-2.97780743e-38j,
         -4.93761552e+01-2.61880274e-38j,
         -4.71730013e+01-3.43967627e-38j,
         -4.47948939e+01-3.04186813e-38j,
         -4.23139098e+01-1.79113018e-38j,
         -3.97929639e+01-1.43996160e-38j,
         -3.72840673e+01-1.63817277e-38j,
         -3.48280092e+01-9.99537738e-39j,
         -3.24550394e+01-3.17721047e-40j,
         -3.01861437e+01-5.81373831e-42j,
         -2.80345785e+01-3.10228717e-39j,
         -2.60074301e+01+1.28594534e-39j,
         -2.41070531e+01+7.60758754e-39j,
         -2.23323155e+01+7.97312927e-39j,
         -2.06796271e+01+5.81693170e-39j,
         -1.91437566e+01+6.56252079e-39j,
         -1.77184618e+01+1.10908840e-38j,
         -1.63969611e+01+1.53872437e-38j,
         -1.51722763e+01+1.39129399e-38j,
         -1.40374741e+01+9.43669477e-39j,
         -1.29858304e+01+9.30920868e-39j,
         -1.20109359e+01+1.23602737e-38j,
         -1.11067589e+01+1.41710050e-38j,
         -1.02676773e+01+1.60717088e-38j,
         -9.48848876e+00+1.77178302e-38j,
         -8.76440610e+00+1.48647842e-38j,
         -8.09104289e+00+1.01146628e-38j,
         -7.46439287e+00+1.11622279e-38j,
         -6.88080593e+00+1.66263627e-38j,
         -6.33696251e+00+1.79488585e-38j,
         -5.82984767e+00+1.46579657e-38j,
         -5.35672567e+00+1.34603496e-38j,
         -4.91511565e+00+1.50574167e-38j,
         -4.50276874e+00+1.54259944e-38j,
         -4.11764669e+00+1.52307339e-38j,
         -3.75790229e+00+1.61072571e-38j,
         -3.42186130e+00+1.52968997e-38j,
         -3.10800611e+00+1.33188351e-38j,
         -2.81496085e+00+1.46531686e-38j,
         -2.54147800e+00+1.65381249e-38j,
         -2.28642630e+00+1.48467159e-38j,
         -2.04877987e+00+1.49987605e-38j,
         -1.82760852e+00+1.83704612e-38j,
         -1.62206896e+00+1.68020109e-38j,
         -1.43139709e+00+1.17510410e-38j,
         -1.25490103e+00+1.25754442e-38j,
         -1.09195489e+00+1.71504952e-38j,
         -9.41993349e-01+1.76972495e-38j,
         -8.04506695e-01+1.53368883e-38j,
         -6.79036552e-01+1.46402303e-38j,
         -5.65172039e-01+1.54974386e-38j,
         -4.62546411e-01+1.60642465e-38j,
         -3.70834105e-01+1.59758147e-38j,
         -2.89748169e-01+1.49361039e-38j,
         -2.19038018e-01+1.32679253e-38j,
         -1.58487515e-01+1.40338570e-38j,
         -1.07913330e-01+1.83256446e-38j,
         -6.71635483e-02+2.05950514e-38j,
         -3.61164846e-02+1.65676365e-38j,
         -1.46794435e-02+1.02374473e-38j,
         -2.78432418e-03+6.69851727e-39j]]])
    ρ['g'] = (-φ['g']/(n+1))**n
    p['g'] = ρ['g']**(1+1/n)
    # Iterations
    def error(perts):
        return np.sum([np.sum(np.abs(pert['c'])) for pert in perts])
    err = np.inf
    while err > tolerance and solver.iteration < 20:
        solver.newton_iteration()
        err = error(solver.perturbations)
        φcen = φ(r=0).evaluate()['g'][0,0,0]
        R = -φcen  / (n+1)**(3/2)
        print(solver.iteration, φcen, R, err)
        dH = solver.subproblems[0].dH_min
        print('%.2e' %np.linalg.cond(dH.A))
    if err > tolerance:
        raise ValueError("Did not converge")
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


