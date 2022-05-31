"""
Test 1D EVP.
"""

import pytest
import numpy as np
import scipy.special as spec
import functools
from dedalus.core import coords, distributor, basis, field, operators, problems, solvers, timesteppers, arithmetic
from .ball_diffusion_analytical_eigenvalues import eigenvalues as analytic_eigenvalues


@pytest.mark.parametrize('dtype', [np.complex128])
@pytest.mark.parametrize('Nx', [32])
@pytest.mark.parametrize('x_basis_class', [basis.ComplexFourier, basis.RealFourier])
def test_heat_1d_periodic(x_basis_class, Nx, dtype):
    # Bases
    c = coords.Coordinate('x')
    d = distributor.Distributor((c,))
    xb = x_basis_class(c, size=Nx, bounds=(0, 2*np.pi))
    x = xb.local_grid(1)
    # Fields
    u = field.Field(name='u', dist=d, bases=(xb,), dtype=dtype)
    a = field.Field(name='a', dist=d, dtype=dtype)
    # Problem
    dx = lambda A: operators.Differentiate(A, c)
    problem = problems.EVP([u], a)
    problem.add_equation((a*u + dx(dx(u)), 0))
    # Solver
    solver = solvers.EigenvalueSolver(problem, matrix_coupling=[True])
    solver.solve_dense(solver.subproblems[0])
    # Check solution
    k = xb.wavenumbers
    if x_basis_class is basis.RealFourier:
        k = k[1:]  # Drop one k=0 for msin
    assert np.allclose(solver.eigenvalues, k**2)


@pytest.mark.parametrize('dtype', [np.complex128])
@pytest.mark.parametrize('Nx', [32])
@pytest.mark.parametrize('x_basis_class', [basis.ChebyshevT])
@pytest.mark.parametrize('sparse', [True, False])
def test_waves_1d(x_basis_class, Nx, dtype, sparse):
    # Bases
    c = coords.Coordinate('x')
    d = distributor.Distributor((c,))
    xb = x_basis_class(c, size=Nx, bounds=(0, np.pi))
    x = xb.local_grid(1)
    # Fields
    u = field.Field(name='u', dist=d, bases=(xb,), dtype=dtype)
    a = field.Field(name='a', dist=d, dtype=dtype)
    τ1 = field.Field(name='τ1', dist=d, dtype=dtype)
    τ2 = field.Field(name='τ2', dist=d, dtype=dtype)
    # Problem
    dx = lambda A: operators.Differentiate(A, c)
    xb2 = dx(dx(u)).domain.bases[0]
    P1 = field.Field(name='P1', dist=d, bases=(xb2,), dtype=dtype)
    P2 = field.Field(name='P2', dist=d, bases=(xb2,), dtype=dtype)
    P1['c'][-1]=1
    P2['c'][-2]=1
    problem = problems.EVP([u,τ1,τ2], a)
    problem.add_equation((a*u + dx(dx(u)) + P1*τ1 + P2*τ2, 0))
    problem.add_equation((u(x=0), 0))
    problem.add_equation((u(x=np.pi), 0))
    # Solver
    solver = solvers.EigenvalueSolver(problem, matrix_coupling=[True])
    Nmodes = 4
    if sparse:
        solver.solve_sparse(solver.subproblems[0], N=Nmodes, target=1.2)
    else:
        solver.solve_dense(solver.subproblems[0])
    i_sort = np.argsort(solver.eigenvalues)
    sorted_eigenvalues = solver.eigenvalues[i_sort]
    # Check solution
    k = np.arange(Nmodes)+1
    assert np.allclose(sorted_eigenvalues[:Nmodes], k**2)


@pytest.mark.parametrize('dtype', [np.complex128])
@pytest.mark.parametrize('Nx', [32])
@pytest.mark.parametrize('x_basis_class', [basis.ChebyshevT])
def test_waves_1d_first_order(x_basis_class, Nx, dtype):
    # Bases
    c = coords.Coordinate('x')
    d = distributor.Distributor((c,))
    xb = x_basis_class(c, size=Nx, bounds=(0, np.pi))
    x = xb.local_grid(1)
    # Fields
    u = field.Field(name='u', dist=d, bases=(xb,), dtype=dtype)
    ux = field.Field(name='ux', dist=d, bases=(xb,), dtype=dtype)
    a = field.Field(name='a', dist=d, dtype=dtype)
    τ1 = field.Field(name='τ1', dist=d, dtype=dtype)
    τ2 = field.Field(name='τ2', dist=d, dtype=dtype)
    # Problem
    dx = lambda A: operators.Differentiate(A, c)
    xb1 = dx(u).domain.bases[0]
    P1 = field.Field(name='P1', dist=d, bases=(xb1,), dtype=dtype)
    P2 = field.Field(name='P2', dist=d, bases=(xb1,), dtype=dtype)
    P1['c'][-1] = 1
    P2['c'][-1] = 1
    problem = problems.EVP([u,ux,τ1,τ2], a)
    problem.add_equation((a*u + dx(ux) + P1*τ1 , 0))
    problem.add_equation((ux - dx(u) + P2*τ2, 0))
    problem.add_equation((u(x=0), 0))
    problem.add_equation((u(x=np.pi), 0))
    # Solver
    solver = solvers.EigenvalueSolver(problem, matrix_coupling=[True])
    solver.solve_dense(solver.subproblems[0])
    i_sort = np.argsort(solver.eigenvalues)
    solver.eigenvalues = solver.eigenvalues[i_sort]
    solver.eigenvectors = solver.eigenvectors[:,i_sort]
    # Check solution
    solver.set_state(0, solver.subproblems[0].subsystems[0])
    eigenfunction = u['g'] / u['g'][0]
    sol = np.sin(x) / np.sin(x[0])
    assert np.allclose(eigenfunction, sol)


@pytest.mark.parametrize('dtype', [np.complex128])
@pytest.mark.parametrize('Nphi', [10])
@pytest.mark.parametrize('Nr', [16])
@pytest.mark.parametrize('m', [0,1,2,4])
@pytest.mark.parametrize('radius', [1, 2])
def test_disk_bessel_zeros(Nphi, Nr, m, radius, dtype):
    # Bases
    c = coords.PolarCoordinates('phi', 'r')
    d = distributor.Distributor((c,))
    b = basis.DiskBasis(c, (Nphi, Nr), radius=radius, dtype=dtype)
    b_S1 = b.S1_basis()
    phi, r = b.local_grids((1, 1))
    # Fields
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    τ_f = field.Field(dist=d, bases=(b_S1,), dtype=dtype)
    k2 = field.Field(name='k2', dist=d, dtype=dtype)
    # Parameters and operators
    lap = lambda A: operators.Laplacian(A, c)
    Lift = lambda A: operators.Lift(A, b, -1)
    # Bessel equation: k^2*f + lap(f) = 0
    problem = problems.EVP([f,τ_f], k2)
    problem.add_equation((k2*f + lap(f) + Lift(τ_f), 0))
    problem.add_equation((f(r=radius), 0))
    # Solver
    solver = solvers.EigenvalueSolver(problem)
    print(solver.subproblems[0].group)
    for sp in solver.subproblems:
        if sp.group[0] == m:
            break
    else:
        raise ValueError("Could not find subproblem with m = %i" %m)
    solver.solve_dense(sp)
    # Compare eigenvalues
    n_compare = 5
    selected_eigenvalues = np.sort(solver.eigenvalues)[:n_compare]
    analytic_eigenvalues = (spec.jn_zeros(m, n_compare) / radius)**2
    assert np.allclose(selected_eigenvalues, analytic_eigenvalues)


@pytest.mark.parametrize('dtype', [np.complex128])
@pytest.mark.parametrize('Lmax', [3])
@pytest.mark.parametrize('Nmax', [31])
@pytest.mark.parametrize('Leig', [0,1,2,3])
@pytest.mark.parametrize('Neig', [0,1,5,10])
@pytest.mark.parametrize('radius', [1,1.5])
def test_ball_bessel_eigenfunction(Lmax, Nmax, Leig, Neig, radius, dtype):
    # Bases
    c = coords.SphericalCoordinates('phi', 'theta', 'r')
    d = distributor.Distributor((c,))
    b = basis.BallBasis(c, (2*(Lmax+1), Lmax+1, Nmax+1), radius=radius, dtype=dtype)
    b_S2 = b.S2_basis()
    phi, theta, r = b.local_grids((1, 1, 1))
    # Fields
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    τ_f = field.Field(dist=d, bases=(b_S2,), dtype=dtype)
    k2 = field.Field(name='k2', dist=d, dtype=dtype)
    # Parameters and operators
    lap = lambda A: operators.Laplacian(A, c)
    Lift = lambda A: operators.Lift(A, b, -1)
    # Bessel equation: k^2*f + lap(f) = 0
    problem = problems.EVP([f,τ_f], k2)
    problem.add_equation((k2*f + lap(f) + Lift(τ_f), 0))
    problem.add_equation((f(r=radius), 0))
    # Solver
    solver = solvers.EigenvalueSolver(problem)
    if not solver.subproblems[Leig].group[1] == Leig:
        raise ValueError("subproblems indexed in a strange way")
    solver.solve_dense(solver.subproblems[Leig])
    i_sort = np.argsort(solver.eigenvalues)
    solver.eigenvalues = solver.eigenvalues[i_sort]
    solver.eigenvectors = solver.eigenvectors[:,i_sort]
    solver.set_state(Neig,solver.subproblems[Leig].subsystems[0]) # m = 0 mode
    f.change_layout(d.layouts[1])
    local_m, local_ell, local_n = f.layout.local_group_arrays(f.domain, f.scales)
    radial_eigenfunction = f.data[(local_m == 0)*(local_ell == Leig)]
    i_max = np.argmax(np.abs(radial_eigenfunction))
    radial_eigenfunction /= radial_eigenfunction[i_max]
    k = np.sqrt(solver.eigenvalues[Neig])
    sol = spec.jv(Leig+1/2,k*r)/np.sqrt(k*r)
    sol = sol.ravel()
    sol /= sol[i_max]
    assert np.allclose(radial_eigenfunction, sol)


@pytest.mark.parametrize('dtype', [np.complex128])
@pytest.mark.parametrize('Lmax', [3])
@pytest.mark.parametrize('Nmax', [31])
@pytest.mark.parametrize('Leig', [1,2,3])
@pytest.mark.parametrize('radius', [1,1.5])
@pytest.mark.parametrize('bc', ['no-slip', 'stress-free', 'potential', 'conducting', 'pseudo'])
def test_ball_diffusion(Lmax, Nmax, Leig, radius, bc, dtype):
    # Bases
    c = coords.SphericalCoordinates('phi', 'theta', 'r')
    d = distributor.Distributor((c,))
    b = basis.BallBasis(c, (2*(Lmax+1), Lmax+1, Nmax+1), radius=radius, dtype=dtype)
    b_S2 = b.S2_basis()
    phi, theta, r = b.local_grids((1, 1, 1))
    # Fields
    A = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
    φ = field.Field(dist=d, bases=(b,), dtype=dtype)
    τ_A = field.Field(dist=d, bases=(b_S2,), tensorsig=(c,), dtype=dtype)
    λ = field.Field(name='λ', dist=d, dtype=dtype)
    # Parameters and operators
    div = lambda A: operators.Divergence(A)
    grad = lambda A: operators.Gradient(A, c)
    curl = lambda A: operators.Curl(A)
    lap = lambda A: operators.Laplacian(A, c)
    trans = lambda A: operators.TransposeComponents(A)
    radial = lambda A, index: operators.RadialComponent(A, index=index)
    angular = lambda A, index: operators.AngularComponent(A, index=index)
    Lift = lambda A: operators.Lift(A, b, -1)
    # Problem
    problem = problems.EVP([φ,A,τ_A], λ)
    problem.add_equation((div(A), 0))
    problem.add_equation((-λ*A + grad(φ) - lap(A) + Lift(τ_A), 0))
    if bc == 'no-slip':
        problem.add_equation((A(r=radius), 0))
    elif bc == 'stress-free':
        E = 1/2*(grad(A) + trans(grad(A)))
        problem.add_equation((radial(A(r=radius),0), 0))
        problem.add_equation((radial(angular(E(r=radius),0),1), 0))
    elif bc == 'potential':
        ell_func = lambda ell: ell+1
        ell_1 = lambda A: operators.SphericalEllProduct(A, c, ell_func)
        problem.add_equation((radial(grad(A)(r=radius),0) + ell_1(A)(r=radius)/radius, 0))
    elif bc == 'conducting':
        problem.add_equation((φ(r=radius), 0))
        problem.add_equation((angular(A(r=radius),0), 0))
    elif bc == 'pseudo':
        problem.add_equation((radial(A(r=radius),0), 0))
        problem.add_equation((angular(curl(A)(r=radius),0), 0))
    # Solver
    solver = solvers.EigenvalueSolver(problem)
    if not solver.subproblems[Leig].group[1] == Leig:
        raise ValueError("subproblems indexed in a strange way")
    solver.solve_dense(solver.subproblems[Leig])
    i_sort = np.argsort(solver.eigenvalues)
    solver.eigenvalues = solver.eigenvalues[i_sort]
    λ_analytic = analytic_eigenvalues(Leig,Nmax+1,bc, r0=radius)
    if (bc == 'stress-free' and Leig == 1):
        # add null space solution
        λ_analytic = np.append(0, λ_analytic)
    assert np.allclose(solver.eigenvalues[:Nmax//4], λ_analytic[:Nmax//4])
