"""Test simple EVPs."""
# TODO: add 2d tests - FC?

import pytest
import numpy as np
import dedalus.public as d3
import scipy.special as spec
from .ball_diffusion_analytical_eigenvalues import eigenvalues as analytic_eigenvalues


@pytest.mark.parametrize('basis', [d3.RealFourier, d3.ComplexFourier])
@pytest.mark.parametrize('N', [32])
@pytest.mark.parametrize('dtype', [np.complex128])
def test_laplace_fourier(basis, N, dtype):
    """Test eigenvalues of Laplacian in Fourier bases."""
    # Bases
    c = d3.Coordinate('x')
    d = d3.Distributor(c, dtype=dtype)
    b = basis(c, size=N, bounds=(0, 2*np.pi))
    # Fields
    u = d.Field(bases=b)
    s = d.Field()
    # Problem
    dx = lambda A: d3.Differentiate(A, c)
    problem = d3.EVP([u], s, namespace=locals())
    problem.add_equation("s*u + dx(dx(u)) = 0")
    # Solver
    solver = problem.build_solver(matrix_coupling=[True])
    solver.solve_dense(solver.subproblems[0])
    # Check solution
    k = b.wavenumbers
    if basis is d3.RealFourier:
        k = k[1:]  # Drop one k=0 for msin
    assert np.allclose(solver.eigenvalues, k**2)


@pytest.mark.parametrize('N', [32])
@pytest.mark.parametrize('a, b', [(-1/2, -1/2), (0, 0)])
@pytest.mark.parametrize('dtype', [np.complex128])
@pytest.mark.parametrize('sparse', [True, False])
def test_laplace_jacobi(N, a, b, dtype, sparse):
    """Test eigenvalues of Laplacian in Jacobi bases."""
    # Bases
    c = d3.Coordinate('x')
    d = d3.Distributor(c, dtype=dtype)
    b = d3.Jacobi(c, size=N, a=a, b=b, bounds=(0, np.pi))
    # Fields
    u = d.Field(bases=b)
    s = d.Field()
    t1 = d.Field()
    t2 = d.Field()
    # Problem
    dx = lambda A: d3.Differentiate(A, c)
    lift = lambda A, n: d3.Lift(A, b.derivative_basis(2), n)
    from numpy import pi
    problem = d3.EVP([u, t1, t2], s, namespace=locals())
    problem.add_equation("s*u + dx(dx(u)) + lift(t1,-1) + lift(t2,-2) = 0")
    problem.add_equation("u(x=0) = 0")
    problem.add_equation("u(x=pi) = 0")
    # Solver
    solver = problem.build_solver()
    Nmodes = 4
    if sparse:
        solver.solve_sparse(solver.subproblems[0], N=Nmodes, target=1.1)
    else:
        solver.solve_dense(solver.subproblems[0])
    # Check eigenvalues
    k = 1 + np.arange(Nmodes)
    assert np.allclose(np.sort(solver.eigenvalues)[:Nmodes], k**2)


@pytest.mark.parametrize('N', [32])
@pytest.mark.parametrize('a, b', [(-1/2, -1/2), (0, 0)])
@pytest.mark.parametrize('dtype', [np.complex128])
def test_laplace_jacobi_first_order(N, a, b, dtype):
    """Test eigenfunctions of Laplacian in Jacobi bases."""
    # Bases
    c = d3.Coordinate('x')
    d = d3.Distributor(c, dtype=dtype)
    b = d3.Jacobi(c, size=N, a=a, b=b, bounds=(0, np.pi))
    x = d.local_grid(b, scale=1)
    # Fields
    u = d.Field(bases=b)
    s = d.Field()
    t1 = d.Field()
    t2 = d.Field()
    # Problem
    dx = lambda A: d3.Differentiate(A, c)
    lift = lambda A, n: d3.Lift(A, b.derivative_basis(1), n)
    ux = dx(u) + lift(t1, -1)
    from numpy import pi
    problem = d3.EVP([u, t1, t2], s, namespace=locals())
    problem.add_equation("s*u + dx(ux) + lift(t2,-1) = 0")
    problem.add_equation("u(x=0) = 0")
    problem.add_equation("u(x=pi) = 0")
    # Solver
    solver = problem.build_solver()
    solver.solve_dense(solver.subproblems[0])
    i_sort = np.argsort(solver.eigenvalues)
    solver.eigenvalues = solver.eigenvalues[i_sort]
    solver.eigenvectors = solver.eigenvectors[:,i_sort]
    # Check first eigenfunction
    solver.set_state(0, solver.subproblems[0].subsystems[0])
    eigenfunction = u['g'] / u['g'][0]
    sol = np.sin(x) / np.sin(x[0])
    assert np.allclose(eigenfunction, sol)


@pytest.mark.parametrize('Nphi', [10])
@pytest.mark.parametrize('Nr', [16])
@pytest.mark.parametrize('radius', [1.5])
@pytest.mark.parametrize('alpha', [0, 1])
@pytest.mark.parametrize('dtype', [np.complex128])
@pytest.mark.parametrize('m', [0, 1, 2, 4])
def test_disk_bessel_zeros(Nphi, Nr, radius, alpha, dtype, m):
    """Test eigenvalues of Bessel equation in disk bases."""
    # Bases
    c = d3.PolarCoordinates('phi', 'r')
    d = d3.Distributor(c, dtype=dtype)
    b = d3.DiskBasis(c, (Nphi, Nr), radius=radius, alpha=alpha, dtype=dtype)
    # Fields
    u = d.Field(bases=b)
    tau = d.Field(bases=b.edge)
    s = d.Field()
    # Problem
    lift = lambda A: d3.Lift(A, b, -1)
    problem = d3.EVP([u, tau], s, namespace=locals())
    problem.add_equation("s*u + lap(u) + lift(tau) = 0")
    problem.add_equation("u(r=radius) = 0")
    # Solver
    solver = problem.build_solver()
    # TODO: clean this up with group selection interface
    for sp in solver.subproblems:
        if sp.group[0] == m:
            break
    else:
        raise ValueError(f"Could not find subproblem with m = {m}")
    solver.solve_dense(sp)
    # Compare eigenvalues
    n_compare = 5
    selected_eigenvalues = np.sort(solver.eigenvalues)[:n_compare]
    analytic_eigenvalues = (spec.jn_zeros(m, n_compare) / radius)**2
    assert np.allclose(selected_eigenvalues, analytic_eigenvalues)


@pytest.mark.parametrize('Nphi', [8])
@pytest.mark.parametrize('Ntheta', [4])
@pytest.mark.parametrize('Nr', [32])
@pytest.mark.parametrize('radius', [1.5])
@pytest.mark.parametrize('alpha', [0, 1])
@pytest.mark.parametrize('dtype', [np.complex128])
@pytest.mark.parametrize('ell', [0, 1, 2, 3])
def test_ball_bessel_eigenfunction(Nphi, Ntheta, Nr, radius, alpha, dtype, ell):
    """Test eigenfunctions of Bessel equation in ball bases."""
    # Bases
    c = d3.SphericalCoordinates('phi', 'theta', 'r')
    d = d3.Distributor(c, dtype=dtype)
    b = d3.BallBasis(c, (Nphi, Ntheta, Nr), radius=radius, alpha=alpha, dtype=dtype)
    phi, theta, r = d.local_grids(b, scales=(1, 1, 1))
    # Fields
    u = d.Field(bases=b)
    tau = d.Field(bases=b.surface)
    s = d.Field()
    # Problem
    lift = lambda A: d3.Lift(A, b, -1)
    problem = d3.EVP([u, tau], s, namespace=locals())
    problem.add_equation("s*u + lap(u) + lift(tau) = 0")
    problem.add_equation("u(r=radius) = 0")
    # Solver
    solver = problem.build_solver()
    # TODO: clean this up with group selection interface
    for sp in solver.subproblems:
        if sp.group[1] == ell:
            break
    else:
        raise ValueError(f"Could not find subproblem with ell = {ell}")
    solver.solve_dense(sp)
    # Compare eigenfunction
    i_sort = np.argsort(solver.eigenvalues)
    solver.eigenvalues = solver.eigenvalues[i_sort]
    solver.eigenvectors = solver.eigenvectors[:,i_sort]
    for Neig in [0, 1, 5, 10]:
        solver.set_state(Neig, sp.subsystems[0]) # m = 0 mode
        u.change_layout(d.layouts[1])
        local_m, local_ell, local_n = u.layout.local_group_arrays(u.domain, u.scales)
        radial_eigenfunction = u.data[(local_m == 0)*(local_ell == ell)]
        i_max = np.argmax(np.abs(radial_eigenfunction))
        radial_eigenfunction /= radial_eigenfunction[i_max]
        k = np.sqrt(solver.eigenvalues[Neig])
        sol = spec.jv(ell+1/2, k*r) / np.sqrt(k*r)
        sol = sol.ravel()
        sol /= sol[i_max]
        assert np.allclose(radial_eigenfunction, sol)


@pytest.mark.parametrize('Nphi', [8])
@pytest.mark.parametrize('Ntheta', [4])
@pytest.mark.parametrize('Nr', [32])
@pytest.mark.parametrize('radius', [1.5])
@pytest.mark.parametrize('alpha', [0, 1])
@pytest.mark.parametrize('dtype', [np.complex128])
@pytest.mark.parametrize('ell', [1, 2, 3])
@pytest.mark.parametrize('bc', ['no-slip', 'stress-free', 'potential', 'conducting', 'pseudo'])
def test_ball_diffusion(Nphi, Ntheta, Nr, radius, alpha, dtype, ell, bc):
    """Test eigenvalues of vector diffusion equation in ball bases with various BCs."""
    # Bases
    c = d3.SphericalCoordinates('phi', 'theta', 'r')
    d = d3.Distributor(c, dtype=dtype)
    b = d3.BallBasis(c, (Nphi, Ntheta, Nr), radius=radius, alpha=alpha, dtype=dtype)
    # Fields
    φ = d.Field(bases=b)
    A = d.VectorField(c, bases=b)
    tA = d.VectorField(c, bases=b.surface)
    λ = d.Field()
    # Problem
    lift = lambda A: d3.Lift(A, b, -1)
    problem = d3.EVP([φ, A, tA], λ, namespace=locals())
    problem.add_equation("div(A) = 0")
    problem.add_equation("-λ*A + grad(φ) - lap(A) + lift(tA) = 0")
    if bc == 'no-slip':
        problem.add_equation("A(r=radius) = 0")
    elif bc == 'stress-free':
        problem.namespace['E'] = (d3.grad(A) + d3.grad(A).T) / 2
        problem.add_equation("radial(A(r=radius)) = 0")
        problem.add_equation("angular(radial(E(r=radius))) = 0")
    elif bc == 'potential':
        ell_func = lambda ell: ell+1
        problem.namespace['ell_1'] = lambda A: d3.SphericalEllProduct(A, c, ell_func)
        problem.add_equation("radial(grad(A)(r=radius)) + ell_1(A)(r=radius)/radius = 0")
    elif bc == 'conducting':
        problem.add_equation("φ(r=radius) = 0")
        problem.add_equation("angular(A(r=radius)) = 0")
    elif bc == 'pseudo':
        problem.add_equation("radial(A(r=radius)) = 0")
        problem.add_equation("angular(curl(A)(r=radius)) = 0")
    # Solver
    solver = problem.build_solver()
    # TODO: clean this up with group selection interface
    for sp in solver.subproblems:
        if sp.group[1] == ell:
            break
    else:
        raise ValueError(f"Could not find subproblem with ell = {ell}")
    solver.solve_dense(sp)
    # Compare eigenvalues
    i_sort = np.argsort(solver.eigenvalues)
    solver.eigenvalues = solver.eigenvalues[i_sort]
    λ_analytic = analytic_eigenvalues(ell, Nr, bc, r0=radius)
    if (bc == 'stress-free' and ell == 1):
        # Add null space solution
        λ_analytic = np.append(0, λ_analytic)
    assert np.allclose(solver.eigenvalues[:Nr//4], λ_analytic[:Nr//4])

