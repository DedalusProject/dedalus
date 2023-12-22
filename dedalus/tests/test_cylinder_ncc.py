"""Tests for NCCs that depend only on radius in the cylinder."""

import pytest
import numpy as np
from dedalus.core import coords, distributor, basis, field, operators, problems, solvers, arithmetic
from dedalus.core import future
from dedalus.tools.array import apply_matrix
from dedalus.tools.cache import CachedFunction


dot = arithmetic.DotProduct
length = 1.88
radius_disk = 1.5
radii_annulus = (0.5, 1.1)

@CachedFunction
def build_periodic_cylinder(Nz, Nphi, Nr, alpha, k, dealias, dtype):
    cz = coords.Coordinate('z')
    cp = coords.PolarCoordinates('phi', 'r')
    c = coords.DirectProduct(cz, cp)
    d = distributor.Distributor(c, dtype=dtype)
    bz = basis.Fourier(cz, Nz, bounds=(0, length), dealias=dealias, dtype=dtype)
    bp = basis.DiskBasis(cp, (Nphi, Nr), dtype=dtype, radius=radius_disk, alpha=alpha, k=k, dealias=dealias)
    z, phi, r = d.local_grids(bz, bp, scales=dealias)
    x, y = cp.cartesian(phi, r)
    return c, d, (bz, bp), z, phi, r, x, y

@CachedFunction
def build_periodic_cylindrical_annulus(Nz, Nphi, Nr, alpha, k, dealias, dtype):
    cz = coords.Coordinate('z')
    cp = coords.PolarCoordinates('phi', 'r')
    c = coords.DirectProduct(cz, cp)
    d = distributor.Distributor(c, dtype=dtype)
    bz = basis.Fourier(cz, Nz, bounds=(0, length), dealias=dealias, dtype=dtype)
    bp = basis.AnnulusBasis(cp, (Nphi, Nr), dtype=dtype, radii=radii_annulus, alpha=alpha, k=k, dealias=dealias)
    z, phi, r = d.local_grids(bz, bp, scales=dealias)
    x, y = cp.cartesian(phi, r)
    return c, d, (bz, bp), z, phi, r, x, y


Nz_range = [8]
Nphi_range = [16]
Nr_range = [16]
alpha_range = [0]
k_range = [0, 1]
dealias_range = [1, 3/2]
basis_range = [build_periodic_cylinder, build_periodic_cylindrical_annulus]
dtype_range = [np.float64, np.complex128]


@pytest.mark.parametrize('Nz', Nz_range)
@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('alpha', alpha_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('basis', basis_range)
@pytest.mark.parametrize('ncc_first', [True, False])
def test_scalar_prod_scalar(Nz, Nphi, Nr, alpha, k, dealias, dtype, basis, ncc_first):
    c, d, b, z, phi, r, x, y = basis(Nz, Nphi, Nr, alpha, k, dealias, dtype)
    f = d.Field(bases=b[1].radial_basis)
    g = d.Field(bases=b)
    f.preset_scales(dealias)
    g.preset_scales(dealias)
    f['g'] = r**4
    g['g'] = 3*x**2 + 2*y + x*np.sin(4*np.pi*z/length)
    vars = [g]
    if ncc_first:
        w0 = f * g
    else:
        w0 = g * f
    w1 = w0.reinitialize(ncc=True, ncc_vars=vars)
    problem = problems.LBVP(vars)
    problem.add_equation((w1, 0))
    solver = solvers.LinearBoundaryValueSolver(problem, matsolver='SuperluNaturalSpsolve', matrix_coupling=[False,False,True])
    w1.store_ncc_matrices(vars, solver.subproblems)
    w0 = w0.evaluate()
    w0.change_scales(1)
    w1 = w1.evaluate_as_ncc()
    assert np.allclose(w0['g'], w1['g'])


@pytest.mark.parametrize('Nz', Nz_range)
@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('alpha', alpha_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('basis', basis_range)
@pytest.mark.parametrize('ncc_first', [True, False])
def test_scalar_prod_vector(Nz, Nphi, Nr, alpha, k, dealias, dtype, basis, ncc_first):
    c, d, b, z, phi, r, x, y = basis(Nz, Nphi, Nr, alpha, k, dealias, dtype)
    f = d.Field(bases=b[1].radial_basis)
    g = d.Field(bases=b)
    f.preset_scales(dealias)
    g.preset_scales(dealias)
    f['g'] = r**4
    g['g'] = 3*x**2 + 2*y + x*np.sin(4*np.pi*z/length)
    u = operators.Gradient(g, c).evaluate()
    vars = [u]
    if ncc_first:
        w0 = f * u
    else:
        w0 = u * f
    w1 = w0.reinitialize(ncc=True, ncc_vars=vars)
    problem = problems.LBVP(vars)
    problem.add_equation((w1, 0))
    solver = solvers.LinearBoundaryValueSolver(problem, matsolver='SuperluNaturalSpsolve', matrix_coupling=[False,False,True])
    w1.store_ncc_matrices(vars, solver.subproblems)
    w0 = w0.evaluate()
    w0.change_scales(1)
    w1 = w1.evaluate_as_ncc()
    assert np.allclose(w0['g'], w1['g'])


@pytest.mark.parametrize('Nz', Nz_range)
@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('alpha', alpha_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('basis', basis_range)
@pytest.mark.parametrize('ncc_first', [True, False])
def test_scalar_prod_tensor(Nz, Nphi, Nr, alpha, k, dealias, dtype, basis, ncc_first):
    c, d, b, z, phi, r, x, y = basis(Nz, Nphi, Nr, alpha, k, dealias, dtype)
    f = d.Field(bases=b[1].radial_basis)
    g = d.Field(bases=b)
    f.preset_scales(dealias)
    g.preset_scales(dealias)
    f['g'] = r**4
    g['g'] = 3*x**2 + 2*y + x*np.sin(4*np.pi*z/length)
    T = operators.Gradient(operators.Gradient(g, c), c).evaluate()
    vars = [T]
    if ncc_first:
        w0 = f * T
    else:
        w0 = T * f
    w1 = w0.reinitialize(ncc=True, ncc_vars=vars)
    problem = problems.LBVP(vars)
    problem.add_equation((w1, 0))
    solver = solvers.LinearBoundaryValueSolver(problem, matsolver='SuperluNaturalSpsolve', matrix_coupling=[False,False,True])
    w1.store_ncc_matrices(vars, solver.subproblems)
    w0 = w0.evaluate()
    w0.change_scales(1)
    w1 = w1.evaluate_as_ncc()
    assert np.allclose(w0['g'], w1['g'])


@pytest.mark.parametrize('Nz', Nz_range)
@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('alpha', alpha_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('basis', basis_range)
@pytest.mark.parametrize('ncc_first', [True, False])
def test_vector_prod_scalar(Nz, Nphi, Nr, alpha, k, dealias, dtype, basis, ncc_first):
    c, d, b, z, phi, r, x, y = basis(Nz, Nphi, Nr, alpha, k, dealias, dtype)
    f = d.Field(bases=b[1].radial_basis)
    g = d.Field(bases=b)
    f.preset_scales(dealias)
    g.preset_scales(dealias)
    f['g'] = r**4
    g['g'] = 3*x**2 + 2*y + x*np.sin(4*np.pi*z/length)
    u = operators.Gradient(f, c).evaluate()
    vars = [g]
    if ncc_first:
        w0 = u * g
    else:
        w0 = g * u
    w1 = w0.reinitialize(ncc=True, ncc_vars=vars)
    problem = problems.LBVP(vars)
    problem.add_equation((dot(u,u)*g, 0))
    solver = solvers.LinearBoundaryValueSolver(problem, matsolver='SuperluNaturalSpsolve', matrix_coupling=[False,False,True])
    w1.store_ncc_matrices(vars, solver.subproblems)
    w0 = w0.evaluate()
    w0.change_scales(1)
    w1 = w1.evaluate_as_ncc()
    assert np.allclose(w0['g'], w1['g'])


@pytest.mark.parametrize('Nz', Nz_range)
@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('alpha', alpha_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('basis', basis_range)
@pytest.mark.parametrize('ncc_first', [True, False])
def test_vector_prod_vector(Nz, Nphi, Nr, alpha, k, dealias, dtype, basis, ncc_first):
    c, d, b, z, phi, r, x, y = basis(Nz, Nphi, Nr, alpha, k, dealias, dtype)
    f = d.Field(bases=b[1].radial_basis)
    g = d.Field(bases=b)
    f.preset_scales(dealias)
    g.preset_scales(dealias)
    f['g'] = r**4
    g['g'] = 3*x**2 + 2*y + x*np.sin(4*np.pi*z/length)
    u = operators.Gradient(f, c).evaluate()
    v = operators.Gradient(g, c).evaluate()
    vars = [v]
    if ncc_first:
        w0 = u * v
    else:
        w0 = v * u
    w1 = w0.reinitialize(ncc=True, ncc_vars=vars)
    problem = problems.LBVP(vars)
    problem.add_equation((dot(u,u)*v, 0))
    solver = solvers.LinearBoundaryValueSolver(problem, matsolver='SuperluNaturalSpsolve', matrix_coupling=[False,False,True])
    w1.store_ncc_matrices(vars, solver.subproblems)
    w0 = w0.evaluate()
    w0.change_scales(1)
    w1 = w1.evaluate_as_ncc()
    assert np.allclose(w0['g'], w1['g'])


@pytest.mark.parametrize('Nz', Nz_range)
@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('alpha', alpha_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('basis', basis_range)
@pytest.mark.parametrize('ncc_first', [True, False])
def test_vector_dot_vector(Nz, Nphi, Nr, alpha, k, dealias, dtype, basis, ncc_first):
    c, d, b, z, phi, r, x, y = basis(Nz, Nphi, Nr, alpha, k, dealias, dtype)
    f = d.Field(bases=b[1].radial_basis)
    g = d.Field(bases=b)
    f.preset_scales(dealias)
    g.preset_scales(dealias)
    f['g'] = r**4
    g['g'] = 3*x**2 + 2*y + x*np.sin(4*np.pi*z/length)
    u = operators.Gradient(f, c).evaluate()
    v = operators.Gradient(g, c).evaluate()
    vars = [v]
    if ncc_first:
        w0 = dot(u, v)
    else:
        w0 = dot(v, u)
    w1 = w0.reinitialize(ncc=True, ncc_vars=vars)
    problem = problems.LBVP(vars)
    problem.add_equation((dot(u,u)*v, 0))
    solver = solvers.LinearBoundaryValueSolver(problem, matsolver='SuperluNaturalSpsolve', matrix_coupling=[False,False,True])
    w1.store_ncc_matrices(vars, solver.subproblems)
    w0 = w0.evaluate()
    w0.change_scales(1)
    w1 = w1.evaluate_as_ncc()
    assert np.allclose(w0['g'], w1['g'])


@pytest.mark.parametrize('Nz', Nz_range)
@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('alpha', alpha_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('basis', basis_range)
@pytest.mark.parametrize('ncc_first', [True, False])
def test_vector_dot_tensor(Nz, Nphi, Nr, alpha, k, dealias, dtype, basis, ncc_first):
    c, d, b, z, phi, r, x, y = basis(Nz, Nphi, Nr, alpha, k, dealias, dtype)
    f = d.Field(bases=b[1].radial_basis)
    g = d.Field(bases=b)
    f.preset_scales(dealias)
    g.preset_scales(dealias)
    f['g'] = r**4
    g['g'] = 3*x**2 + 2*y + x*np.sin(4*np.pi*z/length)
    u = operators.Gradient(f, c).evaluate()
    T = operators.Gradient(operators.Gradient(g, c), c).evaluate()
    vars = [T]
    if ncc_first:
        w0 = dot(u, T)
    else:
        w0 = dot(T, u)
    w1 = w0.reinitialize(ncc=True, ncc_vars=vars)
    problem = problems.LBVP(vars)
    problem.add_equation((dot(u,u)*T, 0))
    solver = solvers.LinearBoundaryValueSolver(problem, matsolver='SuperluNaturalSpsolve', matrix_coupling=[False,False,True])
    w1.store_ncc_matrices(vars, solver.subproblems)
    w0 = w0.evaluate()
    w0.change_scales(1)
    w1 = w1.evaluate_as_ncc()
    assert np.allclose(w0['g'], w1['g'])


@pytest.mark.parametrize('Nz', Nz_range)
@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('alpha', alpha_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('basis', basis_range)
@pytest.mark.parametrize('ncc_first', [True, False])
def test_tensor_prod_scalar(Nz, Nphi, Nr, alpha, k, dealias, dtype, basis, ncc_first):
    c, d, b, z, phi, r, x, y = basis(Nz, Nphi, Nr, alpha, k, dealias, dtype)
    f = d.Field(bases=b[1].radial_basis)
    g = d.Field(bases=b)
    f.preset_scales(dealias)
    g.preset_scales(dealias)
    f['g'] = r**4
    g['g'] = 3*x**2 + 2*y + x*np.sin(4*np.pi*z/length)
    T = operators.Gradient(operators.Gradient(f, c), c).evaluate()
    vars = [g]
    if ncc_first:
        U0 = T * g
    else:
        U0 = g * T
    U1 = U0.reinitialize(ncc=True, ncc_vars=vars)
    problem = problems.LBVP(vars)
    problem.add_equation((f*g, 0))
    solver = solvers.LinearBoundaryValueSolver(problem, matsolver='SuperluNaturalSpsolve', matrix_coupling=[False,False,True])
    U1.store_ncc_matrices(vars, solver.subproblems)
    U0 = U0.evaluate()
    U0.change_scales(1)
    U1 = U1.evaluate_as_ncc()
    assert np.allclose(U0['g'], U1['g'])


@pytest.mark.parametrize('Nz', Nz_range)
@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('alpha', alpha_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('basis', basis_range)
@pytest.mark.parametrize('ncc_first', [True, False])
def test_tensor_dot_vector(Nz, Nphi, Nr, alpha, k, dealias, dtype, basis, ncc_first):
    c, d, b, z, phi, r, x, y = basis(Nz, Nphi, Nr, alpha, k, dealias, dtype)
    f = d.Field(bases=b[1].radial_basis)
    g = d.Field(bases=b)
    f.preset_scales(dealias)
    g.preset_scales(dealias)
    f['g'] = r**4
    g['g'] = 3*x**2 + 2*y + x*np.sin(4*np.pi*z/length)
    T = operators.Gradient(operators.Gradient(f, c), c).evaluate()
    u = operators.Gradient(g, c).evaluate()
    vars = [u]
    if ncc_first:
        w0 = dot(T, u)
    else:
        w0 = dot(u, T)
    w1 = w0.reinitialize(ncc=True, ncc_vars=vars)
    problem = problems.LBVP(vars)
    problem.add_equation((f*u, 0))
    solver = solvers.LinearBoundaryValueSolver(problem, matsolver='SuperluNaturalSpsolve', matrix_coupling=[False,False,True])
    w1.store_ncc_matrices(vars, solver.subproblems)
    w0 = w0.evaluate()
    w0.change_scales(1)
    w1 = w1.evaluate_as_ncc()
    assert np.allclose(w0['g'], w1['g'])


@pytest.mark.parametrize('Nz', Nz_range)
@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('alpha', alpha_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('basis', basis_range)
@pytest.mark.parametrize('ncc_first', [True, False])
def test_tensor_dot_tensor(Nz, Nphi, Nr, alpha, k, dealias, dtype, basis, ncc_first):
    c, d, b, z, phi, r, x, y = basis(Nz, Nphi, Nr, alpha, k, dealias, dtype)
    f = d.Field(bases=b[1].radial_basis)
    g = d.Field(bases=b)
    f.preset_scales(dealias)
    g.preset_scales(dealias)
    f['g'] = r**4
    g['g'] = 3*x**2 + 2*y + x*np.sin(4*np.pi*z/length)
    U = operators.Gradient(operators.Gradient(f, c), c).evaluate()
    T = operators.Gradient(operators.Gradient(g, c), c).evaluate()
    vars = [T]
    if ncc_first:
        W0 = dot(T, U)
    else:
        W0 = dot(U, T)
    W1 = W0.reinitialize(ncc=True, ncc_vars=vars)
    problem = problems.LBVP(vars)
    problem.add_equation((f*T, 0))
    solver = solvers.LinearBoundaryValueSolver(problem, matsolver='SuperluNaturalSpsolve', matrix_coupling=[False,False,True])
    W1.store_ncc_matrices(vars, solver.subproblems)
    W0 = W0.evaluate()
    W0.change_scales(1)
    W1 = W1.evaluate_as_ncc()
    assert np.allclose(W0['g'], W1['g'])

