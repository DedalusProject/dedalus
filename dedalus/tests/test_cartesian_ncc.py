"""Test Fourier and Jacobi NCCs."""
# TODO: add tests inovlving constant NCCs/operands

import pytest
import numpy as np
import dedalus.public as d3
from dedalus.tools.cache import CachedMethod


def xfail_param(param, reason, run=True):
    return pytest.param(param, marks=pytest.mark.xfail(reason=reason, run=run))


@CachedMethod
def build_fourier(N, bounds, dealias, dtype):
    c = d3.Coordinate('x')
    d = d3.Distributor(c, dtype=dtype)
    b = d3.Fourier(c, size=N, bounds=bounds, dealias=dealias, dtype=dtype)
    return c, d, b


@CachedMethod
def build_jacobi(N, a0, b0, bounds, dealias, dtype):
    c = d3.Coordinate('x')
    d = d3.Distributor(c, dtype=dtype)
    b = d3.Jacobi(c, size=N, a=a0, b=b0, bounds=bounds, dealias=dealias)
    return c, d, b


@pytest.mark.parametrize('N', [16])
@pytest.mark.parametrize('a0, b0', [(-1/2, -1/2), (0, 0)])
@pytest.mark.parametrize('k_ncc', [0, 1])
@pytest.mark.parametrize('k_arg', [0, 1])
@pytest.mark.parametrize('dealias', [3/2])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_eval_jacobi_ncc(N, a0, b0, k_ncc, k_arg, dealias, dtype):
    """Compares f(x) * g(x) calculated pseudospectrally and as an NCC operator."""
    c, d, b = build_jacobi(N, a0, b0, (0, 1), dealias, dtype)
    b_ncc = b.clone_with(a=a0+k_ncc, b=b0+k_ncc)
    b_arg = b.clone_with(a=a0+k_arg, b=b0+k_arg)
    f = d.Field(bases=b_ncc)
    g = d.Field(bases=b_arg)
    f.fill_random('g')
    g.fill_random('g')
    vars = [g]
    w0 = f * g
    w1 = w0.reinitialize(ncc=True, ncc_vars=vars)
    problem = d3.LBVP(vars)
    problem.add_equation((w1, 0))
    solver = problem.build_solver()
    w1.store_ncc_matrices(vars, solver.subproblems)
    w0 = w0.evaluate()
    w1 = w1.evaluate_as_ncc()
    # Remove last 2*k_arg coeffs which have dealias-before-conversion errors
    w2 = (w1 - w0).evaluate()
    assert np.allclose(w2['c'][:-2*k_arg], 0)


@pytest.mark.parametrize('N', [24])
@pytest.mark.parametrize('dealias', [3/2])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_eval_fourier_ncc(N, dealias, dtype):
    """Compares f(x) * g(x) calculated as an NCC operator and pseudospectrally."""
    c, d, b = build_fourier(N, (0, 1), dealias, dtype)
    f = d.Field(bases=b)
    g = d.Field(bases=b)
    f.fill_random('g')
    g.fill_random('g')
    vars = [g]
    w0 = f * g
    w1 = w0.reinitialize(ncc=True, ncc_vars=vars)
    problem = d3.LBVP(vars)
    problem.add_equation((w1, 0))
    solver = problem.build_solver()
    w1.store_ncc_matrices(vars, solver.subproblems)
    w0 = w0.evaluate()
    w1 = w1.evaluate_as_ncc()
    w0.change_scales(1)
    w1.change_scales(1)
    assert np.allclose(w0['g'], w1['g'])


@pytest.mark.parametrize('N', [16])
@pytest.mark.parametrize('a0, b0', [(-1/2, -1/2)])
@pytest.mark.parametrize('f_rank', [0, 1])
@pytest.mark.parametrize('g_rank', [0, 1])
@pytest.mark.parametrize('dealias', [3/2])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_eval_fourier_jacobi_ncc(N, a0, b0, f_rank, g_rank, dealias, dtype):
    """Compares f(x) * g(x) calculated pseudospectrally and as an NCC operator."""
    c = d3.CartesianCoordinates('x', 'y')
    d = d3.Distributor(c, dtype=dtype)
    xb = d3.Fourier(c['x'], size=N, bounds=(0, 1), dealias=dealias, dtype=dtype)
    yb = d3.Jacobi(c['y'], size=N, bounds=(0, 1), a=a0, b=b0, dealias=dealias)
    s = d.Field(bases=(xb, yb))
    f = d.TensorField((c,)*f_rank, bases=(xb, yb))
    g = d.TensorField((c,)*g_rank, bases=(xb, yb))
    s.fill_random('g')
    f.fill_random('g')
    g.fill_random('g')
    vars = [g]
    w0 = f * g
    w1 = w0.reinitialize(ncc=True, ncc_vars=vars)
    problem = d3.LBVP(vars)
    problem.add_equation((s*g, 0))
    solver = problem.build_solver()
    w1.store_ncc_matrices(vars, solver.subproblems)
    w0 = w0.evaluate()
    w1 = w1.evaluate_as_ncc()
    w0.change_scales(1)
    w1.change_scales(1)
    assert np.allclose(w0['g'], w1['g'])


@pytest.mark.parametrize('N', [16])
@pytest.mark.parametrize('a0, b0', [(-1/2, -1/2), (0, 0)])
@pytest.mark.parametrize('k_ncc', [0, 1])
@pytest.mark.parametrize('k_arg', [0, xfail_param(1, "dealias before truncation", run=True)])
@pytest.mark.parametrize('dealias', [3/2])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_solve_jacobi_ncc(N, a0, b0, k_ncc, k_arg, dealias, dtype):
    """Solve f(x) * u(x) = f(x) * g(x)."""
    c, d, b = build_jacobi(N, a0, b0, (0, 1), dealias, dtype)
    b_ncc = b.clone_with(a=a0+k_ncc, b=b0+k_ncc)
    b_arg = b.clone_with(a=a0+k_arg, b=b0+k_arg)
    f = d.Field(bases=b_ncc)
    g = d.Field(bases=b_arg)
    u = d.Field(bases=b_arg)
    f.fill_random('g')
    g.fill_random('g')
    problem = d3.LBVP([u])
    problem.add_equation((f*u, f*g))
    solver = problem.build_solver()
    solver.solve()
    assert np.allclose(u['c'], g['c'])


@pytest.mark.parametrize('N', [16])
@pytest.mark.parametrize('dealias', [3/2])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_solve_fourier_ncc(N, dealias, dtype):
    """Solve f(x) * u(x) = f(x) * g(x)."""
    c, d, b = build_fourier(N, (0, 1), dealias, dtype)
    f = d.Field(bases=b)
    g = d.Field(bases=b)
    u = d.Field(bases=b)
    f.fill_random('g')
    g.fill_random('g')
    # Todo: fix for Nyquist
    if dtype == np.complex128:
        f.low_pass_filter(scales=0.5)
        g.low_pass_filter(scales=0.5)
    problem = d3.LBVP([u])
    problem.add_equation((f*u, f*g))
    solver = problem.build_solver()
    solver.solve()
    u.change_scales(1)
    g.change_scales(1)
    assert np.allclose(u['g'], g['g'])

