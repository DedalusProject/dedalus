
import pytest
import numpy as np
import dedalus.public as d3
from dedalus.tools.cache import CachedFunction


def xfail_param(param, reason, run=True):
    return pytest.param(param, marks=pytest.mark.xfail(reason=reason, run=run))

radius_ball = 1.5
radii_shell = (0.6, 1.7)
ncc_cutoff = 1e-6

@CachedFunction
def build_ball(Nphi, Ntheta, Nr, alpha, dealias, dtype):
    c = d3.SphericalCoordinates('phi', 'theta', 'r')
    d = d3.Distributor(c, dtype=dtype)
    b = d3.BallBasis(c, (Nphi, Ntheta, Nr), alpha=alpha, radius=radius_ball, dealias=(dealias, dealias, dealias), dtype=dtype)
    return c, d, b

@CachedFunction
def build_shell(Nphi, Ntheta, Nr, alpha, dealias, dtype):
    c = d3.SphericalCoordinates('phi', 'theta', 'r')
    d = d3.Distributor(c, dtype=dtype)
    b = d3.ShellBasis(c, (Nphi, Ntheta, Nr), alpha=(-0.5+alpha,-0.5+alpha), radii=radii_shell, dealias=(dealias, dealias, dealias), dtype=dtype)
    return c, d, b

Nphi_range = [8]
Ntheta_range = [4]
Nr_range = [16]
basis_range = [build_shell]
alpha_range = [0]
#k_range = [0, 1, 2, 3]
k_range = [0, 3]
dealias_range = [2]
dtype_range = [np.float64, np.complex128]

def k_out_comp(basis, k_ncc, k_arg, f, g):
    if isinstance(basis, d3.BallBasis):
        rank_f = len(f.tensorsig)
        rank_g = len(g.tensorsig)
        return (max(k_ncc, k_arg) + rank_f + rank_g + 1) // 2
    else:
        return k_ncc + k_arg

def norm2(field):
    rank = len(field.tensorsig)
    if rank == 0:
        return field**2
    if rank == 1:
        return field @ field
    if rank == 2:
        return d3.Trace(field @ field)


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('basis', basis_range)
@pytest.mark.parametrize('alpha', alpha_range)
@pytest.mark.parametrize('k_ncc', k_range)
@pytest.mark.parametrize('k_arg', k_range)
@pytest.mark.parametrize('rank_ncc', [0, 1])
@pytest.mark.parametrize('rank_arg', [0, 1])
@pytest.mark.parametrize('ell_coupling', [False, xfail_param(True, "Not working")])
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_radial_multiply(Nphi, Ntheta, Nr, basis, alpha, k_ncc, k_arg, rank_ncc, rank_arg, ell_coupling, dealias, dtype):
    c, d, b = basis(Nphi, Ntheta, Nr, alpha, dealias=dealias, dtype=dtype)
    # Fields
    b_ncc = b.radial_basis.clone_with(k=k_ncc)
    b_arg = b.clone_with(k=k_arg)
    f = d.Field(bases=b_ncc, tensorsig=(c,)*rank_ncc)
    g = d.Field(bases=b_arg, tensorsig=(c,)*rank_arg)
    f.fill_random('g')
    g.fill_random('g')
    # Dummy problem to build subproblems with correct coupling/dependence
    problem = d3.LBVP([g])
    problem.add_equation((norm2(f)*g, 0))
    solver = problem.build_solver(matrix_coupling=(False, ell_coupling, True))
    # NCC operators
    w0 = f * g
    w1 = w0.reinitialize(ncc=True, ncc_vars=[g])
    w1.store_ncc_matrices([g], solver.subproblems, ncc_cutoff=ncc_cutoff)
    w0 = w0.evaluate()
    w1 = w1.evaluate_as_ncc()
    assert np.allclose(w0['c'], w1['c'])


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('basis', basis_range)
@pytest.mark.parametrize('alpha', alpha_range)
@pytest.mark.parametrize('k_ncc', k_range)
@pytest.mark.parametrize('k_arg', k_range)
@pytest.mark.parametrize('rank_ncc', [1, 2])
@pytest.mark.parametrize('rank_arg', [1, 2])
@pytest.mark.parametrize('ell_coupling', [False, xfail_param(True, "Not working")])
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_radial_dot(Nphi, Ntheta, Nr, basis, alpha, k_ncc, k_arg, rank_ncc, rank_arg, ell_coupling, dealias, dtype):
    c, d, b = basis(Nphi, Ntheta, Nr, alpha, dealias=dealias, dtype=dtype)
    # Fields
    b_ncc = b.radial_basis.clone_with(k=k_ncc)
    b_arg = b.clone_with(k=k_arg)
    f = d.Field(bases=b_ncc, tensorsig=(c,)*rank_ncc)
    g = d.Field(bases=b_arg, tensorsig=(c,)*rank_arg)
    f.fill_random('g')
    g.fill_random('g')
    # Dummy problem to build subproblems with correct coupling/dependence
    problem = d3.LBVP([g])
    problem.add_equation((norm2(f)*g, 0))
    solver = problem.build_solver(matrix_coupling=(False, ell_coupling, True))
    # NCC operators
    w0 = f @ g
    w1 = w0.reinitialize(ncc=True, ncc_vars=[g])
    w1.store_ncc_matrices([g], solver.subproblems, ncc_cutoff=ncc_cutoff)
    w0 = w0.evaluate()
    w1 = w1.evaluate_as_ncc()
    assert np.allclose(w0['c'], w1['c'])


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('basis', basis_range)
@pytest.mark.parametrize('alpha', alpha_range)
@pytest.mark.parametrize('k_ncc', k_range)
@pytest.mark.parametrize('k_arg', k_range)
@pytest.mark.parametrize('ell_coupling', [False, xfail_param(True, "Not working")])
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_radial_cross(Nphi, Ntheta, Nr, basis, alpha, k_ncc, k_arg, ell_coupling, dealias, dtype):
    c, d, b = basis(Nphi, Ntheta, Nr, alpha, dealias=dealias, dtype=dtype)
    # Fields
    b_ncc = b.radial_basis.clone_with(k=k_ncc)
    b_arg = b.clone_with(k=k_arg)
    f = d.Field(bases=b_ncc, tensorsig=(c,))
    g = d.Field(bases=b_arg, tensorsig=(c,))
    f.fill_random('g')
    g.fill_random('g')
    # Dummy problem to build subproblems with correct coupling/dependence
    problem = d3.LBVP([g])
    problem.add_equation((norm2(f)*g, 0))
    solver = problem.build_solver(matrix_coupling=(False, ell_coupling, True))
    # NCC operators
    w0 = d3.CrossProduct(f, g)
    w1 = w0.reinitialize(ncc=True, ncc_vars=[g])
    w1.store_ncc_matrices([g], solver.subproblems, ncc_cutoff=ncc_cutoff)
    w0 = w0.evaluate()
    w1 = w1.evaluate_as_ncc()
    assert np.allclose(w0['c'], w1['c'])


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('basis', [build_shell, xfail_param(build_ball, reason="Ball meridional NCCs not implemented.", run=False)])
@pytest.mark.parametrize('alpha', alpha_range)
@pytest.mark.parametrize('k_ncc', k_range)
@pytest.mark.parametrize('k_arg', k_range)
@pytest.mark.parametrize('rank_ncc', [0, 1])
@pytest.mark.parametrize('rank_arg', [0, 1])
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', [xfail_param(np.float64, reason="Real meridional NCCs not implemented.", run=False), np.complex128])
def test_meridional_multiply(Nphi, Ntheta, Nr, basis, alpha, k_ncc, k_arg, rank_ncc, rank_arg, dealias, dtype):
    c, d, b = basis(Nphi, Ntheta, Nr, alpha, dealias=dealias, dtype=dtype)
    # Fields
    b_ncc = b.meridional_basis.clone_with(k=k_ncc)
    b_arg = b.clone_with(k=k_arg)
    f = d.Field(bases=b_ncc, tensorsig=(c,)*rank_ncc)
    g = d.Field(bases=b_arg, tensorsig=(c,)*rank_arg)
    f.fill_random('g')
    g.fill_random('g')
    # Dummy problem to build subproblems with correct coupling/dependence
    problem = d3.LBVP([g])
    problem.add_equation((norm2(f)*g, 0))
    solver = problem.build_solver()
    # NCC operators
    w0 = f * g
    w1 = w0.reinitialize(ncc=True, ncc_vars=[g])
    w1.store_ncc_matrices([g], solver.subproblems, ncc_cutoff=ncc_cutoff)
    w0 = w0.evaluate()
    w1 = w1.evaluate_as_ncc()
    assert np.allclose(w0['c'], w1['c'])


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('basis', [build_shell, xfail_param(build_ball, reason="Ball meridional NCCs not implemented.", run=False)])
@pytest.mark.parametrize('alpha', alpha_range)
@pytest.mark.parametrize('k_ncc', k_range)
@pytest.mark.parametrize('k_arg', k_range)
@pytest.mark.parametrize('rank_ncc', [1, 2])
@pytest.mark.parametrize('rank_arg', [1, 2])
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', [xfail_param(np.float64, reason="Real meridional NCCs not implemented.", run=False), np.complex128])
def test_meridional_dot(Nphi, Ntheta, Nr, basis, alpha, k_ncc, k_arg, rank_ncc, rank_arg, dealias, dtype):
    c, d, b = basis(Nphi, Ntheta, Nr, alpha, dealias=dealias, dtype=dtype)
    # Fields
    b_ncc = b.meridional_basis.clone_with(k=k_ncc)
    b_arg = b.clone_with(k=k_arg)
    f = d.Field(bases=b_ncc, tensorsig=(c,)*rank_ncc)
    g = d.Field(bases=b_arg, tensorsig=(c,)*rank_arg)
    f.fill_random('g')
    g.fill_random('g')
    # Dummy problem to build subproblems with correct coupling/dependence
    problem = d3.LBVP([g])
    problem.add_equation((norm2(f)*g, 0))
    solver = problem.build_solver()
    # NCC operators
    w0 = f @ g
    w1 = w0.reinitialize(ncc=True, ncc_vars=[g])
    w1.store_ncc_matrices([g], solver.subproblems, ncc_cutoff=ncc_cutoff)
    w0 = w0.evaluate()
    w1 = w1.evaluate_as_ncc()
    assert np.allclose(w0['c'], w1['c'])


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('basis', [build_shell, xfail_param(build_ball, reason="Ball meridional NCCs not implemented.", run=False)])
@pytest.mark.parametrize('alpha', alpha_range)
@pytest.mark.parametrize('k_ncc', k_range)
@pytest.mark.parametrize('k_arg', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', [xfail_param(np.float64, reason="Real meridional NCCs not implemented.", run=False), np.complex128])
def test_meridional_cross(Nphi, Ntheta, Nr, basis, alpha, k_ncc, k_arg, dealias, dtype):
    c, d, b = basis(Nphi, Ntheta, Nr, alpha, dealias=dealias, dtype=dtype)
    # Fields
    b_ncc = b.meridional_basis.clone_with(k=k_ncc)
    b_arg = b.clone_with(k=k_arg)
    f = d.Field(bases=b_ncc, tensorsig=(c,))
    g = d.Field(bases=b_arg, tensorsig=(c,))
    f.fill_random('g')
    g.fill_random('g')
    # Dummy problem to build subproblems with correct coupling/dependence
    problem = d3.LBVP([g])
    problem.add_equation((norm2(f)*g, 0))
    solver = problem.build_solver()
    # NCC operators
    w0 = d3.CrossProduct(f, g)
    w1 = w0.reinitialize(ncc=True, ncc_vars=[g])
    w1.store_ncc_matrices([g], solver.subproblems, ncc_cutoff=ncc_cutoff)
    w0 = w0.evaluate()
    w1 = w1.evaluate_as_ncc()
    assert np.allclose(w0['c'], w1['c'])

