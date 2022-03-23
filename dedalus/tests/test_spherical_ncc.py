
import pytest
import numpy as np
from dedalus.core import coords, distributor, basis, field, operators, problems, solvers, arithmetic
from dedalus.core.basis import BallBasis, ShellBasis
from dedalus.tools.cache import CachedFunction


def xfail_param(param, reason, run=True):
    return pytest.param(param, marks=pytest.mark.xfail(reason=reason, run=run))


dot = arithmetic.DotProduct
radius_ball = 1.5
radii_shell = (0.6, 1.7)

@CachedFunction
def build_ball(Nphi, Ntheta, Nr, alpha, dealias, dtype):
    c = coords.SphericalCoordinates('phi', 'theta', 'r')
    d = distributor.Distributor((c,))
    b = BallBasis(c, (Nphi, Ntheta, Nr), alpha=alpha, radius=radius_ball, dealias=(dealias, dealias, dealias), dtype=dtype)
    phi, theta, r = b.local_grids()
    x, y, z = c.cartesian(phi, theta, r)
    return c, d, b, phi, theta, r, x, y, z

@CachedFunction
def build_shell(Nphi, Ntheta, Nr, alpha, dealias, dtype):
    c = coords.SphericalCoordinates('phi', 'theta', 'r')
    d = distributor.Distributor(c, dtype=dtype)
    b = ShellBasis(c, (Nphi, Ntheta, Nr), alpha=(-0.5+alpha,-0.5+alpha), radii=radii_shell, dealias=(dealias, dealias, dealias), dtype=dtype)
    phi, theta, r = b.local_grids()
    x, y, z = c.cartesian(phi, theta, r)
    return c, d, b, phi, theta, r, x, y, z

Nphi_range = [8]
Ntheta_range = [4]
Nr_range = [16]
basis_range = [build_ball, build_shell]
alpha_range = [0]
#k_range = [0, 1, 2, 3]
k_range = [0, 3]
dealias_range = [3/2, 2]
dtype_range = [np.float64, np.complex128]

def k_out_ball(k_ncc, k_arg, f, g):
    rank_f = len(f.tensorsig)
    rank_g = len(g.tensorsig)
    return (max(k_ncc, k_arg) + rank_f + rank_g + 1) // 2

def k_out_shell(k_ncc, k_arg):
    return k_ncc + k_arg

@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('basis', [build_shell, xfail_param(build_ball, reason="Ball meridional NCCs not implemented.", run=False)])
@pytest.mark.parametrize('alpha', alpha_range)
@pytest.mark.parametrize('k_ncc', k_range)
@pytest.mark.parametrize('k_arg', k_range)
@pytest.mark.parametrize('rank_ncc', [0, 1])
@pytest.mark.parametrize('rank_arg', [0, 1])
@pytest.mark.parametrize('dealias', [2])
@pytest.mark.parametrize('dtype', [xfail_param(np.float64, reason="Real meridional NCCs not implemented.", run=False), np.complex128])
def test_meridional_multiply(Nphi, Ntheta, Nr, basis, alpha, k_ncc, k_arg, rank_ncc, rank_arg, dealias, dtype):
    c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, alpha, dealias=dealias, dtype=dtype)
    # Fields
    b_ncc = b.meridional_basis.clone_with(k=k_ncc)
    b_arg = b.clone_with(k=k_arg)
    f = d.Field(bases=b_ncc, tensorsig=(c,)*rank_ncc)
    g = d.Field(bases=b_arg, tensorsig=(c,)*rank_arg)
    f.fill_random('g')
    g.fill_random('g')
    # Dummy problem just to build subproblems
    problem = problems.LBVP([g])
    problem.add_equation((g, 0))
    solver = problem.build_solver(matrix_coupling=(False, True, True))
    # NCC operators
    w0 = f * g
    w1 = w0.reinitialize(ncc=True, ncc_vars=[g])
    w1.store_ncc_matrices([g], solver.subproblems, ncc_cutoff=1e-10)
    w0 = w0.evaluate()
    w1 = w1.evaluate_as_ncc()
    assert np.allclose(w0['c'], w1['c'])


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('alpha', alpha_range)
@pytest.mark.parametrize('k_ncc', k_range)
@pytest.mark.parametrize('k_arg', k_range)
@pytest.mark.parametrize('rank_ncc', [1, 2])
@pytest.mark.parametrize('rank_arg', [1, 2])
@pytest.mark.parametrize('dealias', [2])
@pytest.mark.parametrize('dtype', [np.complex128])
def test_shell2D_dot(Nphi, Ntheta, Nr, alpha, k_ncc, k_arg, rank_ncc, rank_arg, dealias, dtype):
    c, d, b, phi, theta, r, x, y, z = build_shell(Nphi, Ntheta, Nr, alpha, dealias=dealias, dtype=dtype)
    # Fields
    b_ncc = b.meridional_basis.clone_with(k=k_ncc)
    b_arg = b.clone_with(k=k_arg)
    f = d.Field(bases=b_ncc, tensorsig=(c,)*rank_ncc)
    g = d.Field(bases=b_arg, tensorsig=(c,)*rank_arg)
    f.fill_random('g')
    g.fill_random('g')
    # Dummy problem just to build subproblems
    problem = problems.LBVP([g])
    problem.add_equation((g, 0))
    solver = problem.build_solver(matrix_coupling=(False, True, True))
    # NCC operators
    w0 = f @ g
    w1 = w0.reinitialize(ncc=True, ncc_vars=[g])
    w1.store_ncc_matrices([g], solver.subproblems, ncc_cutoff=1e-10)
    w0 = w0.evaluate()
    w1 = w1.evaluate_as_ncc()
    assert np.allclose(w0['c'], w1['c'])


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('alpha', alpha_range)
@pytest.mark.parametrize('k_ncc', k_range)
@pytest.mark.parametrize('k_arg', k_range)
@pytest.mark.parametrize('rank_ncc', [1])
@pytest.mark.parametrize('rank_arg', [1])
@pytest.mark.parametrize('dealias', [2])
@pytest.mark.parametrize('dtype', [np.complex128])
def test_shell2D_cross(Nphi, Ntheta, Nr, alpha, k_ncc, k_arg, rank_ncc, rank_arg, dealias, dtype):
    c, d, b, phi, theta, r, x, y, z = build_shell(Nphi, Ntheta, Nr, alpha, dealias=dealias, dtype=dtype)
    # Fields
    b_ncc = b.meridional_basis.clone_with(k=k_ncc)
    b_arg = b.clone_with(k=k_arg)
    f = d.Field(bases=b_ncc, tensorsig=(c,)*rank_ncc)
    g = d.Field(bases=b_arg, tensorsig=(c,)*rank_arg)
    f.fill_random('g')
    g.fill_random('g')
    # Dummy problem just to build subproblems
    problem = problems.LBVP([g])
    problem.add_equation((g, 0))
    solver = problem.build_solver(matrix_coupling=(False, True, True))
    # NCC operators
    w0 = arithmetic.CrossProduct(f, g)
    w1 = w0.reinitialize(ncc=True, ncc_vars=[g])
    w1.store_ncc_matrices([g], solver.subproblems, ncc_cutoff=1e-10)
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
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_scalar_prod_scalar(Nphi, Ntheta, Nr, basis, alpha, k_ncc, k_arg, dealias, dtype):
    c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, alpha, dealias=dealias, dtype=dtype)
    b_ncc = b.radial_basis.clone_with(k=k_ncc)
    b_arg = b.clone_with(k=k_arg)
    f = field.Field(dist=d, bases=(b_ncc,), dtype=dtype)
    g = field.Field(dist=d, bases=(b_arg,), dtype=dtype)
    f['g'] = np.random.randn(*f['g'].shape)
    g['g'] = np.random.randn(*g['g'].shape)
    if isinstance(b, BallBasis):
        k_out = k_out_ball(k_ncc, k_arg, f, g)
    else:
        k_out = k_out_shell(k_ncc, k_arg)
    if k_out > 0 and dealias < 2:
        f['c'][:, :, -k_out:] = 0
        g['c'][:, :, -k_out:] = 0
    vars = [g]
    w0 = f * g
    w1 = w0.reinitialize(ncc=True, ncc_vars=vars)
    problem = problems.LBVP(vars)
    problem.add_equation((w1, 0))
    solver = solvers.LinearBoundaryValueSolver(problem)
    w1.store_ncc_matrices(vars, solver.subproblems)
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
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_scalar_prod_vector(Nphi, Ntheta, Nr, basis, alpha, k_ncc, k_arg, dealias, dtype):
    c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, alpha, dealias=dealias, dtype=dtype)
    b_ncc = b.radial_basis.clone_with(k=k_ncc)
    b_arg = b.clone_with(k=k_arg)
    f = field.Field(dist=d, bases=(b_ncc,), dtype=dtype)
    u = field.Field(dist=d, bases=(b_arg,), tensorsig=(c,), dtype=dtype)
    f['g'] = np.random.randn(*f['g'].shape)
    u['g'] = np.random.randn(*u['g'].shape)
    if isinstance(b, BallBasis):
        k_out = k_out_ball(k_ncc, k_arg, f, u)
    else:
        k_out = k_out_shell(k_ncc, k_arg)
    if k_out > 0 and dealias < 2:
        f['c'][:, :, -k_out:] = 0
        u['c'][:, :, :, -k_out:] = 0
    vars = [u]
    w0 = f * u
    w1 = w0.reinitialize(ncc=True, ncc_vars=vars)
    problem = problems.LBVP(vars)
    problem.add_equation((w1, 0))
    solver = solvers.LinearBoundaryValueSolver(problem)
    w1.store_ncc_matrices(vars, solver.subproblems)
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
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_scalar_prod_tensor(Nphi, Ntheta, Nr, basis, alpha, k_ncc, k_arg, dealias, dtype):
    c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, alpha, dealias=dealias, dtype=dtype)
    b_ncc = b.radial_basis.clone_with(k=k_ncc)
    b_arg = b.clone_with(k=k_arg)
    f = field.Field(dist=d, bases=(b_ncc,), dtype=dtype)
    T = field.Field(dist=d, bases=(b_arg,), tensorsig=(c,c), dtype=dtype)
    f['g'] = np.random.randn(*f['g'].shape)
    T['g'] = np.random.randn(*T['g'].shape)
    if isinstance(b, BallBasis):
        k_out = k_out_ball(k_ncc, k_arg, f, T)
    else:
        k_out = k_out_shell(k_ncc, k_arg)
    if k_out > 0 and dealias < 2:
        f['c'][:, :, -k_out:] = 0
        T['c'][:, :, :, :, -k_out:] = 0
    vars = [T]
    w0 = f * T
    w1 = w0.reinitialize(ncc=True, ncc_vars=vars)
    problem = problems.LBVP(vars)
    problem.add_equation((w1, 0))
    solver = solvers.LinearBoundaryValueSolver(problem)
    w1.store_ncc_matrices(vars, solver.subproblems)
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
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_vector_prod_scalar(Nphi, Ntheta, Nr, basis, alpha, k_ncc, k_arg, dealias, dtype):
    c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, alpha, dealias=dealias, dtype=dtype)
    b_ncc = b.radial_basis.clone_with(k=k_ncc)
    b_arg = b.clone_with(k=k_arg)
    u = field.Field(dist=d, bases=(b_ncc,), tensorsig=(c,), dtype=dtype)
    f = field.Field(dist=d, bases=(b_arg,), dtype=dtype)
    u['g'] = np.random.randn(*u['g'].shape)
    f['g'] = np.random.randn(*f['g'].shape)
    if isinstance(b, BallBasis):
        k_out = k_out_ball(k_ncc, k_arg, u, f)
    else:
        k_out = k_out_shell(k_ncc, k_arg)
    if k_out > 0 and dealias < 2:
        u['c'][:, :, :, -k_out:] = 0
        f['c'][:, :, -k_out:] = 0
    vars = [f]
    w0 = u * f
    w1 = w0.reinitialize(ncc=True, ncc_vars=vars)
    problem = problems.LBVP(vars)
    problem.add_equation((dot(u,u)*f, 0))
    solver = solvers.LinearBoundaryValueSolver(problem)
    w1.store_ncc_matrices(vars, solver.subproblems)
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
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_vector_prod_vector(Nphi, Ntheta, Nr, basis, alpha, k_ncc, k_arg, dealias, dtype):
    c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, alpha, dealias=dealias, dtype=dtype)
    b_ncc = b.radial_basis.clone_with(k=k_ncc)
    b_arg = b.clone_with(k=k_arg)
    u = field.Field(dist=d, bases=(b_ncc,), tensorsig=(c,), dtype=dtype)
    v = field.Field(dist=d, bases=(b_arg,), tensorsig=(c,), dtype=dtype)
    u['g'] = np.random.randn(*u['g'].shape)
    v['g'] = np.random.randn(*v['g'].shape)
    if isinstance(b, BallBasis):
        k_out = k_out_ball(k_ncc, k_arg, u, v)
    else:
        k_out = k_out_shell(k_ncc, k_arg)
    if k_out > 0 and dealias < 2:
        u['c'][:, :, :, -k_out:] = 0
        v['c'][:, :, :, -k_out:] = 0
    vars = [v]
    w0 = u * v
    w1 = w0.reinitialize(ncc=True, ncc_vars=vars)
    problem = problems.LBVP(vars)
    problem.add_equation((dot(u,u)*v, 0))
    solver = solvers.LinearBoundaryValueSolver(problem,)
    w1.store_ncc_matrices(vars, solver.subproblems)
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
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_tensor_prod_scalar(Nphi, Ntheta, Nr, basis, alpha, k_ncc, k_arg, dealias, dtype):
    c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, alpha, dealias=dealias, dtype=dtype)
    b_ncc = b.radial_basis.clone_with(k=k_ncc)
    b_arg = b.clone_with(k=k_arg)
    T = field.Field(dist=d, bases=(b_ncc,), tensorsig=(c,c), dtype=dtype)
    f = field.Field(dist=d, bases=(b_arg,), dtype=dtype)
    T['g'] = np.random.randn(*T['g'].shape)
    f['g'] = np.random.randn(*f['g'].shape)
    if isinstance(b, BallBasis):
        k_out = k_out_ball(k_ncc, k_arg, T, f)
    else:
        k_out = k_out_shell(k_ncc, k_arg)
    if k_out > 0 and dealias < 2:
        T['c'][:, :, :, :, -k_out:] = 0
        f['c'][:, :, -k_out:] = 0
    vars = [f]
    w0 = T * f
    w1 = w0.reinitialize(ncc=True, ncc_vars=vars)
    problem = problems.LBVP(vars)
    problem.add_equation((operators.Laplacian(f, c), 0))
    solver = solvers.LinearBoundaryValueSolver(problem)
    w1.store_ncc_matrices(vars, solver.subproblems)
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
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_vector_dot_vector(Nphi, Ntheta, Nr, basis, alpha, k_ncc, k_arg, dealias, dtype):
    c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, alpha, dealias=dealias, dtype=dtype)
    b_ncc = b.radial_basis.clone_with(k=k_ncc)
    b_arg = b.clone_with(k=k_arg)
    u = field.Field(dist=d, bases=(b_ncc,), tensorsig=(c,), dtype=dtype)
    v = field.Field(dist=d, bases=(b_arg,), tensorsig=(c,), dtype=dtype)
    u['g'] = np.random.randn(*u['g'].shape)
    v['g'] = np.random.randn(*v['g'].shape)
    if isinstance(b, BallBasis):
        k_out = k_out_ball(k_ncc, k_arg, u, v)
    else:
        k_out = k_out_shell(k_ncc, k_arg)
    if k_out > 0 and dealias < 2:
        u['c'][:, :, :, -k_out:] = 0
        v['c'][:, :, :, -k_out:] = 0
    vars = [v]
    w0 = dot(u, v)
    w1 = w0.reinitialize(ncc=True, ncc_vars=vars)
    problem = problems.LBVP(vars)
    problem.add_equation((dot(u,u)*v, 0))
    solver = solvers.LinearBoundaryValueSolver(problem,)
    w1.store_ncc_matrices(vars, solver.subproblems)
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
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_vector_dot_tensor(Nphi, Ntheta, Nr, basis, alpha, k_ncc, k_arg, dealias, dtype):
    c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, alpha, dealias=dealias, dtype=dtype)
    b_ncc = b.radial_basis.clone_with(k=k_ncc)
    b_arg = b.clone_with(k=k_arg)
    u = field.Field(dist=d, bases=(b_ncc,), tensorsig=(c,), dtype=dtype)
    T = field.Field(dist=d, bases=(b_arg,), tensorsig=(c,c), dtype=dtype)
    u['g'] = np.random.randn(*u['g'].shape)
    T['g'] = np.random.randn(*T['g'].shape)
    if isinstance(b, BallBasis):
        k_out = k_out_ball(k_ncc, k_arg, u, T)
    else:
        k_out = k_out_shell(k_ncc, k_arg)
    if k_out > 0 and dealias < 2:
        u['c'][:, :, :, -k_out:] = 0
        T['c'][:, :, :, :, -k_out:] = 0
    vars = [T]
    w0 = dot(u, T)
    w1 = w0.reinitialize(ncc=True, ncc_vars=vars)
    problem = problems.LBVP(vars)
    problem.add_equation((dot(u,u)*T, 0))
    solver = solvers.LinearBoundaryValueSolver(problem,)
    w1.store_ncc_matrices(vars, solver.subproblems)
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
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_tensor_dot_vector(Nphi, Ntheta, Nr, basis, alpha, k_ncc, k_arg, dealias, dtype):
    c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, alpha, dealias=dealias, dtype=dtype)
    b_ncc = b.radial_basis.clone_with(k=k_ncc)
    b_arg = b.clone_with(k=k_arg)
    T = field.Field(dist=d, bases=(b_ncc,), tensorsig=(c,c), dtype=dtype)
    u = field.Field(dist=d, bases=(b_arg,), tensorsig=(c,), dtype=dtype)
    T['g'] = np.random.randn(*T['g'].shape)
    u['g'] = np.random.randn(*u['g'].shape)
    if isinstance(b, BallBasis):
        k_out = k_out_ball(k_ncc, k_arg, T, u)
    else:
        k_out = k_out_shell(k_ncc, k_arg)
    if k_out > 0 and dealias < 2:
        T['c'][:, :, :, :, -k_out:] = 0
        u['c'][:, :, :, -k_out:] = 0
    vars = [u]
    w0 = dot(T, u)
    w1 = w0.reinitialize(ncc=True, ncc_vars=vars)
    problem = problems.LBVP(vars)
    problem.add_equation((operators.Laplacian(u, c), 0))
    solver = solvers.LinearBoundaryValueSolver(problem,)
    w1.store_ncc_matrices(vars, solver.subproblems)
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
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_tensor_dot_tensor(Nphi, Ntheta, Nr, basis, alpha, k_ncc, k_arg, dealias, dtype):
    c, d, b, phi, theta, r, x, y, z = basis(Nphi, Ntheta, Nr, alpha, dealias=dealias, dtype=dtype)
    b_ncc = b.radial_basis.clone_with(k=k_ncc)
    b_arg = b.clone_with(k=k_arg)
    T = field.Field(dist=d, bases=(b_ncc,), tensorsig=(c,c), dtype=dtype)
    U = field.Field(dist=d, bases=(b_arg,), tensorsig=(c,c), dtype=dtype)
    T['g'] = np.random.randn(*T['g'].shape)
    U['g'] = np.random.randn(*U['g'].shape)
    if isinstance(b, BallBasis):
        k_out = k_out_ball(k_ncc, k_arg, T, U)
    else:
        k_out = k_out_shell(k_ncc, k_arg)
    if k_out > 0 and dealias < 2:
        T['c'][:, :, :, :, -k_out:] = 0
        U['c'][:, :, :, :, -k_out:] = 0
    vars = [U]
    w0 = dot(T, U)
    w1 = w0.reinitialize(ncc=True, ncc_vars=vars)
    problem = problems.LBVP(vars)
    problem.add_equation((operators.Laplacian(U, c), 0))
    solver = solvers.LinearBoundaryValueSolver(problem,)
    w1.store_ncc_matrices(vars, solver.subproblems)
    w0 = w0.evaluate()
    w1 = w1.evaluate_as_ncc()
    assert np.allclose(w0['c'], w1['c'])

