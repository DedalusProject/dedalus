
import pytest
import numpy as np
from dedalus.core import coords, distributor, basis, field, operators, problems, solvers, arithmetic
from dedalus.core import future
from dedalus.tools.array import apply_matrix
from dedalus.tools.cache import CachedFunction

# TODO: add in alpha and k
# Seem to need higher resolution and allclose tolerance for k=1
# or lowering annulus radius

dot = arithmetic.DotProduct
dtypes = [np.float64, np.complex128]
Nphi_range = [16]
Nr_range = [8]
dealias = [1, 3/2]
radius_disk = [1.5,]
radii_annulus = [(0.5, 3),]

@CachedFunction
def build_disk(Nphi, Nr, dealias, dtype=np.float64):
    c = coords.PolarCoordinates('phi', 'r')
    d = distributor.Distributor((c,))
    b = basis.DiskBasis(c, (Nphi, Nr), radius=radius_disk[0], dealias=(dealias, dealias), dtype=dtype)
    phi, r = d.local_grids(b)
    x, y = c.cartesian(phi, r)
    return c, d, b, phi, r, x, y

@CachedFunction
def build_annulus(Nphi, Nr, dealias, dtype):
    c = coords.PolarCoordinates('phi', 'r')
    d = distributor.Distributor((c,))
    b = basis.AnnulusBasis(c, (Nphi, Nr), radii=radii_annulus[0], dealias=(dealias, dealias), dtype=dtype)
    phi, r = d.local_grids(b)
    x, y = c.cartesian(phi, r)
    return c, d, b, phi, r, x, y


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('basis', [build_disk, build_annulus])
@pytest.mark.parametrize('ncc_first', [True,False])
@pytest.mark.parametrize('dealias', dealias)
@pytest.mark.parametrize('dtype', dtypes)
def test_scalar_prod_scalar(Nphi, Nr, basis, ncc_first, dealias, dtype):
    c, d, b, phi, r, x, y = basis(Nphi, Nr, dealias=dealias, dtype=dtype)
    f = field.Field(dist=d, bases=(b.radial_basis,), dtype=dtype)
    g = field.Field(dist=d, bases=(b,), dtype=dtype)
    f['g'] = r**4
    g['g'] = (3*x**2 + 2*y)
    vars = [g]
    if ncc_first:
        w0 = f * g
    else:
        w0 = g * f
    w1 = w0.reinitialize(ncc=True, ncc_vars=vars)
    problem = problems.LBVP(vars)
    problem.add_equation((w1, 0))
    solver = solvers.LinearBoundaryValueSolver(problem, matsolver='SuperluNaturalSpsolve', matrix_coupling=[False,True])
    w1.store_ncc_matrices(vars, solver.subproblems)
    w0 = w0.evaluate()
    w0.change_scales(1)
    w1 = w1.evaluate_as_ncc()
    assert np.allclose(w0['g'], w1['g'])


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('basis', [build_disk, build_annulus])
@pytest.mark.parametrize('ncc_first', [True,False])
@pytest.mark.parametrize('dealias', dealias)
@pytest.mark.parametrize('dtype', dtypes)
def test_scalar_prod_vector(Nphi, Nr, basis, ncc_first, dealias, dtype):
    c, d, b, phi, r, x, y = basis(Nphi, Nr, dealias=dealias, dtype=dtype)
    f = field.Field(dist=d, bases=(b.radial_basis,), dtype=dtype)
    g = field.Field(dist=d, bases=(b,), dtype=dtype)

    f['g'] = r**2
    g['g'] = 3*x**2 + 2*y
    u = operators.Gradient(g, c).evaluate()

    vars = [u]
    if ncc_first:
        w0 = f * u
    else:
        w0 = u * f
    w1 = w0.reinitialize(ncc=True, ncc_vars=vars)

    problem = problems.LBVP(vars)
    problem.add_equation((w1, 0))
    solver = solvers.LinearBoundaryValueSolver(problem, matsolver='SuperluNaturalSpsolve', matrix_coupling=[False,True])
    w1.store_ncc_matrices(vars, solver.subproblems)

    w0 = w0.evaluate()
    w0.change_scales(1)
    w1 = w1.evaluate_as_ncc()
    assert np.allclose(w0['g'], w1['g'])


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('basis', [build_disk, build_annulus])
@pytest.mark.parametrize('ncc_first', [True,False])
@pytest.mark.parametrize('dealias', dealias)
@pytest.mark.parametrize('dtype', dtypes)
def test_scalar_prod_tensor(Nphi, Nr, basis, ncc_first, dealias, dtype):
    c, d, b, phi, r, x, y = basis(Nphi, Nr, dealias=dealias, dtype=dtype)
    f = field.Field(dist=d, bases=(b.radial_basis,), dtype=dtype)
    g = field.Field(dist=d, bases=(b,), dtype=dtype)

    f['g'] = r**4
    g['g'] = 3*x**2 + 2*y
    T = operators.Gradient(operators.Gradient(g, c), c).evaluate()

    vars = [T]
    if ncc_first:
        w0 = f * T
    else:
        w0 = T * f
    w1 = w0.reinitialize(ncc=True, ncc_vars=vars)

    problem = problems.LBVP(vars)
    problem.add_equation((w1, 0))
    solver = solvers.LinearBoundaryValueSolver(problem, matsolver='SuperluNaturalSpsolve', matrix_coupling=[False, True])
    w1.store_ncc_matrices(vars, solver.subproblems)

    w0 = w0.evaluate()
    w0.change_scales(1)
    w1 = w1.evaluate_as_ncc()
    assert np.allclose(w0['g'], w1['g'])


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('basis', [build_disk, build_annulus])
@pytest.mark.parametrize('ncc_first', [True,False])
@pytest.mark.parametrize('dealias', dealias)
@pytest.mark.parametrize('dtype', dtypes)
def test_vector_prod_scalar(Nphi, Nr, basis, ncc_first, dealias, dtype):
    c, d, b, phi, r, x, y = basis(Nphi, Nr, dealias=dealias, dtype=dtype)
    f = field.Field(dist=d, bases=(b.radial_basis,), dtype=dtype)
    g = field.Field(dist=d, bases=(b,), dtype=dtype)

    f['g'] = r**4
    g['g'] = 3*x**2 + 2*y
    u = operators.Gradient(f, c).evaluate()
    vars = [g]
    if ncc_first:
        w0 = u * g
    else:
        w0 = g * u
    w1 = w0.reinitialize(ncc=True, ncc_vars=vars)
    problem = problems.LBVP(vars)
    problem.add_equation((dot(u,u)*g, 0))
    solver = solvers.LinearBoundaryValueSolver(problem, matsolver='SuperluNaturalSpsolve', matrix_coupling=[False, True])
    w1.store_ncc_matrices(vars, solver.subproblems)

    w0 = w0.evaluate()
    w0.change_scales(1)
    w1 = w1.evaluate_as_ncc()
    assert np.allclose(w0['g'], w1['g'])


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('basis', [build_disk, build_annulus])
@pytest.mark.parametrize('ncc_first', [True,False])
@pytest.mark.parametrize('dealias', dealias)
@pytest.mark.parametrize('dtype', dtypes)
def test_vector_prod_vector(Nphi, Nr, basis, ncc_first, dealias, dtype):
    c, d, b, phi, r, x, y = basis(Nphi, Nr, dealias=dealias, dtype=dtype)
    f = field.Field(dist=d, bases=(b.radial_basis,), dtype=dtype)
    g = field.Field(dist=d, bases=(b,), dtype=dtype)

    f['g'] = r**4
    g['g'] = 3*x**2 + 2*y
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
    solver = solvers.LinearBoundaryValueSolver(problem, matsolver='SuperluNaturalSpsolve', matrix_coupling=[False, True])
    w1.store_ncc_matrices(vars, solver.subproblems)

    w0 = w0.evaluate()
    w0.change_scales(1)
    w1 = w1.evaluate_as_ncc()
    assert np.allclose(w0['g'], w1['g'])


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('basis', [build_disk, build_annulus])
@pytest.mark.parametrize('ncc_first', [True,False])
@pytest.mark.parametrize('dealias', dealias)
@pytest.mark.parametrize('dtype', dtypes)
def test_vector_dot_vector(Nphi, Nr, basis, ncc_first, dealias, dtype):
    c, d, b, phi, r, x, y = basis(Nphi, Nr, dealias=dealias, dtype=dtype)

    f = field.Field(dist=d, bases=(b.radial_basis,), dtype=dtype)
    g = field.Field(dist=d, bases=(b,), dtype=dtype)

    f['g'] = r**4
    g['g'] = 3*x**2 + 2*y
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
    solver = solvers.LinearBoundaryValueSolver(problem, matsolver='SuperluNaturalSpsolve', matrix_coupling=[False, True])
    w1.store_ncc_matrices(vars, solver.subproblems)

    w0 = w0.evaluate()
    w0.change_scales(1)
    w1 = w1.evaluate_as_ncc()
    assert np.allclose(w0['g'], w1['g'])


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('basis', [build_disk, build_annulus])
@pytest.mark.parametrize('ncc_first', [True,False])
@pytest.mark.parametrize('dealias', dealias)
@pytest.mark.parametrize('dtype', dtypes)
def test_vector_dot_tensor(Nphi, Nr, basis, ncc_first, dealias, dtype):
    c, d, b, phi, r, x, y = basis(Nphi, Nr, dealias=dealias, dtype=dtype)

    f = field.Field(dist=d, bases=(b.radial_basis,), dtype=dtype)
    T = field.Field(dist=d, bases=(b,), tensorsig=(c,c,), dtype=dtype)
    z = 0
    theta = np.pi/2.
    T['g'][1,1] = (6*x**2+4*y*z)/r**2
    T['g'][1,0] = T['g'][0,1] = 2*x*(z-3*y)/(r**2*np.sin(theta))
    T['g'][0,0] = 6*y**2/(x**2+y**2)

    f['g'] = r**6
    u = operators.Gradient(f, c).evaluate()

    vars = [T]
    if ncc_first:
        w0 = dot(u, T)
    else:
        w0 = dot(T, u)
    w1 = w0.reinitialize(ncc=True, ncc_vars=vars)

    problem = problems.LBVP(vars)
    problem.add_equation((dot(u,u)*T, 0))
    solver = solvers.LinearBoundaryValueSolver(problem, matsolver='SuperluNaturalSpsolve', matrix_coupling=[False, True])
    w1.store_ncc_matrices(vars, solver.subproblems)

    w0 = w0.evaluate()
    w0.change_scales(1)
    w1 = w1.evaluate_as_ncc()
    assert np.allclose(w0['g'], w1['g'])


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('basis', [build_disk, build_annulus])
@pytest.mark.parametrize('ncc_first', [True,False])
@pytest.mark.parametrize('dealias', dealias)
@pytest.mark.parametrize('dtype', dtypes)
def test_tensor_prod_scalar(Nphi, Nr, basis, ncc_first, dealias, dtype):
    c, d, b, phi, r, x, y = basis(Nphi, Nr, dealias=dealias, dtype=dtype)

    f = field.Field(dist=d, bases=(b.radial_basis,), dtype=dtype)
    g = field.Field(dist=d, bases=(b,), dtype=dtype)

    f['g'] = r**4
    g['g'] = 3*x**2 + 2*y
    T = operators.Gradient(operators.Gradient(f, c), c).evaluate()

    vars = [g]
    if ncc_first:
        U0 = T * g
    else:
        U0 = g * T
    U1 = U0.reinitialize(ncc=True, ncc_vars=vars)

    problem = problems.LBVP(vars)
    problem.add_equation((f*g, 0))
    solver = solvers.LinearBoundaryValueSolver(problem, matsolver='SuperluNaturalSpsolve', matrix_coupling=[False, True])
    U1.store_ncc_matrices(vars, solver.subproblems)

    U0 = U0.evaluate()
    U0.change_scales(1)
    U1 = U1.evaluate_as_ncc()
    assert np.allclose(U0['g'], U1['g'])


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('basis', [build_disk, build_annulus])
@pytest.mark.parametrize('ncc_first', [True,False])
@pytest.mark.parametrize('dealias', dealias)
@pytest.mark.parametrize('dtype', dtypes)
def test_tensor_dot_vector(Nphi, Nr, basis, ncc_first, dealias, dtype):
    c, d, b, phi, r, x, y = basis(Nphi, Nr, dealias=dealias, dtype=dtype)

    f = field.Field(dist=d, bases=(b.radial_basis,), dtype=dtype)
    g = field.Field(dist=d, bases=(b,), dtype=dtype)

    f['g'] = r**4
    g['g'] = 3*x**2 + 2*y
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
    solver = solvers.LinearBoundaryValueSolver(problem, matsolver='SuperluNaturalSpsolve', matrix_coupling=[False, True])
    w1.store_ncc_matrices(vars, solver.subproblems)

    w0 = w0.evaluate()
    w0.change_scales(1)
    w1 = w1.evaluate_as_ncc()
    assert np.allclose(w0['g'], w1['g'])


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('basis', [build_disk, build_annulus])
@pytest.mark.parametrize('ncc_first', [True,False])
@pytest.mark.parametrize('dealias', dealias)
@pytest.mark.parametrize('dtype', dtypes)
def test_tensor_dot_tensor(Nphi, Nr, basis, ncc_first, dealias, dtype):
    c, d, b, phi, r, x, y = basis(Nphi, Nr, dealias=dealias, dtype=dtype)

    f = field.Field(dist=d, bases=(b.radial_basis,), dtype=dtype)
    f['g'] = r**6
    U = operators.Gradient(operators.Gradient(f, c), c).evaluate()

    T = field.Field(dist=d, bases=(b,), tensorsig=(c,c,), dtype=dtype)
    z = 0
    theta = np.pi/2.
    T['g'][1,1] = (6*x**2+4*y*z)/r**2
    T['g'][1,0] = T['g'][0,1] = 2*x*(z-3*y)/(r**2*np.sin(theta))
    T['g'][0,0] = 6*y**2/(x**2+y**2)

    vars = [T]
    if ncc_first:
        W0 = dot(T, U)
    else:
        W0 = dot(U, T)
    W1 = W0.reinitialize(ncc=True, ncc_vars=vars)

    problem = problems.LBVP(vars)
    problem.add_equation((f*T, 0))
    solver = solvers.LinearBoundaryValueSolver(problem, matsolver='SuperluNaturalSpsolve', matrix_coupling=[False, True])
    W1.store_ncc_matrices(vars, solver.subproblems)

    W0 = W0.evaluate()
    W0.change_scales(1)
    W1 = W1.evaluate_as_ncc()
    assert np.allclose(W0['g'], W1['g'])

