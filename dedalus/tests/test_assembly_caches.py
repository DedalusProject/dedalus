"""
Tests of the assembly speedups outside the NCC sums: the vectorized Distributor.local_groupset_slices, the shell
derivative operators built from their ell-independent pieces, and the cached Cartesian NCC matrices.
"""

import itertools
import pytest
import numpy as np
from scipy import sparse
import dedalus.public as d3
from dedalus.core import arithmetic, basis
from dedalus.libraries import dedalus_sphere


def reference_groupset_slices(layout, groupset, domain, scales):
    # Distributor.local_groupset_slices before the vectorization (one np.any per group-shaped block)
    groups = layout.local_group_arrays(domain, scales)
    dim = groups.shape[0]
    group_shape = layout.group_shape(domain)
    selections = np.ones(groups[0].shape, dtype=int)
    for i, subgroup in enumerate(groupset):
        if subgroup is not None:
            selections *= (subgroup == groups[i])
    slices = []
    for i, subgroup in enumerate(groupset):
        if subgroup is None:
            subslices = [slice(None)]
        else:
            subslices = [slice(j, j+group_shape[i]) for j in range(0, groups.shape[i+1], group_shape[i])]
        slices.append(subslices)
    group_slices = []
    for s in itertools.product(*slices):
        sliced_selections = selections[tuple(s)]
        if np.any(sliced_selections):
            lefts = list(map(np.min, np.where(sliced_selections)))
            rights = list(map(np.max, np.where(sliced_selections)))
            group_slice = []
            for i in range(dim):
                if s[i] != slice(None):
                    group_slice.append(s[i])
                else:
                    group_slice.append(slice(lefts[i], rights[i]+1))
            group_slices.append(tuple(group_slice))
    return group_slices


def _problem(geometry, dtype):
    if geometry == 'cartesian':
        c = d3.CartesianCoordinates('x', 'y', 'z')
        dist = d3.Distributor(c, dtype=dtype)
        Fourier = d3.RealFourier if dtype == np.float64 else d3.ComplexFourier
        bases = (Fourier(c['x'], 8, bounds=(0, 1)), Fourier(c['y'], 6, bounds=(0, 1)), d3.ChebyshevT(c['z'], 10, bounds=(0, 1)))
        edge = bases[:2]
    elif geometry in ('shell', 'ball'):
        c = d3.SphericalCoordinates('phi', 'theta', 'r')
        dist = d3.Distributor(c, dtype=dtype)
        b = (d3.ShellBasis(c, (8, 6, 10), radii=(0.5, 1.5), dtype=dtype) if geometry == 'shell'
             else d3.BallBasis(c, (8, 6, 10), radius=1, dtype=dtype))
        bases, edge = (b,), (b.S2_basis(),)
    elif geometry in ('disk', 'annulus'):
        c = d3.PolarCoordinates('phi', 'r')
        dist = d3.Distributor(c, dtype=dtype)
        b = (d3.DiskBasis(c, (8, 10), radius=1, dtype=dtype) if geometry == 'disk'
             else d3.AnnulusBasis(c, (8, 10), radii=(0.5, 1.5), dtype=dtype))
        bases, edge = (b,), (b.S1_basis(),)
    else:
        c = d3.S2Coordinates('phi', 'theta')
        dist = d3.Distributor(c, dtype=dtype)
        bases, edge = (d3.SphereBasis(c, (8, 6), radius=1, dtype=dtype),), ()
    f = dist.Field(name='f', bases=bases)
    u = dist.VectorField(c, name='u', bases=bases)
    t = dist.VectorField(c, name='t', bases=edge) if edge else dist.Field(name='t')
    s = dist.Field(name='s')
    return dist, [f, u, t, s]


@pytest.mark.parametrize('geometry', ['cartesian', 'shell', 'ball', 'disk', 'annulus', 'sphere'])
@pytest.mark.parametrize('dtype', [np.complex128, np.float64])
def test_groupset_slices(geometry, dtype):
    dist, fields = _problem(geometry, dtype)
    layout = dist.coeff_layout
    domains = {f.domain for f in fields}
    groupsets = set()
    for domain in domains:
        for coupling in itertools.product([False, True], repeat=dist.dim):
            groupsets |= set(layout.local_groupsets(tuple(coupling), domain, scales=1))
    assert groupsets
    for domain in domains:
        for groupset in groupsets:
            assert layout.local_groupset_slices(groupset, domain, scales=1) == reference_groupset_slices(layout, groupset, domain, 1)


def reference_ell_operator(radii, alpha, k, op, l, size):
    # ShellRadialBasis._ell_operator_matrix before the ell-linear pieces
    if op in ['D+', 'D-']:
        p = int(op[-1]+'1')
        operator = dedalus_sphere.shell.operator(3, radii, 'D', alpha)(p, l)
    else:
        D = dedalus_sphere.shell.operator(3, radii, 'D', alpha)
        operator = D(-1, l+1) @ D(+1, l)
    return operator(size, k).square.astype(np.float64)


@pytest.mark.parametrize('radii', [(0.5, 1.5), (1, 3)])
@pytest.mark.parametrize('alpha', [(-0.5, -0.5), (0, 0)])
@pytest.mark.parametrize('k', [0, 1, 2])
@pytest.mark.parametrize('size', [9, 16])
@pytest.mark.parametrize('op', ['D+', 'D-', 'L'])
def test_shell_ell_operators(radii, alpha, k, size, op):
    c = d3.SphericalCoordinates('phi', 'theta', 'r')
    d3.Distributor(c, dtype=np.complex128)
    shell = d3.ShellBasis(c, (8, 6, size), radii=radii, alpha=alpha, k=k, dtype=np.complex128)
    rb = shell.radial_basis
    for l in [0, 1, 2, 5, 17, 40]:
        new = sparse.csr_matrix(rb._ell_operator_matrix(op, l, size))
        ref = sparse.csr_matrix(reference_ell_operator(radii, alpha, k, op, l, size))
        scale = max(abs(ref).max(), 1e-300)
        assert new.shape == ref.shape
        assert (abs(new - ref).max() if (new - ref).nnz else 0) <= 1e-13 * scale


def _cartesian_ivp(dtype):
    c = d3.CartesianCoordinates('x', 'z')
    dist = d3.Distributor(c, dtype=dtype)
    Fourier = d3.RealFourier if dtype == np.float64 else d3.ComplexFourier
    xb = Fourier(c['x'], size=16, bounds=(0, 4))
    zb = d3.ChebyshevT(c['z'], size=12, bounds=(0, 1))
    b = dist.Field(name='b', bases=(xb, zb))
    u = dist.VectorField(c, name='u', bases=(xb, zb))
    p = dist.Field(name='p', bases=(xb, zb))
    tau_p = dist.Field(name='tau_p')
    tb1 = dist.Field(name='tb1', bases=xb); tb2 = dist.Field(name='tb2', bases=xb)
    tu1 = dist.VectorField(c, name='tu1', bases=xb); tu2 = dist.VectorField(c, name='tu2', bases=xb)
    x, z = dist.local_grids(xb, zb)
    ex, ez = c.unit_vector_fields(dist)
    rho0 = dist.Field(name='rho0', bases=zb); rho0['g'] = np.exp(-2*z)
    U0 = dist.VectorField(c, name='U0', bases=zb); U0['g'][0] = np.sin(np.pi*z)
    lift = lambda A: d3.Lift(A, zb.derivative_basis(1), -1)
    grad_u = d3.grad(u) + ez*lift(tu1)
    grad_b = d3.grad(b) + ez*lift(tb1)
    problem = d3.IVP([p, b, u, tau_p, tb1, tb2, tu1, tu2], namespace=locals())
    problem.add_equation("trace(grad_u) + tau_p = 0")
    problem.add_equation("rho0*dt(b) - div(grad_b) + lift(tb2) + U0@grad(b) = 0")
    problem.add_equation("dt(u) - div(grad_u) + grad(p) - rho0*b*ez + lift(tu2) + U0@grad(u) + u@grad(U0) = 0")
    problem.add_equation("b(z=0) = 0"); problem.add_equation("u(z=0) = 0")
    problem.add_equation("b(z=1) = 0"); problem.add_equation("u(z=1) = 0")
    problem.add_equation("integ(p) = 0")
    return problem, rho0, z


@pytest.mark.parametrize('dtype', [np.complex128, np.float64])
def test_cartesian_ncc_cache(dtype, monkeypatch):
    # cached matrices equal fresh ones, subproblem by subproblem
    problem, rho0, z = _cartesian_ivp(dtype)
    solver = problem.build_solver(d3.SBDF1)
    cached = [(sp.L_min.copy(), sp.M_min.copy()) for sp in solver.subproblems]
    monkeypatch.setattr(arithmetic.Product, 'build_cartesian_ncc_matrix', arithmetic.Product._build_cartesian_ncc_matrix)
    problem2, _, _ = _cartesian_ivp(dtype)
    solver2 = problem2.build_solver(d3.SBDF1)
    for (L, M), sp in zip(cached, solver2.subproblems):
        assert (abs(L - sp.L_min).max() if (L - sp.L_min).nnz else 0) == 0
        assert (abs(M - sp.M_min).max() if (M - sp.M_min).nnz else 0) == 0


def test_cartesian_ncc_cache_follows_data():
    # changing the NCC field and rebuilding gives the new matrices, not the cached old ones
    problem, rho0, z = _cartesian_ivp(np.complex128)
    solver = problem.build_solver(d3.SBDF1)
    sp = solver.subproblems[1]
    M_old = sp.M_min.copy()
    rho0['g'] = np.exp(-3*z)
    solver.build_matrices(solver.subproblems, ['M', 'L'])
    M_new = sp.M_min.copy()
    assert abs(M_new - M_old).max() > 1e-3
    problem3, rho3, z3 = _cartesian_ivp(np.complex128)
    rho3['g'] = np.exp(-3*z3)
    fresh = problem3.build_solver(d3.SBDF1).subproblems[1].M_min
    assert (abs(M_new - fresh).max() if (M_new - fresh).nnz else 0) <= 1e-14 * abs(fresh).max()
