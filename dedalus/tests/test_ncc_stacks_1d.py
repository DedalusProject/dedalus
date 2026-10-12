"""
Test the cached-polynomial NCC assembly of the one-dimensional bases (Jacobi, annulus, disk, ball) against the
sparse Clenshaw evaluation it replaces (kept here as the reference).
"""

import pytest
import numpy as np
from scipy import sparse
import dedalus.public as d3
from dedalus.core import basis
from dedalus.tools import clenshaw
from dedalus.libraries import dedalus_sphere


class FakeSubproblem:
    def __init__(self, m):
        self.group = (m, None, None)


class FakeSubproblemDist:
    def __init__(self, dist, group):
        self.dist = dist
        self.group = group


def assert_close(A, B):
    A = sparse.csr_matrix(A)
    B = sparse.csr_matrix(B)
    assert A.shape == B.shape
    scale = max(abs(B).max(), 1e-300)
    diff = abs(A - B).max() if (A - B).nnz else 0
    assert diff <= 1e-13 * scale


def reference_jacobi(ncc_basis, arg_basis, out_basis, coeffs, cutoff):
    # Jacobi._last_axis_component_ncc_matrix before the cached stacks
    a_ncc, b_ncc = ncc_basis.a, ncc_basis.b
    N = arg_basis.size
    da = int(np.round(out_basis.a - arg_basis.a))
    db = int(np.round(out_basis.b - arg_basis.b))
    Nmat = 3*((N+1)//2) + min((N+1)//2, (da+db+1)//2)
    J = arg_basis.Jacobi_matrix(size=Nmat)
    A, B = clenshaw.jacobi_recursion(Nmat, a_ncc, b_ncc, J)
    f0 = dedalus_sphere.jacobi.polynomials(1, a_ncc, b_ncc, 1)[0] * sparse.identity(Nmat)
    matrix = clenshaw.matrix_clenshaw(coeffs.ravel(), A, B, f0, cutoff=cutoff)
    convert = basis.jacobi.conversion_matrix(Nmat, arg_basis.a, arg_basis.b, out_basis.a, out_basis.b)
    return (convert @ matrix)[:N, :N]


@pytest.mark.parametrize('N', [16, 33])
@pytest.mark.parametrize('kind', ['chebyshev', 'legendre'])
@pytest.mark.parametrize('dk', [0, 1, 2])
@pytest.mark.parametrize('dtype', [np.complex128, np.float64])
def test_jacobi_ncc_matrix(N, kind, dk, dtype):
    c = d3.Coordinate('z')
    d3.Distributor(c, dtype=dtype)
    arg = (d3.ChebyshevT if kind == 'chebyshev' else d3.Legendre)(c, size=N, bounds=(-1, 3))
    out = arg.derivative_basis(dk)
    rng = np.random.default_rng(3)
    coeffs = rng.standard_normal(N) + (1j*rng.standard_normal(N) if dtype == np.complex128 else 0)
    coeffs[2::5] *= 1e-12   # below the cutoff
    cutoff = 1e-6
    new = basis.Jacobi._last_axis_component_ncc_matrix(None, arg, arg, out, coeffs, None, None, None, None, None, None, cutoff)
    assert_close(new, reference_jacobi(arg, arg, out, coeffs, cutoff))
    zero = basis.Jacobi._last_axis_component_ncc_matrix(None, arg, arg, out, 1e-12*coeffs, None, None, None, None, None, None, cutoff)
    assert sparse.csr_matrix(zero).count_nonzero() == 0


def reference_annulus(subproblem, ncc_basis, arg_basis, out_basis, coeffs, arg_comp, arg_ts, cutoff):
    first_axis = subproblem.dist.first_axis(out_basis)
    m = subproblem.group[first_axis]
    spintotal_arg = out_basis.spintotal(arg_ts, arg_comp)
    a_ncc = ncc_basis.k + ncc_basis.alpha[0]
    b_ncc = ncc_basis.k + ncc_basis.alpha[1]
    N = ncc_basis.n_size(m)
    N0 = ncc_basis.n_size(0)
    Nmat = 3*((N0+1)//2) + ncc_basis.k
    J = arg_basis.operator_matrix('Z', m, spintotal_arg, size=Nmat)
    A, B = clenshaw.jacobi_recursion(Nmat, a_ncc, b_ncc, J)
    f0 = dedalus_sphere.jacobi.polynomials(1, a_ncc, b_ncc, 1)[0] * sparse.identity(Nmat)
    prefactor = arg_basis.jacobi_conversion(m, dk=ncc_basis.k, size=Nmat)
    if ncc_basis.dtype == np.float64:
        mc = (prefactor @ clenshaw.matrix_clenshaw(coeffs[0].ravel()[:N0], A, B, f0, cutoff=cutoff))[:N, :N]
        ms = (prefactor @ clenshaw.matrix_clenshaw(coeffs[1].ravel()[:N0], A, B, f0, cutoff=cutoff))[:N, :N]
        return sparse.bmat([[mc, -ms], [ms, mc]], format='csr')
    return (prefactor @ clenshaw.matrix_clenshaw(coeffs.ravel()[:N0], A, B, f0, cutoff=cutoff))[:N, :N]


def reference_disk(subproblem, ncc_basis, arg_basis, out_basis, coeffs, ncc_comp, arg_comp, out_comp, ncc_ts, arg_ts, out_ts, cutoff):
    first_axis = subproblem.dist.first_axis(out_basis)
    m = subproblem.group[first_axis]
    spintotal_ncc = out_basis.spintotal(ncc_ts, ncc_comp)
    spintotal_arg = out_basis.spintotal(arg_ts, arg_comp)
    spintotal_out = out_basis.spintotal(out_ts, out_comp)
    regtotal_ncc = abs(spintotal_ncc)
    regtotal_arg = abs(m + spintotal_arg)
    regtotal_out = abs(m + spintotal_out)
    diff_regtotal = regtotal_out - regtotal_arg
    a_ncc = ncc_basis.alpha + ncc_basis.k
    b_ncc = regtotal_ncc
    N = ncc_basis.n_size(m)
    d = regtotal_ncc - abs(diff_regtotal)
    J = arg_basis.operator_matrix('Z', m, spintotal_arg)
    A, B = clenshaw.jacobi_recursion(N, a_ncc, b_ncc, J)
    f0 = dedalus_sphere.zernike.polynomials(2, 1, a_ncc, b_ncc, 1)[0] * sparse.identity(N)
    prefactor = arg_basis.radius_multiplication_matrix(m, spintotal_arg, diff_regtotal, d)
    if ncc_basis.dtype == np.float64:
        cf = coeffs.ravel()[:2*N]
        mc = prefactor @ clenshaw.matrix_clenshaw(cf[:N], A, B, f0, cutoff=cutoff)
        ms = prefactor @ clenshaw.matrix_clenshaw(cf[N:], A, B, f0, cutoff=cutoff)
        return sparse.bmat([[mc, -ms], [ms, mc]], format='csr')
    return prefactor @ clenshaw.matrix_clenshaw(coeffs.ravel()[:N], A, B, f0, cutoff=cutoff)


def _polar(kind, Nphi, Nr, k, dtype):
    c = d3.PolarCoordinates('phi', 'r')
    dist = d3.Distributor(c, dtype=dtype)
    if kind == 'annulus':
        b = basis.AnnulusBasis(c, (Nphi, Nr), dtype=dtype, radii=(0.5, 3), k=k)
    else:
        b = basis.DiskBasis(c, (Nphi, Nr), dtype=dtype, radius=1.5, k=k)
    return c, dist, b


@pytest.mark.parametrize('kind', ['annulus', 'disk'])
@pytest.mark.parametrize('k', [0, 1])
@pytest.mark.parametrize('m', [0, 1, 3])
@pytest.mark.parametrize('arg_comp', [None, 0, 1])
@pytest.mark.parametrize('dtype', [np.complex128, np.float64])
def test_polar_ncc_matrix(kind, k, m, arg_comp, dtype):
    c, dist, b = _polar(kind, 16, 12, k, dtype)
    sp = FakeSubproblemDist(dist, (m, None))
    arg_ts = () if arg_comp is None else (c,)
    comp = () if arg_comp is None else (arg_comp,)
    rng = np.random.default_rng(5 + m)
    n = 2 * b.n_size(0) + 4
    def rnd():
        v = rng.standard_normal(n) + (1j*rng.standard_normal(n) if dtype == np.complex128 else 0)
        v[3::4] *= 1e-12
        return v
    coeffs = (rnd(), rnd()) if (kind == 'annulus' and dtype == np.float64) else rnd()
    cutoff = 1e-6
    args = (sp, b, b, b, coeffs, (), comp, comp, (), arg_ts, arg_ts)
    new = type(b)._last_axis_component_ncc_matrix(*args, cutoff=cutoff)
    if kind == 'annulus':
        ref = reference_annulus(sp, b, b, b, coeffs, comp, arg_ts, cutoff)
    else:
        ref = reference_disk(*args, cutoff)
    assert_close(new, ref)


def reference_ball(subproblem, ncc_basis, arg_radial_basis, coeffs, ncc_comp, arg_comp, out_comp, cutoff):
    ell = subproblem.group[1]
    regtotal_ncc = basis.BallRadialBasis.regtotal(ncc_comp)
    regtotal_arg = basis.BallRadialBasis.regtotal(arg_comp)
    regtotal_out = basis.BallRadialBasis.regtotal(out_comp)
    diff_regtotal = regtotal_out - regtotal_arg
    a_ncc = ncc_basis.alpha + ncc_basis.k
    b_ncc = regtotal_ncc + 1/2
    N = ncc_basis.n_size(ell)
    N0 = ncc_basis.n_size(0)
    d = regtotal_ncc - abs(diff_regtotal)
    dk = max(ncc_basis.k, arg_radial_basis.k) - arg_radial_basis.k
    Nmat = 3*((N0+1)//2) + (dk+1)//2
    J = arg_radial_basis.operator_matrix('Z', ell, regtotal_arg, size=Nmat)
    A, B = clenshaw.jacobi_recursion(N0, a_ncc, b_ncc, J)
    f0 = dedalus_sphere.zernike.polynomials(3, 1, a_ncc, regtotal_ncc, 1)[0] * sparse.identity(Nmat)
    radial_factor = arg_radial_basis.radius_multiplication_matrix(ell, regtotal_arg, diff_regtotal, d, size=Nmat)
    conversion = arg_radial_basis.conversion_matrix(ell, regtotal_out, dk, size=Nmat)
    prefactor = conversion @ radial_factor
    if ncc_basis.dtype == np.float64:
        mc = (prefactor @ clenshaw.matrix_clenshaw(coeffs[0].ravel()[:N0], A, B, f0, cutoff=cutoff))[:N, :N]
        ms = (prefactor @ clenshaw.matrix_clenshaw(coeffs[1].ravel()[:N0], A, B, f0, cutoff=cutoff))[:N, :N]
        return sparse.bmat([[mc, -ms], [ms, mc]], format='csr')
    return (prefactor @ clenshaw.matrix_clenshaw(coeffs.ravel()[:N0], A, B, f0, cutoff=cutoff))[:N, :N]


@pytest.mark.parametrize('k', [0, 1])
@pytest.mark.parametrize('ell', [0, 1, 4])
@pytest.mark.parametrize('comp', [(), (0,), (1,), (2,)])
@pytest.mark.parametrize('dtype', [np.complex128, np.float64])
def test_ball_ncc_matrix(k, ell, comp, dtype):
    c = d3.SphericalCoordinates('phi', 'theta', 'r')
    d3.Distributor(c, dtype=dtype)
    ball = d3.BallBasis(c, (8, 6, 12), radius=1.5, k=k, dtype=dtype)
    rb = ball.radial_basis
    sp = FakeSubproblem(0)
    sp.group = (0, ell, None)
    rng = np.random.default_rng(9 + ell)
    n = rb.n_size(0)
    def rnd():
        v = rng.standard_normal(n) + (1j*rng.standard_normal(n) if dtype == np.complex128 else 0)
        v[3::4] *= 1e-12
        return v
    coeffs = (rnd(), rnd()) if dtype == np.float64 else rnd()
    cutoff = 1e-6
    new = basis.BallRadialBasis._last_axis_component_ncc_matrix(sp, rb, ball, ball, coeffs, (), comp, comp, (), (), (), cutoff=cutoff)
    ref = reference_ball(sp, rb, rb, coeffs, (), comp, comp, cutoff)
    assert_close(new, ref)
