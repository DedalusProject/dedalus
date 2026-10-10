"""
Test the cached-polynomial NCC assembly for sphere and shell bases against the
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


def build_shell(Nphi, Ntheta, Nr, k):
    coords = d3.SphericalCoordinates('phi', 'theta', 'r')
    dist = d3.Distributor(coords, dtype=np.complex128)
    shell = d3.ShellBasis(coords, shape=(Nphi, Ntheta, Nr), radii=(0.5, 1.5), k=k, dtype=np.complex128)
    return coords, shell


def reference_S2(subproblem, ncc_basis, arg_basis, out_basis, coeffs, ncc_comp, arg_comp, out_comp, ncc_ts, arg_ts, out_ts, cutoff):
    # The sparse Clenshaw evaluation of SphereBasis._last_axis_component_ncc_matrix (complex branch) before the cached stacks
    m = subproblem.group[0]
    s_arg = out_basis.spintotal(arg_ts, arg_comp)
    s_ncc = out_basis.spintotal(ncc_ts, ncc_comp)
    s_out = out_basis.spintotal(out_ts, out_comp)
    a_ncc = b_ncc = abs(s_ncc)
    N = ncc_basis.ell_size(m)
    N0 = ncc_basis.ell_size(0)
    Nmat = 3*((N0+1)//2)
    J = arg_basis.operator_matrix('Cos', m, s_arg, size=Nmat)
    A, B = clenshaw.jacobi_recursion(Nmat, a_ncc, b_ncc, J)
    f0 = dedalus_sphere.jacobi.polynomials(1, a_ncc, b_ncc, 1)[0] * sparse.identity(Nmat)
    prefactor = ncc_basis.sine_multiplication_matrix(m, s_arg, s_ncc, size=Nmat)
    c = coeffs.ravel()[abs(s_ncc):N0]
    lmin_out = max(abs(s_out) - abs(m), 0)
    lmin_arg = max(abs(s_arg) - abs(m), 0)
    matrix = sparse.lil_matrix((N, N), dtype=np.complex128)
    matrix[lmin_out:, lmin_arg:] = (prefactor @ clenshaw.matrix_clenshaw(c, A, B, f0, cutoff=cutoff))[:N-lmin_out, :N-lmin_arg]
    if m < 0:
        matrix = matrix[::-1, ::-1]
    return matrix.tocsr()


def reference_meridional(ncc_basis, arg_basis, coeff_vals, coeff_norms, cutoff):
    # The Kronecker-Clenshaw evaluation of ShellRadialBasis._last_axis_component_ncc_matrix (meridional branch) before the cached stacks
    ell = 0
    arg_radial_basis = arg_basis.radial_basis
    a_ncc = ncc_basis.k + ncc_basis.alpha[0]
    b_ncc = ncc_basis.k + ncc_basis.alpha[1]
    N = ncc_basis.n_size(ell)
    N0 = ncc_basis.n_size(0)
    Nmat = 3*((N0+1)//2) + ncc_basis.k
    J = arg_radial_basis.operator_matrix('Z', ell, 0, size=Nmat)
    A, B = clenshaw.jacobi_recursion(Nmat, a_ncc, b_ncc, J)
    f0 = dedalus_sphere.jacobi.polynomials(1, a_ncc, b_ncc, 1)[0] * sparse.identity(Nmat)
    prefactor = arg_radial_basis.jacobi_conversion(ell, dk=ncc_basis.k, size=Nmat)
    i0, i1 = coeff_vals[0].shape
    I0 = sparse.identity(i0)
    I1 = sparse.identity(i1)
    matrix = sparse.kron(I0, prefactor) @ clenshaw.kronecker_clenshaw(coeff_vals, coeff_norms, A, B, f0, cutoff=cutoff)
    dealias0 = sparse.kron(I0, sparse.eye(N, Nmat))
    dealias1 = sparse.kron(I1, sparse.eye(N, Nmat))
    return (dealias0 @ matrix @ dealias1.T).tocsr()


def assert_close(A, B):
    A = sparse.csr_matrix(A)
    B = sparse.csr_matrix(B)
    assert A.shape == B.shape
    scale = max(abs(B).max(), 1e-300)
    diff = abs(A - B).max() if (A - B).nnz else 0
    assert diff <= 1e-13 * scale


@pytest.mark.parametrize('Ntheta', [12, 17])
@pytest.mark.parametrize('m', [0, 1, 3, -2])
@pytest.mark.parametrize('ncc_comp', [None, 0, 1])
@pytest.mark.parametrize('arg_comp', [None, 0, 2])
def test_S2_component_ncc_matrix(Ntheta, m, ncc_comp, arg_comp):
    coords, shell = build_shell(2*Ntheta, Ntheta, 8, 0)
    S2 = shell.S2_basis()
    rng = np.random.default_rng(42)
    ncc_ts = () if ncc_comp is None else (coords,)
    arg_ts = () if arg_comp is None else (coords,)
    nc = () if ncc_comp is None else (ncc_comp,)
    ac = () if arg_comp is None else (arg_comp,)
    out_ts, oc = arg_ts, ac
    N0 = S2.ell_size(0)
    coeffs = rng.standard_normal(N0) + 1j*rng.standard_normal(N0)
    coeffs[N0//2] = 1e-12   # below the cutoff
    sp = FakeSubproblem(m)
    cutoff = 1e-6
    new = basis.SphereBasis._last_axis_component_ncc_matrix(sp, S2, S2, S2, coeffs, nc, ac, oc, ncc_ts, arg_ts, out_ts, cutoff)
    ref = reference_S2(sp, S2, S2, S2, coeffs, nc, ac, oc, ncc_ts, arg_ts, out_ts, cutoff)
    assert_close(new, ref)


@pytest.mark.parametrize('Nr', [8, 13])
@pytest.mark.parametrize('k', [0, 1])
@pytest.mark.parametrize('threads', [1, 3])
def test_meridional_component_ncc_matrix(Nr, k, threads, monkeypatch):
    coords, shell = build_shell(8, 6, Nr, k)
    monkeypatch.setattr(basis, 'NCC_THREADS', threads)
    rng = np.random.default_rng(7)
    i0, i1 = 7, 5
    nterms = shell.n_size(0)
    vals, norms = [], []
    for n in range(nterms):
        V = sparse.random(i0, i1, density=0.4, random_state=rng, dtype=np.complex128, format='csr')
        V = V + 1j*sparse.random(i0, i1, density=0.4, random_state=rng, format='csr')
        if n % 4 == 3:
            V = 1e-12 * V   # below the cutoff
        vals.append(V)
        norms.append(abs(V).max() if V.nnz else 0)
    norms = np.array(norms)
    cutoff = 1e-6
    new = basis.ShellRadialBasis._last_axis_component_ncc_matrix(None, shell, shell, shell, (vals, norms), (), (), (), (), (), (), cutoff=cutoff)
    ref = reference_meridional(shell, shell, vals, norms, cutoff)
    assert_close(new, ref)
    new = sparse.csr_matrix(new)
    assert new.has_sorted_indices
