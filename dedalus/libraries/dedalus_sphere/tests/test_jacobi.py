"""Test Jacobi transforms and operators."""


import pytest
import numpy as np
from . import jacobi128


N_range = [1, 2, 3, 4, 8, 16]
ab_range = [-0.5, 0, 0.5, 1]


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('a', ab_range)
@pytest.mark.parametrize('b', ab_range)
def test_forward_backward_loop(N, a, b):
    """Test round-trip transforms from grid space."""
    # Setup
    grid, weights = jacobi128.quadrature(N, a, b)
    envelope = jacobi128.envelope(a, b, a, b, grid)
    polynomials = jacobi128.recursion(N, a, b, grid, envelope)
    # Build transform matrices
    forward = weights * polynomials
    backward = polynomials.T.copy()
    assert np.allclose(backward @ forward, np.identity(N+1))


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('a', ab_range)
@pytest.mark.parametrize('b', ab_range)
def test_backward_forward_loop(N, a, b):
    """Test round-trip transforms from coeff space."""
    # Setup
    grid, weights = jacobi128.quadrature(N, a, b)
    envelope = jacobi128.envelope(a, b, a, b, grid)
    polynomials = jacobi128.recursion(N, a, b, grid, envelope)
    # Build transform matrices
    forward = weights * polynomials
    backward = polynomials.T.copy()
    assert np.allclose(forward @ backward, np.identity(N+1))


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('a', ab_range)
@pytest.mark.parametrize('b', ab_range)
def test_Ap_loop(N, a, b):
    """Test round-trip transforms from grid space with conversion."""
    # Setup
    grid0, weights0 = jacobi128.quadrature(N, a, b)
    envelope0 = jacobi128.envelope(a, b, a, b, grid0)
    polynomials0 = jacobi128.recursion(N, a, b, grid0, envelope0)
    envelope1 = jacobi128.envelope(a+1, b, a+1, b, grid0)
    polynomials1 = jacobi128.recursion(N, a+1, b, grid0, envelope1)
    # Build matrices
    forward = weights0 * polynomials0
    conversion = jacobi128.operator('A+', N, a, b)
    backward = polynomials1.T.copy()
    assert np.allclose(backward @ conversion @ forward, np.identity(N+1))


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('a', ab_range)
@pytest.mark.parametrize('b', ab_range)
def test_Bp_loop(N, a, b):
    """Test round-trip transforms from grid space with conversion."""
    # Setup
    grid0, weights0 = jacobi128.quadrature(N, a, b)
    envelope0 = jacobi128.envelope(a, b, a, b, grid0)
    polynomials0 = jacobi128.recursion(N, a, b, grid0, envelope0)
    envelope1 = jacobi128.envelope(a, b+1, a, b+1, grid0)
    polynomials1 = jacobi128.recursion(N, a, b+1, grid0, envelope1)
    # Build matrices
    forward = weights0 * polynomials0
    conversion = jacobi128.operator('B+', N, a, b)
    backward = polynomials1.T.copy()
    assert np.allclose(backward @ conversion @ forward, np.identity(N+1))


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('a', ab_range)
@pytest.mark.parametrize('b', ab_range)
def test_Ap_Bp_commutation(N, a, b):
    # Build matrices
    Ap00 = jacobi128.operator('A+', N, a, b)
    Bp10 = jacobi128.operator('B+', N, a+1, b)
    path1 = Bp10 @ Ap00
    Bp00 = jacobi128.operator('B+', N, a, b)
    Ap01 = jacobi128.operator('A+', N, a, b+1)
    path2 = Ap01 @ Bp00
    assert np.allclose(path1.toarray(), path2.toarray())


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('a', ab_range)
@pytest.mark.parametrize('b', ab_range)
def test_App_Bpp_commutation(N, a, b):
    # Build matrices
    Ap00 = jacobi128.operator('A+', N, a, b)
    Ap10 = jacobi128.operator('A+', N, a+1, b)
    Bp20 = jacobi128.operator('B+', N, a+2, b)
    Bp21 = jacobi128.operator('B+', N, a+2, b+1)
    path1 = Bp21 @ Bp20 @ Ap10 @ Ap00
    Bp00 = jacobi128.operator('B+', N, a, b)
    Bp01 = jacobi128.operator('B+', N, a, b+1)
    Ap02 = jacobi128.operator('A+', N, a, b+2)
    Ap12 = jacobi128.operator('A+', N, a+1, b+2)
    path2 = Ap12 @ Ap02 @ Bp01 @ Bp00
    assert np.allclose(path1.toarray(), path2.toarray())

