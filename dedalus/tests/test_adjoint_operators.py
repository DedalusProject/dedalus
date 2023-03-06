"""Test Jacobi conversion, differentiation, interpolation, and integration."""

import pytest
import numpy as np
from dedalus.core import coords, distributor, basis, field, operators, problems, solvers

# Fourier tests

N_range = [8, 10, 12]
bounds_range = [(0, 2*np.pi), (0.5, 1.5)]
dtype_range = [np.float64, np.complex128]

@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('bounds', bounds_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_fourier_differentiate(N, bounds, dtype):
    c = coords.Coordinate('x')
    d = distributor.Distributor((c,))
    if dtype == np.float64:
        b = basis.RealFourier(c, size=N, bounds=bounds)
    elif dtype == np.complex128:
        b = basis.ComplexFourier(c, size=N, bounds=bounds)
    x = b.local_grid(1)

    u    = field.Field(dist=d, bases=b,dtype=dtype)
    Cu   = field.Field(dist=d, bases=b,dtype=dtype)
    v    = field.Field(dist=d, bases=b,dtype=dtype) 
    CHv  = field.Field(dist=d, bases=b,dtype=dtype) 
    x = b.local_grid(1)

    u.fill_random()
    v.fill_random()
    # Must be in coefficient space for now
    u['c']
    v['c']
    operators.Differentiate(u, c).operate(Cu)
    operators.Differentiate(v, c).operate_adjoint(CHv)
    term1 = np.vdot(v['c'],Cu['c'])
    term2 = np.vdot(CHv['c'],u['c'])

    assert np.allclose(term1,term2)



# Jacobi tests
N_range = [8, 9]
ab_range = [-0.5, 0]
k_range = [0, 1]
dtype_range = [np.float64, np.complex128]
d_range = [1,2,3]

@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('a', ab_range)
@pytest.mark.parametrize('b', ab_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('d_range', d_range)
def test_jacobi_convert_adjoint(N, a, b, k, dtype, d_range):
    c = coords.Coordinate('x')
    d = distributor.Distributor((c,))
    b = basis.Jacobi(c, size=N, a0=a, b0=b, a=a+k, b=b+k, bounds=(0, 1))
    b1 = b.derivative_basis(d_range)

    u    = field.Field(dist=d, bases=b,dtype=dtype)
    Cu   = field.Field(dist=d, bases=b,dtype=dtype)
    v    = field.Field(dist=d, bases=b,dtype=dtype) 
    CHv  = field.Field(dist=d, bases=b,dtype=dtype) 
    x = b.local_grid(1)

    u.fill_random()
    v.fill_random()
    # Must be in coefficient space for now
    u['c']
    v['c']
    operators.convert(u, (b1,)).operate(Cu)
    operators.convert(v, (b1,)).operate_adjoint(CHv)
    term1 = np.vdot(v['c'],Cu['c'])
    term2 = np.vdot(CHv['c'],u['c'])

    assert np.allclose(term1,term2)

@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('a', ab_range)
@pytest.mark.parametrize('b', ab_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dtype', dtype_range)
def test_jacobi_differentiate_adjoint(N, a, b, k, dtype):
    c = coords.Coordinate('x')
    d = distributor.Distributor((c,))
    b = basis.Jacobi(c, size=N, a0=a, b0=b, a=a+k, b=b+k, bounds=(0, 1))
    x = b.local_grid(1)
    
    u    = field.Field(dist=d, bases=b,dtype=dtype)
    Cu   = field.Field(dist=d, bases=b,dtype=dtype)
    v    = field.Field(dist=d, bases=b,dtype=dtype) 
    CHv  = field.Field(dist=d, bases=b,dtype=dtype) 
    x = b.local_grid(1)

    u.fill_random()
    v.fill_random()
    # Must be in coefficient space for now
    u['c']
    v['c']
    operators.Differentiate(u, c).operate(Cu)
    operators.Differentiate(v, c).operate_adjoint(CHv)
    term1 = np.vdot(v['c'],Cu['c'])
    term2 = np.vdot(CHv['c'],u['c'])

    assert np.allclose(term1,term2)
