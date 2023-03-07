"""Test cartesian adjoint differentiation and Jacobi conversion."""

import pytest
import numpy as np
from dedalus.core import coords, distributor, basis, field, operators, problems, solvers

# Fourier tests

N_range = [8, 10, 12]
bounds_range = [(0, 2*np.pi), (0.5, 1.5)]
dtype_range = [np.float64]

@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('bounds', bounds_range)
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('layout', ['g','c'])
def test_fourier_adjoint_differentiate(N, bounds, dtype, layout):
    c = coords.Coordinate('x')
    d = distributor.Distributor((c,))
    if dtype == np.float64:
        b_dir = basis.RealFourier(c, size=N, bounds=bounds)
        b_adj = basis.RealFourier(c, size=N, bounds=bounds, adjoint=True)
    elif dtype == np.complex128:
        b_dir = basis.ComplexFourier(c, size=N, bounds=bounds)
        b_adj = basis.ComplexFourier(c, size=N, bounds=bounds, adjoint=True) 

    u    = field.Field(dist=d, bases=(b_dir,),dtype=dtype)
    Cu   = field.Field(dist=d, bases=(b_dir,),dtype=dtype)
    v    = field.Field(dist=d, bases=(b_adj,),dtype=dtype) 
    CHv  = field.Field(dist=d, bases=(b_adj,),dtype=dtype) 

    u.fill_random(layout=layout)
    v.fill_random(layout=layout)
    # Real doesn't work in grid space...
    u[layout]
    v[layout]
    differentiate = operators.Differentiate(u, c) 
    differentiate.operate(Cu)
    differentiate.operate_adjoint(v,CHv)
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
@pytest.mark.parametrize('layout', ['g','c'])
def test_jacobi_convert_adjoint(N, a, b, k, dtype, d_range, layout):
    c = coords.Coordinate('x')
    d = distributor.Distributor((c,))
    b_dir = basis.Jacobi(c, size=N, a0=a, b0=b, a=a+k, b=b+k, bounds=(0, 1))
    b_adj = basis.Jacobi(c, size=N, a0=a, b0=b, a=a+k, b=b+k, bounds=(0, 1), adjoint=True)
    b1     = b_dir.derivative_basis(d_range)
    b1_adj = b_adj.derivative_basis(d_range)

       
    # For adjoint v should be in the adjoint basis of Cu, 
    # and CHv will be in the adjoint basis of u.
    u    = field.Field(dist=d, bases=b_dir,dtype=dtype)
    Cu   = field.Field(dist=d, bases=b1,dtype=dtype)
    v  = field.Field(dist=d, bases=b1_adj,dtype=dtype)
    CHv  = field.Field(dist=d, bases=b_adj,dtype=dtype) 

    u.fill_random(layout=layout)
    v.fill_random(layout=layout)

    u[layout]
    v[layout]
    
    # Convert is based on the basis of first argument...
    # So need a new input for operate adjoint, but must use original
    # convert operator
    convert = operators.convert(u, (b1,))
    convert.operate(Cu)
    convert.operate_adjoint(v,CHv)
    term1 = np.vdot(v['c'],Cu['c'])
    term2 = np.vdot(CHv['c'],u['c'])

    assert np.allclose(term1,term2)

@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('a', ab_range)
@pytest.mark.parametrize('b', ab_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('layout', ['g','c'])
def test_jacobi_differentiate_adjoint(N, a, b, k, dtype, layout):
    c = coords.Coordinate('x')
    d = distributor.Distributor((c,))
    b_dir = basis.Jacobi(c, size=N, a0=a, b0=b, a=a+k, b=b+k, bounds=(0, 1))
    b_adj = basis.Jacobi(c, size=N, a0=a, b0=b, a=a+k, b=b+k, bounds=(0, 1),adjoint=True)

    b1     = b_dir.derivative_basis(1)
    b1_adj = b_adj.derivative_basis(1)
    
    u    = field.Field(dist=d, bases=(b_dir,),dtype=dtype)
    Cu   = field.Field(dist=d, bases=(b1,),dtype=dtype)
    v    = field.Field(dist=d, bases=(b1_adj,),dtype=dtype)
    CHv  = field.Field(dist=d, bases=(b_adj,),dtype=dtype) 

    u.fill_random(layout=layout)
    v.fill_random(layout=layout)

    u[layout]
    v[layout]
    differentiate = operators.Differentiate(u, c) 
    differentiate.operate(Cu)
    differentiate.operate_adjoint(v,CHv)
    term1 = np.vdot(v['c'],Cu['c'])
    term2 = np.vdot(CHv['c'],u['c'])

    assert np.allclose(term1,term2)
