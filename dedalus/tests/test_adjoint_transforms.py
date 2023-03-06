
import pytest
import numpy as np
from dedalus.core import coords, distributor, basis, field, operators
from dedalus.tools.cache import CachedMethod, CachedFunction
from mpi4py import MPI

comm = MPI.COMM_WORLD

## Adjoint tests

# Cartesian
# Forward
@pytest.mark.parametrize('N', [16])
@pytest.mark.parametrize('dealias', [0.5, 1, 1.5])
@pytest.mark.parametrize('dtype', [np.float64])
@pytest.mark.parametrize('library', ['fftpack', 'scipy', 'fftw', 'fftw_hc'])
def test_real_fourier_adjoint_forward(N, dealias, dtype, library):
    """Tests adjoint forward real fourier transforms"""

    # Ensure that <y,Tx> = <T^Hy,x> where T is the transform
    c = coords.Coordinate('x')
    d = distributor.Distributor([c])
    b = basis.RealFourier(c, size=N, bounds=(0, 2*np.pi), dealias=dealias, library=library)
    u = field.Field(dist=d, bases=(b,), dtype=dtype)
    
    g_shape = u['g'].shape[0]
    c_shape = u['c'].shape[0]  
    
    transform = b.transform_plan(N)

    x_g = np.random.rand(g_shape)
    x_c = np.zeros(c_shape,dtype=dtype)

    y_g = np.random.rand(g_shape)
    y_c = np.zeros(c_shape,dtype=dtype)

    transform.forward(x_g,x_c,0)
    transform.forward_adjoint(y_g,y_c,0)

    inner_1 = np.vdot(y_g,x_c)
    inner_2 = np.vdot(y_c,x_g)
    rel_err = np.linalg.norm(inner_1-inner_2)/np.linalg.norm(inner_1)
    assert(rel_err<1e-13)

@pytest.mark.parametrize('N', [16])
@pytest.mark.parametrize('dealias', [0.5,1,1.5])
@pytest.mark.parametrize('dtype', [np.complex128])
@pytest.mark.parametrize('library', ['scipy', 'fftw'])
def test_complex_fourier_adjoint_forward(N, dealias, dtype, library):
    """Tests adjoint forward complex fourier transforms"""

    # Ensure that <y,Tx> = <T^Hy,x> where T is the transform
    c = coords.Coordinate('x')
    d = distributor.Distributor([c])

    b = basis.Fourier(c, size=N, bounds=(0, 2*np.pi), dealias=dealias, library=library,dtype=dtype)
    u = field.Field(dist=d, bases=(b,), dtype=dtype)
    
    g_shape = u['g'].shape[0]
    c_shape = u['c'].shape[0]  
    
    transform = b.transform_plan(N)

    x_g = np.random.rand(g_shape) + 1j*np.random.rand(g_shape)
    x_c = np.zeros(c_shape,dtype=dtype)

    y_g = np.random.rand(g_shape) + 1j*np.random.rand(g_shape)
    y_c = np.zeros(c_shape,dtype=dtype)

    transform.forward(x_g,x_c,0)
    transform.forward_adjoint(y_g,y_c,0)

    inner_1 = np.vdot(y_g,x_c)
    inner_2 = np.vdot(y_c,x_g)
    rel_err = np.linalg.norm(inner_1-inner_2)/np.linalg.norm(inner_1)
    assert(rel_err<1e-13)

@pytest.mark.parametrize('N', [15, 16])
@pytest.mark.parametrize('alpha', [0, 1, 2])
@pytest.mark.parametrize('dealias', [0.5, 1, 1.5])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('library', ['scipy_dct', 'fftw_dct'])

def test_chebyshev_adjoint_forward(N, alpha, dealias, dtype, library):
    """Tests adjoint forward Chebyshev transforms"""

    # Ensure that <y,Tx> = <T^Hy,x> where T is the transform
    c = coords.Coordinate('x')
    d = distributor.Distributor([c])

    b = basis.Ultraspherical(c, size=N, alpha0=0, alpha=alpha, bounds=(-1, 1), dealias=dealias, library=library)
    u = field.Field(dist=d, bases=(b,), dtype=dtype)
    
    transform = b.transform_plan(N)

    g_shape = u['g'].shape[0]
    c_shape = u['c'].shape[0]  
    
    if(dtype==np.float64):
        x_g = np.random.rand(g_shape)
    else:
        x_g = np.random.rand(g_shape) + 1j*np.random.rand(g_shape) 
    x_c = np.zeros(c_shape,dtype=dtype)

    if(dtype==np.float64):
        y_g = np.random.rand(g_shape)
    else:
        y_g = np.random.rand(g_shape) + 1j*np.random.rand(g_shape)
    y_c = np.zeros(c_shape,dtype=dtype)

    transform.forward(x_g,x_c,0)
    transform.forward_adjoint(y_g,y_c,0)

    inner_1 = np.vdot(y_g,x_c)
    inner_2 = np.vdot(y_c,x_g)
    rel_err = np.linalg.norm(inner_1-inner_2)/np.linalg.norm(inner_1)
    assert(rel_err<1e-13)

# Backwards

@pytest.mark.parametrize('N', [16])
@pytest.mark.parametrize('dealias', [0.5, 1, 1.5])
@pytest.mark.parametrize('dtype', [np.float64])
@pytest.mark.parametrize('library', ['fftpack', 'scipy', 'fftw', 'fftw_hc'])
def test_real_fourier_adjoint_backward(N, dealias, dtype, library):
    """Tests adjoint backward real fourier transforms"""

    # Ensure that <y,Tx> = <T^Hy,x> where T is the transform
    c = coords.Coordinate('x')
    d = distributor.Distributor([c])
    b = basis.RealFourier(c, size=N, bounds=(0, 2*np.pi), dealias=dealias, library=library)
    u = field.Field(dist=d, bases=(b,), dtype=dtype)
    
    g_shape = u['g'].shape[0]
    c_shape = u['c'].shape[0]  
    
    transform = b.transform_plan(N)

    x_g = np.zeros(g_shape)
    x_c = np.random.rand(c_shape)

    y_g = np.zeros(g_shape)
    y_c = np.random.rand(c_shape)

    transform.backward(x_c,x_g,0)
    transform.backward_adjoint(y_c,y_g,0)

    inner_1 = np.vdot(y_c,x_g)
    inner_2 = np.vdot(y_g,x_c)
    rel_err = np.linalg.norm(inner_1-inner_2)/np.linalg.norm(inner_1)
    assert(rel_err<1e-13)

@pytest.mark.parametrize('N', [16])
@pytest.mark.parametrize('dealias', [0.5,1,1.5])
@pytest.mark.parametrize('dtype', [np.complex128])
@pytest.mark.parametrize('library', ['scipy', 'fftw'])
def test_complex_fourier_adjoint_backward(N, dealias, dtype, library):
    """Tests adjoint backward complex fourier transforms"""

    # Ensure that <y,Tx> = <T^Hy,x> where T is the transform
    c = coords.Coordinate('x')
    d = distributor.Distributor([c])

    b = basis.Fourier(c, size=N, bounds=(0, 2*np.pi), dealias=dealias, library=library,dtype=dtype)
    u = field.Field(dist=d, bases=(b,), dtype=dtype)
    
    g_shape = u['g'].shape[0]
    c_shape = u['c'].shape[0]  
    
    transform = b.transform_plan(N)

    x_g = np.zeros(g_shape,dtype=dtype)
    x_c = np.random.rand(c_shape) + 1j*np.random.rand(c_shape)

    y_g = np.zeros(g_shape,dtype=dtype)
    y_c = np.random.rand(c_shape) + 1j*np.random.rand(c_shape)

    transform.backward(x_c,x_g,0)
    transform.backward_adjoint(y_c,y_g,0)

    inner_1 = np.vdot(y_c,x_g)
    inner_2 = np.vdot(y_g,x_c)
    rel_err = np.linalg.norm(inner_1-inner_2)/np.linalg.norm(inner_1)
    assert(rel_err<1e-13)

@pytest.mark.parametrize('N', [15, 16])
@pytest.mark.parametrize('alpha', [0, 1, 2])
@pytest.mark.parametrize('dealias', [0.5, 1, 1.5])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('library', ['scipy_dct', 'fftw_dct'])

def test_chebyshev_adjoint_backward(N, alpha, dealias, dtype, library):
    """Tests adjoint backward Chebyshev transforms"""

    # Ensure that <y,Tx> = <T^Hy,x> where T is the transform
    c = coords.Coordinate('x')
    d = distributor.Distributor([c])
    
    b = basis.Ultraspherical(c, size=N, alpha0=0, alpha=alpha, bounds=(-1, 1), dealias=dealias, library=library)
    u = field.Field(dist=d, bases=(b,), dtype=dtype)

    transform = b.transform_plan(N)

    g_shape = u['g'].shape[0]
    c_shape = u['c'].shape[0]  
    
    if(dtype==np.float64):
        x_c = np.random.rand(c_shape)
    else:
        x_c = np.random.rand(c_shape) + 1j*np.random.rand(c_shape) 
    x_g = np.zeros(g_shape,dtype=dtype)

    if(dtype==np.float64):
        y_c = np.random.rand(c_shape)
    else:
        y_c = np.random.rand(c_shape) + 1j*np.random.rand(c_shape)
    y_g = np.zeros(g_shape,dtype=dtype)

    # Note: backward plan changes xc so make a copy
    x_cc = x_c.copy()
    transform.backward(x_cc,x_g,0)
    transform.backward_adjoint(y_c,y_g,0)

    inner_1 = np.vdot(y_c,x_g)
    inner_2 = np.vdot(y_g,x_c)
    rel_err = np.linalg.norm(inner_1-inner_2)/np.linalg.norm(inner_1)
    assert(rel_err<1e-13)

## Matrix-based tests
# Forward
@pytest.mark.parametrize('N', [16])
@pytest.mark.parametrize('dealias', [0.5, 1, 1.5])
@pytest.mark.parametrize('dtype', [np.float64])
@pytest.mark.parametrize('library', ['fftpack', 'scipy', 'fftw', 'fftw_hc'])
def test_real_fourier_adjoint_forward_matrix(N, dealias, dtype, library):
    """Tests adjoint forward real fourier transforms"""

    # Ensure that forward adjoint transforms match the matrix implementations
    c = coords.Coordinate('x')
    d = distributor.Distributor([c])

    b = basis.RealFourier(c, size=N, bounds=(0, 2*np.pi), dealias=dealias, library=library)
    u = field.Field(dist=d, bases=(b,), dtype=dtype)

    b_mat = basis.RealFourier(c, size=N, bounds=(0, 2*np.pi), dealias=dealias, library='matrix')
    
    g_shape = u['g'].shape[0]
    c_shape = u['c'].shape[0]  
    
    transform     = b.transform_plan(N)
    transform_mat = b_mat.transform_plan(N)

    x_g = np.random.rand(g_shape)
    x_c = np.zeros(c_shape,dtype=dtype)
    y_c = np.zeros(c_shape,dtype=dtype)

    transform.forward_adjoint(x_g,x_c,0)
    transform_mat.forward_adjoint(x_g,y_c,0)

    assert np.allclose(x_c, y_c)

@pytest.mark.parametrize('N', [16])
@pytest.mark.parametrize('dealias', [0.5,1,1.5])
@pytest.mark.parametrize('dtype', [np.complex128])
@pytest.mark.parametrize('library', ['scipy', 'fftw'])
def test_complex_fourier_adjoint_forward_matrix(N, dealias, dtype, library):
    """Tests adjoint forward complex fourier transforms"""

    # Ensure that forward adjoint transforms match the matrix implementations
    c = coords.Coordinate('x')
    d = distributor.Distributor([c])

    b = basis.Fourier(c, size=N, bounds=(0, 2*np.pi), dealias=dealias, library=library,dtype=dtype)
    u = field.Field(dist=d, bases=(b,), dtype=dtype)

    b_mat = basis.Fourier(c, size=N, bounds=(0, 2*np.pi), dealias=dealias, dtype=dtype, library='matrix')
    
    g_shape = u['g'].shape[0]
    c_shape = u['c'].shape[0]  
    
    transform     = b.transform_plan(N)
    transform_mat = b_mat.transform_plan(N)

    x_g = np.random.rand(g_shape) + 1j*np.random.rand(g_shape)
    x_c = np.zeros(c_shape,dtype=dtype)
    y_c = np.zeros(c_shape,dtype=dtype)

    transform.forward_adjoint(x_g,x_c,0)
    transform_mat.forward_adjoint(x_g,y_c,0)

    assert np.allclose(x_c, y_c)

@pytest.mark.parametrize('N', [15, 16])
@pytest.mark.parametrize('alpha', [0, 1, 2])
@pytest.mark.parametrize('dealias', [0.5, 1, 1.5])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('library', ['scipy_dct', 'fftw_dct'])

def test_chebyshev_adjoint_forward_matrix(N, alpha, dealias, dtype, library):
    """Tests adjoint forward Chebyshev transforms"""

    # Ensure that forward adjoint transforms match the matrix implementations
    c = coords.Coordinate('x')
    d = distributor.Distributor([c])
   
    b = basis.Ultraspherical(c, size=N, alpha0=0, alpha=alpha, bounds=(-1, 1), dealias=dealias, library=library)
    u = field.Field(dist=d, bases=(b,), dtype=dtype)

    b_mat = basis.Ultraspherical(c, size=N, alpha0=0, alpha=alpha, bounds=(-1, 1), dealias=dealias, library='matrix')
    
    transform     = b.transform_plan(N)
    transform_mat = b_mat.transform_plan(N)

    g_shape = u['g'].shape[0]
    c_shape = u['c'].shape[0]  
    
    if(dtype==np.float64):
        x_g = np.random.rand(g_shape)
    else:
        x_g = np.random.rand(g_shape) + 1j*np.random.rand(g_shape)

    x_c = np.zeros(c_shape,dtype=dtype)
    y_c = np.zeros(c_shape,dtype=dtype)

    transform.forward_adjoint(x_g,x_c,0)
    transform_mat.forward_adjoint(x_g,y_c,0)

    assert np.allclose(x_c, y_c)

# Backwards

@pytest.mark.parametrize('N', [16])
@pytest.mark.parametrize('dealias', [0.5, 1, 1.5])
@pytest.mark.parametrize('dtype', [np.float64])
@pytest.mark.parametrize('library', ['fftpack', 'scipy', 'fftw', 'fftw_hc'])
def test_real_fourier_adjoint_backward_matrix(N, dealias, dtype, library):
    """Tests adjoint backward real fourier transforms"""

    # Ensure that forward adjoint transforms match the matrix implementations
    c = coords.Coordinate('x')
    d = distributor.Distributor([c])
    b = basis.RealFourier(c, size=N, bounds=(0, 2*np.pi), dealias=dealias, library=library)
    u = field.Field(dist=d, bases=(b,), dtype=dtype)

    b_mat = basis.RealFourier(c, size=N, bounds=(0, 2*np.pi), dealias=dealias, library='matrix')
    
    g_shape = u['g'].shape[0]
    c_shape = u['c'].shape[0]  
    
    transform     = b.transform_plan(N)
    transform_mat = b_mat.transform_plan(N)

    x_g = np.zeros(g_shape)
    y_g = np.zeros(g_shape)
    x_c = np.random.rand(c_shape)

    transform.backward_adjoint(x_c,x_g,0)
    transform_mat.backward_adjoint(x_c,y_g,0)

    assert np.allclose(x_g,y_g)

@pytest.mark.parametrize('N', [16])
@pytest.mark.parametrize('dealias', [0.5,1,1.5])
@pytest.mark.parametrize('dtype', [np.complex128])
@pytest.mark.parametrize('library', ['scipy', 'fftw'])
def test_complex_fourier_adjoint_backward_matrix(N, dealias, dtype, library):
    """Tests adjoint backward complex fourier transforms"""

    # Ensure that forward adjoint transforms match the matrix implementations
    c = coords.Coordinate('x')
    d = distributor.Distributor([c])

    b = basis.Fourier(c, size=N, bounds=(0, 2*np.pi), dealias=dealias, library=library,dtype=dtype)
    u = field.Field(dist=d, bases=(b,), dtype=dtype)

    b_mat = basis.Fourier(c, size=N, bounds=(0, 2*np.pi), dealias=dealias, library='matrix',dtype=dtype)
    
    g_shape = u['g'].shape[0]
    c_shape = u['c'].shape[0]  
    
    transform     = b.transform_plan(N)
    transform_mat = b_mat.transform_plan(N)

    x_g = np.zeros(g_shape,dtype=dtype)
    y_g = np.zeros(g_shape,dtype=dtype)
    x_c = np.random.rand(c_shape) + 1j*np.random.rand(c_shape)
    
    transform.backward_adjoint(x_c,x_g,0)
    transform_mat.backward_adjoint(x_c,y_g,0)

    assert np.allclose(x_g,y_g)

@pytest.mark.parametrize('N', [15, 16])
@pytest.mark.parametrize('alpha', [0, 1, 2])
@pytest.mark.parametrize('dealias', [0.5, 1, 1.5])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('library', ['scipy_dct', 'fftw_dct'])

def test_chebyshev_adjoint_backward_matrix(N, alpha, dealias, dtype, library):
    """Tests adjoint backward Chebyshev transforms"""

    # Ensure that forward adjoint transforms match the matrix implementations
    c = coords.Coordinate('x')
    d = distributor.Distributor([c])
    
    b = basis.Ultraspherical(c, size=N, alpha0=0, alpha=alpha, bounds=(-1, 1), dealias=dealias, library=library)
    u = field.Field(dist=d, bases=(b,), dtype=dtype)

    b_mat = basis.Ultraspherical(c, size=N, alpha0=0, alpha=alpha, bounds=(-1, 1), dealias=dealias, library='matrix')
    
    transform = b.transform_plan(N)
    transform_mat = b_mat.transform_plan(N)

    g_shape = u['g'].shape[0]
    c_shape = u['c'].shape[0]  
    
    if(dtype==np.float64):
        x_c = np.random.rand(c_shape)
    else:
        x_c = np.random.rand(c_shape) + 1j*np.random.rand(c_shape) 
    x_g = np.zeros(g_shape,dtype=dtype)
    y_g = np.zeros(g_shape,dtype=dtype)

    # Note: backward plan changes xc so make a copy
    x_cc = x_c.copy()
    transform.backward_adjoint(x_cc,x_g,0)
    x_cc = x_c.copy()
    transform_mat.backward_adjoint(x_cc,y_g,0)

    assert np.allclose(x_g,y_g)