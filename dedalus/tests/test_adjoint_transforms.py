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

    b_adj = b.adjoint_basis()
    u_adj = field.Field(dist=d, bases=(b_adj,), dtype=dtype)
    
    u.fill_random(layout='g')
    u_adj.fill_random(layout='c')

    inner_1 = np.vdot(u_adj['c'],u['c'])
    inner_2 = np.vdot(u_adj['g'],u['g'])

    assert np.allclose(inner_1,inner_2)

@pytest.mark.parametrize('N', [16])
@pytest.mark.parametrize('dealias', [0.5,1,1.5])
@pytest.mark.parametrize('dtype', [np.complex128])
@pytest.mark.parametrize('library', ['scipy', 'fftw'])
def test_complex_fourier_adjoint_forward(N, dealias, dtype, library):
    """Tests adjoint forward complex fourier transforms"""

    # Ensure that <y,Tx> = <T^Hy,x> where T is the transform
    c = coords.Coordinate('x')
    d = distributor.Distributor([c])

    b = basis.Fourier(c, size=N, bounds=(0, 2*np.pi), dealias=dealias, library=library, dtype=dtype)
    u = field.Field(dist=d, bases=(b,), dtype=dtype)

    b_adj = b.adjoint_basis()
    u_adj = field.Field(dist=d, bases=(b_adj,), dtype=dtype)
    
    u.fill_random(layout='g')
    u_adj.fill_random(layout='c')

    inner_1 = np.vdot(u_adj['c'],u['c'])
    inner_2 = np.vdot(u_adj['g'],u['g'])

    assert np.allclose(inner_1,inner_2)

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
    
    b_adj = b.adjoint_basis()
    u_adj = field.Field(dist=d, bases=(b_adj,), dtype=dtype)
    
    u.fill_random(layout='g')
    u_adj.fill_random(layout='c')

    inner_1 = np.vdot(u_adj['c'],u['c'])
    inner_2 = np.vdot(u_adj['g'],u['g'])

    assert np.allclose(inner_1,inner_2)

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
    
    b_adj = b.adjoint_basis()
    u_adj = field.Field(dist=d, bases=(b_adj,), dtype=dtype)
    
    u.fill_random(layout='c')
    u_adj.fill_random(layout='g')

    inner_1 = np.vdot(u_adj['g'],u['g'])
    inner_2 = np.vdot(u_adj['c'],u['c'])

    assert np.allclose(inner_1,inner_2)

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
    
    b_adj = b.adjoint_basis()
    u_adj = field.Field(dist=d, bases=(b_adj,), dtype=dtype)
    
    u.fill_random(layout='c')
    u_adj.fill_random(layout='g')

    inner_1 = np.vdot(u_adj['g'],u['g'])
    inner_2 = np.vdot(u_adj['c'],u['c'])

    assert np.allclose(inner_1,inner_2)

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

    b_adj = b.adjoint_basis()
    u_adj = field.Field(dist=d, bases=(b_adj,), dtype=dtype)
    
    u.fill_random(layout='c')
    u_adj.fill_random(layout='g')

    inner_1 = np.vdot(u_adj['g'],u['g'])
    inner_2 = np.vdot(u_adj['c'],u['c'])

    assert np.allclose(inner_1,inner_2)

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
    b_adj = basis.RealFourier(c, size=N, bounds=(0, 2*np.pi), dealias=dealias, library=library, adjoint=True)
    u_adj = field.Field(dist=d, bases=(b_adj,), dtype=dtype)

    b_mat_adj = basis.RealFourier(c, size=N, bounds=(0, 2*np.pi), dealias=dealias, library='matrix', adjoint=True)
    u_mat_adj = field.Field(dist=d, bases=(b_mat_adj,), dtype=dtype)
    
    u_adj.fill_random(layout='g')
    u_mat_adj['g'] = u_adj['g']

    assert np.allclose(u_adj['c'], u_mat_adj['c'])

@pytest.mark.parametrize('N', [16])
@pytest.mark.parametrize('dealias', [0.5,1,1.5])
@pytest.mark.parametrize('dtype', [np.complex128])
@pytest.mark.parametrize('library', ['scipy', 'fftw'])
def test_complex_fourier_adjoint_forward_matrix(N, dealias, dtype, library):
    """Tests adjoint forward complex fourier transforms"""

    # Ensure that forward adjoint transforms match the matrix implementations
    c = coords.Coordinate('x')
    d = distributor.Distributor([c])

    b_adj = basis.Fourier(c, size=N, bounds=(0, 2*np.pi), dealias=dealias, library=library, dtype=dtype, adjoint=True)
    u_adj = field.Field(dist=d, bases=(b_adj,), dtype=dtype)

    b_mat_adj = basis.Fourier(c, size=N, bounds=(0, 2*np.pi), dealias=dealias, library='matrix', dtype=dtype, adjoint=True)
    u_mat_adj = field.Field(dist=d, bases=(b_mat_adj,), dtype=dtype)

    u_adj.fill_random(layout='g')
    u_mat_adj['g'] = u_adj['g']

    assert np.allclose(u_adj['c'], u_mat_adj['c'])

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
   
    b_adj = basis.Ultraspherical(c, size=N, alpha0=0, alpha=alpha, bounds=(-1, 1), dealias=dealias, library=library, adjoint=True)
    u_adj = field.Field(dist=d, bases=(b_adj,), dtype=dtype)

    b_mat_adj = basis.Ultraspherical(c, size=N, alpha0=0, alpha=alpha, bounds=(-1, 1), dealias=dealias, library='matrix', adjoint=True)
    u_mat_adj = field.Field(dist=d, bases=(b_mat_adj,), dtype=dtype)

    u_adj.fill_random(layout='g')
    u_mat_adj['g'] = u_adj['g']

    assert np.allclose(u_adj['c'], u_mat_adj['c'])

# Backwards

@pytest.mark.parametrize('N', [16])
@pytest.mark.parametrize('dealias', [0.5, 1, 1.5])
@pytest.mark.parametrize('dtype', [np.float64])
@pytest.mark.parametrize('library', ['fftpack', 'scipy', 'fftw', 'fftw_hc'])
def test_real_fourier_adjoint_backward_matrix(N, dealias, dtype, library):
    """Tests adjoint backward real fourier transforms"""

    # Ensure that backward adjoint transforms match the matrix implementations
    c = coords.Coordinate('x')
    d = distributor.Distributor([c])
    b_adj = basis.RealFourier(c, size=N, bounds=(0, 2*np.pi), dealias=dealias, library=library, adjoint=True)
    u_adj = field.Field(dist=d, bases=(b_adj,), dtype=dtype)

    b_mat_adj = basis.RealFourier(c, size=N, bounds=(0, 2*np.pi), dealias=dealias, library='matrix', adjoint=True)
    u_mat_adj = field.Field(dist=d, bases=(b_mat_adj,), dtype=dtype)
    
    u_adj.fill_random(layout='c')
    u_mat_adj['c'] = u_adj['c']

    assert np.allclose(u_adj['g'], u_mat_adj['g'])

@pytest.mark.parametrize('N', [16])
@pytest.mark.parametrize('dealias', [0.5,1,1.5])
@pytest.mark.parametrize('dtype', [np.complex128])
@pytest.mark.parametrize('library', ['scipy', 'fftw'])
def test_complex_fourier_adjoint_backward_matrix(N, dealias, dtype, library):
    """Tests adjoint backward complex fourier transforms"""

    # Ensure that backward adjoint transforms match the matrix implementations
    c = coords.Coordinate('x')
    d = distributor.Distributor([c])

    b_adj = basis.Fourier(c, size=N, bounds=(0, 2*np.pi), dealias=dealias, library=library, dtype=dtype, adjoint=True)
    u_adj = field.Field(dist=d, bases=(b_adj,), dtype=dtype)

    b_mat_adj = basis.Fourier(c, size=N, bounds=(0, 2*np.pi), dealias=dealias, library='matrix', dtype=dtype, adjoint=True)
    u_mat_adj = field.Field(dist=d, bases=(b_mat_adj,), dtype=dtype)

    u_adj.fill_random(layout='c')
    u_mat_adj['c'] = u_adj['c']

    assert np.allclose(u_adj['g'], u_mat_adj['g'])

@pytest.mark.parametrize('N', [15, 16])
@pytest.mark.parametrize('alpha', [0, 1, 2])
@pytest.mark.parametrize('dealias', [0.5, 1, 1.5])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('library', ['scipy_dct', 'fftw_dct'])

def test_chebyshev_adjoint_backward_matrix(N, alpha, dealias, dtype, library):
    """Tests adjoint backward Chebyshev transforms"""

    # Ensure that backward adjoint transforms match the matrix implementations
    c = coords.Coordinate('x')
    d = distributor.Distributor([c])
   
    b_adj = basis.Ultraspherical(c, size=N, alpha0=0, alpha=alpha, bounds=(-1, 1), dealias=dealias, library=library, adjoint=True)
    u_adj = field.Field(dist=d, bases=(b_adj,), dtype=dtype)

    b_mat_adj = basis.Ultraspherical(c, size=N, alpha0=0, alpha=alpha, bounds=(-1, 1), dealias=dealias, library='matrix', adjoint=True)
    u_mat_adj = field.Field(dist=d, bases=(b_mat_adj,), dtype=dtype)

    u_adj.fill_random(layout='c')
    u_mat_adj['c'] = u_adj['c']

    assert np.allclose(u_adj['g'], u_mat_adj['g'])