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

    u_adj = field.Field(dist=d, bases=(b,), dtype=dtype,adjoint=True)
    
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

    u_adj = field.Field(dist=d, bases=(b,), dtype=dtype, adjoint=True)
    
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
    
    u_adj = field.Field(dist=d, bases=(b,), dtype=dtype, adjoint=True)
    
    u.fill_random(layout='g')
    u_adj.fill_random(layout='c')

    inner_1 = np.vdot(u_adj['c'],u['c'])
    inner_2 = np.vdot(u_adj['g'],u['g'])

    assert np.allclose(inner_1,inner_2)

# # Backwards

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
    
    u_adj = field.Field(dist=d, bases=(b,), dtype=dtype,adjoint=True)
    
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
    
    u_adj = field.Field(dist=d, bases=(b,), dtype=dtype, adjoint=True)
    
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

    u_adj = field.Field(dist=d, bases=(b,), dtype=dtype, adjoint=True)
    
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
    b = basis.RealFourier(c, size=N, bounds=(0, 2*np.pi), dealias=dealias, library=library)
    u_adj = field.Field(dist=d, bases=(b,), dtype=dtype, adjoint=True)

    b_mat = basis.RealFourier(c, size=N, bounds=(0, 2*np.pi), dealias=dealias, library='matrix')
    u_mat_adj = field.Field(dist=d, bases=(b_mat,), dtype=dtype, adjoint=True)
    
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

    b = basis.Fourier(c, size=N, bounds=(0, 2*np.pi), dealias=dealias, library=library, dtype=dtype)
    u_adj = field.Field(dist=d, bases=(b,), dtype=dtype, adjoint=True)

    b_mat = basis.Fourier(c, size=N, bounds=(0, 2*np.pi), dealias=dealias, library='matrix', dtype=dtype)
    u_mat_adj = field.Field(dist=d, bases=(b_mat,), dtype=dtype, adjoint=True)

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
   
    b = basis.Ultraspherical(c, size=N, alpha0=0, alpha=alpha, bounds=(-1, 1), dealias=dealias, library=library)
    u_adj = field.Field(dist=d, bases=(b,), dtype=dtype, adjoint=True)

    b_mat = basis.Ultraspherical(c, size=N, alpha0=0, alpha=alpha, bounds=(-1, 1), dealias=dealias, library='matrix')
    u_mat_adj = field.Field(dist=d, bases=(b_mat,), dtype=dtype, adjoint=True)

    u_adj.fill_random(layout='g')
    u_mat_adj['g'] = u_adj['g']

    assert np.allclose(u_adj['c'], u_mat_adj['c'])

# # Backwards

@pytest.mark.parametrize('N', [16])
@pytest.mark.parametrize('dealias', [0.5, 1, 1.5])
@pytest.mark.parametrize('dtype', [np.float64])
@pytest.mark.parametrize('library', ['fftpack', 'scipy', 'fftw', 'fftw_hc'])
def test_real_fourier_adjoint_backward_matrix(N, dealias, dtype, library):
    """Tests adjoint backward real fourier transforms"""

    # Ensure that backward adjoint transforms match the matrix implementations
    c = coords.Coordinate('x')
    d = distributor.Distributor([c])
    b = basis.RealFourier(c, size=N, bounds=(0, 2*np.pi), dealias=dealias, library=library)
    u_adj = field.Field(dist=d, bases=(b,), dtype=dtype, adjoint=True)

    b_mat = basis.RealFourier(c, size=N, bounds=(0, 2*np.pi), dealias=dealias, library='matrix')
    u_mat_adj = field.Field(dist=d, bases=(b_mat,), dtype=dtype, adjoint=True)
    
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

    b = basis.Fourier(c, size=N, bounds=(0, 2*np.pi), dealias=dealias, library=library, dtype=dtype)
    u_adj = field.Field(dist=d, bases=(b,), dtype=dtype, adjoint=True)

    b_mat = basis.Fourier(c, size=N, bounds=(0, 2*np.pi), dealias=dealias, library='matrix', dtype=dtype)
    u_mat_adj = field.Field(dist=d, bases=(b_mat,), dtype=dtype, adjoint=True)

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
   
    b = basis.Ultraspherical(c, size=N, alpha0=0, alpha=alpha, bounds=(-1, 1), dealias=dealias, library=library)
    u_adj = field.Field(dist=d, bases=(b,), dtype=dtype, adjoint=True)

    b_mat = basis.Ultraspherical(c, size=N, alpha0=0, alpha=alpha, bounds=(-1, 1), dealias=dealias, library='matrix')
    u_mat_adj = field.Field(dist=d, bases=(b_mat,), dtype=dtype, adjoint=True)

    u_adj.fill_random(layout='c')
    u_mat_adj['c'] = u_adj['c']

    assert np.allclose(u_adj['g'], u_mat_adj['g'])

# Curvilinear

# Disk
@pytest.mark.parametrize('k', [0, 1, 2])
@pytest.mark.parametrize('dealias', [0.5, 1, 1.5])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_disk_scalar(k, dealias, dtype):
    """Tests adjoint disk transforms on scalars"""
    Nphi = 4
    Nr   = 8
    c = coords.PolarCoordinates('phi', 'r')
    dist = distributor.Distributor(c, dtype=dtype)
    disk = basis.DiskBasis(c, shape=(Nphi, Nr), radius=1.4, dealias=dealias, dtype=dtype, k=k)
    f     = dist.Field(name='f', bases=disk)
    f_adj = dist.Field(name='f_adj', bases=disk, adjoint=True)
    f.fill_random(layout='c')
    f_adj.fill_random(layout='c')
    inner_1 = np.vdot(f_adj['g'], f['g'])
    inner_2 = np.vdot(f_adj['c'], f['c'])
    assert np.allclose(inner_1, inner_2)

@pytest.mark.parametrize('k', [0, 1, 2])
@pytest.mark.parametrize('dealias', [0.5, 1, 1.5])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_disk_vector(k, dealias, dtype):
    """Tests adjoint disk transforms on vectors"""
    Nphi = 4
    Nr   = 8
    c = coords.PolarCoordinates('phi', 'r')
    dist = distributor.Distributor(c, dtype=dtype)
    disk = basis.DiskBasis(c, shape=(Nphi, Nr), radius=1.4, dealias=dealias, dtype=dtype, k=k)
    f     = dist.VectorField(c, name='f', bases=disk)
    f_adj = dist.VectorField(c, name='f_adj', bases=disk, adjoint=True)
    f.fill_random(layout='c')
    f_adj.fill_random(layout='c')
    inner_1 = np.vdot(f_adj['g'], f['g'])
    inner_2 = np.vdot(f_adj['c'], f['c'])
    assert np.allclose(inner_1, inner_2)

# Annulus
@pytest.mark.parametrize('k', [0, 1, 2])
@pytest.mark.parametrize('dealias', [0.5, 1, 1.5])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_annulus_scalar(k, dealias, dtype):
    """Tests adjoint annulus transforms on scalars"""
    Nphi = 4
    Nr   = 8
    c = coords.PolarCoordinates('phi', 'r')
    dist = distributor.Distributor(c, dtype=dtype)
    annulus = basis.AnnulusBasis(c, shape=(Nphi, Nr), radii=(0.6,1.4), dealias=dealias, dtype=dtype, k=k)
    f     = dist.Field(name='f', bases=annulus)
    f_adj = dist.Field(name='f_adj', bases=annulus, adjoint=True)
    f.fill_random(layout='c')
    f_adj.fill_random(layout='c')
    inner_1 = np.vdot(f_adj['g'], f['g'])
    inner_2 = np.vdot(f_adj['c'], f['c'])
    assert np.allclose(inner_1, inner_2)

@pytest.mark.parametrize('k', [0, 1, 2])
@pytest.mark.parametrize('dealias', [0.5, 1, 1.5])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_annulus_vector(k, dealias, dtype):
    """Tests adjoint annulus transforms on vectors"""
    Nphi = 4
    Nr   = 8
    c = coords.PolarCoordinates('phi', 'r')
    dist = distributor.Distributor(c, dtype=dtype)
    annulus = basis.AnnulusBasis(c, shape=(Nphi, Nr), radii=(0.6,1.4), dealias=dealias, dtype=dtype, k=k)
    f     = dist.VectorField(c, name='f', bases=annulus)
    f_adj = dist.VectorField(c, name='f_adj', bases=annulus, adjoint=True)
    f.fill_random(layout='c')
    f_adj.fill_random(layout='c')
    inner_1 = np.vdot(f_adj['g'], f['g'])
    inner_2 = np.vdot(f_adj['c'], f['c'])
    assert np.allclose(inner_1, inner_2)

# Sphere

@pytest.mark.parametrize('dealias', [0.5, 1, 1.5])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_sphere_scalar(dealias, dtype):
    """Tests adjoint sphere transforms on scalars"""
    lmax = 7
    Ntheta = lmax+1
    Nphi = 2*(lmax+1)
    c = coords.S2Coordinates('phi', 'theta')
    dist = distributor.Distributor(c, dtype=dtype)
    sphere = basis.SphereBasis(c, shape=(Nphi, Ntheta), radius=1.4, dealias=dealias, dtype=dtype)
    f     = dist.Field(name='f', bases=sphere)
    f_adj = dist.Field(name='f_adj', bases=sphere, adjoint=True)
    f.fill_random(layout='c')
    f_adj.fill_random(layout='c')
    inner_1 = np.vdot(f_adj['g'], f['g'])
    inner_2 = np.vdot(f_adj['c'], f['c'])
    assert np.allclose(inner_1, inner_2)

@pytest.mark.parametrize('dealias', [0.5, 1, 1.5])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_sphere_vector(dealias, dtype):
    """Tests adjoint sphere transforms on vectors"""
    lmax = 7
    Ntheta = lmax+1
    Nphi = 2*(lmax+1)
    c = coords.S2Coordinates('phi', 'theta')
    dist = distributor.Distributor(c, dtype=dtype)
    sphere = basis.SphereBasis(c, shape=(Nphi, Ntheta), radius=1.4, dealias=dealias, dtype=dtype)
    f     = dist.VectorField(c, name='f', bases=sphere)
    f_adj = dist.VectorField(c, name='f_adj', bases=sphere, adjoint=True)
    f.fill_random(layout='c')
    f_adj.fill_random(layout='c')
    inner_1 = np.vdot(f_adj['g'], f['g'])
    inner_2 = np.vdot(f_adj['c'], f['c'])
    assert np.allclose(inner_1, inner_2)

# Ball
@pytest.mark.parametrize('k', [0, 1, 2])
@pytest.mark.parametrize('dealias', [0.5, 1, 1.5])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_ball_vector(k, dealias, dtype):
    """Tests adjoint ball transforms on vectors"""
    lmax = 7
    Ntheta = lmax+1
    Nphi = 2*(lmax+1)
    Nr = 8
    c = coords.SphericalCoordinates('phi', 'theta', 'r')
    dist = distributor.Distributor(c, dtype=dtype)
    sphere = basis.BallBasis(c, shape=(Nphi, Ntheta, Nr), radius=1.4, dealias=dealias, dtype=dtype, k=k)
    f     = dist.VectorField(c, name='f', bases=sphere)
    f_adj = dist.VectorField(c, name='f_adj', bases=sphere, adjoint=True)
    f.fill_random(layout='c')
    f_adj.fill_random(layout='c')
    inner_1 = np.vdot(f_adj['g'], f['g'])
    inner_2 = np.vdot(f_adj['c'], f['c'])
    assert np.allclose(inner_1, inner_2)

# Shell
@pytest.mark.parametrize('k', [0, 1, 2])
@pytest.mark.parametrize('dealias', [0.5, 1, 1.5])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_shell_vector(k, dealias, dtype):
    """Tests adjoint shell transforms on vectors"""
    lmax = 7
    Ntheta = lmax+1
    Nphi = 2*(lmax+1)
    Nr = 8
    c = coords.SphericalCoordinates('phi', 'theta', 'r')
    dist = distributor.Distributor(c, dtype=dtype)
    sphere = basis.ShellBasis(c, shape=(Nphi, Ntheta, Nr), radii=(0.8, 1.4), dealias=dealias, dtype=dtype, k=k)
    f     = dist.VectorField(c, name='f', bases=sphere)
    f_adj = dist.VectorField(c, name='f_adj', bases=sphere, adjoint=True)
    f.fill_random(layout='c')
    f_adj.fill_random(layout='c')
    inner_1 = np.vdot(f_adj['g'], f['g'])
    inner_2 = np.vdot(f_adj['c'], f['c'])
    assert np.allclose(inner_1, inner_2)