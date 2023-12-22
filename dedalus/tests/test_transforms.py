
import pytest
import numpy as np
from dedalus.core import coords, distributor, basis, field, operators
from dedalus.tools.cache import CachedFunction
from mpi4py import MPI

comm = MPI.COMM_WORLD


## Cartesian
N_range = [8]
dealias_range = [0.5, 1, 3/2]
jacobi_range = [-0.5 , 0]
dtypes = [np.float64, np.complex128]


@pytest.mark.parametrize('N', [16])
@pytest.mark.parametrize('dealias', [0.5, 1, 1.5])
@pytest.mark.parametrize('dtype', [np.float64])
@pytest.mark.parametrize('library', ['fftpack', 'scipy', 'fftw', 'fftw_hc'])
def test_real_fourier_libraries_backward(N, dealias, dtype, library):
    """Tests that fast real Fourier transforms match matrix transforms."""
    c = coords.Coordinate('x')
    d = distributor.Distributor([c])
    # Matrix
    b_mat = basis.RealFourier(c, size=N, bounds=(0, 2*np.pi), dealias=dealias, library='matrix')
    u_mat = field.Field(dist=d, bases=(b_mat,), dtype=dtype)
    u_mat.preset_scales(dealias)
    u_mat['c'] = np.random.randn(N)
    # Library
    b_lib = basis.RealFourier(c, size=N, bounds=(0, 2*np.pi), dealias=dealias, library=library)
    u_lib = field.Field(dist=d, bases=(b_lib,), dtype=dtype)
    u_lib.preset_scales(dealias)
    u_lib['c'] = u_mat['c']
    assert np.allclose(u_mat['g'], u_lib['g'])


@pytest.mark.parametrize('N', [16])
@pytest.mark.parametrize('dealias', [0.5, 1, 1.5])
@pytest.mark.parametrize('dtype', [np.float64])
@pytest.mark.parametrize('library', ['fftpack', 'scipy', 'fftw', 'fftw_hc'])
def test_real_fourier_libraries_forward(N, dealias, dtype, library):
    """Tests that fast real Fourier transforms match matrix transforms."""
    c = coords.Coordinate('x')
    d = distributor.Distributor([c])
    # Matrix
    b_mat = basis.RealFourier(c, size=N, bounds=(0, 2*np.pi), dealias=dealias, library='matrix')
    u_mat = field.Field(dist=d, bases=(b_mat,), dtype=dtype)
    u_mat.preset_scales(dealias)
    u_mat['g'] = np.random.randn(int(np.ceil(dealias * N)))
    # Library
    b_lib = basis.RealFourier(c, size=N, bounds=(0, 2*np.pi), dealias=dealias, library=library)
    u_lib = field.Field(dist=d, bases=(b_lib,), dtype=dtype)
    u_lib.preset_scales(dealias)
    u_lib['g'] = u_mat['g']
    assert np.allclose(u_mat['c'], u_lib['c'])


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('dealias', dealias_range)
def test_CF_scalar_roundtrip(N, dealias):
    if comm.size == 1:
        c = coords.Coordinate('x')
        d = distributor.Distributor([c])
        xb = basis.ComplexFourier(c, size=N, bounds=(0, 1), dealias=dealias)
        x = d.local_grid(xb, scale=dealias)
        # Scalar transforms
        u = field.Field(dist=d, bases=(xb,), dtype=np.complex128)
        u.preset_scales(dealias)
        u['g'] = ug = np.exp(2*np.pi*1j*x)
        u['c']
        assert np.allclose(u['g'], ug)
    else:
        pytest.skip("Can only test 1D transform in serial")


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('dealias', dealias_range)
def test_RF_scalar_roundtrip(N, dealias):
    if comm.size == 1:
        c = coords.Coordinate('x')
        d = distributor.Distributor([c])
        xb = basis.RealFourier(c, size=N, bounds=(0, 1), dealias=dealias)
        x = d.local_grid(xb, scale=dealias)
        # Scalar transforms
        u = field.Field(dist=d, bases=(xb,), dtype=np.float64)
        u.preset_scales(dealias)
        u['g'] = ug = np.cos(2*np.pi*x + np.pi/4)
        u['c']
        assert np.allclose(u['g'], ug)
    else:
        pytest.skip("Can only test 1D transform in serial")


@pytest.mark.parametrize('a', jacobi_range)
@pytest.mark.parametrize('b', jacobi_range)
@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtypes)
def test_J_scalar_roundtrip(a, b, N, dealias, dtype):
    if comm.size == 1:
        c = coords.Coordinate('x')
        d = distributor.Distributor([c])
        xb = basis.Jacobi(c, a=a, b=b, size=N, bounds=(0, 1), dealias=dealias)
        x = d.local_grid(xb, scale=dealias)
        # Scalar transforms
        u = field.Field(dist=d, bases=(xb,), dtype=dtype)
        u.preset_scales(dealias)
        u['g'] = ug = 2*x**2 - 1
        u['c']
        assert np.allclose(u['g'], ug)
    else:
        pytest.skip("Can only test 1D transform in serial")


@pytest.mark.parametrize('N', [15, 16])
@pytest.mark.parametrize('alpha', [0, 1, 2])
@pytest.mark.parametrize('dealias', [0.5, 1, 1.5])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('library', ['scipy_dct', 'fftw_dct'])
def test_chebyshev_libraries_backward(N, alpha, dealias, dtype, library):
    """Tests that fast Chebyshev transforms match matrix transforms."""
    c = coords.Coordinate('x')
    d = distributor.Distributor([c])
    # Matrix
    b_mat = basis.Ultraspherical(c, size=N, alpha0=0, alpha=alpha, bounds=(-1, 1), dealias=dealias, library='matrix')
    u_mat = field.Field(dist=d, bases=(b_mat,), dtype=dtype)
    u_mat.preset_scales(dealias)
    u_mat['c'] = np.random.randn(N)
    # Library
    b_lib = basis.Ultraspherical(c, size=N, alpha0=0, alpha=alpha, bounds=(-1, 1), dealias=dealias, library=library)
    u_lib = field.Field(dist=d, bases=(b_lib,), dtype=dtype)
    u_lib.preset_scales(dealias)
    u_lib['c'] = u_mat['c']
    assert np.allclose(u_mat['g'], u_lib['g'])


@pytest.mark.parametrize('N', [15, 16])
@pytest.mark.parametrize('alpha', [0, 1, 2])
@pytest.mark.parametrize('dealias', [0.5, 1, 1.5])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('library', ['scipy_dct', 'fftw_dct'])
def test_chebyshev_libraries_forward(N, alpha, dealias, dtype, library):
    """Tests that fast Chebyshev transforms match matrix transforms."""
    c = coords.Coordinate('x')
    d = distributor.Distributor([c])
    # Matrix
    b_mat = basis.Ultraspherical(c, size=N, alpha0=0, alpha=alpha, bounds=(-1, 1), dealias=dealias, library='matrix')
    u_mat = field.Field(dist=d, bases=(b_mat,), dtype=dtype)
    u_mat.preset_scales(dealias)
    u_mat['g'] = np.random.randn(int(np.ceil(dealias * N)))
    # Library
    b_lib = basis.Ultraspherical(c, size=N, alpha0=0, alpha=alpha, bounds=(-1, 1), dealias=dealias, library=library)
    u_lib = field.Field(dist=d, bases=(b_lib,), dtype=dtype)
    u_lib.preset_scales(dealias)
    u_lib['g'] = u_mat['g']
    assert np.allclose(u_mat['c'], u_lib['c'])


@CachedFunction
def build_CF_CF(Nx, Ny, dealias_x, dealias_y):
    c = coords.CartesianCoordinates('x', 'y')
    d = distributor.Distributor((c,))
    xb = basis.ComplexFourier(c.coords[0], size=Nx, bounds=(0, np.pi), dealias=dealias_x)
    yb = basis.ComplexFourier(c.coords[1], size=Ny, bounds=(0, np.pi), dealias=dealias_y)
    x = d.local_grid(xb, scale=dealias_x)
    y = d.local_grid(yb, scale=dealias_y)
    return c, d, xb, yb, x, y


@pytest.mark.parametrize('Nx', N_range)
@pytest.mark.parametrize('Ny', N_range)
@pytest.mark.parametrize('dealias_x', dealias_range)
@pytest.mark.parametrize('dealias_y', dealias_range)
def test_CF_CF_scalar_roundtrip(Nx, Ny, dealias_x, dealias_y):
    c, d, xb, yb, x, y = build_CF_CF(Nx, Ny, dealias_x, dealias_y)
    f = field.Field(dist=d, bases=(xb,yb), dtype=np.complex128)
    f.preset_scales((dealias_x, dealias_y))
    f['g'] = fg = np.exp(2j*x) * np.exp(2j*y + 1j*np.pi/3) + 3 + np.exp(2j*y)
    f['c']
    assert np.allclose(f['g'], fg)


@CachedFunction
def build_RF_RF(Nx, Ny, dealias_x, dealias_y):
    c = coords.CartesianCoordinates('x', 'y')
    d = distributor.Distributor((c,))
    xb = basis.RealFourier(c.coords[0], size=Nx, bounds=(0, np.pi), dealias=dealias_x)
    yb = basis.RealFourier(c.coords[1], size=Ny, bounds=(0, np.pi), dealias=dealias_y)
    x = d.local_grid(xb, scale=dealias_x)
    y = d.local_grid(yb, scale=dealias_y)
    return c, d, xb, yb, x, y


@pytest.mark.parametrize('Nx', N_range)
@pytest.mark.parametrize('Ny', N_range)
@pytest.mark.parametrize('dealias_x', dealias_range)
@pytest.mark.parametrize('dealias_y', dealias_range)
def test_RF_RF_scalar_roundtrip(Nx, Ny, dealias_x, dealias_y):
    c, d, xb, yb, x, y = build_RF_RF(Nx, Ny, dealias_x, dealias_y)
    f = field.Field(dist=d, bases=(xb,yb), dtype=np.float64)
    f.preset_scales((dealias_x, dealias_y))
    f['g'] = fg = np.sin(2*x) + np.cos(2*y + np.pi/3) + 3 + np.sin(2*y)
    f['c']
    assert np.allclose(f['g'], fg)


@CachedFunction
def build_CF_J(a, b, Nx, Ny, dealias_x, dealias_y):
    c = coords.CartesianCoordinates('x', 'y')
    d = distributor.Distributor((c,))
    xb = basis.ComplexFourier(c.coords[0], size=Nx, bounds=(0, np.pi), dealias=dealias_x)
    yb = basis.Jacobi(c.coords[1], a=a, b=b, size=Ny, bounds=(0, 1), dealias=dealias_y)
    x = d.local_grid(xb, scale=dealias_x)
    y = d.local_grid(yb, scale=dealias_y)
    return c, d, xb, yb, x, y


@pytest.mark.parametrize('a', jacobi_range)
@pytest.mark.parametrize('b', jacobi_range)
@pytest.mark.parametrize('Nx', N_range)
@pytest.mark.parametrize('Ny', N_range)
@pytest.mark.parametrize('dealias_x', dealias_range)
@pytest.mark.parametrize('dealias_y', dealias_range)
def test_CF_J_scalar_roundtrip(a, b, Nx, Ny, dealias_x, dealias_y):
    c, d, xb, yb, x, y = build_CF_J(a, b, Nx, Ny, dealias_x, dealias_y)
    f = field.Field(dist=d, bases=(xb,yb,), dtype=np.complex128)
    f.preset_scales((dealias_x, dealias_y))
    f['g'] = fg = np.sin(2*x) * y**5
    f['c']
    assert np.allclose(f['g'], fg)


@pytest.mark.parametrize('a', jacobi_range)
@pytest.mark.parametrize('b', jacobi_range)
@pytest.mark.parametrize('Nx', N_range)
@pytest.mark.parametrize('Ny', N_range)
@pytest.mark.parametrize('dealias_x', dealias_range)
@pytest.mark.parametrize('dealias_y', dealias_range)
def test_CF_J_vector_roundtrip(a, b, Nx, Ny, dealias_x, dealias_y):
    c, d, xb, yb, x, y = build_CF_J(a, b, Nx, Ny, dealias_x, dealias_y)
    u = field.Field(dist=d, bases=(xb,yb,), tensorsig=(c,), dtype=np.complex128)
    u.preset_scales((dealias_x, dealias_y))
    u['g'] = ug = np.array([np.cos(2*x) * 2 * y**2, np.sin(2*x) * y + y])
    u['c']
    assert np.allclose(u['g'], ug)


@pytest.mark.parametrize('a', jacobi_range)
@pytest.mark.parametrize('b', jacobi_range)
@pytest.mark.parametrize('Nx', N_range)
@pytest.mark.parametrize('Ny', N_range)
@pytest.mark.parametrize('dealias_x', dealias_range)
@pytest.mark.parametrize('dealias_y', dealias_range)
def test_CF_J_1d_vector_roundtrip(a, b, Nx, Ny, dealias_x, dealias_y):
    c, d, xb, yb, x, y = build_CF_J(a, b, Nx, Ny, dealias_x, dealias_y)
    v = field.Field(dist=d, bases=(xb,), tensorsig=(c,), dtype=np.complex128)
    v.preset_scales((dealias_x, dealias_y))
    v['g'] = vg = np.array([np.cos(2*x) * 2, np.sin(2*x) + 1])
    v['c']
    assert np.allclose(v['g'], vg)


## Sphere

@CachedFunction
def build_sphere_2d(Nphi, Ntheta, radius, dealias, dtype):
    c = coords.S2Coordinates('phi', 'theta')
    d = distributor.Distributor((c,))
    b = basis.SphereBasis(c, (Nphi, Ntheta), radius=radius, dealias=(dealias, dealias), dtype=dtype)
    phi, theta = d.local_grids(b, scales=(dealias, dealias))
    return c, d, b, phi, theta

@CachedFunction
def build_sphere_3d(Nphi, Ntheta, radius, dealias, dtype):
    c = coords.SphericalCoordinates('phi', 'theta', 'r')
    d = distributor.Distributor((c,))
    b = basis.SphereBasis(c, (Nphi, Ntheta), radius=radius, dealias=(dealias, dealias), dtype=dtype)
    phi, theta = d.local_grids(b, scales=(dealias, dealias, dealias))
    return c, d, b, phi, theta

Nphi_range = [16]
Ntheta_range = [16]
radius_range = [1.5]
basis_range = [build_sphere_2d, build_sphere_3d]
dealias_range = [1/2, 1, 3/2]
dtype_range = [np.float64, np.complex128]
layout_range = ['g', 'c']
rank_range = [0, 1, 2]

@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('radius', radius_range)
@pytest.mark.parametrize('basis', basis_range)
@pytest.mark.parametrize('dealias', dealias_range)
def test_sphere_complex_scalar_backward(Nphi, Ntheta, radius, basis, dealias):
    c, d, b, phi, theta = basis(Nphi, Ntheta, radius, dealias, np.complex128)
    f = field.Field(dist=d, bases=(b,), dtype=np.complex128)
    f.preset_scales(dealias)
    m, ell, *_ = d.coeff_layout.local_group_arrays(b.domain(d), scales=1)
    f['c'][(m == -2) * (ell == 2)] = 1
    fg = np.sqrt(15) / 4 * np.sin(theta)**2 * np.exp(-2j*phi)
    assert np.allclose(fg, f['g'])

@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('radius', radius_range)
@pytest.mark.parametrize('basis', basis_range)
@pytest.mark.parametrize('dealias', dealias_range)
def test_sphere_complex_scalar_forward(Nphi, Ntheta, radius, basis, dealias):
    c, d, b, phi, theta = basis(Nphi, Ntheta, radius, dealias, np.complex128)
    f = field.Field(dist=d, bases=(b,), dtype=np.complex128)
    f.preset_scales(dealias)
    m, ell, *_  = d.coeff_layout.local_group_arrays(b.domain(d), scales=1)
    f['g'] = np.sqrt(15) / 4 * np.sin(theta)**2 * np.exp(-2j*phi)
    fc = np.zeros_like(f['c'])
    fc[(m == -2) * (ell == 2)] = 1
    assert np.allclose(fc, f['c'])

@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('radius', radius_range)
@pytest.mark.parametrize('basis', basis_range)
@pytest.mark.parametrize('dealias', dealias_range)
def test_sphere_real_scalar_backward(Nphi, Ntheta, radius, basis, dealias):
    c, d, b, phi, theta = basis(Nphi, Ntheta, radius, dealias, np.float64)
    f = field.Field(dist=d, bases=(b,), dtype=np.float64)
    f.preset_scales(dealias)
    m, ell, *_  = d.coeff_layout.local_group_arrays(b.domain(d), scales=1)
    f['c'][(m == 2) * (ell == 2)] = 1
    fg = np.sqrt(15) / 4 * np.sin(theta)**2 * (np.cos(2*phi) - np.sin(2*phi))
    assert np.allclose(fg, f['g'])

@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('radius', radius_range)
@pytest.mark.parametrize('basis', basis_range)
@pytest.mark.parametrize('dealias', dealias_range)
def test_sphere_real_scalar_forward(Nphi, Ntheta, radius, basis, dealias):
    c, d, b, phi, theta = basis(Nphi, Ntheta, radius, dealias, np.float64)
    f = field.Field(dist=d, bases=(b,), dtype=np.float64)
    f.preset_scales(dealias)
    m, ell, *_  = d.coeff_layout.local_group_arrays(b.domain(d), scales=1)
    f['g'] = np.sqrt(15) / 4 * np.sin(theta)**2 * (np.cos(2*phi) - np.sin(2*phi))
    fc = np.zeros_like(f['c'])
    fc[(m == 2) * (ell == 2)] = 1
    assert np.allclose(fc, f['c'])

@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('radius', radius_range)
@pytest.mark.parametrize('basis', basis_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('layout', layout_range)
@pytest.mark.parametrize('rank', rank_range)
def test_sphere_roundtrip_noise(Nphi, Ntheta, radius, basis, dealias, dtype, layout, rank):
    c, d, b, phi, theta = basis(Nphi, Ntheta, radius, dealias, dtype)
    tensorsig = (c,) * rank
    f = field.Field(dist=d, bases=(b,), tensorsig=tensorsig, dtype=dtype)
    f.preset_scales(dealias)
    other = {'g':'c', 'c':'g'}[layout]
    f[other] = np.random.randn(*f[other].shape)
    f_layout = f[layout].copy()
    f[other]
    assert np.allclose(f_layout, f[layout])



## Polar bases
Nphi_range = [16]
Nr_range = [16]
radius_range = [2.5]
alpha_range = [0, 1]
k_range = [0, 1, 2]
dealias_range = [0.5, 1, 1.5]


@CachedFunction
def build_disk(Nphi, Nr, radius, alpha, k, dealias, dtype):
    c = coords.PolarCoordinates('phi', 'r')
    d = distributor.Distributor((c,))
    b = basis.DiskBasis(c, (Nphi, Nr), dtype=dtype, radius=radius, alpha=alpha, k=k, dealias=(dealias, dealias))
    return c, d, b


@CachedFunction
def build_annulus(Nphi, Nr, radius, alpha, k, dealias, dtype):
    c = coords.PolarCoordinates('phi', 'r')
    d = distributor.Distributor((c,))
    alpha = (alpha-1/2, alpha-1/2)
    b = basis.AnnulusBasis(c, (Nphi, Nr), radii=(radius,radius+1.3), alpha=alpha, k=k, dealias=(dealias, dealias), dtype=dtype)
    return c, d, b


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('radius', radius_range)
@pytest.mark.parametrize('alpha', alpha_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('build_basis', [build_annulus, build_disk])
def test_polar_scalar_roundtrip(Nphi, Nr, radius, alpha, k, dealias, dtype, build_basis):
    c, d, b = build_basis(Nphi, Nr, radius, alpha, k, dealias, dtype)
    phi, r = d.local_grids(b, scales=(dealias, dealias))
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    f.preset_scales((dealias, dealias))
    f['g'] = (r*np.cos(phi))**3
    fg = f['g'].copy()
    f['c']
    assert np.allclose(f['g'], fg)


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('radius', radius_range)
@pytest.mark.parametrize('alpha', alpha_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('layout', layout_range)
@pytest.mark.parametrize('rank', rank_range)
@pytest.mark.parametrize('build_basis', [build_annulus, build_disk])
def test_polar_roundtrip_noise(Nphi, Nr, radius, alpha, k, dealias, dtype, layout, rank, build_basis):
    c, d, b = build_basis(Nphi, Nr, radius, alpha, k, dealias, dtype)
    tensorsig = (c,) * rank
    f = field.Field(dist=d, bases=(b,), tensorsig=tensorsig, dtype=dtype)
    f.preset_scales((dealias, dealias))
    other = {'g':'c', 'c':'g'}[layout]
    f[other] = np.random.randn(*f[other].shape)
    f_layout = f[layout].copy()
    f[other]
    assert np.allclose(f_layout, f[layout])


@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('radius', radius_range)
@pytest.mark.parametrize('alpha', alpha_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('build_basis', [build_annulus, build_disk])
def test_polar_scalar_roundtrip_mmax0(Nr, radius, alpha, k, dealias, dtype, build_basis):
    c, d, b = build_basis(1, Nr, radius, alpha, k, dealias, dtype)
    phi, r = d.local_grids(b, scales=(dealias, dealias))
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    f.preset_scales((dealias, dealias))
    f['g'] = r**4 + 0*phi
    fg = f['g'].copy()
    f['c']
    assert np.allclose(f['g'], fg)


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('radius', radius_range)
@pytest.mark.parametrize('alpha', alpha_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('build_basis', [build_annulus, build_disk])
def test_polar_vector_roundtrip(Nphi, Nr, radius, alpha, k, dealias, dtype, build_basis):
    c, d, b = build_basis(Nphi, Nr, radius, alpha, k, dealias, dtype=dtype)
    phi, r = d.local_grids(b, scales=(dealias, dealias))
    x = r*np.cos(phi)
    y = r*np.sin(phi)
    ex = np.array([-np.sin(phi)+0*r, np.cos(phi)+0*r])
    f = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
    f.preset_scales((dealias, dealias))
    f['g'] = x * ex
    fg = f['g'].copy()
    f['c']
    assert np.allclose(f['g'], fg)


@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('radius', radius_range)
@pytest.mark.parametrize('alpha', alpha_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('build_basis', [build_annulus, build_disk])
def test_polar_vector_roundtrip_mmax0(Nr, radius, alpha, k, dealias, dtype, build_basis):
    c, d, b = build_basis(1, Nr, radius, alpha, k, dealias, dtype=dtype)
    phi, r = d.local_grids(b, scales=(dealias, dealias))
    f = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
    f.preset_scales((dealias, dealias))
    f['g'][1] = 6*r**5 + 0*phi
    fg = f['g'].copy()
    f['c']
    assert np.allclose(f['g'], fg)


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('radius', radius_range)
@pytest.mark.parametrize('alpha', alpha_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('build_basis', [build_annulus, build_disk])
def test_polar_tensor_roundtrip(Nphi, Nr, radius, alpha, k, dealias, dtype, build_basis):
    c, d, b = build_basis(Nphi, Nr, radius, alpha, k, dealias, dtype=dtype)
    phi, r = d.local_grids(b, scales=(dealias, dealias))
    x = r*np.cos(phi)
    ex = np.array([-np.sin(phi)+0*r, np.cos(phi)+0*r])
    exex = ex[None, :, ...] * ex[:, None, ...]
    f = field.Field(dist=d, bases=(b,), tensorsig=(c,c), dtype=dtype)
    f.preset_scales((dealias, dealias))
    f['g'] = 6*x * exex
    fg = f['g'].copy()
    f['c']
    assert np.allclose(f['g'], fg)


@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('radius', radius_range)
@pytest.mark.parametrize('alpha', alpha_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('build_basis', [build_annulus, build_disk])
def test_polar_tensor_roundtrip_mmax0(Nr, radius, alpha, k, dealias, dtype, build_basis):
    c, d, b = build_basis(1, Nr, radius, alpha, k, dealias, dtype=dtype)
    phi, r = d.local_grids(b, scales=(dealias, dealias))
    f = field.Field(dist=d, bases=(b,), tensorsig=(c,c), dtype=dtype)
    f.preset_scales((dealias, dealias))
    f['g'][1,1] = r**2 + 0*phi
    fg = f['g'].copy()
    f['c']
    assert np.allclose(f['g'], fg)


## Cylinders

@CachedFunction
def build_periodic_cylinder(Nz, Nphi, Nr, length, radius, alpha, k, dealias, dtype):
    cz = coords.Coordinate('z')
    cp = coords.PolarCoordinates('phi', 'r')
    c = coords.DirectProduct(cz, cp)
    d = distributor.Distributor(c)
    bz = basis.Fourier(cz, Nz, bounds=(0, length), dealias=dealias, dtype=dtype)
    bp = basis.DiskBasis(cp, (Nphi, Nr), dtype=dtype, radius=radius, alpha=alpha, k=k, dealias=(dealias, dealias))
    return c, d, (bz, bp)

@CachedFunction
def build_periodic_cylindrical_annulus(Nz, Nphi, Nr, length, radius, alpha, k, dealias, dtype):
    cz = coords.Coordinate('z')
    cp = coords.PolarCoordinates('phi', 'r')
    c = coords.DirectProduct(cz, cp)
    d = distributor.Distributor(c)
    bz = basis.Fourier(cz, Nz, bounds=(0, length), dealias=dealias, dtype=dtype)
    bp = basis.AnnulusBasis(cp, (Nphi, Nr), dtype=dtype, radii=(radius,radius+1.3), alpha=alpha, k=k, dealias=(dealias, dealias))
    return c, d, (bz, bp)

Nz_range = [8]
Nphi_range = [16]
Nr_range = [16]
length_range = [1.7]
radius_range = [2.5]
alpha_range = [0, 1]
k_range = [0, 1, 2]
dealias_range = [1/2, 1, 3/2]
dtype_range = [np.float64, np.complex128]
layout_range = ['g', 'c']
rank_range = [0, 1, 2]

@pytest.mark.parametrize('Nz', Nr_range)
@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('length', length_range)
@pytest.mark.parametrize('radius', radius_range)
@pytest.mark.parametrize('alpha', alpha_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('layout', layout_range)
@pytest.mark.parametrize('rank', rank_range)
@pytest.mark.parametrize('build_basis', [build_periodic_cylinder, build_periodic_cylindrical_annulus])
def test_cylinder_roundtrip_noise(Nz, Nphi, Nr, length, radius, alpha, k, dealias, dtype, layout, rank, build_basis):
    c, d, b = build_basis(Nz, Nphi, Nr, length, radius, alpha, k, dealias, dtype)
    tensorsig = (c,) * rank
    f = field.Field(dist=d, bases=b, tensorsig=tensorsig, dtype=dtype)
    f.preset_scales(dealias)
    other = {'g':'c', 'c':'g'}[layout]
    f[other] = np.random.randn(*f[other].shape)
    f_layout = f[layout].copy()
    f[other]
    assert np.allclose(f_layout, f[layout])

@pytest.mark.parametrize('Nz', Nr_range)
@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('length', length_range)
@pytest.mark.parametrize('radius', radius_range)
@pytest.mark.parametrize('alpha', alpha_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('layout', layout_range)
@pytest.mark.parametrize('build_basis', [build_periodic_cylinder, build_periodic_cylindrical_annulus])
def test_cylinder_axial_vector_roundtrip_noise(Nz, Nphi, Nr, length, radius, alpha, k, dealias, dtype, layout, build_basis):
    c, d, b = build_basis(Nz, Nphi, Nr, length, radius, alpha, k, dealias, dtype)
    tensorsig = (c,)
    f = field.Field(dist=d, bases=b, tensorsig=tensorsig, dtype=dtype)
    f.preset_scales(dealias)
    other = {'g':'c', 'c':'g'}[layout]
    f[other][0] = np.random.randn(*f[other][0].shape)
    assert np.allclose(f[layout][1:], 0)


## Shell

@CachedFunction
def build_shell(Nphi, Ntheta, Nr, radii, alpha, k, dealias, dtype):
    c = coords.SphericalCoordinates('phi', 'theta', 'r')
    d = distributor.Distributor((c,))
    b  = basis.ShellBasis(c, (Nphi, Ntheta, Nr), radii=radii, alpha=alpha, k=k, dealias=(dealias, dealias, dealias), dtype=dtype)
    phi, theta, r = d.local_grids(b, scales=(dealias, dealias, dealias))
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return c, d, b, phi, theta, r, x, y, z

Nphi_range = [8]
Ntheta_range = [8]
Nr_range = [8]
radii_range = [(0.5, 3)]
alpha_range = [(-1/2, -1/2), (0, 0)]
k_range = [0, 1, 2]
dealias_range = [1/2, 1, 3/2]
dtype_range = [np.float64, np.complex128]
layout_range = ['g', 'c']
rank_range = [0, 1, 2]

@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('radii', radii_range)
@pytest.mark.parametrize('alpha', alpha_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('layout', layout_range)
@pytest.mark.parametrize('rank', rank_range)
def test_shell_radial_roundtrip_noise(Nr, radii, alpha, k, dealias, dtype, layout, rank):
    c, d, b, phi, theta, r, x, y, z = build_shell(4, 4, Nr, radii, alpha, k, dealias, dtype)
    tensorsig = (c,) * rank
    f = field.Field(dist=d, bases=(b.radial_basis,), tensorsig=tensorsig, dtype=dtype)
    f.preset_scales((dealias, dealias, dealias))
    other = {'g':'c', 'c':'g'}[layout]
    f[other] = np.random.randn(*f[other].shape)
    f_layout = f[layout].copy()
    f[other]
    assert np.allclose(f_layout, f[layout])

@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('radii', radii_range)
@pytest.mark.parametrize('alpha', alpha_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('layout', layout_range)
@pytest.mark.parametrize('rank', rank_range)
def test_shell_roundtrip_noise(Nphi, Ntheta, Nr, radii, alpha, k, dealias, dtype, layout, rank):
    c, d, b, phi, theta, r, x, y, z = build_shell(Nphi, Ntheta, Nr, radii, alpha, k, dealias, dtype)
    tensorsig = (c,) * rank
    f = field.Field(dist=d, bases=(b,), tensorsig=tensorsig, dtype=dtype)
    f.preset_scales((dealias, dealias, dealias))
    other = {'g':'c', 'c':'g'}[layout]
    f[other] = np.random.randn(*f[other].shape)
    f_layout = f[layout].copy()
    f[other]
    assert np.allclose(f_layout, f[layout])


## Ball

@CachedFunction
def build_ball(Nphi, Ntheta, Nr, radius, alpha, k, dealias, dtype):
    c = coords.SphericalCoordinates('phi', 'theta', 'r')
    d = distributor.Distributor((c,))
    b = basis.BallBasis(c, (Nphi, Ntheta, Nr), radius=radius, alpha=alpha, k=k, dealias=(dealias, dealias, dealias), dtype=dtype)
    phi, theta, r = d.local_grids(b, scales=(dealias, dealias, dealias))
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return c, d, b, phi, theta, r, x, y, z

Nphi_range = [8]
Ntheta_range = [8]
Nr_range = [16]
radius_range = [1.5]
alpha_range = [0, 1]
k_range = [0, 1, 2]
dealias_range = [1/2, 1, 3/2]
dtype_range = [np.float64, np.complex128]
layout_range = ['g', 'c']
rank_range = [0, 1, 2]

@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('radius', radius_range)
@pytest.mark.parametrize('alpha', alpha_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('layout', layout_range)
@pytest.mark.parametrize('rank', rank_range)
def test_ball_radial_roundtrip_noise(Nr, radius, alpha, k, dealias, dtype, layout, rank):
    c, d, b, phi, theta, r, x, y, z = build_ball(4, 4, Nr, radius, alpha, k, dealias, dtype)
    tensorsig = (c,) * rank
    f = field.Field(dist=d, bases=(b.radial_basis,), tensorsig=tensorsig, dtype=dtype)
    f.preset_scales((dealias, dealias, dealias))
    other = {'g':'c', 'c':'g'}[layout]
    f[other] = np.random.randn(*f[other].shape)
    f_layout = f[layout].copy()
    f[other]
    assert np.allclose(f_layout, f[layout])

@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('radius', radius_range)
@pytest.mark.parametrize('alpha', alpha_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', dtype_range)
@pytest.mark.parametrize('layout', layout_range)
@pytest.mark.parametrize('rank', rank_range)
def test_ball_roundtrip_noise(Nphi, Ntheta, Nr, radius, alpha, k, dealias, dtype, layout, rank):
    c, d, b, phi, theta, r, x, y, z = build_ball(Nphi, Ntheta, Nr, radius, alpha, k, dealias, dtype)
    tensorsig = (c,) * rank
    f = field.Field(dist=d, bases=(b,), tensorsig=tensorsig, dtype=dtype)
    f.preset_scales((dealias, dealias, dealias))
    other = {'g':'c', 'c':'g'}[layout]
    f[other] = np.random.randn(*f[other].shape)
    f_layout = f[layout].copy()
    f[other]
    assert np.allclose(f_layout, f[layout])

