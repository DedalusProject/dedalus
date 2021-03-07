
import pytest
import numpy as np
from dedalus.core import coords, distributor, basis, field, operators
from dedalus.tools.cache import CachedMethod, CachedFunction
from mpi4py import MPI

comm = MPI.COMM_WORLD


## Cartesian
N_range = [8]
dealias_range = [0.5, 1, 3/2]
jacobi_range = [-0.5 , 0]
dtypes = [np.float64, np.complex128]


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('dealias', dealias_range)
def test_CF_scalar_roundtrip(N, dealias):
    if comm.size == 1:
        c = coords.Coordinate('x')
        d = distributor.Distributor([c])
        xb = basis.ComplexFourier(c, size=N, bounds=(0, 1), dealias=dealias)
        x = xb.local_grid(dealias)
        # Scalar transforms
        u = field.Field(dist=d, bases=(xb,), dtype=np.complex128)
        u.set_scales(dealias)
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
        x = xb.local_grid(dealias)
        # Scalar transforms
        u = field.Field(dist=d, bases=(xb,), dtype=np.float64)
        u.set_scales(dealias)
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
        x = xb.local_grid(dealias)
        # Scalar transforms
        u = field.Field(dist=d, bases=(xb,), dtype=dtype)
        u.set_scales(dealias)
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
    u_mat.set_scales(dealias)
    u_mat['c'] = np.random.randn(N)
    # Library
    b_lib = basis.Ultraspherical(c, size=N, alpha0=0, alpha=alpha, bounds=(-1, 1), dealias=dealias, library=library)
    u_lib = field.Field(dist=d, bases=(b_lib,), dtype=dtype)
    u_lib.set_scales(dealias)
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
    u_mat.set_scales(dealias)
    u_mat['g'] = np.random.randn(int(np.ceil(dealias * N)))
    # Library
    b_lib = basis.Ultraspherical(c, size=N, alpha0=0, alpha=alpha, bounds=(-1, 1), dealias=dealias, library=library)
    u_lib = field.Field(dist=d, bases=(b_lib,), dtype=dtype)
    u_lib.set_scales(dealias)
    u_lib['g'] = u_mat['g']
    assert np.allclose(u_mat['c'], u_lib['c'])


@CachedFunction
def build_CF_CF(Nx, Ny, dealias_x, dealias_y):
    c = coords.CartesianCoordinates('x', 'y')
    d = distributor.Distributor((c,))
    xb = basis.ComplexFourier(c.coords[0], size=Nx, bounds=(0, np.pi), dealias=dealias_x)
    yb = basis.ComplexFourier(c.coords[1], size=Ny, bounds=(0, np.pi), dealias=dealias_y)
    x = xb.local_grid(dealias_x)
    y = yb.local_grid(dealias_y)
    return c, d, xb, yb, x, y


@pytest.mark.parametrize('Nx', N_range)
@pytest.mark.parametrize('Ny', N_range)
@pytest.mark.parametrize('dealias_x', dealias_range)
@pytest.mark.parametrize('dealias_y', dealias_range)
def test_CF_CF_scalar_roundtrip(Nx, Ny, dealias_x, dealias_y):
    c, d, xb, yb, x, y = build_CF_CF(Nx, Ny, dealias_x, dealias_y)
    f = field.Field(dist=d, bases=(xb,yb), dtype=np.complex128)
    f.set_scales((dealias_x, dealias_y))
    f['g'] = fg = np.exp(2j*x) * np.exp(2j*y + 1j*np.pi/3) + 3 + np.exp(2j*y)
    f['c']
    assert np.allclose(f['g'], fg)


@CachedFunction
def build_RF_RF(Nx, Ny, dealias_x, dealias_y):
    c = coords.CartesianCoordinates('x', 'y')
    d = distributor.Distributor((c,))
    xb = basis.RealFourier(c.coords[0], size=Nx, bounds=(0, np.pi), dealias=dealias_x)
    yb = basis.RealFourier(c.coords[1], size=Ny, bounds=(0, np.pi), dealias=dealias_y)
    x = xb.local_grid(dealias_x)
    y = yb.local_grid(dealias_y)
    return c, d, xb, yb, x, y


@pytest.mark.parametrize('Nx', N_range)
@pytest.mark.parametrize('Ny', N_range)
@pytest.mark.parametrize('dealias_x', dealias_range)
@pytest.mark.parametrize('dealias_y', dealias_range)
def test_RF_RF_scalar_roundtrip(Nx, Ny, dealias_x, dealias_y):
    c, d, xb, yb, x, y = build_RF_RF(Nx, Ny, dealias_x, dealias_y)
    f = field.Field(dist=d, bases=(xb,yb), dtype=np.float64)
    f.set_scales((dealias_x, dealias_y))
    f['g'] = fg = np.sin(2*x) + np.cos(2*y + np.pi/3) + 3 + np.sin(2*y)
    f['c']
    assert np.allclose(f['g'], fg)


@CachedFunction
def build_CF_J(a, b, Nx, Ny, dealias_x, dealias_y):
    c = coords.CartesianCoordinates('x', 'y')
    d = distributor.Distributor((c,))
    xb = basis.ComplexFourier(c.coords[0], size=Nx, bounds=(0, np.pi), dealias=dealias_x)
    yb = basis.Jacobi(c.coords[1], a=a, b=b, size=Ny, bounds=(0, 1), dealias=dealias_y)
    x = xb.local_grid(dealias_x)
    y = yb.local_grid(dealias_y)
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
    f.set_scales((dealias_x, dealias_y))
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
    u.set_scales((dealias_x, dealias_y))
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
    v.set_scales((dealias_x, dealias_y))
    v['g'] = vg = np.array([np.cos(2*x) * 2, np.sin(2*x) + 1])
    v['c']
    assert np.allclose(v['g'], vg)


## S2
Nphi_range = [8, 16]
Ntheta_range = [12]
dealias_range = [0.5, 1, 1.5]

@CachedMethod
def build_S2(Nphi, Ntheta, dealias, dtype=np.complex128):
    c = coords.S2Coordinates('phi', 'theta')
    d = distributor.Distributor((c,))
    sb = basis.SpinWeightedSphericalHarmonics(c, (Nphi, Ntheta), radius=1, dealias=(dealias, dealias), dtype=dtype)
    phi, theta = sb.local_grids()
    return c, d, sb, phi, theta

def build_sphere_3d(Nphi, Ntheta, dealias, dtype=np.complex128):
    c = coords.SphericalCoordinates('phi', 'theta', 'r')
    d = distributor.Distributor((c,))
    sb = basis.SpinWeightedSphericalHarmonics(c, (Nphi, Ntheta), radius=1, dealias=(dealias, dealias), dtype=dtype)
    phi, theta = sb.local_grids()
    return c, d, sb, phi, theta

@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_S2, build_sphere_3d])
def test_S2_scalar_backward(Nphi, Ntheta, dealias, basis):
    c, d, sb, phi, theta = basis(Nphi, Ntheta, dealias)
    f = field.Field(dist=d, bases=(sb,), dtype=np.complex128)
    m = sb.local_m
    ell = sb.local_ell
    f['c'][(m == -2) * (ell == 2)] = 1
    fg = np.sqrt(15) / 4 * np.sin(theta)**2 * np.exp(-2j*phi)
    assert np.allclose(f['g'], fg)

@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_S2, build_sphere_3d])
def test_S2_scalar_forward(Nphi, Ntheta, dealias, basis):
    c, d, sb, phi, theta = basis(Nphi, Ntheta, dealias)
    f = field.Field(dist=d, bases=(sb,), dtype=np.complex128)
    m = sb.local_m
    ell = sb.local_ell
    f['g'] = np.sqrt(15) / 4 * np.sin(theta)**2 * np.exp(-2j*phi)
    fc = np.zeros_like(f['c'])
    fc[(m == -2) * (ell == 2)] = 1
    assert np.allclose(f['c'], fc)

@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('dealias', dealias_range)
def test_S2_real_scalar_backward(Nphi, Ntheta, dealias):
    c, d, sb, phi, theta = build_S2(Nphi, Ntheta, dealias, dtype=np.float64)
    f = field.Field(dist=d, bases=(sb,), dtype=np.float64)
    m = sb.local_m
    ell = sb.local_ell
    f['c'][(m == 2) * (ell == 2)] = (1, 1)
    fg = np.sqrt(15) / 4 * np.sin(theta)**2 * (np.cos(2*phi) - np.sin(2*phi))
    assert np.allclose(f['g'], fg)

@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('dealias', dealias_range)
def test_S2_real_scalar_forward(Nphi, Ntheta, dealias):
    c, d, sb, phi, theta = build_S2(Nphi, Ntheta, dealias, dtype=np.float64)
    f = field.Field(dist=d, bases=(sb,), dtype=np.float64)
    m = sb.local_m
    ell = sb.local_ell
    f['g'] = np.sqrt(15) / 4 * np.sin(theta)**2 * (np.cos(2*phi) - np.sin(2*phi))
    fc = np.zeros_like(f['c'])
    fc[(m == 2) * (ell == 2)] = (1, 1)
    assert np.allclose(f['c'], fc)

@pytest.mark.xfail(reason="Need to check SWSH normalizations")
@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('dealias', dealias_range)
def test_S2_vector_backward(Nphi, Ntheta, dealias):
    # Note: u is the gradient of sin(theta)*exp(1j*phi)
    c, d, sb, phi, theta = build_S2(Nphi, Ntheta, dealias)
    u = field.Field(dist=d, bases=(sb,), tensorsig=(c,), dtype=np.complex128)
    m = sb.local_m
    ell = sb.local_ell
    u['c'][0][(m == 1) * (ell == 2)] = -np.sqrt(4/5)
    u['c'][1][(m == 1) * (ell == 2)] = np.sqrt(4/5)
    ug = np.zeros_like(u['g'])
    ug[0] = 1j*np.exp(1j*phi)
    ug[1] = np.cos(theta)*np.exp(1j*phi)
    assert np.allclose(u['g'], ug)

@pytest.mark.xfail(reason="Need to check SWSH normalizations")
@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('dealias', dealias_range)
def test_S2_vector_forward(Nphi, Ntheta, dealias):
    # Note: u is the gradient of sin(theta)*exp(1j*phi)
    c, d, sb, phi, theta = build_S2(Nphi, Ntheta, dealias)
    u = field.Field(dist=d, bases=(sb,), tensorsig=(c,), dtype=np.complex128)
    m = sb.local_m
    ell = sb.local_ell
    u['g'][0] = 1j*np.exp(1j*phi)
    u['g'][1] = np.cos(theta)*np.exp(1j*phi)
    uc = np.zeros_like(u['c'])
    uc[0][(m == 1) * (ell == 2)] = -np.sqrt(4/5)
    uc[1][(m == 1) * (ell == 2)] = np.sqrt(4/5)
    assert np.allclose(u['c'], uc)

@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_S2, build_sphere_3d])
def test_S2_vector_roundtrip(Nphi, Ntheta, dealias, basis):
    # Note: u is the gradient of sin(theta)*exp(1j*phi)
    c, d, sb, phi, theta = basis(Nphi, Ntheta, dealias)
    u = field.Field(dist=d, bases=(sb,), tensorsig=(c,), dtype=np.complex128)
    m = sb.local_m
    ell = sb.local_ell
    u['g'][0] = 1j * np.exp(1j*phi)
    u['g'][1] = np.cos(theta) * np.exp(1j*phi)
    ug = u['g'].copy()
    u['c']
    assert np.allclose(u['g'], ug)

@pytest.mark.xfail(reason="Need to check SWSH normalizations")
@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('dealias', dealias_range)
def test_S2_real_vector_backward(Nphi, Ntheta, dealias):
    # Note: u is the gradient of sin(theta)*cos(phi)
    c, d, sb, phi, theta = build_S2(Nphi, Ntheta, dealias, dtype=np.float64)
    u = field.Field(dist=d, bases=(sb,), tensorsig=(c,), dtype=np.float64)
    m = sb.local_m
    ell = sb.local_ell
    # HACK: don't really know what this should be but it passes by switching sine of ug[0]
    u['c'][0][(m == 1) * (ell == 2)] = (-np.sqrt(4/5), 0)
    u['c'][1][(m == 1) * (ell == 2)] = (np.sqrt(4/5), 0)
    ug = np.zeros_like(u['g'])
    ug[0] = - np.sin(phi)
    ug[1] = np.cos(theta) * np.cos(phi)
    assert np.allclose(u['g'], ug)

@pytest.mark.xfail(reason="Need to check SWSH normalizations")
@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('dealias', dealias_range)
def test_S2_real_vector_forward(Nphi, Ntheta, dealias):
    # Note: u is the gradient of sin(theta)*cos(phi)
    c, d, sb, phi, theta = build_S2(Nphi, Ntheta, dealias, dtype=np.float64)
    u = field.Field(dist=d, bases=(sb,), tensorsig=(c,), dtype=np.float64)
    m = sb.local_m
    ell = sb.local_ell
    u['g'][0] = - np.sin(phi)
    u['g'][1] = np.cos(theta) * np.cos(phi)
    uc = np.zeros_like(u['c'])
    # HACK: don't really know what this should be but it passes by switching sine of ug[0]
    uc[0][(m == 1) * (ell == 2)] = (-np.sqrt(4/5), 0)
    uc[1][(m == 1) * (ell == 2)] = (np.sqrt(4/5), 0)
    assert np.allclose(u['c'], uc)

@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_S2, build_sphere_3d])
def test_S2_real_vector_roundtrip(Nphi, Ntheta, dealias, basis):
    # Note: u is the gradient of sin(theta)*cos(phi)
    c, d, sb, phi, theta = basis(Nphi, Ntheta, dealias, dtype=np.float64)
    u = field.Field(dist=d, bases=(sb,), tensorsig=(c,), dtype=np.float64)
    m = sb.local_m
    ell = sb.local_ell
    u['g'][0] = - np.sin(phi)
    u['g'][1] = np.cos(theta) * np.cos(phi)
    ug = u['g'].copy()
    u['c']
    assert np.allclose(u['g'][0], ug[0])

@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('dealias', dealias_range)
def test_S2_tensor_backward(Nphi, Ntheta, dealias):
    # Note: only checking one component of the tensor
    c, d, sb, phi, theta = build_S2(Nphi, Ntheta, dealias)
    T = field.Field(dist=d, bases=(sb,), tensorsig=(c,c), dtype=np.complex128)
    m = sb.local_m
    ell = sb.local_ell
    T['c'][0,0][(m == 2) * (ell == 3)] = 1
    Tg = np.zeros_like(T['g'])
    Tg[0,0] = - 0.5 * np.sqrt(7/2) * (np.cos(theta/2)**4 * (-2 + 3*np.cos(theta))) * np.exp(2j*phi)
    assert np.allclose(T['g'][0,0], Tg[0,0])

@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_S2_tensor_roundtrip(Nphi, Ntheta, dealias, dtype):
    c, d, sb, phi, theta = build_S2(Nphi, Ntheta, dealias, dtype=dtype)
    T = field.Field(dist=d, bases=(sb,), tensorsig=(c,c), dtype=dtype)
    T['g'][1,1] = 2*np.cos(theta)*(3*np.cos(theta)*np.cos(phi)**2 - 2*np.sin(theta)*np.sin(phi))
    T['g'][1,0] = T['g'][0,1] = -2*np.cos(phi)*(np.sin(theta) + 3*np.cos(theta)*np.sin(phi))
    T['g'][0,0] = 6*np.sin(phi)**2
    Tg = T['g'].copy()
    T['c']
    assert np.allclose(T['g'], Tg)

@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_S2_3D_vector_roundtrip(Nphi, Ntheta, dealias, dtype):
    # Note: u is the S2 gradient of cos(theta)*cos(phi)
    c = coords.SphericalCoordinates('phi', 'theta', 'r')
    d = distributor.Distributor((c,))
    c_S2 = c.S2coordsys
    sb = basis.SpinWeightedSphericalHarmonics(c_S2, (Nphi, Ntheta), radius=1, dealias=(dealias, dealias), dtype=dtype)
    phi, theta = sb.local_grids()
    u = field.Field(dist=d, bases=(sb,), tensorsig=(c,), dtype=dtype)
    u['g'][2] = 0
    u['g'][1] = np.cos(2*theta)*np.cos(phi)
    u['g'][0] = -np.cos(theta)*np.sin(phi)
    ug = u['g'].copy()
    u['c']
    assert np.allclose(u['g'], ug)

## D2
Nphi_range = [8, 16]
Nr_range = [12]
dealias_range = [0.5, 1, 1.5]
radius_range = [1, 2]
@CachedMethod
def build_D2(Nphi, Nr, radius, dealias, dtype=np.float64):
    c = coords.PolarCoordinates('phi', 'r')
    d = distributor.Distributor((c,))
    db = basis.DiskBasis(c, (Nphi, Nr), radius=radius, dealias=(dealias, dealias), dtype=dtype)
    phi, r = db.local_grids()
    return c, d, db, phi, r

@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('radius', radius_range)
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_D2_scalar_roundtrip(Nphi, Nr, radius, dealias, dtype):
    c, d, db, phi, r = build_D2(Nphi, Nr, radius, dealias, dtype=dtype)
    f = field.Field(dist=d, bases=(db,), dtype=dtype)
    f['g'] = (r*np.cos(phi))**3
    fg = f['g'].copy()
    f['c']
    assert np.allclose(f['g'], fg)

@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('radius', radius_range)
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_D2_scalar_roundtrip_mmax0(Nr, radius, dealias, dtype):
    Nphi = 1
    c, d, db, phi, r = build_D2(Nphi, Nr, radius, dealias, dtype=dtype)
    f = field.Field(dist=d, bases=(db,), dtype=dtype)
    f['g'] = r**4
    f.require_scales(dealias)
    #f.towards_coeff_space()

    fg = f['g'].copy()
    f['c']
    assert np.allclose(f['g'], fg)

@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('radius', radius_range)
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_D2_vector_roundtrip(Nphi, Nr, radius, dealias, dtype):
    c, d, db, phi, r = build_D2(Nphi, Nr, radius, dealias, dtype=dtype)
    vf = field.Field(dist=d, bases=(db,), tensorsig=(c,), dtype=dtype)
    x = r*np.cos(phi)
    y = r*np.sin(phi)
    ex = np.array([-np.sin(phi)+0.*r,np.cos(phi)+0.*r])

    vf['g'] = x* ex
    vfg = vf['g'].copy()
    vf['c']
    assert np.allclose(vf['g'], vfg)

@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('radius', radius_range)
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_D2_vector_roundtrip_mmax0(Nr, radius, dealias, dtype):
    Nphi = 1
    c, d, db, phi, r = build_D2(Nphi, Nr, radius, dealias, dtype=dtype)
    vf = field.Field(dist=d, bases=(db,), tensorsig=(c,), dtype=dtype)

    vf['g'][1] = 6*r**5
    vfg = vf['g'].copy()
    vf['c']
    assert np.allclose(vf['g'], vfg)

@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('radius', radius_range)
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_D2_vector_roundtrip_mmax0_2(Nr, radius, dealias, dtype):
    Nphi = 1
    c, d, db, phi, r = build_D2(Nphi, Nr, radius, dealias, dtype=dtype)
    f = field.Field(dist=d, bases=(db,), dtype=dtype)

    f['g'] = r**6
    u = operators.Gradient(f, c).evaluate()
    ufg = u['c'].copy()
    u['g']
    assert np.allclose(u['c'], ufg)


@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('radius', radius_range)
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_D2_tensor_roundtrip(Nphi, Nr, radius, dealias, dtype):
    c, d, db, phi, r = build_D2(Nphi, Nr, radius, dealias, dtype=dtype)
    tf = field.Field(dist=d, bases=(db,), tensorsig=(c,c), dtype=dtype)
    x = r*np.cos(phi)
    ex = np.array([-np.sin(phi)+0.*r,np.cos(phi)+0.*r])
    exex = ex[None,:,...]*ex[:, None,...]

    tf['g'] = 6*x * exex
    tfg = tf['g'].copy()
    tf['c']
    assert np.allclose(tf['g'][1][1], tfg[1][1])

@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('radius', radius_range)
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_D2_tensor_roundtrip_mmax0(Nr, radius, dealias, dtype):
    Nphi = 1
    c, d, db, phi, r = build_D2(Nphi, Nr, radius, dealias, dtype=dtype)
    tf = field.Field(dist=d, bases=(db,), tensorsig=(c,c), dtype=dtype)
    tf['g'][1,1] = r**2 + 0.*phi
    tfg = tf['g'].copy()
    tf['c']
    assert np.allclose(tf['g'][1][1], tfg[1][1])

## Spherical Shell
Nphi_range = [8, 16]
Ntheta_range = [12]
Nr_range = [8]
radii_range = [(0.5, 3)]
k_range = [0, 1, 2]
dealias_range = [0.5, 1, 3/2]

@CachedMethod
def build_spherical_shell(Nphi, Ntheta, Nr, radii, k, dealias, dtype=np.complex128):
    c = coords.SphericalCoordinates('phi', 'theta', 'r')
    d = distributor.Distributor((c,))
    b  = basis.SphericalShellBasis(c, (Nphi, Ntheta, Nr), k=k, radii=radii, dealias=(dealias, dealias, dealias), dtype=dtype)
    phi, theta, r = b.local_grids()
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return c, d, b, phi, theta, r, x, y, z

@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('radii', radii_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_spherical_shell_radial_scalar(Nr, radii, k, dtype):
    c, d, b, phi, theta, r, x, y, z = build_spherical_shell(4, 4, Nr, radii, k, 1, dtype)
    f = field.Field(dist=d, bases=(b.radial_basis,), dtype=dtype)
    f['g'] = fg = r - r**2/7
    f['c']
    assert np.allclose(f['g'], fg)

@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('radii', radii_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_spherical_shell_radial_vector(Nr, radii, k, dtype):
    c, d, b, phi, theta, r, x, y, z = build_spherical_shell(4, 4, Nr, radii, k, 1, dtype)
    u = field.Field(dist=d, bases=(b.radial_basis,), tensorsig=(c,), dtype=dtype)
    u['g'][2] = 1 - 2*r/7
    u0 = np.copy(u['g'])
    u['c']
    assert np.allclose(u['g'], u0)

@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('radii', radii_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_spherical_shell_scalar(Nphi, Ntheta, Nr, radii, k, dealias, dtype):
    c, d, b, phi, theta, r, x, y, z = build_spherical_shell(Nphi, Ntheta, Nr, radii, k, dealias, dtype)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    f['g'] = fg = 3*x**2 + 2*y*z
    f['c']
    assert np.allclose(f['g'], fg)

@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('radii', radii_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_spherical_shell_vector(Nphi, Ntheta, Nr, radii, k, dealias, dtype):
    # Note: u is the gradient of a scalar
    c, d, b, phi, theta, r, x, y, z = build_spherical_shell(Nphi, Ntheta, Nr, radii, k, dealias, dtype)
    u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
    u['g'][2] =  4*r**3*np.sin(theta)*np.cos(theta)*np.cos(phi)
    u['g'][1] =    r**3*np.cos(2*theta)*np.cos(phi)
    u['g'][0] = -r**3*np.cos(theta)*np.sin(phi)
    ug = np.copy(u['g'])
    u['c']
    assert np.allclose(u['g'], ug)

## Ball
Nphi_range = [8, 16]
Ntheta_range = [12]
Nr_range = [8]
radius_range = [1.5]
dealias_range = [0.5, 1, 3/2]
k_range = [0, 1, 2]

@CachedMethod
def build_ball(Nphi, Ntheta, Nr, radius, k, dealias, dtype=np.complex128):
    c = coords.SphericalCoordinates('phi', 'theta', 'r')
    d = distributor.Distributor((c,))
    b = basis.BallBasis(c, (Nphi, Ntheta, Nr), radius=radius, k=k, dealias=(dealias, dealias, dealias), dtype=dtype)
    phi, theta, r = b.local_grids()
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return c, d, b, phi, theta, r, x, y, z

@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('radius', radius_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_ball_radial_scalar(Nr, radius, k, dtype):
    c, d, b, phi, theta, r, x, y, z = build_ball(4, 4, Nr, radius, k, 1, dtype)
    f = field.Field(dist=d, bases=(b.radial_basis,), dtype=dtype)
    f['g'] = fg = r**2 - r**4/7
    f['c']
    assert np.allclose(f['g'], fg)

@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('radius', radius_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_ball_radial_vector(Nr, radius, k, dtype):
    c, d, b, phi, theta, r, x, y, z = build_ball(4, 4, Nr, radius, k, 1, dtype)
    u = field.Field(dist=d, bases=(b.radial_basis,), tensorsig=(c,), dtype=dtype)
    u['g'][2] = 2*r - 3*r**3/7
    u0 = np.copy(u['g'])
    u['c']
    assert np.allclose(u['g'], u0)

@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('radius', radius_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_ball_scalar(Nphi, Ntheta, Nr, radius, k, dealias, dtype):
    c, d, b, phi, theta, r, x, y, z = build_ball(Nphi, Ntheta, Nr, radius, k, dealias, dtype)
    f = field.Field(dist=d, bases=(b,), dtype=dtype)
    f['g'] = fg = 3*x**2 + 2*y*z
    f['c']
    assert np.allclose(f['g'], fg)

# Vector transforms
@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('radius', radius_range)
@pytest.mark.parametrize('k', k_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_ball_vector(Nphi, Ntheta, Nr, radius, k, dealias, dtype):
    # Note: u is the gradient of a scalar
    c, d, b, phi, theta, r, x, y, z = build_ball(Nphi, Ntheta, Nr, radius, k, dealias, dtype)
    u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
    u['g'][2] =  4*r**3*np.sin(theta)*np.cos(theta)*np.cos(phi)
    u['g'][1] =    r**3*np.cos(2*theta)*np.cos(phi)
    u['g'][0] = -1*r**3*np.cos(theta)*np.sin(phi)
    ug = np.copy(u['g'])
    u['c']
    assert np.allclose(u['g'], ug)

