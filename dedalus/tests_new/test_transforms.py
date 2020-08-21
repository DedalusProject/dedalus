
import pytest
import numpy as np
from dedalus.core import coords, distributor, basis, field, operators
from dedalus.tools.cache import CachedMethod, CachedFunction
from mpi4py import MPI

comm = MPI.COMM_WORLD

## Cartesian
N_range = [8, 10]
dealias_range = [0.5, 1, 3/2]

@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('dealias', dealias_range)
def test_1D_complex_fourier_scalar(N, dealias):
    if comm.size == 1:
        c = coords.Coordinate('x')
        d = distributor.Distributor([c])
        xb = basis.ComplexFourier(c, size=N, bounds=(0, 1), dealias=dealias)
        x = xb.local_grid()
        # Scalar transforms
        u = field.Field(dist=d, bases=(xb,), dtype=np.complex128)
        ug = np.exp(2*np.pi*1j*x)
        u['g'] = ug
        u['c']
        assert np.allclose(u['g'], ug)
    else:
        pytest.skip("Can only test 1D transform in serial")


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('dealias', dealias_range)
def test_1D_real_fourier_scalar(N, dealias):
    if comm.size == 1:
        c = coords.Coordinate('x')
        d = distributor.Distributor([c])
        xb = basis.RealFourier(c, size=N, bounds=(0, 1), dealias=dealias)
        x = xb.local_grid()
        # Scalar transforms
        u = field.Field(dist=d, bases=(xb,), dtype=np.float64)
        ug = np.cos(2*np.pi*x + np.pi/4)
        u['g'] = ug
        u['c']
        assert np.allclose(u['g'], ug)
    else:
        pytest.skip("Can only test 1D transform in serial")


@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('dealias', dealias_range)
def test_1D_chebyshev_scalar(N, dealias):
    if comm.size == 1:
        c = coords.Coordinate('x')
        d = distributor.Distributor([c])
        xb = basis.ChebyshevT(c, size=N, bounds=(0, 1), dealias=dealias)
        x = xb.local_grid()
        # Scalar transforms
        u = field.Field(dist=d, bases=(xb,), dtype=np.complex128)
        u['g'] = ug = 2*x**2 - 1
        u['c']
        assert np.allclose(u['g'], ug)
    else:
        pytest.skip("Can only test 1D transform in serial")

@CachedFunction
def build_2D_real_fourier(Nx, Ny, dealias_x, dealias_y):
    c = coords.CartesianCoordinates('x', 'y')
    d = distributor.Distributor((c,))
    xb = basis.RealFourier(c.coords[0], size=Nx, bounds=(0, 2*np.pi), dealias=dealias_x)
    yb = basis.RealFourier(c.coords[1], size=Ny, bounds=(0, 2*np.pi), dealias=dealias_y)
    x = xb.local_grid()
    y = yb.local_grid()
    return c, d, xb, yb, x, y

@pytest.mark.parametrize('Nx', N_range)
@pytest.mark.parametrize('Ny', N_range)
@pytest.mark.parametrize('dealias_x', dealias_range)
@pytest.mark.parametrize('dealias_y', dealias_range)
def test_2D_real_fourier_scalar(Nx, Ny, dealias_x, dealias_y):
    c, d, xb, yb, x, y = build_2D_real_fourier(Nx, Ny, dealias_x, dealias_y)
    f = field.Field(dist=d, bases=(xb,yb), dtype=np.float64)
    f['g'] = fg = np.sin(x) * np.cos(2*y + np.pi/3) + 3 + np.sin(y)
    f['c']
    assert np.allclose(f['g'], fg)

@CachedFunction
def build_2D_fourier_chebyshev(Nx, Ny, dealias_x, dealias_y):
    c = coords.CartesianCoordinates('x', 'y')
    d = distributor.Distributor((c,))
    xb = basis.ComplexFourier(c.coords[0], size=Nx, bounds=(0, 2*np.pi), dealias=dealias_x)
    yb = basis.ChebyshevT(c.coords[1], size=Ny, bounds=(0, 1), dealias=dealias_y)
    x = xb.local_grid()
    y = yb.local_grid()
    return c, d, xb, yb, x, y

@pytest.mark.parametrize('Nx', N_range)
@pytest.mark.parametrize('Ny', N_range)
@pytest.mark.parametrize('dealias_x', dealias_range)
@pytest.mark.parametrize('dealias_y', dealias_range)
def test_2D_fourier_chebyshev_scalar(Nx, Ny, dealias_x, dealias_y):
    c, d, xb, yb, x, y = build_2D_fourier_chebyshev(Nx, Ny, dealias_x, dealias_y)
    f = field.Field(dist=d, bases=(xb,yb,), dtype=np.complex128)
    f['g'] = fg = np.sin(x) * y**5
    f['c']
    assert np.allclose(f['g'], fg)

@pytest.mark.parametrize('Nx', N_range)
@pytest.mark.parametrize('Ny', N_range)
@pytest.mark.parametrize('dealias_x', dealias_range)
@pytest.mark.parametrize('dealias_y', dealias_range)
def test_2D_fourier_chebyshev_vector(Nx, Ny, dealias_x, dealias_y):
    c, d, xb, yb, x, y = build_2D_fourier_chebyshev(Nx, Ny, dealias_x, dealias_y)
    u = field.Field(dist=d, bases=(xb,yb,), tensorsig=(c,), dtype=np.complex128)
    u['g'] = ug = np.array([np.cos(x) * 2 * y**2, np.sin(x) * y + y])
    u['c']
    assert np.allclose(u['g'], ug)

@pytest.mark.parametrize('Nx', N_range)
@pytest.mark.parametrize('Ny', N_range)
@pytest.mark.parametrize('dealias_x', dealias_range)
@pytest.mark.parametrize('dealias_y', dealias_range)
def test_2D_fourier_chebyshev_1D_vector(Nx, Ny, dealias_x, dealias_y):
    c, d, xb, yb, x, y = build_2D_fourier_chebyshev(Nx, Ny, dealias_x, dealias_y)
    v = field.Field(dist=d, bases=(xb,), tensorsig=(c,), dtype=np.complex128)
    v['g'] = vg = np.array([np.cos(x) * 2, np.sin(x) + 1])
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

@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('dealias', dealias_range)
def test_S2_scalar_backward(Nphi, Ntheta, dealias):
    c, d, sb, phi, theta = build_S2(Nphi, Ntheta, dealias)
    f = field.Field(dist=d, bases=(sb,), dtype=np.complex128)
    m = sb.local_m
    ell = sb.local_ell
    f['c'][(m == -2) * (ell == 2)] = 1
    fg = np.sqrt(15) / 4 * np.sin(theta)**2 * np.exp(-2j*phi)
    assert np.allclose(f['g'], fg)

@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('dealias', dealias_range)
def test_S2_scalar_forward(Nphi, Ntheta, dealias):
    c, d, sb, phi, theta = build_S2(Nphi, Ntheta, dealias)
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
def test_S2_vector_roundtrip(Nphi, Ntheta, dealias):
    # Note: u is the gradient of sin(theta)*exp(1j*phi)
    c, d, sb, phi, theta = build_S2(Nphi, Ntheta, dealias)
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
def test_S2_real_vector_roundtrip(Nphi, Ntheta, dealias):
    # Note: u is the gradient of sin(theta)*cos(phi)
    c, d, sb, phi, theta = build_S2(Nphi, Ntheta, dealias, dtype=np.float64)
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

