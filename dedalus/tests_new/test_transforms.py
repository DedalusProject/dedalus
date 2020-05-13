
import pytest
import numpy as np
from dedalus.core import coords, distributor, basis, field, operators
from dedalus.tools.cache import CachedMethod
from mpi4py import MPI

comm = MPI.COMM_WORLD

## Cartesian
N_range = [8, 10]

@pytest.mark.parametrize('N', N_range)
def test_1D_complex_fourier_scalar(N):
    if comm.size == 1:
        c = coords.Coordinate('x')
        d = distributor.Distributor([c])
        xb = basis.ComplexFourier(c, size=16, bounds=(0, 1))
        x = xb.local_grid(1)
        # Scalar transforms
        u = field.Field(dist=d, bases=(xb,), dtype=np.complex128)
        ug = np.exp(2*np.pi*1j*x)
        u['g'] = ug
        u['c']
        assert np.allclose(u['g'], ug)
    else:
        pytest.skip("Can only test 1D transform in serial")

@pytest.mark.parametrize('N', N_range)
def test_1D_chebyshev_scalar(N):
    if comm.size == 1:
        c = coords.Coordinate('x')
        d = distributor.Distributor([c])
        xb = basis.ChebyshevT(c, size=16, bounds=(0, 1))
        x = xb.local_grid(1)
        # Scalar transforms
        u = field.Field(dist=d, bases=(xb,), dtype=np.complex128)
        u['g'] = ug = 2*x**2 - 1
        u['c']
        assert np.allclose(u['g'], ug)
    else:
        pytest.skip("Can only test 1D transform in serial")

@CachedMethod
def build_2D_fourier_chebyshev(Nx, Ny):
    c = coords.CartesianCoordinates('x', 'y')
    d = distributor.Distributor((c,))
    xb = basis.ComplexFourier(c.coords[0], size=Nx, bounds=(0, 2*np.pi))
    yb = basis.ChebyshevT(c.coords[1], size=Ny, bounds=(0, 1))
    x = xb.local_grid(1)
    y = yb.local_grid(1)
    return c, d, xb, yb, x, y

@pytest.mark.parametrize('Nx', N_range)
@pytest.mark.parametrize('Ny', N_range)
def test_2D_fourier_chebyshev_scalar(Nx, Ny):
    c, d, xb, yb, x, y = build_2D_fourier_chebyshev(Nx, Ny)
    f = field.Field(dist=d, bases=(xb,yb,), dtype=np.complex128)
    f['g'] = fg = np.sin(x) * y**5
    f['c']
    assert np.allclose(f['g'], fg)

@pytest.mark.parametrize('Nx', N_range)
@pytest.mark.parametrize('Ny', N_range)
def test_2D_fourier_chebyshev_vector(Nx, Ny):
    c, d, xb, yb, x, y = build_2D_fourier_chebyshev(Nx, Ny)
    u = field.Field(dist=d, bases=(xb,yb,), tensorsig=(c,), dtype=np.complex128)
    u['g'] = ug = np.array([np.cos(x) * 2 * y**2, np.sin(x) * y + y])
    u['c']
    assert np.allclose(u['g'], ug)

@pytest.mark.parametrize('Nx', N_range)
@pytest.mark.parametrize('Ny', N_range)
def test_2D_fourier_chebyshev_1D_vector(Nx, Ny):
    c, d, xb, yb, x, y = build_2D_fourier_chebyshev(Nx, Ny)
    v = field.Field(dist=d, bases=(xb,), tensorsig=(c,), dtype=np.complex128)
    v['g'] = vg = np.array([np.cos(x) * 2, np.sin(x) + 1])
    v['c']
    assert np.allclose(v['g'], vg)

## S2
Nphi_range = [8]
Ntheta_range = [10]

@CachedMethod
def build_S2(Nphi, Ntheta):
    c = coords.S2Coordinates('phi', 'theta')
    d = distributor.Distributor((c,))
    sb = basis.SpinWeightedSphericalHarmonics(c, (32,16), radius=1)
    phi, theta = sb.local_grids((1, 1))
    return c, d, sb, phi, theta

@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
def test_S2_scalar(Nphi, Ntheta):
    c, d, sb, phi, theta = build_S2(Nphi, Ntheta)
    f = field.Field(dist=d, bases=(sb,), dtype=np.complex128)
    f['c'][-2,2] = 1
    fg = np.sqrt(15) / 4 * np.sin(theta)**2 * np.exp(-2j*phi)
    assert np.allclose(f['g'], fg)

@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
def test_S2_vector(Nphi, Ntheta):
    # Note: u is the gradient of cos(theta)*exp(1j*phi)
    c, d, sb, phi, theta = build_S2(Nphi, Ntheta)
    u = field.Field(dist=d, bases=(sb,), tensorsig=(c,), dtype=np.complex128)
    u['g'][1] =    np.cos(2*theta)*np.exp(1j*phi)
    u['g'][0] = 1j*np.cos(theta)*np.exp(1j*phi)
    ug0 = np.copy(u['g'])
    u['c']
    assert np.allclose(u['g'], ug0)

@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
def test_S2_tensor(Nphi, Ntheta):
    # Note: only checking one component of the tensor
    c, d, sb, phi, theta = build_S2(Nphi, Ntheta)
    T = field.Field(dist=d, bases=(sb,), tensorsig=(c,c), dtype=np.complex128)
    T['c'][0,0,2,3] = 1
    Tg00 = - 0.5 * np.sqrt(7/2) * (np.cos(theta/2)**4 * (-2 + 3*np.cos(theta))) * np.exp(2j*phi)
    assert np.allclose(T['g'][0,0], Tg00)

@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
def test_S2_3D_vector(Nphi, Ntheta):
    # Note: u is the S2 gradient of cos(theta)*exp(1j*phi)
    c = coords.SphericalCoordinates('phi', 'theta', 'r')
    d = distributor.Distributor( (c,) )
    c_S2 = c.S2coordsys
    sb = basis.SpinWeightedSphericalHarmonics(c_S2, (32,16), radius=1)
    phi, theta = sb.local_grids((1, 1))
    u = field.Field(dist=d, bases=(sb,), tensorsig=(c,), dtype=np.complex128)
    u['g'][2] = 0
    u['g'][1] =    np.cos(2*theta)*np.exp(1j*phi)
    u['g'][0] = 1j*np.cos(theta)*np.exp(1j*phi)
    ug0 = np.copy(u['g'])
    u['c']
    assert np.allclose(u['g'], ug0)

## Spherical Shell
Nphi_range = [8]
Ntheta_range = [10]
Nr_range = [6]
radii_range = [(0.5, 3)]
k_range = [0, 1]

@CachedMethod
def build_spherical_shell(Nphi, Ntheta, Nr, radii, k):
    c = coords.SphericalCoordinates('phi', 'theta', 'r')
    d = distributor.Distributor((c,))
    b  = basis.SphericalShellBasis(c, (Nphi, Ntheta, Nr), k=k, radii=radii)
    phi, theta, r = b.local_grids((1, 1, 1))
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return c, d, b, phi, theta, r, x, y, z

@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('radii', radii_range)
@pytest.mark.parametrize('k', k_range)
def test_spherical_shell_scalar(Nphi, Ntheta, Nr, radii, k):
    c, d, b, phi, theta, r, x, y, z = build_spherical_shell(Nphi, Ntheta, Nr, radii, k)
    f = field.Field(dist=d, bases=(b,), dtype=np.complex128)
    f['g'] = fg = 3*x**2 + 2*y*z
    f['c']
    assert np.allclose(f['g'], fg)

@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('radii', radii_range)
@pytest.mark.parametrize('k', k_range)
def test_spherical_shell_vector(Nphi, Ntheta, Nr, radii, k):
    # Note: u is the gradient of a scalar
    c, d, b, phi, theta, r, x, y, z = build_spherical_shell(Nphi, Ntheta, Nr, radii, k)
    u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
    u['g'][2] =  4*r**3*np.sin(theta)*np.cos(theta)*np.exp(1j*phi)
    u['g'][1] =    r**3*np.cos(2*theta)*np.exp(1j*phi)
    u['g'][0] = 1j*r**3*np.cos(theta)*np.exp(1j*phi)
    ug = np.copy(u['g'])
    u['c']
    assert np.allclose(u['g'], ug)

## Ball
Nphi_range = [8]
Ntheta_range = [10]
Nr_range = [6]
radius_range = [1, 1.5]

@CachedMethod
def build_ball(Nphi, Ntheta, Nr, radius):
    c = coords.SphericalCoordinates('phi', 'theta', 'r')
    d = distributor.Distributor((c,))
    b = basis.BallBasis(c, (Nphi, Ntheta, Nr), radius=radius)
    phi, theta, r = b.local_grids((1, 1, 1))
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return c, d, b, phi, theta, r, x, y, z

@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('radius', radius_range)
def test_ball_scalar(Nphi, Ntheta, Nr, radius):
    c, d, b, phi, theta, r, x, y, z = build_ball(Nphi, Ntheta, Nr, radius)
    f = field.Field(dist=d, bases=(b,), dtype=np.complex128)
    f['g'] = fg = 3*x**2 + 2*y*z
    f['c']
    assert np.allclose(f['g'], fg)

# Vector transforms
@pytest.mark.parametrize('Nphi', Nphi_range)
@pytest.mark.parametrize('Ntheta', Ntheta_range)
@pytest.mark.parametrize('Nr', Nr_range)
@pytest.mark.parametrize('radius', radius_range)
def test_ball_scalar(Nphi, Ntheta, Nr, radius):
    # Note: u is the gradient of a scalar
    c, d, b, phi, theta, r, x, y, z = build_ball(Nphi, Ntheta, Nr, radius)
    u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
    u['g'][2] =  4*r**3*np.sin(theta)*np.cos(theta)*np.exp(1j*phi)
    u['g'][1] =    r**3*np.cos(2*theta)*np.exp(1j*phi)
    u['g'][0] = 1j*r**3*np.cos(theta)*np.exp(1j*phi)
    ug = np.copy(u['g'])
    u['c']
    assert np.allclose(u['g'], ug)



