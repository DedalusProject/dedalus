import pytest
import numpy as np
from dedalus.core import coords, distributor, basis, field, operators, arithmetic
from dedalus.tools.cache import CachedMethod
from mpi4py import MPI

comm = MPI.COMM_WORLD

Lx=2
Ly=2
Lz=1
@CachedMethod
def build_3d_box(Nx, Ny, Nz, dealias, k=0):
    c = coords.CartesianCoordinates('x', 'y', 'z')
    d = distributor.Distributor((c,))
    xb = basis.ComplexFourier(c.coords[0], size=Nx, bounds=(0, Lx))
    yb = basis.ComplexFourier(c.coords[1], size=Ny, bounds=(0, Ly))
    zb = basis.ChebyshevT(c.coords[2], size=Nz, bounds=(0, Lz))
    b = (xb, yb, zb)
    x = xb.local_grid(1)
    y = yb.local_grid(1)
    z = zb.local_grid(1)
    return c, d, b, x, y, z

@CachedMethod
def build_2d_box(Nx, Nz, dealias, k=0):
    c = coords.CartesianCoordinates('x', 'z')
    d = distributor.Distributor((c,))
    xb = basis.ComplexFourier(c.coords[0], size=Nx, bounds=(0, Lx))
    zb = basis.ChebyshevT(c.coords[1], size=Nz, bounds=(0, Lz))
    b = (xb, zb)
    x = xb.local_grid(1)
    z = zb.local_grid(1)
    return c, d, b, x, z

Nx_range = [16]
Ny_range = [8]
Nz_range = [8]
dealias_range = [1, 3/2]

@pytest.mark.parametrize('Nx', Nx_range)
@pytest.mark.parametrize('Ny', Ny_range)
@pytest.mark.parametrize('Nz', Nz_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_3d_box])
def test_transpose_3d_grid_tensor(Nx, Ny, Nz, dealias, basis):
    c, d, b, x, y, z = basis(Nx, Ny, Nz, dealias)
    u = field.Field(dist=d, bases=(*b,), tensorsig=(c,), dtype=np.complex128)
    u['g'][0] = (np.sin(2*x)+np.sin(x))*np.cos(y)*np.sin(z)
    u['g'][1] = (np.cos(2*x)+np.cos(x))*np.sin(y)*np.sin(z)
    u['g'][2] = np.sin(x)*np.cos(y)*np.cos(z)
    T = operators.Gradient(u, c).evaluate()
    T.require_grid_space()
    Tg = np.transpose(np.copy(T['g']),(1,0,2,3,4))
    T = operators.TransposeComponents(T, c).evaluate()
    assert np.allclose(T['g'], Tg)

@pytest.mark.parametrize('Nx', Nx_range)
@pytest.mark.parametrize('Nz', Nz_range)
@pytest.mark.parametrize('dealias', dealias_range)
@pytest.mark.parametrize('basis', [build_2d_box])
def test_transpose_2d_grid_tensor(Nx, Nz, dealias, basis):
    c, d, b, x, z = basis(Nx, Nz, dealias)
    u = field.Field(dist=d, bases=(*b,), tensorsig=(c,), dtype=np.complex128)
    u['g'][0] = (np.sin(2*x)+np.sin(x))*np.sin(z)
    u['g'][1] = np.sin(x)*np.cos(z)
    T = operators.Gradient(u, c).evaluate()
    T.require_grid_space()
    Tg = np.transpose(np.copy(T['g']),(1,0,2,3))
    T = operators.TransposeComponents(T, c).evaluate()
    assert np.allclose(T['g'], Tg)
