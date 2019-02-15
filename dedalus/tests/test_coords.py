

import numpy as np
from dedalus.core import distributor, spaces, basis, field, vectorspaces


def test_sphere_2vec():
    """2-vectors on sphere"""
    dist = Distributor(dim=2)
    phi, theta = SphericalCoords(['phi', 'theta'], dist, axis=0)
    m1 = Sphere((phi, theta), radius=1)
    # b1 = SWSH(m1, shape=(32,32))
    # f1 = Field(b1, tensor=[c1])

def test_sphere_3vec():
    """3-vectors on sphere"""
    dist = Distributor(dim=3)
    phi, theta, r = SphericalCoords(['phi', 'theta', 'r'], dist, axis=0)
    m1 = Sphere((phi, theta), radius=1)
    # b1 = SWSH(m1, shape=(32,32))
    # f1 = Field(b1, tensor=[c1])

def test_ball_3vec():
    """3-vectors on ball"""
    dist = Distributor(dim=3)
    phi, theta, r = SphericalCoords(['phi', 'theta', 'r'], dist, axis=0)
    m1 = Ball((phi, theta, r), radius=1)
    #b1 = BallBasis(m1, shape=(32,32,32))
    #f1 = Field(b1, tensor=[c1])

def test_spherical_shell_3vec():
    """3-vectors on spherical shell"""
    dist = Distributor(dim=3)
    phi, theta, r = SphericalCoords(['phi', 'theta', 'r'], dist, axis=0)
    m1 = Sphere(phi, theta)
    m2 = Interval(r, bounds=(1,2))
    # b1 = SWSH(m1, shape=(32,32))
    # b2 = Chebyshev(m2, 32)
    # f1 = Field(b1*b2, tensor=[c1])

def test_disk_2vec():
    """2-vectors on disk"""
    dist = Distributor(dim=3)
    phi, r = PolarCoords(['phi', 'r'], dist, axis=0)
    m1 = Disk((phi, r), radius=1)

def test_disk_3vec():
    """3-vectors on disk"""
    dist = Distributor(dim=3)
    phi, r = PolarCoords(['phi', 'r'], dist, axis=0)
    z = Coordinate('z', dist, axis=2)
    m1 = Disk((phi, r), radius=1)

def test_annulus_2vec():
    """2-vectors on annulus"""
    dist = Distributor(dim=3)
    phi, r = PolarCoords(['phi', 'r'], dist, axis=0)
    m1 = PeriodicInterval(phi, bounds=(0, 2*np.pi))
    m2 = Interval(r, bounds=(1,2))

def test_annulus_3vec():
    """3-vectors on annulus"""
    dist = Distributor(dim=3)
    phi, r = PolarCoords(['phi', 'r'], dist, axis=0)
    z = Coordinate('z', dist, axis=2)
    m1 = PeriodicInterval(phi, bounds=(0, 2*np.pi))
    m2 = Interval(r, bounds=(1,2))

def test_cylinder_3vec():
    """3-vectors on cylinder"""
    dist = Distributor(dim=3)
    phi, r = PolarCoords(['phi', 'r'], dist, axis=0)
    z = Coordinate('z', dist, axis=2)
    m1 = Disk((phi, r), radius=1)
    m2 = Interval(z, bounds=(-1,1))
    # b1 = DiskBasis(m1, (32,32))
    # b2 = Legendre(m2, 32)
    # f1 = Field(b1*b2, tensor=[c1*c2])

def test_cylindrical_shell_3vec():
    dist = Distributor(dim=3)
    phi, r = PolarCoords(['phi', 'r'], dist, axis=0)
    z = Coordinate('z', dist, axis=2)
    m1 = PeriodicInterval(phi, bounds=(0, 2*np.pi))
    m2 = Interval(r, bounds=(1,2))
    m3 = Interval(z, bounds=(-1,1))
    # b1 = Fourier(m1, 32)
    # b2 = Chebyshev(m2, 32)
    # b3 = Legendre(m3, 32)
    # v1 = VectorSpace(phi, r, z)
    # f1 = Field(b1*b2*b3, tensor=[v1])
    # f2 = grad(f1, v1)

def test_box_3vec():
    """3-vectors on box"""
    x, y, z = CartesianCoords(['x','y','z'], dist, axis=0)
    m1 = PeriodicInterval(x, (0,1))
    m2 = Interval(y, (0,1))
    m3 = Interval(z, (0,1))
    # b1 = Fourier(m1, 32)
    # b2 = Chebyshev(m2, 32)
    # b3 = Legendre(m3, 32)
    # v1 = VectorSpace(x, y, z)
    # f1 = Field(b1*b2*b3, tensor=[v1])
    # f2 = grad(f1, v1)

def test_rad():
    """Radiative transfer setup."""
    dist = Distributor(dim=5)
    x, y, z = CartesianCoords(['x', 'y', 'z'], dist, axis=0)
    phi, theta = SphericalCoords(['phi', 'theta'], dist, axis=3)
    m1 = Interval(x, (0,1))
    m2 = Interval(y, (0,1))
    m3 = Interval(z, (0,1))
    m4 = Sphere(phi, theta)
    # b1 = Chebyshev(m1, 32)
    # b2 = Chebyshev(m2, 32)
    # b3 = Chebyshev(m3, 32)
    # b4 = SWSH(m4, (32,32))
    # v1 = VectorSpace(x, y, z)
    # v2 = VectorSpace(phi, theta)
    # I = Field(b1*b2*b3*b4)
    # Omega = Field(b4, tensor=[v1])
    # Omega['g'][2] = np.cos(theta)  # Omega.z...
    # Omega @ grad(I, v1)

# d = distributor.Distributor(dim=2)
# s = spaces.Sphere(('phi', 'theta'), Lmax=7, radius=1, dist=d, axis=0)
# b = basis.SWSH(s)

# X = vectorspaces.VectorSpace([s])
# u = field.Field(d, bases=[b], name='u', tensorsig=[X,X], dtype=np.complex128)

# print(s.grids(1))

# print(u.data.shape)
# print(u['g'])
# print(u['c'])