"""Quick setup of common domains."""


import numpy as np
import dedalus.public as d3


def fourier(N, dealias=3/2, dtype=np.float64):
    coord = d3.Coordinate('x')
    dist = d3.Distributor(coord, dtype=dtype)
    xbasis = d3.Fourier(coord, size=N, bounds=(0, 2*np.pi), dealias=dealias, dtype=dtype)
    return coord, dist, xbasis


def chebyshev(N, dealias=3/2, dtype=np.float64):
    coord = d3.Coordinate('x')
    dist = d3.Distributor(coord, dtype=dtype)
    xbasis = d3.Chebyshev(coord, size=N, bounds=(-1, 1), dealias=dealias)
    return coord, dist, xbasis


def fourier_2d(N, dealias=3/2, dtype=np.float64):
    coords = d3.CartesianCoordinates('x', 'y')
    dist = d3.Distributor(coords, dtype=dtype)
    xbasis = d3.Fourier(coords[0], size=N, bounds=(0, 2*np.pi), dealias=dealias, dtype=dtype)
    ybasis = d3.Fourier(coords[1], size=N, bounds=(0, 2*np.pi), dealias=dealias, dtype=dtype)
    return coords, dist, (xbasis, ybasis)


def fourier_3d(N, dealias=3/2, dtype=np.float64):
    coords = d3.CartesianCoordinates('x', 'y', 'z')
    dist = d3.Distributor(coords, dtype=dtype)
    xbasis = d3.Fourier(coords[0], size=N, bounds=(0, 2*np.pi), dealias=dealias, dtype=dtype)
    ybasis = d3.Fourier(coords[1], size=N, bounds=(0, 2*np.pi), dealias=dealias, dtype=dtype)
    zbasis = d3.Fourier(coords[2], size=N, bounds=(0, 2*np.pi), dealias=dealias, dtype=dtype)
    return coords, dist, (xbasis, ybasis, zbasis)


def channel_2d(N, dealias=3/2, dtype=np.float64):
    coords = d3.CartesianCoordinates('x', 'y')
    dist = d3.Distributor(coords, dtype=dtype)
    xbasis = d3.Fourier(coords[0], size=N, bounds=(0, 2*np.pi), dealias=dealias, dtype=dtype)
    ybasis = d3.Chebyshev(coords[1], size=N, bounds=(-1, 1), dealias=dealias)
    return coords, dist, (xbasis, ybasis)


def channel_3d(N, dealias=3/2, dtype=np.float64):
    coords = d3.CartesianCoordinates('x', 'y', 'z')
    dist = d3.Distributor(coords, dtype=dtype)
    xbasis = d3.Fourier(coords[0], size=N, bounds=(0, 2*np.pi), dealias=dealias, dtype=dtype)
    ybasis = d3.Fourier(coords[1], size=N, bounds=(0, 2*np.pi), dealias=dealias, dtype=dtype)
    zbasis = d3.Chebyshev(coords[2], size=N, bounds=(-1, 1), dealias=dealias)
    return coords, dist, (xbasis, ybasis, zbasis)

