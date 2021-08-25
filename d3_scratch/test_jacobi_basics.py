"""
Test basic forward operations on Jacobi basis.
"""

import numpy as np
from dedalus_burns import dev
import logging
logger = logging.getLogger(__name__)


def general_test(a, b, Nx, bounds):

    # Domain
    dist = dev.distributor.Distributor(dim=1, comm=comm)
    xs = dev.spaces.FiniteInterval(a=a, b=b, name='x', coeff_size=Nx, bounds=bounds, dist=dist, axis=0, dealias=3/2)
    xb0 = xs.Jacobi(da=0, db=0)

    # Build simple field
    u = dev.field.Field(bases=[xb0])
    x = xs.local_grid()
    Lx = bounds[1] - bounds[0]
    kx = 3 * (2 * np.pi) / Lx
    u['g'] = 1 + np.sin(kx*x)

    # Test differentiation
    f1 = u.differentiate('x')
    f1.require_scales(1)
    diff_test = np.allclose(f1['g'], kx*np.cos(kx*x))

    # Test interpolation
    rand = np.random.RandomState(seed=9)
    xi = rand.uniform(*bounds)
    f2 = u.interpolate(x=xi)
    f2.require_scales(1)
    interp_test = np.allclose(f2['g'], 1 + np.sin(kx*xi))

    # Test integration
    f3 = u.integrate('x')
    f3.require_scales(1)
    integ_test = np.allclose(f3['g'], Lx)

    return (diff_test, interp_test, integ_test)


def test_legendre():
    assert all(general_test(a=0, b=0, Nx=32, bounds=(2,3)))


def test_chebyshevT():
    assert all(general_test(a=-1/2, b=-1/2, Nx=32, bounds=(2,3)))


def test_chebyshevU():
    assert all(general_test(a=1/2, b=1/2, Nx=32, bounds=(2,3)))


if __name__ == '__main__':
    defaults = {'a':0, 'b':0, 'Nx':32, 'bounds':(2,3)}
    print(general_test(**defaults))

