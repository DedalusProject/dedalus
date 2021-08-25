"""
Test integrating a random polynomial using Jacobi basis.
"""

import numpy as np
from dedalus_burns import dev
import logging
logger = logging.getLogger(__name__)


def jacobi_integrate_poly(a, b, Nx, bounds, poly_order):

    # Domain
    domain = dev.domain.Domain(dim=1, dtype=np.float64)
    xs = dev.spaces.FiniteInterval(a=a, b=b, name='x', coeff_size=Nx, bounds=bounds, domain=domain, axis=0, dealias=1)
    xb0 = xs.Jacobi(da=0, db=0)
    xb1 = xs.Jacobi(da=1, db=1)

    # Build random polynomial
    rand = np.random.RandomState(seed=14)
    poly = rand.random_integers(-10, 11, poly_order)
    deriv = np.polyder(poly)
    x = xs.local_grid()
    F = dev.field.Field(domain=domain, bases=[xb0], name='F')
    F['g'] = np.polyval(deriv, x)

    # Problem
    p1 = xb1[-1]
    p1.name = 'p1'
    problem = dev.problems.LBVP(domain)
    problem.add_variable('u', bases=[xb0])
    problem.add_variable('t1', domain=domain)
    problem.parameters['p1'] = p1
    problem.parameters['F'] = F
    problem.parameters['G'] = np.polyval(poly, bounds[0])
    problem.add_equation('dx(u) + t1*p1 = F')
    problem.add_equation("u(x='left') = G")

    # Solve
    solver = problem.build_solver()
    solver.solve()

    # Test
    u = solver.state[0]
    return np.allclose(u['g'], np.polyval(poly, x))


def test_legendre():
    assert jacobi_integrate_poly(a=0, b=0, Nx=32, bounds=(2,3), poly_order=10)
    assert jacobi_integrate_poly(a=0, b=0, Nx=33, bounds=(2,3), poly_order=10)


def test_chebyshevT():
    assert jacobi_integrate_poly(a=-1/2, b=-1/2, Nx=32, bounds=(2,3), poly_order=10)
    assert jacobi_integrate_poly(a=-1/2, b=-1/2, Nx=33, bounds=(2,3), poly_order=10)


def test_chebyshevU():
    assert jacobi_integrate_poly(a=1/2, b=1/2, Nx=32, bounds=(2,3), poly_order=10)
    assert jacobi_integrate_poly(a=1/2, b=1/2, Nx=33, bounds=(2,3), poly_order=10)


if __name__ == '__main__':
    defaults = {'a':0, 'b':0, 'Nx':32, 'bounds':(2,3), 'poly_order':10}
    print(jacobi_integrate_poly(**defaults))

