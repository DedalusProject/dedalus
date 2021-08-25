"""
Test integrating a random trigonometric polynomials using Fourier bases.
"""

import numpy as np
from dedalus_burns import dev
import logging
logger = logging.getLogger(__name__)


def fourier_integrate_trigpoly(Nx, bounds, poly_order):

    # Domain
    domain = dev.domain.Domain(dim=1, dtype=np.float64)
    xs = dev.spaces.PeriodicInterval(name='x', coeff_size=Nx, bounds=bounds, domain=domain, axis=0, dealias=3/2)
    xb = xs.Fourier

    # Build random polynomial
    Lx = bounds[1] - bounds[0]
    k0 = 2 * np.pi / Lx
    rand = np.random.RandomState(seed=66)
    psin = rand.random_integers(-10, 11, poly_order)
    pcos = rand.random_integers(-10, 11, poly_order)
    dsin = -np.arange(poly_order) * k0 * pcos
    dcos = np.arange(poly_order) * k0 * psin

    x = xs.local_grid()
    p = d = 0
    for n in range(poly_order):
        p += psin[n] * np.sin(n*k0*x)
        p += pcos[n] * np.cos(n*k0*x)
        d += dsin[n] * np.sin(n*k0*x)
        d += dcos[n] * np.cos(n*k0*x)

    F = dev.field.Field(domain=domain, bases=[xb])
    F['g'] = d

    # Problem
    problem = dev.problems.LBVP(domain)
    problem.add_variable('u', bases=[xb])
    problem.add_variable('t1', domain=domain)
    problem.parameters['F'] = F
    problem.parameters['G'] = p[0]
    problem.add_equation('dx(u) + t1 = F')
    problem.add_equation("u(x='left') = G")

    # Solve
    solver = problem.build_solver()
    solver.solve()

    # Test
    u = solver.state[0]
    return np.allclose(u['g'], p)


def sine_integrate_trigpoly(Nx, bounds, poly_order):

    # Domain
    domain = dev.domain.Domain(dim=1, dtype=np.float64)
    xs = dev.spaces.ParityInterval(name='x', base_grid_size=Nx, bounds=bounds, domain=domain, axis=0, dealias=3/2)

    # Build random polynomial
    Lx = bounds[1] - bounds[0]
    k0 = np.pi / Lx
    rand = np.random.RandomState(seed=66)
    psin = rand.random_integers(-10, 11, poly_order)
    dcos = np.arange(poly_order) * k0 * psin

    x = xs.local_grid()
    p = d = 0
    for n in range(poly_order):
        p += psin[n] * np.sin(n*k0*x)
        d += dcos[n] * np.cos(n*k0*x)

    F = dev.field.Field(domain=domain, bases=[xs.Cosine])
    F['g'] = d

    # Problem
    problem = dev.problems.LBVP(domain)
    problem.add_variable('u', bases=[xs.Sine])
    problem.add_variable('t1', domain=domain)
    problem.parameters['F'] = F
    problem.add_equation('dx(u) + t1 = F')

    # Solve
    solver = problem.build_solver()
    solver.solve()

    # Test
    u = solver.state[0]
    return np.allclose(u['g'], p)


def test_sine():
    assert sine_integrate_trigpoly(Nx=64, bounds=(5,10), poly_order=10)





if __name__ == '__main__':
    defaults = {'Nx':256, 'bounds':(5,10), 'poly_order':10}
    print(fourier_integrate_trigpoly(**defaults))

