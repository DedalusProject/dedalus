"""
Test solving Legendre's differential equation using Jacobi basis.
"""

import numpy as np
from scipy import special
from dedalus_burns import dev
import logging
logger = logging.getLogger(__name__)


def general_test(a, b, Nx, n):

    # Domain
    domain = dev.domain.Domain(dim=1, dtype=np.float64)
    xs = dev.spaces.FiniteInterval(a=a, b=b, name='x', coeff_size=Nx, bounds=[-1,1], domain=domain, axis=0, dealias=3/2)
    xb0 = xs.Jacobi(da=0, db=0)
    xb2 = xs.Jacobi(da=0, db=0)

    # Problem
    problem = dev.problems.LBVP(domain)
    problem.add_variable('u', bases=[xb0])
    problem.add_variable('t1', domain=domain)
    problem.add_variable('t2', domain=domain)
    problem.parameters['p1'] = xb2[-1]
    problem.parameters['p2'] = xb2[-2]
    problem.parameters['n'] = n
    problem.add_equation('dx((1-x*x)*dx(u)) + n*(n+1)*u + t1*p1 + t2*p2 = 0')
    problem.add_equation("u(x='left') = (-1)**n")
    problem.add_equation("u(x='right') = 1")

    # Solve
    solver = problem.build_solver()
    solver.solve()

    # Test
    u = solver.state[0]
    t1 = solver.state[1]
    t2 = solver.state[2]
    x = xs.local_grid()
    exact = special.eval_legendre(n, x)
    return np.allclose(u['g'], exact)


def test_legendre():
    assert general_test(a=0, b=0, Nx=32, n=7)


def test_chebyshevT():
    assert general_test(a=-1/2, b=-1/2, Nx=32, n=7)


def test_chebyshevU():
    assert general_test(a=1/2, b=1/2, Nx=32, n=7)


if __name__ == '__main__':
    defaults = {'a':0, 'b':0, 'Nx':32, 'n':7}
    print(general_test(**defaults))

