import pytest
import numpy as np
from dedalus.core import coords, distributor, basis, field, operators, arithmetic
from dedalus.tools.cache import CachedMethod
from mpi4py import MPI

N_range = [8, 9, 10]
ab_range = [-0.5, 0, 0.5]
func_range = operators.UnaryGridFunction.supported.values()

@pytest.mark.parametrize('N', N_range)
@pytest.mark.parametrize('a', ab_range)
@pytest.mark.parametrize('b', ab_range)
@pytest.mark.parametrize('func', func_range)
def test_jacobi_ufunc(N, a, b, func):
    c = coords.Coordinate('x')
    d = distributor.Distributor((c,))
    b = basis.Jacobi(c, size=N, a=a, b=b, bounds=(0, 1))
    x = b.local_grid(1)
    f = field.Field(dist=d, bases=(b,), dtype=np.complex128)
    f['g'] = np.sin(x)
    f0 = np.copy(f['g'])
    dedalus_func = lambda A: operators.UnaryGridFunctionField(func, A)
    g = dedalus_func(f).evaluate()
    assert np.allclose(g['g'], func(f0))
