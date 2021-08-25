"""
Test arithmetic operators on 1D Jacobi fields.
"""

import numpy as np
from dedalus_burns import dev
import logging
logger = logging.getLogger(__name__)


# Parameters
a = -1/2
b = -1/2
Nx = 32
bounds = (0,1)

# Domain
domain = dev.domain.Domain(dim=1, dtype=np.float64)
xs = dev.spaces.FiniteInterval(a=a, b=b, name='x', coeff_size=Nx, bounds=bounds, domain=domain, axis=0, dealias=3/2)
xb0 = xs.Jacobi(da=0, db=0)
x = xs.local_grid()

def test_add_const_const():
    f1 = dev.field.Field(domain=domain)
    f2 = dev.field.Field(domain=domain)
    f1['g'] = 3
    f2['g'] = 5
    test = (f1 + f2).evaluate()
    assert np.allclose(test['g'], 8)

def test_add_const_field():
    f1 = dev.field.Field(domain=domain)
    f2 = dev.field.Field(bases=[xb0])
    f1['g'] = 2
    f2['g'] = 1 + np.sin(x)
    test = (f1 + f2).evaluate()
    test.require_scales(1)
    exact = 3 + np.sin(x)
    assert np.allclose(test['g'], exact)

def test_add_field_field():
    f1 = dev.field.Field(bases=[xb0])
    f2 = dev.field.Field(bases=[xb0])
    f1['g'] = 1 + x
    f2['g'] = x**2
    test = (f1 + f2).evaluate()
    test.require_scales(1)
    exact = 1 + x + x**2
    assert np.allclose(test['g'], exact)

def test_multiply_const_const():
    f1 = dev.field.Field(domain=domain)
    f2 = dev.field.Field(domain=domain)
    f1['g'] = 3
    f2['g'] = 5
    test = (f1 * f2).evaluate()
    assert np.allclose(test['g'], 15)

def test_multiply_const_field():
    f1 = dev.field.Field(domain=domain)
    f2 = dev.field.Field(bases=[xb0])
    f1['g'] = 2
    f2['g'] = 1 + np.sin(x)
    test = (f1 * f2).evaluate()
    test.require_scales(1)
    exact = 2 + 2*np.sin(x)
    assert np.allclose(test['g'], exact)

def test_multiply_field_field():
    f1 = dev.field.Field(bases=[xb0])
    f2 = dev.field.Field(bases=[xb0])
    f1['g'] = 1 + x
    f2['g'] = x**2
    test = (f1 * f2).evaluate()
    test.require_scales(1)
    exact = x**2 + x**3
    assert np.allclose(test['g'], exact)
