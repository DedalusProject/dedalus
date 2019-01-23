

import numpy as np
from dedalus.core import distributor, spaces, basis, field, vectorspaces


d = distributor.Distributor(dim=4)
s1 = spaces.Ball(('phi', 'theta', 'r'), shape=(15,8,8), radius=2, dist=d, axis=0)
s2 = spaces.FiniteInterval('a', size=64, bounds=(0,10), dist=d, axis=3)
b1 = basis.BallBasis(s1)
b2 = basis.Jacobi(s2, 0, 0)

X = vectorspaces.VectorSpace([s1])
u = field.Field(d, bases=[b1, b2], name='u', tensorsig=[X,X], dtype=np.complex128)

u['c']
print('Coeff')
u['g']
print('Grid')
u['c']
print('Coeff')

