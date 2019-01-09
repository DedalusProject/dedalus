

import numpy as np
from dedalus.core import distributor, spaces, basis, field, vectorspaces


d = distributor.Distributor(dim=3)
s = spaces.Ball(('phi', 'theta', 'r'), Lmax=8, Nmax=8, radius=2, dist=d, axis=0)
b = basis.BallBasis(s)

X = vectorspaces.VectorSpace([s])
u = field.Field(d, bases=[b], name='u', tensorsig=[X,X], dtype=np.complex128)

print(s.grids(1))
u['c']
print('Coeff')
u['g']
print('Grid')
u['c']
print('Coeff')