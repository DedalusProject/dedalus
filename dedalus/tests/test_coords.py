

import numpy as np
from dedalus.core import distributor, spaces, basis, field, coordinates


d = distributor.Distributor(dim=2)
s = spaces.Sphere(('phi', 'theta'), Lmax=7, radius=1, dist=d, axis=0)
b = basis.SWSH(s)

X = coordinates.VectorSpace([s])
u = field.Field(d, bases=[b], name='u', tensorsig=[X,X], dtype=np.complex128)

print(u.data.shape)
print(u['g'])
print(u['c'])