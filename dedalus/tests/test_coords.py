

import numpy as np
from dedalus.core import distributor, spaces, basis, field


d = distributor.Distributor(dim=3)
s = spaces.Sphere(('phi', 'theta'), Lmax=32, radius=1, dist=d, axis=0)
#z = spaces.FiniteInterval('z', dist=d, axis=2)
b = basis.SWSH(s)

u = field.Field(d, bases=[b], name='u', tensorsig=None)
print(u['g'])

