

import numpy as np
from dedalus.core import distributor, spaces, basis, field


d = distributor.Distributor(dim=2)
s = spaces.Sphere(('phi', 'theta'), Lmax=32, radius=1, dist=d, axis=0)
b = basis.SWSH(s)

u = field.Field(d, bases=[b], name='u', tens=None)
print(u.data)
