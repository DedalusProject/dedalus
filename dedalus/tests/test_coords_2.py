

import numpy as np
from dedalus.core import spaces
from dedalus.core import basis
from dedalus.core import distributor
from dedalus.core import field


dist = distributor.Distributor(dim=3)
ss = spaces.Sphere(('phi','theta'), Lmax=32, radius=4, dist=dist, axis=0)
zs = spaces.FiniteInterval('z', size=32, bounds=(-1,1), dist=dist, axis=2)


u = field.Field(dist, bases=[basis.SWSH(ss)])
print(u)



