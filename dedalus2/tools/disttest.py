

import numpy as np
from dedalus2.dev import *


zb = basis.Chebyshev(20)
yb = basis.Fourier(20)
xb = basis.Fourier(20)

dom = domain.Domain([xb, yb, zb], dtype=np.float64)
mesh = [2, 2]

dist = dist.Distributor(dom, mesh)

print('*'*10)
for l in dist.layouts:
    print('local:', l.local)
    print('real :', l.grid_space)
    print('shape:', l.shape)
    print('dtype:', l.dtype)
    print('*'*10)

