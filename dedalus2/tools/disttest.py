

import numpy as np
from dedalus2_burns.dev import *


zb = basis.Chebyshev(20)
yb = basis.Fourier(20)
xb = basis.Fourier(20)

dom = domain.Domain([xb, yb, zb], grid_dtype=np.float64)

if dom.distributor.local_process == 0:
    print('*'*10)
    for l in dom.distributor.layouts:
        print('local:', l.local)
        print('grid :', l.grid_space)
        print('shape:', l.shape)
        print('dtype:', l.dtype)
        print('*'*10)

