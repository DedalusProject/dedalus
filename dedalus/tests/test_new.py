
import numpy as np
from dedalus.core import coords, distributor, basis, field

c = coords.CartesianCoordinates('x', 'y')
d = distributor.Distributor(c.coords)
xb = basis.ComplexFourier(c.coords[0], size=8, bounds=(1,2))
yb = basis.ComplexFourier(c.coords[1], size=8, bounds=(1,2))
#yb = basis.Jacobi(c.coords[1], size=8, a=0, b=0, a0=0, b0=0, bounds=(3,4))
u = field.Field(dist=d, bases=(xb,yb), dtype=np.complex128)

u.set_scales(0.5)
u['c'][1, 1] = 1
uc0 = u['c'].copy()
u['g']
print(np.allclose(u['c'], uc0))

c = coords.S2Coordinates('phi','theta')
d = distributor.Distributor(c.coords)
sb = basis.SpinWeightedSphericalHarmonics(c, 15, 1, fourier_library='matrix')

v = field.Field(dist=d, bases=(sb,), dtype=np.complex128)

