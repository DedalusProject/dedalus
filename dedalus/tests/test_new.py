
import numpy as np
import matplotlib
matplotlib.use('Agg')

from dedalus.core import coords, distributor, basis, field

c = coords.CartesianCoordinates('x', 'y')
d = distributor.Distributor(c.coords)
xb = basis.ComplexFourier(c.coords[0], size=8, bounds=(1,2))
yb = basis.ComplexFourier(c.coords[1], size=8, bounds=(1,2))
#yb = basis.Jacobi(c.coords[1], size=8, a=0, b=0, a0=0, b0=0, bounds=(3,4))
u = field.Field(dist=d, bases=(xb,yb), dtype=np.complex128)

u['c'][1, 1] = 1
uc0 = u['c'].copy()
u['g']
print('Complex Fourier transform check:')
print(np.allclose(u['c'], uc0))

c = coords.S2Coordinates('phi','theta')
d = distributor.Distributor(c.coords)
sb = basis.SpinWeightedSphericalHarmonics(c, 7, 1, fourier_library='matrix')

phi, theta = sb.local_grids((1, 1))

v = field.Field(dist=d, bases=(sb,), dtype=np.complex128)
vg = np.sin(theta)**2 * np.exp(-2j*phi)
v['g'] = vg
v['c']
print('SWSH scalar transform check:')
print(np.allclose(v['g'], vg))

w = field.Field(dist=d, bases=(sb,), tensorsig=(c,), dtype=np.complex128)

w['c'][0,2,3] = 1.
wc0 = w['c'].copy()
wg = -1j*np.sqrt(35/512)/2*(np.sin(theta) - 4*np.sin(2*theta) - 3*np.sin(3*theta)) * np.exp(2j*phi)
print('SWSH vector transform check:')
print(np.allclose(w['g'][0], wg))

T = field.Field(dist=d, bases=(sb,), tensorsig=(c,c), dtype=np.complex128)

T['c'][0,0,2,3] = 1.
Tc0 = T['c'].copy()
Tg = -0.5*np.sqrt(7/2)*(np.cos(theta/2)**4 * ( -2 + 3*np.cos(theta) ) ) * np.exp(2j*phi)
print('SWSH tensor transform check:')
print(np.allclose(T['g'][0,0], Tg))



