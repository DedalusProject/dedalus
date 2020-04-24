
import numpy as np
from dedalus.core import coords, distributor, basis, field, operators
from dedalus.core import future
from dedalus.tools.array import apply_matrix
from mpi4py import MPI

comm = MPI.COMM_WORLD

results = []

## Ball
c = coords.SphericalCoordinates('phi', 'theta', 'r')
d = distributor.Distributor((c,))
b = basis.BallBasis(c, (16,16,16), radius=1)
phi, theta, r = b.local_grids((1, 1, 1))
x = r * np.sin(theta) * np.cos(phi)
y = r * np.sin(theta) * np.sin(phi)
z = r * np.cos(theta)

class Subproblem:

    def __init__(self,ell):
        self.ell = ell

## Scalar x Scalar
#f = field.Field(dist=d, bases=(b,), dtype=np.complex128)
#g = field.Field(dist=d, bases=(b,), dtype=np.complex128)
#h = field.Field(dist=d, bases=(b,), dtype=np.complex128)
#f['g'] = r**4
#g['g'] = 3*x**2 + 2*y*z
#h0 = f['g'] * g['g']
#
#for ell in b.local_l:
#    sp = Subproblem(ell)
#    matrix = b.tensor_product_ncc(b, f['c'][0,0,:], (), (), (), sp, ncc_first=True)
#    slice = b.n_slice((),ell)
#    apply_matrix(matrix/np.sqrt(2), g['c'][:,ell,slice], axis=1, out=h['c'][:,ell,slice])
#
#result = np.allclose(h['g'], h0)
#results.append(result)
#print(len(results), ':', result)

## Scalar x Vector
#f = field.Field(dist=d, bases=(b,), dtype=np.complex128)
#g = field.Field(dist=d, bases=(b,), dtype=np.complex128)
#
#bk1 = basis.BallBasis(c, (16,16,16), k=1, radius=1)
#v = field.Field(dist=d, bases=(bk1,), tensorsig=(c,), dtype=np.complex128)
#
#f['g'] = 2*r**2-1
#g['g'] = 3*x**2 + 2*y*z
#u = operators.Gradient(g, c).evaluate()
##u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
##u['g'][2] = (6*x**2+4*y*z)/r
##u['g'][1] = -2*(y**3+x**2*(y-3*z)-y*z**2)/(r**2*np.sin(theta))
##u['g'][0] = 2*x*(-3*y+z)/(r*np.sin(theta))
#
#v['g'] = f['g']*u['g']
#
#w = field.Field(dist=d, bases=u.domain.bases, tensorsig=(c,), dtype=np.complex128)
#
#for ell in b.local_l:
#    sp = Subproblem(ell)
#    slice = b.n_slice((),ell)
#    vector = np.transpose(u['c'][:,:,ell,slice], axes=(1,0,2))
#    shape = vector.shape
#    view = vector.reshape((shape[0],shape[1]*shape[2]))
#    matrix = b.tensor_product_ncc(u.domain.bases[0], f['c'][0,0,:], (), (c,), (c,), sp, ncc_first=True)
#    apply_matrix(matrix/np.sqrt(2), view, axis=1, out=view)
#    vector = view.reshape(shape)
#    w['c'][:,:,ell,slice] = np.transpose(vector, axes=(1,0,2))
#
#result = np.allclose(v['g'], w['g'])
#results.append(result)
#print(len(results), ':', result)

## Vector x Scalar
#f = field.Field(dist=d, bases=(b,), dtype=np.complex128)
#g = field.Field(dist=d, bases=(b,), dtype=np.complex128)
#v = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
#
#f['g'] = r**6
#g['g'] = 3*x**2 + 2*y*z
#u = operators.Gradient(f, c).evaluate()
#
#v['g'] = g['g']*u['g']
#
#w = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
#
#ncc_basis = u.domain.bases[0]
#
#for ell in ncc_basis.local_l:
#    sp = Subproblem(ell)
#    matrix = ncc_basis.tensor_product_ncc(b, u['c'][:,0,0,:], (c,), (), (c,), sp, ncc_first=True)
#    slice = b.n_slice((),ell)
#    view = apply_matrix(matrix, g['c'][:,ell,slice], axis=1)
#    shape = w['c'][:,:,ell,slice].shape
#    vector = view.reshape(shape[1],shape[0],shape[2])
#    w['c'][:,:,ell,slice] = np.transpose(vector, axes=(1,0,2))
#
#result = np.allclose(v['g'], w['g'])
#results.append(result)
#print(len(results), ':', result)

## Vector x Vector
#f = field.Field(dist=d, bases=(b,), dtype=np.complex128)
#g = field.Field(dist=d, bases=(b,), dtype=np.complex128)
#T = field.Field(dist=d, bases=(b,), tensorsig=(c,c), dtype=np.complex128)
#
#f['g'] = r**6
#g['g'] = 3*x**2 + 2*y*z
#u = operators.Gradient(f, c).evaluate()
#v = operators.Gradient(g, c).evaluate()
#
#T = field.Field(dist=d, bases=v.domain.bases, tensorsig=(c,c), dtype=np.complex128)
#
#for i in range(3):
#    for j in range(3):
#        T['g'][j,i] = u['g'][i]*v['g'][j]
#
#ncc_basis = u.domain.bases[0]
#arg_basis = v.domain.bases[0]
#
#W = field.Field(dist=d, bases=v.domain.bases, tensorsig=(c,c), dtype=np.complex128)
#
#for ell in b.local_l:
#    sp = Subproblem(ell)
#    slice = b.n_slice((),ell)
#    vector = np.transpose(v['c'][:,:,ell,slice], axes=(1,0,2))
#    shape = vector.shape
#    view = vector.reshape((shape[0],shape[1]*shape[2]))
#    matrix = ncc_basis.tensor_product_ncc(arg_basis, u['c'][:,0,0,:], (c,), (c,), (c,c), sp, ncc_first=False)
#    view = apply_matrix(matrix, view, axis=1)
#    shape = T['c'][:,:,:,ell,slice].shape
#    vector = view.reshape(shape[2],shape[0],shape[1],shape[3])
#    W['c'][:,:,:,ell,slice] = np.transpose(vector, axes=(1,2,0,3))
#
#result = np.allclose(T['g'], W['g'])
#results.append(result)
#print(len(results), ':', result)

## Vector dot Vector
#
#f = field.Field(dist=d, bases=(b,), dtype=np.complex128)
#g = field.Field(dist=d, bases=(b,), dtype=np.complex128)
#
#f['g'] = r**6
#g['g'] = 3*x**2 + 2*y*z
#u = operators.Gradient(f, c).evaluate()
#v = operators.Gradient(g, c).evaluate()
#
#h = field.Field(dist=d, bases=v.domain.bases, dtype=np.complex128)
#
#for i in range(3):
#    h['g'] = u['g'][i]*v['g'][i]
#
#ncc_basis = u.domain.bases[0]
#arg_basis = v.domain.bases[0]
#
#t = field.Field(dist=d, bases=v.domain.bases, dtype=np.complex128)
#
#for ell in b.local_l:
#    sp = Subproblem(ell)
#    slice = b.n_slice((),ell)
#    vector = np.transpose(v['c'][:,:,ell,slice], axes=(1,0,2))
#    shape = vector.shape
#    view = vector.reshape((shape[0],shape[1]*shape[2]))
#    matrix = ncc_basis.dot_product_ncc(arg_basis, u['c'][:,0,0,:], (c,), (c,), (), sp, ncc_first=False, indices=(0,0))
#    t['c'][:,ell,slice] = apply_matrix(matrix, view, axis=1)
#
#result = np.allclose(t['g'], h['g'])
#results.append(result)
#print(len(results), ':', result)

# Vector dot Tensor

f = field.Field(dist=d, bases=(b,), dtype=np.complex128)
T = field.Field(dist=d, bases=(b,), tensorsig=(c,c,), dtype=np.complex128)

T['g'][2,2] = (6*x**2+4*y*z)/r**2
T['g'][2,1] = T['g'][1,2] = -2*(y**3+x**2*(y-3*z)-y*z**2)/(r**3*np.sin(theta))
T['g'][2,0] = T['g'][0,2] = 2*x*(z-3*y)/(r**2*np.sin(theta))
T['g'][1,1] = 6*x**2/(r**2*np.sin(theta)**2) - (6*x**2+4*y*z)/r**2
T['g'][1,0] = T['g'][0,1] = -2*x*(x**2+y**2+3*y*z)/(r**3*np.sin(theta)**2)
T['g'][0,0] = 6*y**2/(x**2+y**2)

f['g'] = r**6
u = operators.Gradient(f, c).evaluate()

v = field.Field(dist=d, bases=T.domain.bases, tensorsig=(c,), dtype=np.complex128)

for i in range(3):
    for j in range(3):
        v['g'][i] = u['g'][j]*T['g'][j,i]

ncc_basis = u.domain.bases[0]
arg_basis = T.domain.bases[0]

w = field.Field(dist=d, bases=T.domain.bases, tensorsig=(c,), dtype=np.complex128)

for ell in b.local_l:
    sp = Subproblem(ell)
    slice = b.n_slice((),ell)
    vector = np.transpose(T['c'][:,:,:,ell,slice], axes=(2,0,1,3))
    shape = vector.shape
    view = vector.reshape((shape[0],shape[1]*shape[2]*shape[3]))
    matrix = ncc_basis.dot_product_ncc(arg_basis, u['c'][:,0,0,:], (c,), (c,c), (c,), sp, ncc_first=True, indices=(0,0))
    view = apply_matrix(matrix, view, axis=1)
    shape = w['c'][:,:,ell,slice].shape
    vector = view.reshape(shape[1],shape[0],shape[2])
    w['c'][:,:,ell,slice] = np.transpose(vector, axes=(1,0,2))

result = np.allclose(v['g'], w['g'])
results.append(result)
print(len(results), ':', result)


