
import numpy as np
from dedalus.core import coords, distributor, basis, field, operators
from scipy import sparse
from dedalus_sphere import intertwiner, ball

from mpi4py import MPI

comm = MPI.COMM_WORLD

results = []

## Ball
c = coords.SphericalCoordinates('phi', 'theta', 'r')
d = distributor.Distributor(c.coords)
b = basis.BallBasis(c, (16,16,16), radius=1)
bk2 = basis.BallBasis(c, (16,16,16), radius=1, k=2)

f = field.Field(dist=d, bases=(b,), dtype=np.complex128)
u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)

class Subproblem:

    def __init__(self,ell):
        self.ell = ell

# Helper functions for old matrix construction
N = 15
def D(sp,mu,i,deg):
    if mu == +1: return ball.operator(3,'D+',N,i,sp.ell,deg)
    if mu == -1: return ball.operator(3,'D-',N,i,sp.ell,deg)

def E(sp,i,deg): return ball.operator(3,'E',N,i,sp.ell,deg)

def Z(sp,deg_out,deg_in): return ball.zeros(N,sp.ell,deg_out,deg_in)

sp = Subproblem(3)

xim = intertwiner.xi(-1,sp.ell)
xip = intertwiner.xi(+1,sp.ell)

# Gradient of a scalar
gradf = operators.Gradient(f, c)
L1 = gradf.subproblem_matrix(sp)

L03 = xim*D(sp,-1,0,0)
L13 = xip*D(sp,+1,0,0)
L23 = Z(sp,0,0)
L1_old = sparse.bmat([[L03],
                      [L13],
                      [L23]])

result = np.allclose(L1.todense(),L1_old.todense())
results.append(result)
print(len(results), ':', result)

# Laplacian of a vector
lapu = operators.Laplacian(u, c)
L2 = lapu.subproblem_matrix(sp)

Z01 = Z(sp,-1,+1)
Z02 = Z(sp,-1, 0)
Z10 = Z(sp,+1,-1)
Z12 = Z(sp,+1, 0)
Z20 = Z(sp, 0,-1)
Z21 = Z(sp, 0,+1)

L00 = D(sp,-1,1, 0).dot(D(sp,+1, 0,-1))
L11 = D(sp,+1,1, 0).dot(D(sp,-1, 0,+1))
L22 = D(sp,-1,1,+1).dot(D(sp,+1, 0, 0))

L2_old=sparse.bmat([[L00, Z01, Z02],
                    [Z10, L11, Z12],
                    [Z20, Z21, L22]])

result = np.allclose(L2.todense(),L2_old.todense())
results.append(result)
print(len(results), ':', result)

# Divergence of vector
divu = operators.Divergence(u)
L3 = divu.subproblem_matrix(sp)

L30 = xim*D(sp,+1,0,-1)
L31 = xip*D(sp,-1,0,+1)
Z32 = Z(sp, 0, 0)

L3_old=sparse.bmat([[L30, L31, Z32]])

result = np.allclose(L3.todense(),L3_old.todense())
results.append(result)
print(len(results), ':', result)

# Laplacian of scalar
lapf = operators.Laplacian(f, c)
L4 = lapf.subproblem_matrix(sp)

L4_old = D(sp,-1,1,+1).dot(D(sp,+1, 0,0))

result = np.allclose(L4.todense(),L4_old.todense())
results.append(result)
print(len(results), ':', result)

# Conversion

convertu = operators.convert(u, (bk2,)) 
L5 = convertu.subproblem_matrix(sp)

M00 = E(sp,1,-1).dot(E(sp, 0,-1))
M11 = E(sp,1,+1).dot(E(sp, 0,+1))
M22 = E(sp,1, 0).dot(E(sp, 0, 0))

L5_old=sparse.bmat([[M00, Z01, Z02],
                    [Z10, M11, Z12],
                    [Z20, Z21, M22]])

result = np.allclose(L5.todense(),L5_old.todense())
results.append(result)
print(len(results), ':', result)

