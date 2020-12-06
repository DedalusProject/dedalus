
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
T = field.Field(dist=d, bases=(b,), tensorsig=(c,c,), dtype=np.complex128)

class Subproblem:

    def __init__(self,ell):
        self.ell = ell

# Helper functions for old matrix construction
N = 15
def D(sp,mu,i,deg):
    if mu == +1: return ball.operator(3,'D+',N,i,sp.ell,deg)
    if mu == -1: return ball.operator(3,'D-',N,i,sp.ell,deg)

def E(sp,i,deg): return ball.operator(3,'E',N,i,sp.ell,deg)

def zeros(sp): return ball.operator(3,'0',N,0,sp.ell,0)

sp = Subproblem(3)

xim = intertwiner.xi(-1,sp.ell)
xip = intertwiner.xi(+1,sp.ell)

# Gradient of a scalar
gradf = operators.Gradient(f, c)
L1 = gradf.subproblem_matrix(sp)

Z = zeros(sp)

L03 = xim*D(sp,-1,0,0)
L13 = xip*D(sp,+1,0,0)
L1_old = sparse.bmat([[L03],
                      [L13],
                      [Z]])

result = np.allclose(L1.todense(),L1_old.todense())
results.append(result)
print(len(results), ':', result)

# Laplacian of a vector
lapu = operators.Laplacian(u, c)
L2 = lapu.subproblem_matrix(sp)

Z = zeros(sp)

L00 = D(sp,-1,1, 0).dot(D(sp,+1, 0,-1))
L11 = D(sp,+1,1, 0).dot(D(sp,-1, 0,+1))
L22 = D(sp,-1,1,+1).dot(D(sp,+1, 0, 0))

L2_old=sparse.bmat([[L00,   Z,   Z],
                    [  Z, L11,   Z],
                    [  Z,   Z, L22]])

result = np.allclose(L2.todense(),L2_old.todense())
results.append(result)
print(len(results), ':', result)

# Divergence of vector
divu = operators.Divergence(u)
L3 = divu.subproblem_matrix(sp)

L30 = xim*D(sp,+1,0,-1)
L31 = xip*D(sp,-1,0,+1)
Z = zeros(sp)

L3_old=sparse.bmat([[L30, L31, Z]])

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

L5_old=sparse.bmat([[M00,   Z,   Z],
                    [  Z, M11,   Z],
                    [  Z,   Z, M22]])

result = np.allclose(L5.todense(),L5_old.todense())
results.append(result)
print(len(results), ':', result)

# Composition
op = operators.convert(operators.Gradient(f, c) + operators.Laplacian(u, c), (bk2,))
op_matrices = op.expression_matrices(sp, (f,u,))
L6 = op_matrices[f]

L03 = xim*E(sp,+1,-1)@D(sp,-1,0,0)
L13 = xip*E(sp,+1,+1)@D(sp,+1,0,0)
Z = zeros(sp)
L6_old = sparse.bmat([[L03],
                      [L13],
                      [Z]])

result = np.allclose(L6.todense(),L6_old.todense())
results.append(result)
print(len(results), ':', result)

L7 = op_matrices[u]
L7_old = L2_old

result = np.allclose(L7.todense(),L7_old.todense())
results.append(result)
print(len(results), ':', result)

# Matrix of Zeros
Z = zeros(sp)
L22 = D(sp,-1,1,+1).dot(D(sp,+1, 0, 0))

L8_old=0*sparse.bmat([[Z, Z, L22]])

zero = operators.ZeroMatrix(u, ())
L8 = zero.subproblem_matrix(sp)

result = np.allclose(L8.todense(),L8_old.todense())
results.append(result)
print(len(results), ':', result)

# Scalar Interpolation
op = operators.interpolate(f,r=1)
L9 = op.subproblem_matrix(sp)

L9_old = ball.operator(3,'r=R',N,0,sp.ell,0)

result = np.allclose(L9,L9_old)
results.append(result)
print(len(results), ':', result)

# Vector Interpolation
op = operators.interpolate(u,r=1)
L10 = op.subproblem_matrix(sp)

N0 = b.n_size((),sp.ell)
N1 = 2*N0
N2 = 3*N0
N3 = 4*N0

row0=np.concatenate((             ball.operator(3,'r=R',N,0,sp.ell,-1),np.zeros(N2-N0)))
row1=np.concatenate((np.zeros(N0),ball.operator(3,'r=R',N,0,sp.ell,+1),np.zeros(N2-N1)))
row2=np.concatenate((np.zeros(N1),ball.operator(3,'r=R',N,0,sp.ell, 0)))
L10_old = np.vstack((row0,row1,row2))
Q = b.radial_recombinations((c,),ell_list=(sp.ell,))[0]
L10_old = Q @ L10_old

result = np.allclose(L10,L10_old)
results.append(result)
print(len(results), ':', result)

# Vector of Zeros
zero = operators.ZeroVector(u, (c,))
L11 = zero.subproblem_matrix(sp)

row0=np.concatenate((             ball.operator(3,'r=R',N,0,sp.ell,-1),np.zeros(N2-N0)))
row1=np.concatenate((np.zeros(N0),ball.operator(3,'r=R',N,0,sp.ell,+1),np.zeros(N2-N1)))
row2=np.concatenate((np.zeros(N1),ball.operator(3,'r=R',N,0,sp.ell, 0)))
L11_old = 0*np.vstack((row0,row1,row2))

result = np.allclose(L11,L11_old)
results.append(result)
print(len(results), ':', result)

# Transpose
op = operators.TransposeComponents(T)
L12 = op.subproblem_matrix(sp)

Z = zeros(sp)
eye = np.eye(Z.shape[0])

transpose = np.zeros((9,9))
transpose[0,0] = 1
transpose[1,3] = 1
transpose[2,6] = 1
transpose[3,1] = 1
transpose[4,4] = 1
transpose[5,7] = 1
transpose[6,2] = 1
transpose[7,5] = 1
transpose[8,8] = 1
Q = b.radial_recombinations((c,c,),ell_list=(sp.ell,))[0]
trans = Q.T @ transpose @ Q
L12_old = np.kron(trans, eye)

result = np.allclose(L12.todense(),L12_old)
results.append(result)
print(len(results), ':', result)

# Radial Component
op = operators.RadialComponent(operators.interpolate(u,r=1))
L10 = op.expression_matrices(sp,(u,))
L10 = L10[u]

N0 = b.n_size((),sp.ell)
N1 = 2*N0
N2 = 3*N0
N3 = 4*N0

row0=np.concatenate((             ball.operator(3,'r=R',N,0,sp.ell,-1),np.zeros(N2-N0)))
row1=np.concatenate((np.zeros(N0),ball.operator(3,'r=R',N,0,sp.ell,+1),np.zeros(N2-N1)))
row2=np.concatenate((np.zeros(N1),ball.operator(3,'r=R',N,0,sp.ell, 0)))
L10_old = np.vstack((row0,row1,row2))
Q = b.radial_recombinations((c,),ell_list=(sp.ell,))[0]
L10_old = Q @ L10_old
L10_old = L10_old[2]

result = np.allclose(L10,L10_old)
results.append(result)
print(len(results), ':', result)

