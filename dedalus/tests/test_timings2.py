
import numpy as np
from dedalus.core import coords, distributor, basis, field, operators
from mpi4py import MPI
import time

comm = MPI.COMM_WORLD

c = coords.SphericalCoordinates('phi', 'theta', 'r')
d = distributor.Distributor(c.coords)
b = basis.BallBasis(c, (16,16,16), radius=1)
phi, theta, r = b.local_grids((1, 1, 1))
x = r * np.sin(theta) * np.cos(phi)
y = r * np.sin(theta) * np.sin(phi)
z = r * np.cos(theta)

f = field.Field(dist=d, bases=(b,), dtype=np.complex128)
u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)

# IC
f['g'] = x**4*y**2*z + 5*x**2*z**6 - 2*y**4*z**5
u['g'] = operators.Gradient(f,c).evaluate()['g']

neg = field.Field(dist=d, bases=(b,), dtype=np.complex128)
neg['g'] = -1
negOm = field.Field(dist=d, bases=(b,), dtype=np.complex128)
negOm['g'] = -20
ez = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
ez['g'][1] = -np.sin(theta)
ez['g'][2] =  np.cos(theta)

bk1 = basis.BallBasis(c, (16,16,16), radius=1, k=1)
bk2 = basis.BallBasis(c, (16,16,16), radius=1, k=2)

Du = field.Field(dist=d, bases=(bk1,), tensorsig=(c,c,), dtype=np.complex128)
u_rhs = field.Field(dist=d, bases=(bk2,), tensorsig=(c,), dtype=np.complex128)

# first formulation
op1a = operators.Gradient(u, c)
op1a.out = Du
op1b = neg*operators.DotProduct(u,Du)
op1c = negOm*operators.CrossProduct(ez,u)
def calculate1():
    Du.set_layout(Du.dist.coeff_layout)
    Du['c'] = 0
    op1a.evaluate()
    u_rhs.set_layout(u_rhs.dist.grid_layout)
    u_rhs['g'] = op1b.evaluate()['g']
    # R = ez cross u
    u_rhs['g'] += op1c.evaluate()['g']

# second formulation
op2a = neg*operators.DotProduct(u,operators.Gradient(u, c))
op2b = negOm*operators.CrossProduct(ez,u)
def calculate2():
    u_rhs.set_layout(u_rhs.dist.grid_layout)
    u_rhs['g'] = op2a.evaluate()['g']
    # R = ez cross u
    u_rhs['g'] += op2b.evaluate()['g']

# third formulation
op3a = neg*operators.DotProduct(u,operators.Gradient(u, c)) + negOm*operators.CrossProduct(ez,u)
def calculate3():
    u_rhs.set_layout(u_rhs.dist.grid_layout)
    u_rhs['g'] = op3a.evaluate()['g']

# fourth formulation
op4a = neg*operators.DotProduct(u,operators.Gradient(u, c)) + negOm*operators.CrossProduct(ez,u)
op4b = operators.convert(op4a,(bk2,))
def calculate4():
    u_rhs = op4b.evaluate()

iter = 15

start_time = time.time()
for i in range(iter):
    calculate1()
end_time = time.time()
print('first formulation took: %f' %(end_time-start_time))

start_time = time.time()
for i in range(iter):
    calculate2()
end_time = time.time()
print('second formulation took: %f' %(end_time-start_time))

start_time = time.time()
for i in range(iter):
    calculate3()
end_time = time.time()
print('third formulation took: %f' %(end_time-start_time))

start_time = time.time()
for i in range(iter):
    calculate4()
end_time = time.time()
print('fourth formulation took: %f' %(end_time-start_time))

