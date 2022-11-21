import numpy as np
import matplotlib.pyplot as plt

Nx = 64
h0 = 1/128
dt = 1.8*h0

x = np.linspace(0,1,Nx+1)[:-1]
dx = x[1]-x[0]
nu = 0.1
u0 = np.sin(2*np.pi*x)

Dx = np.zeros((Nx,Nx))
for i in range(Nx):
    Dx[i,np.mod(i+1,Nx)] = 1/(2*dx)
    Dx[i,np.mod(i-1,Nx)] = -1/(2*dx)

def F(u):
    return -np.sin(2*np.pi*x)*(Dx@u)

def FAdj(u):
    return Dx.T@(-np.sin(2*np.pi*x)*u)

L = np.zeros((Nx,Nx))
for i in range(Nx):
    L[i,np.mod(i+1,Nx)] = -nu/(dx**2)
    L[i,np.mod(i,Nx)]   = nu*2/(dx**2)
    L[i,np.mod(i-1,Nx)] = -nu/(dx**2)

M = np.eye(Nx,Nx)

############
## Scheme ##
############

# stages = 1
#
# c = np.array([0, 1])
#
# A = np.array([[0, 0],
#               [1, 0]])
#
# H = np.array([[0, 0],
#               [0, 1]])

stages = 4

c = np.array([0, 1/2, 2/3, 1/2, 1])

A = np.array([[  0  ,   0  ,  0 ,   0 , 0],
              [ 1/2 ,   0  ,  0 ,   0 , 0],
              [11/18,  1/18,  0 ,   0 , 0],
              [ 5/6 , -5/6 , 1/2,   0 , 0],
              [ 1/4 ,  7/4 , 3/4, -7/4, 0]])

H = np.array([[0,   0 ,   0 ,  0 ,  0 ],
              [0,  1/2,   0 ,  0 ,  0 ],
              [0,  1/6,  1/2,  0 ,  0 ],
              [0, -1/2,  1/2, 1/2,  0 ],
              [0,  3/2, -3/2, 1/2, 1/2]])
# stages = 2
#
# γ = (2 - np.sqrt(2)) / 2
# δ = 1 - 1 / γ / 2
#
# c = np.array([0, γ, 1])
#
# A = np.array([[0,  0 , 0],
#               [γ,  0 , 0],
#               [δ, 1-δ, 0]])
#
# H = np.array([[0,  0 , 0],
#               [0,  γ , 0],
#               [0, 1-γ, γ]])
# stages = 3
#
# α1, α2, α3 = (29/96, -3/40, 1/6)
# β1, β2, β3 = (37/160, 5/24, 1/6)
# γ1, γ2, γ3 = (8/15, 5/12, 3/4)
# ζ2, ζ3 = (-17/60, -5/12)
#
# c = np.array([0, 8/15, 2/3, 1])
#
# A = np.array([[    0,     0,  0, 0],
#               [   γ1,     0,  0, 0],
#               [γ1+ζ2,    γ2,  0, 0],
#               [γ1+ζ2, γ2+ζ3, γ3, 0]])
#
# H = np.array([[ 0,     0,     0,  0],
#               [α1,    β1,     0,  0],
#               [α1, β1+α2,    β2,  0],
#               [α1, β1+α2, β2+α3, β3]])
a = np.random.rand(Nx)
b = np.random.rand(Nx)

##############
## Timestep ##
##############
print('Begin direct solve')
NSteps = 100
u = a
for steps in range(NSteps):
    us = [u]
    for i in range(1,stages+1):
        LHS = M + dt*H[i,i]*L
        RHS = M@us[0]
        for j in range(i):
            RHS += dt*(A[i,j]*F(us[j])-H[i,j]*L@us[j])
        us.append(np.linalg.solve(LHS,RHS))
    u = us[-1]

La = u
#############
## Adjoint ##
#############
print('Begin adjoint solve')
# Initial condition
u = b
for steps in range(NSteps):
    for i in reversed(range(1,stages+1)):
        if(i==stages):
            LHS = (M + dt*H[i,i]*L).T
            us = [np.linalg.solve(LHS,u)]
        else:
            LHS = (M + dt*H[i,i]*L).T
            RHS = 0
            for j in range(i,stages):
                RHS += dt*(A[j+1,i]*FAdj(us[j-i])-H[j+1,i]*L.T@us[j-i])
            # us.append(np.linalg.solve(LHS,RHS))
            us.insert(0,np.linalg.solve(LHS,RHS))
    # Get initial time

    u = 0
    for j in range(1,stages+1):
        u += (M.T@us[j-1]+dt*A[j,0]*FAdj(us[j-1])-dt*H[j,0]*L.T@us[j-1])

    # for j in range(1,stages+1):
    #     u += (M.T@us[j-1]+dt*A[j,0]*FAdj(us[j-1])-dt*H[j,0]*L.T@us[j-1])
#         RHS = 0
#         for j in range(i,stages):
#             RHS = dt*(A[j,i]*F(us[j])-H[j,i]*L.T@us[j])
#         us.insert(0,np.linalg.solve(LHS,RHS))
#     uend = 0
#     for j in range(1,stages+1):
#         uend += (-M@us[j]-dt*F(us[j])+H[j,0]*L@us[j])
#     u = uend
LTb = u

norm1 = np.vdot(b,La)
norm2 = np.vdot(LTb,a)
print(norm1,norm2)
print('Error',np.linalg.norm(norm1-norm2)/np.linalg.norm(norm1))
