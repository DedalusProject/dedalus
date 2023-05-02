import numpy as np
import matplotlib.pyplot as plt
from dedalus.core import timesteppers

test_timesteppers = [timesteppers.RK111,timesteppers.RK222,timesteppers.RK443,timesteppers.RKSMR,timesteppers.RKGFY]

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

for timestepper in test_timesteppers:
    A = timestepper.A
    stages = timestepper.stages
    H = timestepper.H
    name = timestepper.__name__
    print('Timestepper:', name)
    a = np.random.rand(Nx)
    b = np.random.rand(Nx)

    ##############
    ## Timestep ##
    ##############
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
    # Initial condition
    u = b
    # for steps in range(NSteps):
    #     for i in reversed(range(1,stages+1)):
    #         if(i==stages):
    #             LHS = (M + dt*H[i,i]*L).T
    #             us = [np.linalg.solve(LHS,u)]
    #         else:
    #             LHS = (M + dt*H[i,i]*L).T
    #             RHS = 0
    #             for j in range(i,stages):
    #                 RHS += dt*(A[j+1,i]*FAdj(us[j-i])-H[j+1,i]*L.T@us[j-i])
    #             us.insert(0,np.linalg.solve(LHS,RHS))
    #     # Get initial time
    #     u = 0
    #     for j in range(1,stages+1):
    #         u += (M.T@us[j-1]+dt*A[j,0]*FAdj(us[j-1])-dt*H[j,0]*L.T@us[j-1])

    LX = [None]*stages
    MX = [None]*stages
    FX = [None]*stages

    for steps in range(NSteps):
        for i in reversed(range(1,stages+1)):
            if(i==stages):
                RHS = u.copy()
            else:
                RHS = 0
            for j in range(i+1,stages+1):
                RHS += dt*(A[j,i]*FX[j-1]-H[j,i]*LX[j-1])
            LHS = (M + dt*H[i,i]*L).T

            u = np.linalg.solve(LHS,RHS)
            LX[i-1] = L.T@u
            FX[i-1] = FAdj(u)
            MX[i-1] = M.T@u 
    # Get initial time
        u = 0
        for j in range(1,stages+1):
            u += (MX[j-1] + dt*A[j,0]*FX[j-1] - dt*H[j,0]*LX[j-1])

    LTb = u

    ##################
    ## Adjoint test ##
    ##################

    norm1 = np.vdot(b,La)
    norm2 = np.vdot(LTb,a)
    print('Relative adjoint error',np.linalg.norm(norm1-norm2)/np.linalg.norm(norm1))
    print()
