import numpy as np
import scipy.sparse      as sparse
import dedalus.public as de
from dedalus.core import arithmetic, timesteppers, operators, problems, solvers
from dedalus.tools.parsing import split_equation
from dedalus.extras.flow_tools import GlobalArrayReducer
from mpi4py import MPI
import time

import matplotlib
import logging
logger = logging.getLogger(__name__)

matplotlib_logger = logging.getLogger('matplotlib')
matplotlib_logger.setLevel(logging.WARNING)

comm = MPI.COMM_WORLD
rank = comm.rank

Lmax = 31
Nmax = 31

# right now can't run with dealiasing
L_dealias = 1
N_dealias = 1

# parameters
Ekman = 1e-3
Prandtl = 1
Rayleigh = 100
r_inner = 7/13
r_outer = 20/13
radii = (r_inner,r_outer)

# mesh must be 2D for plotting
mesh = [8,4]

c = de.coords.SphericalCoordinates('phi', 'theta', 'r')
d = de.distributor.Distributor((c,), mesh=mesh)
b    = de.basis.SphericalShellBasis(c, (2*(Lmax+1),Lmax+1,Nmax+1), radii=radii)
b_inner = b.S2_basis(radius=r_inner)
b_outer = b.S2_basis(radius=r_outer)
phi, theta, r = b.local_grids((1, 1, 1))
phig,thetag,rg= b.global_grids((1,1, 1))
theta_target = thetag[0,(Lmax+1)//2,0]

weight_theta = b.local_colatitude_weights(1)
weight_r = b.local_radial_weights(1)*r**2

u = de.field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
p = de.field.Field(dist=d, bases=(b,), dtype=np.complex128)
T = de.field.Field(dist=d, bases=(b,), dtype=np.complex128)
tau_u_inner = de.field.Field(dist=d, bases=(b_inner,), tensorsig=(c,), dtype=np.complex128)
tau_T_inner = de.field.Field(dist=d, bases=(b_inner,), dtype=np.complex128)
tau_u_outer = de.field.Field(dist=d, bases=(b_outer,), tensorsig=(c,), dtype=np.complex128)
tau_T_outer = de.field.Field(dist=d, bases=(b_outer,), dtype=np.complex128)

ez = de.field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
ez['g'][1] = -np.sin(theta)
ez['g'][2] =  np.cos(theta)

r_vec = de.field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
r_vec['g'][2] = r/r_outer

T_inner = de.field.Field(dist=d, bases=(b_inner,), dtype=np.complex128)
T_inner['g'] = 1.

# initial condition
A = 0.1
x = 2*r-r_inner-r_outer
T['g'] = r_inner*r_outer/r - r_inner + 210*A/np.sqrt(17920*np.pi)*(1-3*x**2+3*x**4-x**6)*np.sin(theta)**4*np.cos(4*phi)

# Parameters and operators
div = lambda A: operators.Divergence(A, index=0)
lap = lambda A: operators.Laplacian(A, c)
grad = lambda A: operators.Gradient(A, c)
dot = lambda A, B: arithmetic.DotProduct(A, B)
cross = lambda A, B: arithmetic.CrossProduct(A, B)
ddt = lambda A: operators.TimeDerivative(A)
LiftTau = lambda A, n: operators.LiftTau(A, b, n)

# Problem
def eq_eval(eq_str):
    return [eval(expr) for expr in split_equation(eq_str)]
problem = problems.IVP([u, p, T, tau_u_inner, tau_T_inner, tau_u_outer, tau_T_outer])
problem.add_equation(eq_eval("Ekman*ddt(u) - Ekman*lap(u) + grad(p) + LiftTau(tau_u_inner,-1) + LiftTau(tau_u_outer,-2) = - Ekman*dot(u,grad(u)) + Rayleigh*r_vec*T - 2*cross(ez, u)"), condition = "ntheta != 0")
problem.add_equation(eq_eval("u = 0"), condition = "ntheta == 0")
problem.add_equation(eq_eval("div(u) = 0"), condition = "ntheta != 0")
problem.add_equation(eq_eval("p = 0"), condition = "ntheta == 0")
problem.add_equation(eq_eval("ddt(T) - lap(T)/Prandtl + LiftTau(tau_T_inner,-1) + LiftTau(tau_T_outer,-2) = - dot(u,grad(T))"))
problem.add_equation(eq_eval("u(r=7/13) = 0"), condition = "ntheta != 0")
problem.add_equation(eq_eval("tau_u_inner = 0"), condition = "ntheta == 0")
problem.add_equation(eq_eval("T(r=7/13) = T_inner"))
problem.add_equation(eq_eval("u(r=20/13) = 0"), condition = "ntheta != 0")
problem.add_equation(eq_eval("tau_u_outer = 0"), condition = "ntheta == 0")
problem.add_equation(eq_eval("T(r=20/13) = 0"))
logger.info("Problem built")

# Solver
solver = solvers.InitialValueSolver(problem, timesteppers.CNAB2)

reducer = GlobalArrayReducer(d.comm_cart)

vol_test = np.sum(weight_r*weight_theta+0*p['g'])*np.pi/(Lmax+1)/L_dealias
vol_test = reducer.reduce_scalar(vol_test, MPI.SUM)
vol = 4*np.pi/3*(r_outer**3-r_inner**3)
vol_correction = vol/vol_test

t = 0.

t_list = []
E_list = []

report_cadence = 10

plot_cadence = 500
dpi = 150

plot = theta_target in theta

include_data = comm.gather(plot)

var = T['g']
name = 'T'
remove_m0 = True
if plot:
    i_theta = np.argmin(np.abs(theta[0,:,0] - theta_target))
    plot_data = var[:,i_theta,:].real
else:
    plot_data = None

plot_data = comm.gather(plot_data, root=0)

import matplotlib.pyplot as plt
def equator_plot(r, phi, data, index=None, pcm=None, cmap=None, title=None):
    if pcm is None:
        r_pad   = np.pad(r[0,0,:], ((0,1)), mode='constant', constant_values=(r_inner,r_outer))
        phi_pad = np.append(phi[:,0,0], 2*np.pi)
        fig, ax = plt.subplots(subplot_kw=dict(polar=True))
        r_plot, phi_plot = np.meshgrid(r_pad,phi_pad)
        pcm = ax.pcolormesh(phi_plot,r_plot,data, cmap=cmap)
        ax.set_rlim(bottom=0, top=r_outer)
        ax.set_rticks([])
        ax.set_aspect(1)

        pmin,pmax = pcm.get_clim()
        cNorm = matplotlib.colors.Normalize(vmin=pmin, vmax=pmax)
        ax_cb = fig.add_axes([0.8, 0.3, 0.03, 1-0.3*2])
        cb = fig.colorbar(pcm, cax=ax_cb, norm=cNorm, cmap=cmap)
        fig.subplots_adjust(left=0.05,right=0.85)
        if title is not None:
            ax_cb.set_title(title)
        pcm.ax_cb = ax_cb
        pcm.cb_cmap = cmap
        pcm.cb = cb
        return fig, pcm
    else:
        pcm.set_array(np.ravel(data))
        pcm.set_clim([np.min(data),np.max(data)])
        cNorm = matplotlib.colors.Normalize(vmin=np.min(data), vmax=np.max(data))
        pcm.cb.mappable.set_norm(cNorm)
        if title is not None:
            pcm.ax_cb.set_title(title)

if rank == 0:
    data = []
    for pd, id in zip(plot_data, include_data):
        if id: data.append(pd)
    data = np.array(data)
    data = np.transpose(data, axes=(1,0,2)).reshape((2*(Lmax+1),Nmax+1))
    if remove_m0:
        data -= np.mean(data, axis=0)
    fig, pcm = equator_plot(rg, phig, data, title=name+"'\n t = {:5.2f}".format(0), cmap = 'RdYlBu_r')
    plt.savefig('frames/%s_%04i.png' %(name, solver.iteration//plot_cadence), dpi=dpi)

# timestepping loop
start_time = time.time()

# Integration parameters
dt = 1.e-4
t_end = 10 #1.25
solver.stop_sim_time = t_end

while solver.ok:

    if solver.iteration % report_cadence == 0:
        E0 = np.sum(vol_correction*weight_r*weight_theta*u['g'].real**2)
        E0 = 0.5*E0*(np.pi)/(Lmax+1)/L_dealias/vol
        E0 = reducer.reduce_scalar(E0, MPI.SUM)
        T0 = np.sum(vol_correction*weight_r*weight_theta*T['g'].real**2)
        T0 = 0.5*T0*(np.pi)/(Lmax+1)/L_dealias/vol
        T0 = reducer.reduce_scalar(T0, MPI.SUM)
        logger.info("iter: {:d}, dt={:e}, t={:e}, E0={:e}, T0={:e}".format(solver.iteration, dt, solver.sim_time, E0, T0))
        t_list.append(solver.sim_time)
        E_list.append(E0)

    if solver.iteration % plot_cadence == 0:
        if plot:
            plot_data = var[:,i_theta,:].real

        plot_data = comm.gather(plot_data, root=0)

        if rank == 0:
            data = []
            for pd, id in zip(plot_data, include_data):
                if id: data.append(pd)
            data = np.array(data)
            data = np.transpose(data, axes=(1,0,2)).reshape((2*(Lmax+1),Nmax+1))
            if remove_m0:
                data -= np.mean(data, axis=0)
            equator_plot(rg, phig, data, title=name+"'\n t = {:5.2f}".format(solver.sim_time), cmap='RdYlBu_r', pcm=pcm)
            fig.savefig('frames/%s_%04i.png' %(name,solver.iteration//plot_cadence), dpi=dpi)

    solver.step(dt)

end_time = time.time()
if rank==0:
    print('simulation took: %f' %(end_time-start_time))
    t_list = np.array(t_list)
    E_list = np.array(E_list)
    np.savetxt('marti_conv.dat',np.array([t_list,E_list]))
