import numpy as np
import scipy.sparse      as sparse
import dedalus.public as de
from dedalus.core import arithmetic, timesteppers, problems, solvers, operators
from dedalus.tools.parsing import split_equation
from dedalus.extras.flow_tools import GlobalArrayReducer
import dedalus_sphere
from mpi4py import MPI
import time
from dedalus_sphere import ball, intertwiner
from dedalus_sphere import jacobi128 as jacobi

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
Pm = 5
r_inner = 7/13
r_outer = 20/13
radii = (r_inner,r_outer)

# ChebyshevV
alpha_BC = (2-1/2, 2-1/2)

def C(N):
    ab = alpha_BC
    cd = (b.alpha[0]+2,b.alpha[1]+2)
    return dedalus_sphere.jacobi128.coefficient_connection(N,ab,cd)

def BC_rows(N, num_comp):
    N_list = (np.arange(num_comp)+1)*(N + 1)
    return N_list


# mesh must be 2D for plotting
mesh = [1,1] #[16,16]

c = de.coords.SphericalCoordinates('phi', 'theta', 'r')
d = de.distributor.Distributor((c,), mesh=mesh)
b    = de.basis.SphericalShellBasis(c, (2*(Lmax+1),Lmax+1,Nmax+1), radii=radii)
bk2  = de.basis.SphericalShellBasis(c, (2*(Lmax+1),Lmax+1,Nmax+1), k=2, radii=radii)
b_inner = b.S2_basis(radius=r_inner)
b_outer = b.S2_basis(radius=r_outer)
phi, theta, r = b.local_grids((1, 1, 1))
phig,thetag,rg= b.global_grids((1,1, 1))
theta_target = thetag[0,(Lmax+1)//2,0]

weight_theta = b.local_colatitude_weights(1)
weight_r = b.local_radius_weights(1)*r**2

u = de.field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
A = de.field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
B = de.field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
p = de.field.Field(dist=d, bases=(b,), dtype=np.complex128)
φ = de.field.Field(dist=d, bases=(b,), dtype=np.complex128)
T = de.field.Field(dist=d, bases=(b,), dtype=np.complex128)
tau_u_inner = de.field.Field(dist=d, bases=(b_inner,), tensorsig=(c,), dtype=np.complex128)
tau_A_inner = de.field.Field(dist=d, bases=(b_inner,), tensorsig=(c,), dtype=np.complex128)
tau_T_inner = de.field.Field(dist=d, bases=(b_inner,), dtype=np.complex128)
tau_u_outer = de.field.Field(dist=d, bases=(b_outer,), tensorsig=(c,), dtype=np.complex128)
tau_A_outer = de.field.Field(dist=d, bases=(b_outer,), tensorsig=(c,), dtype=np.complex128)
tau_T_outer = de.field.Field(dist=d, bases=(b_outer,), dtype=np.complex128)

ez = de.field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
ez['g'][1] = -np.sin(theta)
ez['g'][2] =  np.cos(theta)

r_vec = de.field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
r_vec['g'][2] = r/r_outer

T_inner = de.field.Field(dist=d, bases=(b_inner,), dtype=np.complex128)
T_inner['g'] = 1.

# initial condition
amp = 0.1
x = 2*r-r_inner-r_outer
T['g'] = r_inner*r_outer/r - r_inner + 210*amp/np.sqrt(17920*np.pi)*(1-3*x**2+3*x**4-x**6)*np.sin(theta)**4*np.cos(4*phi)


B['g'][0] = 5*np.sin(np.pi*(r-r_inner))*np.sin(2*theta)
B['g'][1] = 5/8*(9*r -8*r_outer -r_inner**4/r**3)*np.sin(theta)
B['g'][2] = 5/8*(8*r_outer -6*r -2*r_inner**4/r**3)*np.cos(theta)


# Parameters and operators
div = lambda A: de.operators.Divergence(A, index=0)
lap = lambda A: de.operators.Laplacian(A, c)
grad = lambda A: de.operators.Gradient(A, c)
dot = lambda A, B: arithmetic.DotProduct(A, B)
cross = lambda A, B: de.operators.CrossProduct(A, B)
ddt = lambda A: de.operators.TimeDerivative(A)
curl = lambda A: operators.Curl(A)

ell_func = lambda ell: ell+1
A_potential_bc_outer = operators.RadialComponent(operators.interpolate(operators.Gradient(A, c), r=r_outer)) + operators.interpolate(operators.SphericalEllProduct(A, c, ell_func), r=r_outer)/r_outer
A_potential_bc_inner = operators.RadialComponent(operators.interpolate(operators.Gradient(A, c), r=r_inner)) + operators.interpolate(operators.SphericalEllProduct(A, c, ell_func), r=r_inner)/r_inner

# Problem
def eq_eval(eq_str):
    return [eval(expr) for expr in split_equation(eq_str)]

def initial_conditions(B_IC):
    # Initial condtions on A
    # BVP for initial A
    d = de.distributor.Distributor((c,), comm=MPI.COMM_SELF)
    V = de.field.Field(dist=d, bases=(b,), dtype=np.complex128)
    A = de.field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
    B = de.field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.complex128)
    tau_A_inner = de.field.Field(dist=d, bases=(b_inner,), tensorsig=(c,), dtype=np.complex128)
    tau_A_outer = de.field.Field(dist=d, bases=(b_outer,), tensorsig=(c,), dtype=np.complex128)
    A_potential_bc_outer = operators.RadialComponent(operators.interpolate(operators.Gradient(A, c), r=r_outer)) + operators.interpolate(operators.SphericalEllProduct(A, c, ell_func), r=r_outer)/r_outer
    A_potential_bc_inner = operators.RadialComponent(operators.interpolate(operators.Gradient(A, c), r=r_inner)) + operators.interpolate(operators.SphericalEllProduct(A, c, ell_func), r=r_inner)/r_inner

    B['g'] = B_IC

    BVP = problems.LBVP([A, V, tau_A_inner, tau_A_outer])

    def eq_eval(eq_str):
        return [eval(expr) for expr in split_equation(eq_str)]
    BVP.add_equation(eq_eval("curl(A) + grad(V) = B"), condition="ntheta != 0")
    BVP.add_equation(eq_eval("div(A) = 0"), condition="ntheta != 0")
    BVP.add_equation(eq_eval("A_potential_bc_inner = 0"), condition="ntheta != 0")
    BVP.add_equation(eq_eval("A_potential_bc_outer = 0"), condition="ntheta != 0")
    BVP.add_equation(eq_eval("A = 0"), condition="ntheta == 0")
    BVP.add_equation(eq_eval("V = 0"), condition="ntheta == 0")
    BVP.add_equation(eq_eval("tau_A_inner = 0"), condition="ntheta == 0")
    BVP.add_equation(eq_eval("tau_A_outer = 0"), condition="ntheta == 0")

    solver = solvers.LinearBoundaryValueSolver(BVP)

    for subproblem in solver.subproblems:
        ell = subproblem.group[1]
        L = subproblem.L_min
        shape = L.shape
        tau_columns = np.zeros((shape[0], 6))
        BCs         = np.zeros((3, shape[1]))
        N0, N1, N2, N3 = BC_rows(Nmax, 4)
        if ell != 0:
            tau_columns[N0:N1,0] = (C(Nmax))[:,-1]
            tau_columns[N1:N2,1] = (C(Nmax))[:,-1]
            tau_columns[N2:N3,2] = (C(Nmax))[:,-1]
            tau_columns[N0:N1,3] = (C(Nmax))[:,-2]
            tau_columns[N1:N2,4] = (C(Nmax))[:,-2]
            tau_columns[N2:N3,5] = (C(Nmax))[:,-2]
            subproblem.L_min[:,-6:] = tau_columns
        subproblem.L_min.eliminate_zeros()
        subproblem.expand_matrices(['L'])

    logger.info("built BVP")
    solver.solve()
    logger.info("solved BVP")
    return A['g']

slices = d.grid_layout.slices(A.domain,(1,1,1))
A_IC = initial_conditions(B['c'][:,slices[0],slices[1],slices[2]])
A['g'][:,slices[0],slices[1],slices[2]] = A_IC

problem = problems.IVP([p, u, φ, A, T, tau_u_inner, tau_A_inner, tau_T_inner, tau_u_outer, tau_A_outer, tau_T_outer])
problem.add_equation(eq_eval("div(u) = 0"), condition = "ntheta != 0")
problem.add_equation(eq_eval("Ekman*ddt(u) - Ekman*lap(u) + grad(p) = - Ekman*dot(u,grad(u)) + Rayleigh*r_vec*T - 2*cross(ez, u) + 1/Pm*cross(curl(curl(A)),curl(A))"), condition = "ntheta != 0")
problem.add_equation(eq_eval("div(A) = 0"), condition="ntheta != 0")
problem.add_equation(eq_eval("ddt(A) - 1/Pm*lap(A) + grad(φ) = cross(u, curl(A))"), condition="ntheta != 0")

problem.add_equation(eq_eval("p = 0"), condition = "ntheta == 0")
problem.add_equation(eq_eval("u = 0"), condition = "ntheta == 0")
problem.add_equation(eq_eval("φ = 0"), condition="ntheta == 0")
problem.add_equation(eq_eval("A = 0"), condition="ntheta == 0")

problem.add_equation(eq_eval("ddt(T) - lap(T)/Prandtl = - dot(u,grad(T))"))

problem.add_equation(eq_eval("u(r=7/13) = 0"), condition = "ntheta != 0")
problem.add_equation(eq_eval("tau_u_inner = 0"), condition = "ntheta == 0")
problem.add_equation(eq_eval("A_potential_bc_inner = 0"), condition = "ntheta != 0")
problem.add_equation(eq_eval("tau_A_inner = 0"), condition = "ntheta == 0")
problem.add_equation(eq_eval("T(r=7/13) = T_inner"))
problem.add_equation(eq_eval("u(r=20/13) = 0"), condition = "ntheta != 0")
problem.add_equation(eq_eval("tau_u_outer = 0"), condition = "ntheta == 0")
problem.add_equation(eq_eval("A_potential_bc_outer = 0"), condition = "ntheta != 0")
problem.add_equation(eq_eval("tau_A_outer = 0"), condition = "ntheta == 0")
problem.add_equation(eq_eval("T(r=20/13) = 0"))
logger.info("Problem built")

# Solver
solver = solvers.InitialValueSolver(problem, timesteppers.CNAB2)

# Add taus

for subproblem in solver.subproblems:
    ell = subproblem.group[1]
    M = subproblem.M_min
    L = subproblem.L_min
    shape = M.shape
    subproblem.M_min[:,-14:] = 0
    subproblem.M_min.eliminate_zeros()
    N0, N1, N2, N3, N4, N5, N6, N7, N8 = BC_rows(Nmax, 9)
    tau_columns = np.zeros((shape[0], 14))
    if ell != 0:
        # no tau column on p
        tau_columns[N0:N1,0] = (C(Nmax))[:,-1]
        tau_columns[N1:N2,1] = (C(Nmax))[:,-1]
        tau_columns[N2:N3,2] = (C(Nmax))[:,-1]
        # no tau column on φ
        tau_columns[N4:N5,3] = (C(Nmax))[:,-1]
        tau_columns[N5:N6,4] = (C(Nmax))[:,-1]
        tau_columns[N6:N7,5] = (C(Nmax))[:,-1]
        tau_columns[N7:N8,6] = (C(Nmax))[:,-1]
        # no tau column on p
        tau_columns[N0:N1,7] = (C(Nmax))[:,-2]
        tau_columns[N1:N2,8] = (C(Nmax))[:,-2]
        tau_columns[N2:N3,9] = (C(Nmax))[:,-2]
        # no tau column on φ
        tau_columns[N4:N5,10] = (C(Nmax))[:,-2]
        tau_columns[N5:N6,11] = (C(Nmax))[:,-2]
        tau_columns[N6:N7,12] = (C(Nmax))[:,-2]
        tau_columns[N7:N8,13] = (C(Nmax))[:,-2]
        subproblem.L_min[:,-14:] = tau_columns
    else: # ell = 0
        tau_columns[N7:N8,6]  = (C(Nmax))[:,-1]
        tau_columns[N7:N8,13] = (C(Nmax))[:,-2]
        subproblem.L_min[:,-8:-7] = tau_columns[:,6:7]
        subproblem.L_min[:,-1:] = tau_columns[:,13:]
    subproblem.L_min.eliminate_zeros()
    subproblem.expand_matrices(['M','L'])

reducer = GlobalArrayReducer(d.comm_cart)

vol_test = np.sum(weight_r*weight_theta+0*p['g'])*np.pi/(Lmax+1)/L_dealias
vol_test = reducer.reduce_scalar(vol_test, MPI.SUM)
vol = 4*np.pi/3*(r_outer**3-r_inner**3)
vol_correction = vol/vol_test

t = 0.

t_list = []
KE_list = []
ME_list = []

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
        KE = np.sum(vol_correction*weight_r*weight_theta*u['g'].real**2)
        KE = 0.5*KE*(np.pi)/(Lmax+1)/L_dealias/vol
        KE = reducer.reduce_scalar(KE, MPI.SUM)
        T0 = np.sum(vol_correction*weight_r*weight_theta*T['g'].real**2)
        T0 = 0.5*T0*(np.pi)/(Lmax+1)/L_dealias/vol
        T0 = reducer.reduce_scalar(T0, MPI.SUM)
        B = (operators.Curl(A)).evaluate()
        ME = np.sum(vol_correction*weight_r*weight_theta*B['g'].real**2)
        ME = 0.5*ME*(np.pi)/(Lmax+1)/L_dealias
        ME = reducer.reduce_scalar(ME, MPI.SUM)
        ME /= (Ekman*Pm)

        logger.info("iter: {:d}, dt={:e}, t={:e}, E0={:e}, ME={:e}, T0={:e}".format(solver.iteration, dt, solver.sim_time, KE, ME, T0))
        t_list.append(solver.sim_time)
        KE_list.append(KE)
        ME_list.append(ME)

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
    KE_list = np.array(KE_list)
    ME_list = np.array(ME_list)
    np.savetxt('marti_conv.dat',np.array([t_list,KE_list,ME_list]))
