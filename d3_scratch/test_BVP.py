import numpy as np
from dedalus.core import coords, distributor, basis, field, operators, problems, solvers, arithmetic
from dedalus.tools import logging
from dedalus.tools.parsing import split_equation
from dedalus.extras.flow_tools import GlobalArrayReducer
from dedalus.libraries.dedalus_sphere import jacobi
from scipy import sparse
import time
from mpi4py import MPI

import logging
logger = logging.getLogger(__name__)

# Parameters
radius = 1
Lmax = 7
L_dealias = 1
Nmax = 7
N_dealias = 1

dtype = np.complex128

# Bases
c = coords.SphericalCoordinates('phi', 'theta', 'r')
d = distributor.Distributor((c,),comm=MPI.COMM_SELF)
b = basis.BallBasis(c, (2*(Lmax+1), Lmax+1, Nmax+1), radius=radius)
b_S2 = b.S2_basis()
phi, theta, r = b.local_grids((1, 1, 1))

# Fields
A = field.Field(name="A", dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
φ = field.Field(name="φ", dist=d, bases=(b,), dtype=dtype)
B = field.Field(name="B", dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
τ_A = field.Field(name="τ_A", dist=d, bases=(b_S2,), tensorsig=(c,), dtype=dtype)
τ_φ = field.Field(name="τ_φ", dist=d, bases=(b_S2,), dtype=dtype)

# initial toroidal magnetic field
B['g'][1] = -3./2.*r*(-1+4*r**2-6*r**4+3*r**6)*(np.cos(phi)+np.sin(phi))
B['g'][0] = -3./4.*r*(-1+r**2)*np.cos(theta)* \
                 ( 3*r*(2-5*r**2+4*r**4)*np.sin(theta)
                  +2*(1-3*r**2+3*r**4)*(np.cos(phi)-np.sin(phi)))

# Potential BC on A
ell_func = lambda ell: ell + 1

# Parameters and operators
div = lambda A: operators.Divergence(A)
grad = lambda A: operators.Gradient(A, c)
curl = lambda A: operators.Curl(A)
LiftTau = lambda A: operators.LiftTau(A, b, -1)
ellp1 = lambda A: operators.SphericalEllProduct(A, c, ell_func)
radial = lambda A: operators.RadialComponent(A)
angular = lambda A: operators.AngularComponent(A, index=1)

# BVP for initial A
BVP = problems.LBVP([φ, A, τ_A, τ_φ])
#BVP.add_equation((angular(τ_A),0))
BVP.add_equation((div(A) + LiftTau(τ_φ), 0))
BVP.add_equation((curl(A) + grad(φ) + LiftTau(τ_A), B))
BVP.add_equation((radial(grad(A)(r=radius))+ellp1(A)(r=radius)/radius, 0) )#, condition = "ntheta != 0")
BVP.add_equation((φ(r=radius), 0) )#, condition = "ntheta == 0")
solver = solvers.LinearBoundaryValueSolver(BVP)
solver.solve()

plot_matrices = False

if plot_matrices:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure()
    I = 2
    J = 1
    for i, sp in enumerate(solver.subproblems[:I]):
        for j, mat in enumerate(['L_min']):
            axes = plt.subplot(I,J,i*J+j+1)
            A = getattr(sp, mat)
            A = sp.left_perm.T @ A
            im = axes.pcolor(np.log10(np.abs(A.A[::-1])))
            axes.set_title('sp %i, %s' %(i, mat))
            axes.set_aspect('equal')
            plt.colorbar(im)
    plt.tight_layout()
    plt.savefig("nmh_matrices.pdf")

def err(a,b):
    logger.info(np.max(np.abs(a-b))/np.max(np.abs(b)))

B2 = (curl(A) + grad(φ)).evaluate()

logger.info('magnetic field error:')
err(B2['g'],B['g'])

c_par = coords.SphericalCoordinates('phi', 'theta', 'r')
d_par = distributor.Distributor((c_par,))
b_par = basis.BallBasis(c_par, (2*(Lmax+1), Lmax+1, Nmax+1), radius=radius)
phi, theta, r = b_par.local_grids((1, 1, 1))

A_par = field.Field(dist=d_par, bases=(b_par,), tensorsig=(c_par,), dtype=dtype)
slices = d_par.grid_layout.slices(A_par.domain,(1,1,1))
A_par['g'][:] = A['g'][:,slices[0],slices[1],slices[2]]

A_analytic_2 = (3/2*r**2*(1-4*r**2+6*r**4-3*r**6)
                   *np.sin(theta)*(np.sin(phi)-np.cos(phi))
               +3/8*r**3*(2-7*r**2+9*r**4-4*r**6)
                   *(3*np.cos(theta)**2-1)
               +9/160*r**2*(-200/21*r+980/27*r**3-540/11*r**5+880/39*r**7)
                     *(3*np.cos(theta)**2-1)
               +9/80*r*(1-100/21*r**2+245/27*r**4-90/11*r**6+110/39*r**8)
                    *(3*np.cos(theta)**2-1)
               +1/8*r*(-48/5*r+288/7*r**3-64*r**5+360/11*r**7)
                   *np.sin(theta)*(np.sin(phi)-np.cos(phi))
               +1/8*(1-24/5*r**2+72/7*r**4-32/3*r**6+45/11*r**8)
                   *np.sin(theta)*(np.sin(phi)-np.cos(phi)))
A_analytic_1 = (-27/80*r*(1-100/21*r**2+245/27*r**4-90/11*r**6+110/39*r**8)
                        *np.cos(theta)*np.sin(theta)
                +1/8*(1-24/5*r**2+72/7*r**4-32/3*r**6+45/11*r**8)
                    *np.cos(theta)*(np.sin(phi)-np.cos(phi)))
A_analytic_0 = (1/8*(1-24/5*r**2+72/7*r**4-32/3*r**6+45/11*r**8)
                   *(np.cos(phi)+np.sin(phi)))

logger.info('errors in components of A (r, theta, phi):')
err(A_par['g'][2],A_analytic_2)
err(A_par['g'][1],A_analytic_1)
err(A_par['g'][0],A_analytic_0)
