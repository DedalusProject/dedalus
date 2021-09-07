
import numpy     as np
from scipy.sparse import linalg as spla
from . import sphere_wrapper as sph
from . import ball128 as ball
from dedalus.tools.cache import CachedMethod
from dedalus.tools.array import reshape_vector
from . import clenshaw
from . import jacobi128 as jacobi
import time

from dedalus.tools.config import config
STORE_LU_TRANSFORM = config['transforms'].getboolean('STORE_LU_TRANSFORM')

import logging
logger = logging.getLogger(__name__)

barrier = False
timing = False

class Ball:
    def __init__(self,N_max,L_max,R_max=0,a=0,N_r=None,N_theta=None,ell_min=None,ell_max=None,m_min=None,m_max=None):
        self.N_max, self.L_max, self.R_max  = N_max, L_max, R_max
        if N_r == None: self.N_r = self.N_max + 1
        else: self.N_r = N_r
        self.a = a

        if ell_min == None: ell_min =  0
        if ell_max == None: ell_max =  L_max
        if   m_min == None:   m_min =  0
        if   m_max == None:   m_max =  L_max

        self.ell_min, self.ell_max = ell_min, ell_max
        self.m_min,   self.m_max   = m_min, m_max

        # Spherical Harmonic Transforms
        self.S = sph.Sphere(self.L_max,S_max=self.R_max,N_theta=N_theta,m_min=m_min,m_max=m_max)

        self.theta     = self.S.grid
        self.cos_theta = self.S.cos_grid
        self.sin_theta = self.S.sin_grid

        # grid and weights for the radial transforms
        z_projection, weights_projection = ball.quadrature(self.N_r-1,niter=3,a=a,report_error=False)

        # grid and weights for radial integral using volume measure

        z0, weights0 = ball.quadrature(self.N_r-1,a=0.0)

        Q0           = ball.polynomial(self.N_r-1,0,0,z0,a=a)
        Q_projection = ball.polynomial(self.N_r-1,0,0,z_projection,a=a)

        self.dV = ((Q0.dot(weights0)).T).dot(weights_projection*Q_projection)

        self.pushW, self.pullW = {}, {}
        self.Q = {}

        for ell in range( max(ell_min-R_max,0) , ell_max+R_max+1):
            W = ball.polynomial(self.N_max+self.R_max-self.N_min(ell),0,ell,z_projection,a=a)
            self.pushW[(ell)] = (weights_projection*W).astype(np.float64)
            self.pullW[(ell)] = (W.T).astype(np.float64)

        for ell in range( ell_min, ell_max+1):
            self.Q[(ell,0)] = np.array([[1]])
            for deg in range(1,R_max+1):
                self.Q[(ell,deg)] = ball.recurseQ(self.Q[(ell,deg-1)],ell,deg)

        # downcast to double precision
        self.radius = np.sqrt( (z_projection+1)/2 ).astype(np.float64)
        self.dV = self.dV.astype(np.float64)

        self.LU_grad_initialized = []
        self.LU_grad = []
        self.LU_curl_initialized = []
        self.LU_curl = []

        for ell in range( ell_min, ell_max+1):
            # this hard codes an assumption that there are two ranks (e.g., 0 and 1)
            self.LU_grad_initialized.append([False]*2)
            self.LU_grad.append([None]*2)
            self.LU_curl_initialized.append([False]*2)
            self.LU_curl.append([None]*2)

        if timing:
            self.radial_transform_time = 0.
            self.angular_transform_time = 0.
            self.transpose_time = 0

    @CachedMethod
    def grid(self, axis, dimensions=2):
        if axis == 0 and dimensions == 2: grid = self.theta
        if axis == 1 and dimensions == 2: grid = self.radius
        if axis == 1 and dimensions == 3: grid = self.theta
        if axis == 2 and dimensions == 3: grid = self.radius
        return reshape_vector(grid, dimensions, axis)

    @CachedMethod
    def weight(self,axis,dimensions=2):
        if axis == 0 and dimensions == 2: weight = self.S.weights
        if axis == 1 and dimensions == 2: weight = self.dV
        if axis == 1 and dimensions == 3: weight = self.S.weights
        if axis == 2 and dimensions == 3: weight = self.dV
        return reshape_vector(weight, dimensions, axis)

    @CachedMethod
    def op(self,op_name,N,k,ell,dtype=np.float64,a=None):
        if a == None: a = self.a
        return ball.operator(op_name,N,k,ell,a=a).astype(dtype)

    def xi(self,mu,ell):
        # returns xi for ell > 0 or ell = 0 and mu = +1
        # otherwise returns 0.
        if (ell > 0) or (ell == 0 and mu == 1):
            return ball.xi(mu,ell)
        return 0.

    @CachedMethod
    def unitary3D(self,rank=1,adjoint=False):
        return ball.unitary3D(rank=rank,adjoint=adjoint)

    @CachedMethod
    def spins(self,rank):
        return ball.spins(rank)

    def forward_angle(self,m,rank,data_in,data_out):

        if timing: start_time = time.time()

        if rank == 0:
            data_out[0,int(self.S.L_min(m,0)):] = self.S.forward_spin(m,0,data_in[0])
            return

        spins, unitary = self.spins(rank), self.unitary3D(rank=rank,adjoint=True)

        data_in = np.einsum("ij,j...->i...",unitary,data_in)

        # This may benefit from some cython. Maybe, maybe not. Keaton?
        for i in range(3**rank):
            data_out[i,int(self.S.L_min(m,spins[i])):] = self.S.forward_spin(m,spins[i],data_in[i])

        if timing:
            end_time = time.time()
            self.angular_transform_time += (end_time - start_time)

    def backward_angle(self,m,rank,data_in,data_out):

        if timing: start_time = time.time()

        if rank == 0:
            data_out[0] = self.S.backward_spin(m,0,data_in[0,self.S.L_min(m,0):])
            return

        spins = self.spins(rank)

        # This may benefit from some cython. Maybe, maybe not. Keaton?
        for i in range(3**rank):
            data_out[i] = self.S.backward_spin(m,spins[i],data_in[i,int(self.S.L_min(m,spins[i])):])

        if timing:
            end_time = time.time()
            self.angular_transform_time += (end_time - start_time)

    @CachedMethod
    def N_min(self,ell):
        return ball.N_min(ell)

    # data is coeff representation of the ncc
    def ncc_matrix(self, N, k, ell, deg_in, deg_out, data, cutoff=1e-6, name=""):
        q_in = self.a
        m_in = deg_in  + 1/2
        q_out = k + self.a
        m_out= ell + deg_out + 1/2
        n_terms, max_term, matrix = clenshaw.ncc_matrix(N, q_in, m_in, q_out, m_out, data, cutoff=cutoff)
        matrix /=  (0.5)**(3/4)
        logger.debug("Expanded NCC {:s} to mode {:d} with {:d} terms.".format(name, max_term, n_terms))
        return matrix

    def forward_component(self,ell,deg,data):
        # grid --> coefficients
        N = self.N_max - self.N_min(ell-self.R_max) + 1
        if ell+deg >= 0: return (self.pushW[(ell+deg)][:N,:]).dot(data)
        else:
            shape = np.array(data.shape)
            shape[0] = N
            return np.zeros(shape)

    def backward_component(self,ell,deg,data):
        # coefficients --> grid
        N = self.N_max - self.N_min(ell-self.R_max) + 1
        if ell+deg >= 0:
            return self.pullW[(ell+deg)][:,:N].dot(data)
        else:
            shape = np.array(data.shape)
            shape[0] = self.N_r
            return np.zeros(shape)

    def radial_forward(self,ell,rank,data_in,data_out):

        if timing: start_time = time.time()

        if rank == 0:
            np.copyto(data_out,self.forward_component(ell,0,data_in[0]))
            return

        degs = self.spins(rank)
        N = self.N_max - self.N_min(ell-self.R_max) + 1

        data_in = np.einsum("ij,j...->i...",self.Q[(ell,rank)].T,data_in) # note transpose

        for i in range(3**rank): # Geoff thinks python can make this faster (??) "VECTORS MAN!" -- Geoff, 2017
            data_out[i*N:(i+1)*N] = self.forward_component(ell,degs[i],data_in[i])
        if timing:
            end_time = time.time()
            self.radial_transform_time += (end_time - start_time)

    def radial_backward(self,ell,rank,data_in,data_out):

        if timing: start_time = time.time()

        if rank == 0:
            data_out[0] = self.backward_component(ell,0,data_in)
            return

        degs = self.spins(rank)
        N =self.N_max - self.N_min(ell-self.R_max) + 1

        for i in range(3**rank):
            data_out[i] = self.backward_component(ell,degs[i],data_in[i*N:(i+1)*N])

        if timing:
            end_time = time.time()
            self.radial_transform_time += (end_time - start_time)

    def unpack(self,ell,rank,data_in):
        N = self.N_max + 1 - self.N_min(ell-self.R_max)
        data_out = []
        for i in range(3**rank):
            data_out.append(data_in[i*N:(i+1)*N])
        return data_out

    def pack(self,ell,rank,data_in,data_out):
        N = self.N_max + 1 - self.N_min(ell-self.R_max)
        for i in range(3**rank):
            data_out[i*N:(i+1)*N] = data_in[i]

    def rank(self,length):
        if length == 1:
            return 0
        else:
            return 1 + self.rank(length//3)

    def grad(self,ell,rank,data_in):
        shape = np.array(data_in.shape)
        shape[0] *= 3
        data_dtype = data_in.dtype
        data_out = np.zeros(shape,dtype=data_in.dtype)
        if STORE_LU_TRANSFORM:
            i_LU = ell-self.ell_min
            if not self.LU_grad_initialized[i_LU][rank]:
                logger.debug("LU_grad not initialized l={},rank={}".format(ell, rank))
                self.LU_grad[i_LU][rank] = [None]*(4*(3**rank))

        for i in range(3**rank):
            tau_bar = ball.bar(i,rank)
            N = self.N_max - self.N_min(ell-self.R_max)

            if ell+tau_bar >= 1:
                Cm  = self.op('E',N,0,ell+tau_bar-1,data_dtype)
                Dm  = self.op('D-',N,0,ell+tau_bar,data_dtype)
                xim = self.xi(-1,ell+tau_bar)
                index = i
                if STORE_LU_TRANSFORM:
                    if not self.LU_grad_initialized[i_LU][rank]:
                        self.LU_grad[i_LU][rank][index] = spla.splu(Cm)
                    data_out[i*(N+1):(i+1)*(N+1)] = self.LU_grad[i_LU][rank][index].solve(xim*Dm.dot(data_in[i*(N+1):(i+1)*(N+1)]))
                else:
                    data_out[i*(N+1):(i+1)*(N+1)] = spla.spsolve(Cm,xim*Dm.dot(data_in[i*(N+1):(i+1)*(N+1)]))

            if ell+tau_bar >= 0:
                Cp  = self.op('E',N,0,ell+tau_bar+1,data_dtype)
                Dp  = self.op('D+',N,0,ell+tau_bar,data_dtype)
                xip = self.xi(+1,ell+tau_bar)
                index = i+2*(3**rank)
                if STORE_LU_TRANSFORM:
                    if not self.LU_grad_initialized[i_LU][rank]:
                        self.LU_grad[i_LU][rank][index] = spla.splu(Cp)
                    data_out[index*(N+1):(index+1)*(N+1)] = self.LU_grad[i_LU][rank][index].solve(xip*Dp.dot(data_in[i*(N+1):(i+1)*(N+1)]))
                else:
                    data_out[index*(N+1):(index+1)*(N+1)] = spla.spsolve(Cp,xip*Dp.dot(data_in[i*(N+1):(i+1)*(N+1)]))

        if STORE_LU_TRANSFORM and not self.LU_grad_initialized[i_LU][rank]:
            self.LU_grad_initialized[i_LU][rank] = True

        return data_out

    def curl(self,ell,rank,data_in,data_out):

        data_dtype = data_in.dtype
        if STORE_LU_TRANSFORM:
            i_LU = ell-self.ell_min
            if not self.LU_curl_initialized[i_LU][rank]:
                logger.debug("LU_curl not initialized l={},rank={}".format(ell, rank))
                self.LU_curl[i_LU][rank] = [None]*(3)

        N = self.N_max - self.N_min(ell-self.R_max)
        xim = self.xi(-1,ell)
        xip = self.xi(+1,ell)
        if ell >= 1:
            Cm = self.op('E',N,0,ell-1,data_dtype)
            Dm = self.op('D-',N,0,ell,data_dtype)
            if STORE_LU_TRANSFORM:
                index = 0
                if not self.LU_curl_initialized[ell][rank]:
                    self.LU_curl[i_LU][rank][index] = spla.splu(Cm)
                data_out[:N+1] = self.LU_curl[i_LU][rank][index].solve(-1j*xip*Dm.dot(data_in[(N+1):2*(N+1)]))
            else:
                data_out[:N+1] = spla.spsolve(Cm,-1j*xip*Dm.dot(data_in[(N+1):2*(N+1)]))
        else:
            data_out[:N+1] = 0.


        C0 = self.op('E',N,0,ell,data_dtype)
        Dm = self.op('D-',N,0,ell+1,data_dtype)
        if STORE_LU_TRANSFORM:
            index = 1
            if not self.LU_curl_initialized[ell][rank]:
                self.LU_curl[i_LU][rank][index] = spla.splu(C0)

        if ell >= 1:
            Dp = self.op('D+',N,0,ell-1,data_dtype)
            if STORE_LU_TRANSFORM:
                data_out[(N+1):2*(N+1)] = self.LU_curl[i_LU][rank][index].solve( 1j*xim*Dm.dot(data_in[2*(N+1):])
                                                                                -1j*xip*Dp.dot(data_in[:(N+1)]))
            else:
                data_out[(N+1):2*(N+1)] = spla.spsolve(C0, 1j*xim*Dm.dot(data_in[2*(N+1):])
                                                          -1j*xip*Dp.dot(data_in[:(N+1)]))
        else:
            if STORE_LU_TRANSFORM:
                data_out[(N+1):2*(N+1)] = self.LU_curl[i_LU][rank][index].solve(1j*xim*Dm.dot(data_in[2*(N+1):]))
            else:
                data_out[(N+1):2*(N+1)] = spla.spsolve(C0,1j*xim*Dm.dot(data_in[2*(N+1):]))

        Cp = self.op('E',N,0,ell+1,data_dtype)
        Dp = self.op('D+',N,0,ell,data_dtype)
        if STORE_LU_TRANSFORM:
            index = 2
            if not self.LU_curl_initialized[ell][rank]:
                self.LU_curl[i_LU][rank][index] = spla.splu(Cp)
        if STORE_LU_TRANSFORM:
            data_out[2*(N+1):] = self.LU_curl[i_LU][rank][index].solve(1j*xim*Dp.dot(data_in[(N+1):2*(N+1)]))
        else:
            data_out[2*(N+1):] = spla.spsolve(Cp,1j*xim*Dp.dot(data_in[(N+1):2*(N+1)]))

        if STORE_LU_TRANSFORM and not self.LU_curl_initialized[i_LU][rank]:
            self.LU_curl_initialized[i_LU][rank] = True

    def div(self,data_in):

        rank = self.rank(len(data_in))
        data_dtype = data_in[0].dtype

        data_out = [None]*(3**(rank-1))

        for i in range(3**(rank-1)):
            tau_bar   = ball.bar(i,rank-1)
            m_tau_bar = -1 + tau_bar
            p_tau_bar =  1 + tau_bar
            # initialise arrays
            data_out[i] = []
            for ell in range(self.L_max+1):

                N = self.N_max - self.N_min(ell-self.R_max)

                if ell+tau_bar == 0:
                    C   = self.op('E',N,0,ell+tau_bar,data_dtype)
                    Dm  = self.op('D-',N,0,ell+p_tau_bar,data_dtype)
                    xip = self.xi(+1,ell+tau_bar)

                    data_out[i].append(spla.spsolve(C,xip*Dm.dot(data_in[i+2*(3**(rank-1))][ell])))

                elif ell+tau_bar > 0:
                    C   = self.op('E',N,0,ell+tau_bar,data_dtype)
                    Dm  = self.op('D-',N,0,ell+p_tau_bar,data_dtype)
                    Dp  = self.op('D+',N,0,ell+m_tau_bar,data_dtype)
                    xim, xip = self.xi([-1,+1],ell+tau_bar)

                    data_out[i].append(spla.spsolve(C,xip*Dm.dot(data_in[i+2*(3**(rank-1))][ell])+xim*Dp.dot(data_in[i][ell])))

                else:
                    data_out[i].append(0*data_in[i][ell])

        return data_out

    def div_grad(self,data_in,ell_start=0,ell_end=None):
        if ell_end == None: ell_end = self.L_max

        rank = self.rank(len(data_in))

        data_out = [None]*(3**rank)

        for i in range(3**rank):
            tau_bar   = ball.bar(i,rank)
            # initialise arrays
            data_out[i] = []
            for ell in range(ell_start,ell_end+1):
                ell_local = ell - ell_start

                N = self.N_max - self.N_min(ell-self.R_max)

                if ell+tau_bar >= 0:
                    CC  = self.op('E',N,1,ell+tau_bar).dot(self.op('E',N,0,ell+tau_bar))
                    DD  = self.op('D-',N,1,ell+tau_bar+1).dot(self.op('D+',N,0,ell+tau_bar))
                    Lap = spla.spsolve(CC,DD.dot(data_in[i][ell_local]))
                    data_out[i].append(Lap)
                else:
                    data_out[i].append(0*data_in[i][ell_local])

        return data_out

    def cross_grid(self,a,b):
        return np.array([a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]])

    def dot_grid(self,a,b):
        return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

class TensorField:

    def __init__(self,rank,B,domain):
        self.domain = domain
        self.B = B
        self.rank = rank

        self.ell_min, self.ell_max = B.ell_min, B.ell_max

    def __getitem__(self, layout):
        """Return data viewed in specified layout."""

        self.require_layout(layout)
        return self.data

    def __setitem__(self, layout, data):
        """Set data viewed in specified layout."""

        self.layout = layout
        np.copyto(self.data, data)

    @property
    def layout(self):
        return self._layout

    @layout.setter
    def layout(self, layout):
        self._layout = layout
        if self._layout == 'g':
            self.data = self.grid_data
        elif self._layout == 'c':
            self.data = self.coeff_data

    def require_layout(self, layout):

        if layout == 'g' and self._layout == 'c':
            self.require_grid_space()
        elif layout == 'c' and self._layout == 'g':
            self.require_coeff_space()

class TensorField_2D(TensorField):

    def __init__(self,rank,m,B,domain):

        TensorField.__init__(self,rank,B,domain)
        self.m = m

        mesh = self.domain.distributor.mesh
        if len(mesh) == 0: #serial
            self.ell_r_layout = domain.distributor.layouts[1]
            self.r_ell_layout = domain.distributor.layouts[1]
        else:
            self.ell_r_layout = domain.distributor.layouts[2]
            self.r_ell_layout = domain.distributor.layouts[1]

        local_grid_shape = self.ell_r_layout.local_shape(scales=1)
        local_grid_shape = (int(domain.dealias[0]*local_grid_shape[0]),
                            int(domain.dealias[1]*local_grid_shape[1]))
        local_ellr_shape = self.ell_r_layout.local_shape(scales=domain.dealias)
        local_rell_shape = self.r_ell_layout.local_shape(scales=domain.dealias)
        grid_shape = np.append(3**rank,np.array(local_grid_shape))
        ellr_shape = np.append(3**rank,np.array(local_ellr_shape))
        rell_shape = np.append(3**rank,np.array(local_rell_shape))

        self.grid_data = np.zeros(grid_shape,dtype=np.complex128)
        self.ellr_data = np.zeros(ellr_shape,dtype=np.complex128)
        self.rell_data = np.zeros(rell_shape,dtype=np.complex128)
        self.fields = domain.new_fields(3**rank)
        for field in self.fields: field.preset_scales(domain.dealias)
        self.coeff_data = []
        for ell in range(self.ell_min,self.ell_max+1):
            N = B.N_max - B.N_min(ell-B.R_max) + 1
            self.coeff_data.append(np.zeros(N*(3**rank),dtype=np.complex128))

        self._layout = 'g'
        self.data = self.grid_data

    def require_coeff_space(self):
        """Transform from grid space to coeff space"""

        rank = self.rank
        self.B.forward_angle(self.m,rank,self.grid_data,self.ellr_data)

        for i, field in enumerate(self.fields):
            field.layout = self.ell_r_layout
            field.data = self.ellr_data[i]
            field.require_layout(self.r_ell_layout)

        for ell in range(self.ell_min, self.ell_max + 1):
            ell_local = ell - self.ell_min
            self.B.radial_forward(ell,rank,
                                  [self.fields[i].data[ell_local] for i in range(3**rank)],
                                  self.coeff_data[ell_local])

        self.data = self.coeff_data
        self._layout = 'c'

    def require_grid_space(self):
        """Transform from coeff space to grid space"""

        rank = self.rank
        for ell in range(self.ell_min, self.ell_max + 1):
            ell_local = ell - self.ell_min
            self.B.radial_backward(ell,rank,self.coeff_data[ell_local],self.rell_data[:,ell_local,:])
            self.rell_data[:,ell_local,:] = np.einsum("ij,j...->i...",self.B.Q[(ell,rank)],self.rell_data[:,ell_local,:])

        for i, field in enumerate(self.fields):
            field.layout = self.r_ell_layout
            field.data = self.rell_data[i]
            field.require_layout(self.ell_r_layout)

        self.B.backward_angle(self.m,rank,np.array([self.fields[i].data for i in range(3**rank)]),self.grid_data)
        if rank > 0:
            self.grid_data = np.einsum("ij,j...->i...",self.B.unitary3D(rank=rank,adjoint=False),self.grid_data)

        self.data = self.grid_data
        self._layout = 'g'


class TensorField_3D(TensorField):

    def __init__(self,rank,B,domain):

        TensorField.__init__(self,rank,B,domain)

        mesh = self.domain.distributor.mesh

        if len(mesh) == 0: # serial
            self.phi_layout   = domain.distributor.layouts[3]
            self.th_m_layout  = domain.distributor.layouts[2]
            self.ell_r_layout = domain.distributor.layouts[1]
            self.r_ell_layout = domain.distributor.layouts[1]
        elif len(mesh) == 1: # 1D domain decomposition
            self.phi_layout   = domain.distributor.layouts[4]
            self.th_m_layout  = domain.distributor.layouts[2]
            self.ell_r_layout = domain.distributor.layouts[1]
            self.r_ell_layout = domain.distributor.layouts[1]
        elif len(mesh) == 2: # 2D domain decomposition
            self.phi_layout   = domain.distributor.layouts[5]
            self.th_m_layout  = domain.distributor.layouts[3]
            self.ell_r_layout = domain.distributor.layouts[2]
            self.r_ell_layout = domain.distributor.layouts[1]

        # allocating arrays
        local_grid_shape = self.phi_layout.local_shape(scales=domain.dealias)
        grid_shape = np.append(3**rank,np.array(local_grid_shape))
        self.grid_data = np.zeros(grid_shape,dtype=np.float64)

        scales = (1,1,domain.dealias[2])
        local_ellr_shape = self.ell_r_layout.local_shape(scales=scales)
        ellr_shape = np.append(3**rank,np.array(local_ellr_shape))
        self.mlr_ell_data = np.zeros(ellr_shape,dtype=np.complex128)

        local_rell_shape = self.r_ell_layout.local_shape(scales=scales)
        rell_shape = np.append(3**rank,np.array(local_rell_shape))
        self.mlr_r_data   = np.zeros(rell_shape,dtype=np.complex128)
        rlm_shape  = np.append(3**rank,np.array(local_rell_shape)[::-1])
        self.rlm_data     = np.zeros(rlm_shape,dtype=np.complex128)

        scales = (1,domain.dealias[1],self.domain.dealias[2])
        local_mthr_shape = self.th_m_layout.local_shape(scales=scales)
        mthr_shape = np.append(3**rank,np.array(local_mthr_shape))
        self.mthr_data    = np.zeros(mthr_shape,dtype=np.complex128)

        self.fields = domain.new_fields(3**rank)
        for field in self.fields: field.preset_scales(domain.dealias)

        m_size = B.m_max - B.m_min + 1
        self.coeff_data = []
        for ell in range(self.ell_min,self.ell_max+1):
            N = B.N_max - B.N_min(ell-B.R_max) + 1
            self.coeff_data.append(np.zeros( (N*(3**rank),m_size) ,dtype=np.complex128))

        self._layout = 'g'
        self.data = self.grid_data

    def require_coeff_space(self):
        """Transform from grid space to coeff space"""

        while self._layout != 'c':
            self.decrement_layout()

    def decrement_layout(self):

        rank = self.rank
        B = self.B

        if self._layout == 'g':
            if barrier: self.domain.dist.comm_cart.Barrier()
            if timing: start_time = time.time()
            for i, field in enumerate(self.fields):
                field.layout = self.phi_layout
                np.copyto(field.data,self.data[i])
                field.require_layout(self.th_m_layout)
                np.copyto(self.mthr_data[i],field.data)
            self._layout = 3
            if timing:
                end_time = time.time()
                self.B.transpose_time += end_time-start_time
            if barrier: self.domain.dist.comm_cart.Barrier()
        elif self._layout == 3:
            if barrier: self.domain.dist.comm_cart.Barrier()
            for m in range(B.m_min,B.m_max+1):
                m_local = m - B.m_min
                B.forward_angle(m,rank,np.array(self.mthr_data[:,m_local]),self.mlr_ell_data[:,m_local,:,:])
            self._layout = 2
            if barrier: self.domain.dist.comm_cart.Barrier()
        elif self._layout == 2:
            if barrier: self.domain.dist.comm_cart.Barrier()
            if timing: start_time = time.time()
            if self.ell_r_layout != self.r_ell_layout:
                for i, field in enumerate(self.fields):
                    field.layout = self.ell_r_layout
                    np.copyto(field.data,self.mlr_ell_data[i])
                    self.domain.distributor.paths[1].decrement([field])
                    np.copyto(self.rlm_data[i],field.data.T)
            else:
                for i, field in enumerate(self.fields):
                    np.copyto(self.rlm_data[i],self.mlr_ell_data[i].T)
            self._layout = 1
            if timing:
                end_time = time.time()
                self.B.transpose_time += end_time-start_time
            if barrier: self.domain.dist.comm_cart.Barrier()
        elif self._layout == 1:
            if barrier: self.domain.dist.comm_cart.Barrier()
            for ell in range(B.ell_min, B.ell_max + 1):
                ell_local = ell - B.ell_min
                self.B.radial_forward(ell,rank,self.rlm_data[:,:,ell_local,:],self.coeff_data[ell_local])
            self.data = self.coeff_data
            self._layout = 'c'
            if barrier: self.domain.dist.comm_cart.Barrier()

    def require_grid_space(self):
        """Transform from coeff space to grid space"""

        while self._layout != 'g':
            self.increment_layout()

    def increment_layout(self):

        rank = self.rank
        B = self.B

        if self._layout == 'c':
            if barrier: self.domain.dist.comm_cart.Barrier()
            for ell in range(B.ell_min, B.ell_max+1):
                ell_local = ell - B.ell_min
                B.radial_backward(ell,rank,self.data[ell_local],self.rlm_data[:,:,ell_local,:])
                self.rlm_data[:,:,ell_local,:] = np.einsum("ij,j...->i...",self.B.Q[(ell,rank)],self.rlm_data[:,:,ell_local,:])
            np.copyto(self.mlr_r_data,self.rlm_data.transpose(0,3,2,1),casting='no')
            self._layout = 1
            if barrier: self.domain.dist.comm_cart.Barrier()
        elif self._layout == 1:
            if barrier: self.domain.dist.comm_cart.Barrier()
            if timing: start_time = time.time()
            if self.ell_r_layout != self.r_ell_layout:
                for i, field in enumerate(self.fields):
                    field.layout = self.r_ell_layout
                    np.copyto(field.data,self.mlr_r_data[i])
                    self.domain.distributor.paths[1].increment([field])
                    np.copyto(self.mlr_ell_data[i],field.data)
            else:
                for i, field in enumerate(self.fields):
                    np.copyto(self.mlr_ell_data[i],self.mlr_r_data[i])
            self._layout = 2
            if timing:
                end_time = time.time()
                self.B.transpose_time += end_time-start_time
            if barrier: self.domain.dist.comm_cart.Barrier()
        elif self._layout == 2:
            if barrier: self.domain.dist.comm_cart.Barrier()
            for m in range(B.m_min, B.m_max + 1):
                m_local = m - B.m_min
                B.backward_angle(m,rank,self.mlr_ell_data[:,m_local,:,:],self.mthr_data[:,m_local,:,:])
            if rank > 0:
                self.mthr_data = np.einsum("ij,j...->i...",self.B.unitary3D(rank=rank,adjoint=False),self.mthr_data)
            self._layout = 3
            if barrier: self.domain.dist.comm_cart.Barrier()
        elif self._layout == 3:
            if barrier: self.domain.dist.comm_cart.Barrier()
            if timing: start_time = time.time()
            for i, field in enumerate(self.fields):
                field.layout = self.th_m_layout
                np.copyto(field.data,self.mthr_data[i])
                field.require_layout(self.phi_layout)
                np.copyto(self.grid_data[i],field.data)
            self.data = self.grid_data
            if timing:
                end_time = time.time()
                self.B.transpose_time += end_time-start_time
            self._layout = 'g'
            if barrier: self.domain.dist.comm_cart.Barrier()
