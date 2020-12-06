
import numpy     as np
from . import sphere128 as sph
from dedalus.tools.cache import CachedMethod

class Sphere:
    def __init__(self,L_max,S_max=0,N_theta=None,m_min=None,m_max=None):
        self.L_max, self.S_max  = L_max, S_max
        if N_theta == None: N_theta = L_max+1
        self.N_theta = N_theta

        if m_min == None: m_min = -L_max
        if m_max == None: m_max =  L_max

        # grid and weights for the all transforms
        self.cos_grid,self.weights = sph.quadrature(self.N_theta-1,niter=3,report_error=False)
        self.grid = np.arccos(self.cos_grid)
        self.sin_grid = np.sqrt(1-self.cos_grid**2)

        self.pushY, self.pullY = {}, {}

        for s in range(-S_max,S_max+1):
            for m in range(m_min,m_max+1):
                Y = sph.Y(self.L_max,m,s,self.cos_grid)
                self.pushY[(m,s)] = (self.weights*Y).astype(np.float64)
                self.pullY[(m,s)] = (Y.T).astype(np.float64)

        # downcast to double precision
        self.grid     = self.grid.astype(np.float64)
        self.weights  = self.weights.astype(np.float64)
        self.sin_grid = self.sin_grid.astype(np.float64)
        self.cos_grid = self.cos_grid.astype(np.float64)

    @CachedMethod
    def op(self,op_name,m,s):
        return sph.operator(op_name,self.L_max,m,s).astype(np.float64)

    @CachedMethod
    def L_min(self,m,s):
        return sph.L_min(m,s)

    def zeros(self,m,s_out,s_in):
        return sph.zeros(self.L_max,m,s_out,s_in)

    def forward_spin(self,m,s,data):
        # grid --> coefficients
        return self.pushY[(m,s)].dot(data)

    def backward_spin(self,m,s,data):
        # coefficients --> grid
        return self.pullY[(m,s)].dot(data)

    @CachedMethod
    def tensor_index(self,m,rank):
        num = np.arange(2**rank)
        spin = (-1)**num
        for k in range(2,rank+1):
            spin += ((-1)**(num//2**(k-1))).astype(np.int64)

        if rank == 0: spin = [0]

        start_index = [0]
        end_index = []
        for k in range(2**rank):
            end_index.append(start_index[k]+self.L_max-sph.L_min(m,spin[k])+1)
            if k < 2**rank-1:
                start_index.append(end_index[k])

        return (start_index,end_index,spin)

    @CachedMethod
    def unitary(self,rank=1,adjoint=False):
        return sph.unitary(rank=rank,adjoint=adjoint)

    def forward(self,m,rank,data,unitary=None):

        if rank == 0:
            return self.forward_spin(m,0,data)

        (start_index,end_index,spin) = self.tensor_index(m,rank)

        if not unitary:
            unitary = self.unitary(rank=rank,adjoint=True)

        data = np.einsum("ij,j...->i...",unitary,data)

        shape = np.array(np.array(data).shape[1:])
        shape[0] = end_index[-1]

        data_c = np.zeros(shape,dtype=np.complex128)

        for i in range(2**rank):
            data_c[start_index[i]:end_index[i]] = self.forward_spin(m,spin[i],data[i])
        return data_c

    def backward(self,m,rank,data,unitary=None):

        if rank == 0:
            return self.backward_spin(m,0,data)

        (start_index,end_index,spin) = self.tensor_index(m,rank)

        if not unitary:
            unitary = self.unitary(rank=rank,adjoint=False)

        shape = np.array(np.array(data).shape)
        shape = np.concatenate(([2**rank],shape))
        shape[1] = self.N_theta

        data_g = np.zeros(shape,dtype=np.complex128)

        for i in range(2**rank):
            data_g[i] = self.backward_spin(m,spin[i],data[start_index[i]:end_index[i]])
        return np.einsum("ij,j...->i...",unitary,data_g)

    def grad(self,m,rank_in,data,data_out):
        # data and data_out are in coefficient space

        (start_index_in,end_index_in,spin_in) = self.tensor_index(m,rank_in)
        rank_out = rank_in+1
        (start_index_out,end_index_out,spin_out) = self.tensor_index(m,rank_out)

        half = 2**(rank_out-1)
        for i in range(2**(rank_out)):
            if i//half == 0:
                operator = self.op('k+',m,spin_in[i%half])
            else:
                operator = self.op('k-',m,spin_in[i%half])

            np.copyto( data_out[start_index_out[i]:end_index_out[i]],
                       operator.dot(data[start_index_in[i%half]:end_index_in[i%half]]) )


