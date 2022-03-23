import numpy             as np
from . import jacobi            as Jacobi
from scipy.sparse import dia_matrix as banded
from .operators    import Operator, infinite_csr

dtype = 'longdouble'

def quadrature(Lmax,dtype=dtype):
    """Generates the Gauss quadrature grid and weights for spherical harmonics transform.
        Returns cos_theta, weights

        Will integrate polynomials on (-1,+1) exactly up to degree = 2*Lmax+1.

    Parameters
    ----------
    Lmax: int >=0; spherical-harmonic degree.

    """

    return Jacobi.quadrature(Lmax+1,0,0,dtype=dtype)


def spin2Jacobi(Lmax,m,s,ds=None,dm=None):

    n    = Lmax + 1 - max(abs(m),abs(s))
    a, b = abs(m+s), abs(m-s)

    if ds == dm == None:
        return n,a,b

    if ds == None: ds = 0
    if dm == None: dm = 0

    m += dm
    s += ds

    dn    = Lmax + 1 - max(abs(m),abs(s)) - n
    da,db = abs(m+s) - a, abs(m-s) - b

    return n,a,b,dn,da,db


def harmonics(Lmax,m,s,cos_theta,**kwargs):
    """
        Gives spin-wieghted spherical harmonic functions on the Gauss quaduature grid.
        Returns an array with
            shape = ( Lmax - Lmin(m,s) + 1, len(z) )
                 or (Lmax - Lmin(m,s) + 1,) if z is a single point.

        Parameters
        ----------
        Lmax: int >=0; spherical-harmonic degree.
        m,s : int
            spherical harmonic parameters.
        cos_theta: np.ndarray or float.
        dtype: output dtype. internal dtype = 'longdouble'.
        """

    n,a,b = spin2Jacobi(Lmax,m,s)

    init = np.exp(0.5*Jacobi.measure(a,b,cos_theta,log=True))
    init *= ((-1.)**max(m,-s))

    return Jacobi.polynomials(n,a,b,cos_theta,init,**kwargs)


def operator(name,dtype=dtype):
    """
    Interface to base ShereOperator class.

    Parameters
    ----------

    """

    if name == 'Id':
        return SphereOperator.identity(dtype=dtype)

    if name == 'Pi':
        return SphereOperator.parity(dtype=dtype)

    if name == 'L':
        return SphereOperator.L(dtype=dtype)

    if name == 'M':
        return SphereOperator.M(dtype=dtype)

    if name == 'S':
        return SphereOperator.S(dtype=dtype)

    if name == 'Cos':
        def Cos(Lmax,m,s):
            return Jacobi.operator('Z',dtype=dtype)(*spin2Jacobi(Lmax,m,s))
            #return Jacobi.operator('Z',dtype=dtype)(Lmax+1, abs(m+s), abs(m-s))
        return Operator(Cos,SphereCodomain(1,0,0,0))

    return SphereOperator(name,dtype=dtype)


class SphereOperator():

    def __init__(self,name,radius=1,dtype=dtype):

        self.__function   = getattr(self,f'_SphereOperator__{name}')

        self.__radius = radius

        self.__dtype = dtype

    def __call__(self,ds):
        return Operator(*self.__function(ds))

    @property
    def radius(self):
        return self.__radius

    @property
    def dtype(self):
        return self.__dtype

    def __D(self,ds):

        def D(Lmax,m,s):

            n,a,b,dn,da,db = spin2Jacobi(Lmax,m,s,ds=ds)

            D = Jacobi.operator('C' if da+db == 0 else 'D',dtype=self.dtype)(da)

            return  (-ds*np.sqrt(0.5)/self.radius)*D(n,a,b)

        return D, SphereCodomain(0,0,ds,0)

    def __Sin(self,ds):

        def Sin(Lmax,m,s):

            n,a,b,dn,da,db = spin2Jacobi(Lmax,m,s,ds=ds)

            S =     Jacobi.operator('A',dtype=self.dtype)(da)
            S = S @ Jacobi.operator('B',dtype=self.dtype)(db)

            return (da*ds) * S(n,a,b)

        return Sin, SphereCodomain(1,0,ds,0)

    @staticmethod
    def identity(dtype=dtype):

        def I(Lmax,m,s):
            n = spin2Jacobi(Lmax,m,s)[0]
            N = np.ones(n,dtype=dtype)
            return infinite_csr(banded((N,[0]),(max(n,0),max(n,0))))

        return Operator(I,SphereCodomain(0,0,0,0))

    @staticmethod
    def parity(dtype=dtype):

        def Pi(Lmax,m,s):
            return Jacobi.operator('Pi',dtype=dtype)(*spin2Jacobi(Lmax,m,s))

        return Operator(Pi,SphereCodomain(0,0,0,1))

    @staticmethod
    def L(dtype=dtype):

        def L(Lmax,m,s):
            n = spin2Jacobi(Lmax,m,s)[0]
            N = np.arange(Lmax+1-n,Lmax+1,dtype=dtype)
            return infinite_csr(banded((N,[0]),(max(n,0),max(n,0))))

        return Operator(L,SphereCodomain(0,0,0,0))

    @staticmethod
    def M(dtype=dtype):

        def M(Lmax,m,s):
            n = spin2Jacobi(Lmax,m,s)[0]
            N = m*np.ones(n,dtype=dtype)
            return infinite_csr(banded((N,[0]),(max(n,0),max(n,0))))

        return Operator(M,SphereCodomain(0,0,0,0))

    @staticmethod
    def S(dtype=dtype):

        def S(Lmax,m,s):
            n = spin2Jacobi(Lmax,m,s)[0]
            N = abs(s)*np.ones(n,dtype=dtype)
            return infinite_csr(banded((N,[0]),(max(n,0),max(n,0))))

        return Operator(S,SphereCodomain(0,0,0,0))


class SphereCodomain(Jacobi.JacobiCodomain):

    def __init__(self,dL=0,dm=0,ds=0,pi=0):
        Jacobi.JacobiCodomain.__init__(self,*(dL,dm,ds,pi),Output=SphereCodomain)

    def __str__(self):
        s = f'(L->L+{self[0]},m->m+{self[1]},s->s+{self[2]})'
        if self[3]: s = s.replace('s->s','s->-s')
        return s.replace('+0','').replace('+-','-')

    def __call__(self,*args,evaluate=True):
        L,m,s = args[:3]
        if self[3]: s *= -1
        return self[0] + L, self[1] + m, self[2] + s

    def __neg__(self):
        m,s = -self[1],-self[2]
        if self[3]: s *= -1
        return SphereCodomain(-self[0],m,s,self[3])

