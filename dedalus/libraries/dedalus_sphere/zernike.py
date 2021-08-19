import numpy as np
from . import jacobi  as Jacobi
from .operators    import Operator, infinite_csr


# The defalut configuration for the base Jacobi parameter.
alpha = 0

def mass(dimension,k=alpha):
    return Jacobi.mass(k,dimension/2 - 1)/2**( k + dimension/2 + 1 )

def quadrature(dimension,n,k=alpha):
    """
    Weights associated with
        dV = (1-r*r)**k * r**(dimension-1) dr, where 0 <= r <= 1.

    """

    z, w = Jacobi.quadrature(n,k,dimension/2 - 1)

    w /= 2**( k + dimension/2 + 1 )

    return z, w

def min_degree(l):
    return max(l//2,0)

def polynomials(dimension,n,k,l,z):
    """
        Unit normalised:

            integral(Q**2 dV)  = 1

    """

    b = l + dimension/2 - 1

    init  = Jacobi.measure(0,l,z,log=True,probability=False)
    init -= Jacobi.mass(k,b,log=True)  - np.log(2)*(k + dimension/2 + 1)
    init = np.exp(0.5*init)

    return Jacobi.polynomials(n,k,b,z,init)


def operator(dimension, name, radius=1):
    """
    Interface to base ZernikeOperator class.

    Parameters
    ----------

    """

    if name == 'Id':
        def I(n,k,l):
            return Jacobi.operator('Id')(n,k,l+dimension/2 - 1)
        return Operator(I,ZernikeCodomain(0,0,0))

    if name == 'Z':
        def Z(n,k,l):
            return Jacobi.operator('Z')(n,k,l+dimension/2 - 1)
        return Operator(Z,ZernikeCodomain(1,0,0))

    return ZernikeOperator(dimension, name, radius=radius)

class ZernikeOperator():

    def __init__(self,dimension,name,radius=1):

        self.__function   = getattr(self,f'_ZernikeOperator__{name}')
        self.__dimension  = dimension
        self.__radius     = radius

    def __call__(self,p):
        return Operator(*self.__function(p))

    @property
    def dimension(self):
        return self.__dimension

    @property
    def radius(self):
        return self.__radius

    def b(self,l):
        return l + self.dimension/2 - 1

    def __D(self,dl):

        def D(n,k,l):
            D = Jacobi.operator('D' if dl > 0 else 'C')(+1)
            return  (2/self.radius)*D(n,k,self.b(l))

        return D, ZernikeCodomain(-(1+dl)//2,1,dl)

    def __E(self,dk):

        def E(n,k,l):
            E = Jacobi.operator('A')(dk)
            return  np.sqrt(0.5)*E(n,k,self.b(l))

        return E, ZernikeCodomain((1-dk)//2,dk,0)

    def __R(self,dl):

        def R(n,k,l):
            R = Jacobi.operator('B')(dl)
            return (np.sqrt(0.5)*self.radius)*R(n,k,self.b(l))

        return R, ZernikeCodomain((1-dl)//2,0,dl)


class ZernikeCodomain(Jacobi.JacobiCodomain):

    def __init__(self,dn=0,dk=0,dl=0,pi=0):
        Jacobi.JacobiCodomain.__init__(self,dn,dk,dl,0,Output=ZernikeCodomain)

    def __str__(self):
        s = f'(n->n+{self[0]},k->k+{self[1]},l->l+{self[2]})'
        return s.replace('+0','').replace('+-','-')

