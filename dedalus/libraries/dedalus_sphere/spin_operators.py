import numpy as np
from itertools import product
from .tuple_tools import *
from .operators import Operator, Codomain

indexing  = (-1,0,1)
threshold = 1e-12

class TensorOperator(Operator):
    """
    Class for lazy evaluation of spin/regularity tensor operations.

    Attributes
    ----------
    codomain: TensorCodomain object
        keeps track of the difference in rank of TensorOperators.
    indexing: tuple
        must be a permutation of (-1,+1) or (-1,0,+1)
    threshold: float (1e-12 default)
        send smaller values to 0.
    dimension: int
        number of basis indices.

    Methods
    -------
    self(rank):
        all TensorOperator objects are callable on the input rank of a tensor.
    self[sigma,tau]:
        sigma,tau tuples of spin/regularity indices
    self.generator(rank):
        generate all lenght-rank tuples according to a given indexing.
    self.array:
        from self[sigma,tau] compute flattened (dimension**ranks[0],dimension**ranks[1]) np.ndarray.

    """

    def __init__(self,function,codomain,indexing=indexing,threshold=threshold):
        Operator.__init__(self,function,codomain,Output=TensorOperator)
        self.__indexing  = indexing
        self.__threshold = threshold

    @property
    def indexing(self):
        return self.__indexing

    @property
    def threshold(self):
        return self.__threshold

    @property
    def dimension(self):
        return len(self.indexing)

    def __call__(self,*args):
        output = self.function(*args)
        np.where(np.abs(output) < self.threshold,0,output)
        return output

    @int2tuple
    def __getitem__(self,i):
        sigma,tau = i[0],i[1]
        i = tuple2index(sigma,self.indexing)
        j = tuple2index(tau,self.indexing)
        return self(len(tau))[i,j]

    def range(self,rank):
        return product(*(rank*(self.indexing,)))

    def array(self,ranks):
        T = np.zeros(tuple(self.dimension**r for r in ranks))
        for i, sigma in enumerate(self.range(ranks[0])):
            for j, tau in enumerate(self.range(ranks[1])):
                T[i,j] = self[sigma,tau]
        return T

class Identity(TensorOperator):
    """
    Spin/regularity space identity transformation of arbitrary rank.

    Methods
    -------
    self[sigma,tau] = 1 if sigma == tau else 0

    """

    def __init__(self,**kwargs):

        identity = lambda rank: self.array((rank,rank))
        TensorOperator.__init__(self,identity,TensorCodomain(0),**kwargs)

    @int2tuple
    def __getitem__(self,i):
        return int(i[0] == i[1])


class Metric(TensorOperator):
    """
    Spin-space representation of arbitrary-rank local Cartesian metric tensor. E.g.:

    Id = e(+)e(-) + e(0)e(0) + e(-)e(+) = e(x)e(x) + e(y)e(y) + e(z)e(z)

    Methods
    -------
    self[sigma,tau] = 1 if sigma == -tau else 0

    """

    def __init__(self,**kwargs):

        metric = lambda rank: self.array((rank,rank))
        TensorOperator.__init__(self,metric,TensorCodomain(0),**kwargs)

    @int2tuple
    def __getitem__(self,i):
        return int(i[0] == dual(i[1]))


class Transpose(TensorOperator):
    """
    Transpose operator for arbitrary rank tensor.

        T[i,j,...,k] -> T[permutation(i,j,...,k)]

    Default transposes 0 <--> 1 indices.

    Attributes
    ----------
    permutation: tuple
        Relative to natural order, using Cauchy's "one-line notation".

    Methods
    -------
    self[sigma,tau] = self[sigma,permutation(tau)]


    """

    def __init__(self,permutation=(1,0),**kwargs):

        transpose = lambda rank: self.array((rank,rank))
        TensorOperator.__init__(self,transpose,TensorCodomain(0),**kwargs)
        self.__permutation = permutation

    @property
    def permutation(self):
        return self.__permutation

    @int2tuple
    def __getitem__(self,i):
        return int(i[0] == apply(self.permutation)(i[1]))

class Trace(TensorOperator):
    """
    Class for contracting arbitrary indices down to a scalar in those indices:

        sum_(i+j=0) T[..i,..j,..]

    This can generalise to (e.g.):

        sum_(i+j+k+l=0) T[..i,..j,..k,..l,..]

    It might seem like we would prefer to do

        sum_(i+j=0,k+l=0) T[..i,..j,..k,..l,..] = sum_(i+j=0) sum_(k+l=0) T[..i,..j,..k,..l,..]

    However, we can accomplish the latter by multiple operations over two indices.
    Conversly, repeted apllication of 2-index sums cannot sum more than two indices simultaneously.

    For a few examples:

        len(indices) == 0:
            Identity()
            S --> S

        len(indices) == 1:
            Selects spin=0 component from given axis:
                V --> V[0]

        len(indices) == 2:
            Traditional Trace((0,1)):
                T --> T[-,+] + T[0,0] + T[+,-]

            Trace((0,)) @ Trace((1,))
            produces T[0,0] individually.

            Trace((0,1)) - Trace((0,)) @ Trace((1,))
            produces T[-,+] + T[+,-] individually.

        len(indices) == 3:
            R[+,-,0]+R[-,+,0] + R[+,0,-]+R[-,0,+] + R[0,+,-]+R[0,-,+] + R[0,0,0]

            We can select different scalars in this sum by application of lower-rank traces.


    Attributes
    ----------
    indices: tuple of -1,0,+1

    """

    def __init__(self,indices,**kwargs):
        if type(indices) == int: indices = (indices,)

        trace = lambda rank: self.array((rank-len(indices),rank))
        TensorOperator.__init__(self,trace,TensorCodomain(-len(indices)),**kwargs)

        self.__indices  = indices

    @property
    def indices(self):
        return self.__indices

    @int2tuple
    def __getitem__(self,i):

        return int(i[0] == remove(self.indices)(i[1]) and sum_(self.indices)(i[1]) == 0)

class TensorProduct(TensorOperator):
    """
    Action of multiplication by single spin-tensor basis element:

        e(kappa) (X) T = sum_(sigma) T(sigma) e(kappa+sigma)

        or

        T (X) e(kappa) = sum_(sigma) T(sigma) e(sigma+kappa)

    Attributes
    ----------
    element: tuple
        single tensor basis element, kappa
    action: str ('left' or 'right')

    """

    def __init__(self,element,action='left',**kwargs):
        if type(element) == int: element = (element,)

        product = lambda rank: self.array((rank+len(element),rank))
        TensorOperator.__init__(self,product,TensorCodomain(len(element)),**kwargs)
        self.__element = element
        self.__action  = action

    @property
    def element(self):
        return self.__element

    @property
    def action(self):
        return self.__action

    @int2tuple
    def __getitem__(self,i):
        if self.action == 'left':
            return int(i[0] == self.element + i[1])
        if self.action == 'right':
            return int(i[0] == i[1] + self.element)


def xi(mu,ell):
    """
        Normalised derivative scale factors. xi(-1,ell)**2 + xi(+1,ell)**2 = 1.

        Parameters
        ----------
        mu  : int
            regularity; -1,+1,0. xi(0,ell) = 0 by definition.
        ell : int
            spherical-harmonic degree.

        """

    return np.abs(mu)*np.sqrt((1 + mu/(2*ell+1))/2)


class Intertwiner(TensorOperator):
    """
    Regularity-to-spin map.

        Q(ell)[spin,regularity]

    Attributes
    ----------
    L : int
        spherical-harmonic degree

    Methods
    -------
    k: int mu, s
        angular spherical wavenumbers.
    forbidden_spin: tuple spin
        filter spin components that don't exist.
    forbidden_regularity: tuple regularity
        filter regularity components that don't exist.
    self[sigma,a]:
        regularity-to-spin coupling coefficients

    """

    def __init__(self,L,**kwargs):

        intertwiner = lambda rank: self.array((rank,rank))
        TensorOperator.__init__(self,intertwiner,TensorCodomain(0),**kwargs)
        self.__ell = L

    @property
    def L(self):
        return self.__ell

    def k(self,mu,s):
        return -mu*np.sqrt((self.L-s*mu)*(self.L+s*mu+1)/2)

    @int2tuple
    def forbidden_spin(self,spin):
        return self.L < abs(sum(spin))

    @int2tuple
    def forbidden_regularity(self,regularity):
        # Fast return for clearly allowed cases
        if self.L >= len(regularity):
            return False
        # Check other cases
        walk = (self.L,)
        for r in regularity[::-1]:
            walk += (walk[-1] + r,)
            if walk[-1] < 0 or walk[-2:] == (0,0):
                return True
        return False

    @int2tuple
    def __getitem__(self,i):

        spin, regularity = i[0], i[1]

        if len(spin) == 0:
            return 1

        if self.forbidden_spin(spin) or self.forbidden_regularity(regularity):
            return 0

        sigma, a = spin[0],  regularity[0]
        tau,   b = spin[1:], regularity[1:]

        R = 0
        for i,t in enumerate(tau):
            if t+sigma ==  0: R -= self[replace(i,0)(tau),b]
            if t       ==  0: R += self[replace(i,sigma)(tau),b]

        Q  = self[tau,b]
        R -= self.k(sigma,sum(tau))*Q
        J  = self.L + sum(b)

        if sigma != 0:
            Q = 0

        if a == -1: return (Q * J   - R)/np.sqrt(J*(2*J+1))
        if a ==  0: return      sigma*R /np.sqrt(J*(J+1))
        if a == +1: return (Q*(J+1) + R)/np.sqrt((J+1)*(2*J+1))


class TensorCodomain(Codomain):
    """
    Class for keeping track of TensorOperator codomains.


    Attributes
    ----------
    arrow: int
        relative change in rank between input and output.


    """

    def __init__(self,rank_change):
        Codomain.__init__(self,rank_change,Output=TensorCodomain)

    def __str__(self):
        s = f'(rank->rank+{self[0]})'
        return s.replace('+0','').replace('+-','-')
