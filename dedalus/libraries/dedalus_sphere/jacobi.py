import numpy             as np
from scipy.sparse import dia_matrix as banded

from .operators import Operator, Codomain, infinite_csr

dtype='float64'

def coefficient_connection(N,ab,cd,init_ab=1,init_cd=1):
    """The connection matrix between any bases coefficients:
        Pab(z) = Pcd(z) @ Cab2cd    -->    Acd = Cab2cd @ Aab
        The output is always a dense matrix format.
        Parameters
        ----------
        N      :  int
        ab, cd :  tuples of input and output Jacobi parameters.
        """

    a,b = ab
    c,d = cd

    zcd, wcd = quadrature(N,c,d)

    wcd /= np.sum(wcd)

    Pab = polynomials(N,a,b,zcd,init_ab + 0*zcd)
    Pcd = polynomials(N,c,d,zcd,init_cd + 0*zcd)

    return Pcd @ (wcd*Pab).T

def polynomials(n,a,b,z,init=None,Newton=False,normalised=True,dtype=dtype,internal='longdouble'):
    """
    Jacobi polynomials, P(n,a,b,z), of type (a,b) up to degree n-1.

    Jacobi.polynomials(n,a,b,z)[k] gives P(k,a,b,z).

    Newton = True: cubic-converging update of P(n-1,a,b,z) = 0.

    Parameters
    ----------
    a,b > -1
    z: float, np.ndarray.

    init: float, np.ndarray or None -> 1+0*z, or (1+0*z)/sqrt(mass) if normalised.
    normalised: classical or unit-integral normalisation.
    dtype:   'float64','longdouble' output dtype.
    internal: internal dtype.

    """

    if n < 1:
        return np.zeros((0, z.size), dtype=dtype)

    if init is None:
        init = 1 + 0*z
        if normalised:
            init /= np.sqrt(mass(a,b),dtype=internal)

    Z = operator('Z',normalised=normalised,dtype=internal)
    Z = banded(Z(n+1,a,b).T).data

    shape = n+1
    if type(z) == np.ndarray:
        shape = (shape,len(z))

    P     = np.zeros(shape,dtype=internal)
    P[0]  = init

    if len(Z) == 2:
        P[1]  = z*P[0]/Z[1,1]
        for k in range(2,n+1):
            P[k] = (z*P[k-1] - Z[0,k-2]*P[k-2])/Z[1,k]
    else:
        P[1]  = (z-Z[1,0])*P[0]/Z[2,1]
        for k in range(2,n+1):
            P[k] = ((z-Z[1,k-1])*P[k-1] - Z[0,k-2]*P[k-2])/Z[2,k]

    if Newton:
        L = n + (a+b)/2
        return z + (1-z**2)*P[n-1]/(L*Z[-1,n]*P[n]-(L-1)*Z[0,n-2]*P[n-2]), P[:n-1]

    return P[:n].astype(dtype)

def quadrature(n,a,b,days=3,probability=False,dtype=dtype,internal='longdouble'):
    """
    Jacobi 'roots' grid and weights; solutions to

    P(n,a,b,z) = 0; len(z) = n.

    sum(weights*polynomial) = integrate_(-1,+1)(polynomial),
    exactly up to degree(polynomial) = 2n - 1.

    Parameters
    ----------
    n: int > 0.
    a,b: float > -1.
    days: number of Newton updates.
    probability: sum(weights) = 1 or sum(weights) = mass(a,b)
    dtype:   'float64','longdouble' output dtype.
    internal: internal dtype (Newton uses by default).

    """

    z = grid_guess(n,a,b,dtype=internal)

    if probability:
        w = 1
    else:
        w = mass(a,b)

    if (a == b == -1/2):
        return z.astype(dtype), (w/n + 0*z).astype(dtype)
    elif (a == b == +1/2):
        P = polynomials(n+1,a,b,z,dtype=internal)[:n]
    else:
        for _ in range(days):
            z, P = polynomials(n+1,a,b,z,Newton=True)

    P[0] /= np.sqrt(np.sum(P**2,axis=0))
    w *= P[0]**2

    return z.astype(dtype), w.astype(dtype)

def grid_guess(n,a,b,dtype='longdouble',quick=False):
    """
    Approximate solution to

    P(n,a,b,z) = 0

    """

    if a == b == -1/2 :
        quick = True

    if n == 1:
        return operator('Z')(n,a,b).toarray()[0]

    if quick:
        return np.cos(np.pi*(np.arange(4*n-1,2,-4,dtype=dtype)+2*a)/(4*n+2*(a+b+1)))

    from scipy.linalg import eigvalsh_tridiagonal as eigs

    Z = banded(operator('Z')(n,a,b))

    return eigs(Z.diagonal(0),Z.diagonal(1)).astype(dtype)


def measure(a,b,z,probability=True,log=False):
    """

    mu(a,b,z) = (1-z)**a (1+z)**b

    if normalised:  ((1-z)/2)**a ((1+z)/2)**b / (2*Beta(a+1,b+1))

    Parameters
    ----------
    a,b > -1

    """

    if not log:

        w = 1

        if a != 0:
            w *= (1-z)**a

        if b != 0:
            w *= (1+z)**b

        if probability:
            w /= mass(a,b)

        return w

    S = 0

    if a != 0:
        S += a*np.log(1-z)

    if b != 0:
        S += b*np.log(1+z)

    if probability:
        S -= mass(a,b,log=True)

    return S

def mass(a,b,log=False):
    """

    2**(a+b+1)*Beta(a+1,b+1) = integrate_(-1,+1)( (1-z)**a (1+z)**b )

    Parameters
    __________
    a,b > -1
    log: optional

    """

    if not log:
        from scipy.special import beta
        return 2**(a+b+1)*beta(a+1,b+1)

    from scipy.special import betaln
    return (a+b+1)*np.log(2) + betaln(a+1,b+1)

def norm_ratio(dn,da,db,n,a,b,squared=False):
    """
    Ratio of classical Jacobi normalisation.

        N(n,a,b) = integrate_(-1,+1)( (1-z)**a (1+z)**b P(n,a,b,z)**2 )

                                   Gamma(n+a+1) * Gamma(n+b+1)
        N(n,a,b) = 2**(a+b+1) * ----------------------------------
                                 (2n+a+b+1) * Gamma(n+a+b+1) * n!


    The function returns: sqrt(N(n+dn,a+da,b+db)/N(n,a,b))

    This is used in rescaling the input and output of operators that increment n,a,b.

    Parameters
    ----------
    dn,da,db: int
    n: np.ndarray, int > 0
    a,b: float > -1
    squared: return N(n,a,b) or sqrt(N(n,a,b)) (defalut)

    """

    if not all(type(d) == int for d in (dn,da,db)):
        raise TypeError('can only increment by integers.')

    def tricky(n,a,b):
        # 0/0 = 1
        if a+b != -1:
            return (2*n+a+b+1)/(n+a+b+1)
        return 2 - (n==0)

    def n_ratio(d,n,a,b):
        if d <  0: return 1/n_ratio(-d,n+d,a,b)
        if d == 0: return 1 + 0*n
        if d == 1:
            return ((n+a+1)*(n+b+1)/((n+1)*(2*n+a+b+3))) * tricky(n,a,b)
        return n_ratio(1,n+d-1,a,b)*n_ratio(d-1,n,a,b)

    def ab_ratio(d,n,a,b):
        if d <  0: return 1/ab_ratio(-d,n,a+d,b)
        if d == 0: return 1 + 0*n
        if d == 1:
            return (2*(n+a+1)/(2*n+a+b+2)) * tricky(n,a,b)
        return ab_ratio(1,n,a+d-1,b)*ab_ratio(d-1,n,a,b)

    ratio = n_ratio(dn,n,a+da,b+db)*ab_ratio(da,n,a,b+db)*ab_ratio(db,n,b,a)

    if not squared:
        return np.sqrt(ratio)
    return ratio


def operator(name,normalised=True,dtype=dtype):
    """
    Interface to base JacobiOperator class.

    Parameters
    ----------
    name: A, B, C, D, Id, Pi, N, Z (Jacobi matrix)
    normalised: True --> unit-integral, False --> classical.
    dtype: output dtype

    """
    if name == 'Id':
        return JacobiOperator.identity(dtype=dtype)
    if name == 'Pi':
        return JacobiOperator.parity(dtype=dtype)
    if name == 'N':
        return JacobiOperator.number(dtype=dtype)
    if name == 'Z':
        A = JacobiOperator('A',normalised=normalised,dtype=dtype)
        B = JacobiOperator('B',normalised=normalised,dtype=dtype)
        return (B(-1) @ B(+1) - A(-1) @ A(+1))/2
    return JacobiOperator(name,normalised=normalised,dtype=dtype)

class JacobiOperator():
    """
    The base class for primary operators acting on finite row vectors of Jacobi polynomials.

    <n,a,b,z| = [P(0,a,b,z),P(1,a,b,z),...,P(n-1,a,b,z)]

    P(k,a,b,z) = <n,a,b,z|k> if k < n else 0.

    Each oparator takes the form:

    L(a,b,z,d/dz) <n,a,b,z| = <n+dn,a+da,b+db,z| R(n,a,b)

    The Left action is a z-differential operator.
    The Right action is a matrix with n+dn rows and n columns.

    The Right action is encoded with an "infinite_csr" sparse matrix object.
    The parameter increments are encoded with a JacobiCodomain object.

     L(a,b,z,d/dz)  ............................  dn, da, db
    ---------------------------------------------------------
     A(+1) = 1      ............................   0, +1,  0
     A(-1) = 1-z    ............................  +1, -1,  0
     A(0)  = a      ............................   0,  0,  0

     B(+1) = 1      ............................   0,  0, +1
     B(-1) = 1+z    ............................  +1,  0, -1
     B(0)  = b      ............................   0,  0,  0

     C(+1) = b + (1+z)d/dz .....................   0, +1, -1
     C(-1) = a - (1-z)d/dz .....................   0, -1, +1

     D(+1) = d/dz  .............................  -1, +1, +1
     D(-1) = a(1+z) - b(1-z) - (1-z)(1+z)d/dz ..  +1, -1, -1

     Each -1 operator is the adjoint of the coresponding +1 operator.

     In addition there are a few exceptional operators:

        Identity: <n,a,b,z| -> <n,a,b,z|

        Parity:   <n,a,b,z| -> <n,a,b,-z| = <n,b,a,z| Pi
                  The codomain is not additive in this case.

        Number:   <n,a,b,z| -> [0*P(0,a,b,z),1*P(1,a,b,z),...,(n-1)*P(n-1,a,b,z)]
                  This operator doesn't have a local differential Left action.

    Attributes
    ----------
    name: str
        A, B, C, D
    normalised: bool
        True gives operators on unit-integral polynomials, False on classical normalisation.
    dtype: 'float64','longdouble'
        output dtype.


    Methods
    -------
    __call__(p): p=-1,0,1
        returns Operator object depending on p.
        Operator.function is an infinite_csr matrix constructor for n,a,b.
        Operator.codomain is a JacobiCodomain object.

    staticmethods
    -------------
    identity: Operator object for identity matrix
    parity:   Operator object for reflection transformation.
    number:   Operator object for polynomial degree.

    """

    dtype='longdouble'

    def __init__(self,name,normalised=True,dtype=dtype):

        self.__function   = getattr(self,f'_JacobiOperator__{name}')
        self.__normalised = normalised
        self.dtype        = dtype

    @property
    def normalised(self):
        return self.__normalised

    def __call__(self,p):
        return Operator(*self.__function(p))

    def __A(self,p):

        if p == 0:
            def A(n,a,b):
                N = a*np.ones(n,dtype=self.dtype)
                return infinite_csr(banded((N,[0]),(n,n)))
            return A, JacobiCodomain(0,0,0,0)

        def A(n,a,b):

            N = np.arange(n,dtype=self.dtype)
            bands = np.array({+1:[N+(a+b+1),  -(N+b)],
                              -1:[2*(N+a)  ,-2*(N+1)]}[p])
            bands[:,0] = 1 if a+b == -1 else bands[:,0]/(a+b+1)
            bands[:,1:] /= 2*N[1:]+a+b+1

            if self.normalised:
                bands[0] *= norm_ratio(0,p,0,N,a,b)
                bands[1,(1+p)//2:] *= norm_ratio(-p,p,0,N[(1+p)//2:],a,b)

            return infinite_csr(banded((bands,[0,p]),(max(n+(1-p)//2,0),max(n,0))))

        return A, JacobiCodomain((1-p)//2,p,0,0)

    def __B(self,p):

        def B(n,a,b):
            Pi = operator('Pi')
            return (Pi @ operator('A')(p) @ Pi)(n,a,b)

        return B, JacobiCodomain((1-p)//2,0,p,0)

    def __C(self,p):

        def C(n,a,b):

            N = np.arange(n,dtype=self.dtype)
            bands = np.array([N + {+1:b,-1:a}[p]])

            if self.normalised:
                bands[0] *= norm_ratio(0,p,-p,N,a,b)

            return infinite_csr(banded((bands,[0]),(max(n,0),max(n,0))))

        return C, JacobiCodomain(0,p,-p,0)

    def __D(self,p):

        def D(n,a,b):

            N = np.arange(n,dtype=self.dtype)
            bands = np.array([(N + {+1:a+b+1,-1:1}[p])*2**(-p)])

            if self.normalised:
                bands[0,(1+p)//2:] *= norm_ratio(-p,p,p,N[(1+p)//2:],a,b)

            return infinite_csr(banded((bands,[p]),(max(n-p,0),max(n,0))))

        return D, JacobiCodomain(-p,p,p,0)

    @staticmethod
    def identity(dtype=dtype):

        def I(n,a,b):
            N = np.ones(n,dtype=dtype)
            return infinite_csr(banded((N,[0]),(max(n,0),max(n,0))))

        return Operator(I,JacobiCodomain(0,0,0,0))

    @staticmethod
    def parity(dtype=dtype):

        def P(n,a,b):
            N = np.arange(n,dtype=dtype)
            return infinite_csr(banded(((-1)**N,[0]),(max(n,0),max(n,0))))

        return Operator(P,JacobiCodomain(0,0,0,1))

    @staticmethod
    def number(dtype=dtype):

        def N(n,a,b):
            return infinite_csr(banded((np.arange(n,dtype=dtype),[0]),(max(n,0),max(n,0))))

        return Operator(N,JacobiCodomain(0,0,0,0))



class JacobiCodomain(Codomain):
    """
    Base class for Jacobi codomain.

    codomain = JacobiCodomain(dn,da,db,pi)

    n', a', b' = codomain(n,a,b)

    if pi == 0:
        n', a', b' = n+dn, a+da, b+db

    if pi == 1:
        n', a', b' = n+dn, b+da, a+db

    pi_0 + pi_1 = pi_0 XOR pi_1

    Attributes
    ----------
    __arrow: stores dn,da,db,pi.


    Methods
    -------
    self[0:3]: returns dn,da,db,pi respectively.
    str(self): displays codomain mapping.
    self + other: combines codomains.
    self(n,a,b): evaluates current codomain.
    -self: inverse codomain.
    n*self: iterated codomain addition.
    self == other: determines equivalent codomains (a,b,pi).
    self | other: determines codomain compatiblity and returns larger-n space.

    """

    def __init__(self,dn=0,da=0,db=0,pi=0,Output=None):
        if Output == None: Output = JacobiCodomain
        Codomain.__init__(self,*(dn,da,db,pi),Output=Output)

    def __len__(self):
        return 3

    def __str__(self):
        s = f'(n->n+{self[0]},a->a+{self[1]},b->b+{self[2]})'
        if self[3]: s = s.replace('a->a','a->b').replace('b->b','b->a')
        return s.replace('+0','').replace('+-','-')

    def __add__(self,other):
        return self.Output(*self(*other[:3],evaluate=False),self[3]^other[3])

    def __call__(self,*args,evaluate=True):
        n,a,b = args[:3]
        if self[3]: a,b = b,a
        n, a, b = self[0] + n, self[1] + a, self[2] + b
        if evaluate and (a <= -1 or b <= -1):
            raise ValueError('invalid Jacobi parameter.')
        return n,a,b

    def __neg__(self):
        a,b = -self[1],-self[2]
        if self[3]: a,b = b,a
        return self.Output(-self[0],a,b,self[3])

    def __eq__(self,other):
        return self[1:] == other[1:]

    def __or__(self,other):
        if self != other:
            raise TypeError('operators have incompatible codomains.')
        if self[0] >= other[0]:
            return self
        return other
