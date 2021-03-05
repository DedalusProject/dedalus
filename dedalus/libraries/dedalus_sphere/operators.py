from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from scipy.sparse import coo_matrix
from scipy.sparse import identity as id_matrix

class Operator():
    """
    Class for deffered (lazy) evaluation of matrix-valued functions between parameterised vector spaces.

    Over a set of possible vector spaces D = {domains},

        A: domain in D --> codomain(A)(domain) in D
        B: domain in D --> codomain(B)(domain) in D

        A @ B : domain --> codomain(B)(domain) --> codomain(AB)(domain).

    Operator strings are lazily evaluated on a given domin,

        (A @ B)(domain) = A(codomain(B)(domain)) @ B(domain).

    The codomains have a composition rule:

        codomain(A) + codomain(B) = codomain(AB).

    The composition rule need not be commutative, but it often is.

    Operators with compatible codomains form a linear vector space.

    For scalar multiplication:

        codomain(a*A) = codomain(A)

    For addition:

        A + B : domain in D --> codomain(A+B)(domain) in D,

    where codomain(A+B) = codomain(A) or codomain(B), provided they are compatible.

    For a given operator, we can define the inverse codomain such that,

        codomain(A)(domain)  + (-codomain(A)(domain)) = domain.

    This leads to the notion of a transpose operator,

        A.T : domain --> -codomain(A)(domain).

    and A @ A.T , A.T @ A : domain --> domain.

    The specific form of the transpose is given by A(domain).T for each domain.


    Attributes
    ----------
    codomain: an arrow between any given domain and codomain(domain).
    identity: The identity operator with the same type as self.
    Output  : class to cast output into. It should be a subclass of Operator.

    Methods
    -------
    self.data(*args):
        view of the matrix for given domain args.
    self(*args):
        evaluation of an operator object on domain args.
    self.T:
        returns transpose operator.
    self@other:
        operator composition.
    self+other:
        compatible operator addition.
    self*other:
        if self and other are both operators, retrurn the commutator A@B - B@A.
        Otherwise returns scalar multiplication.
    self**n: repeated composition.

    """

    def __init__(self,function,codomain,Output=None):
        if Output == None: Output = Operator

        self.__function = function
        self.__codomain = codomain
        self.__Output   = Output

    @property
    def function(self):
        return self.__function

    @property
    def codomain(self):
        return self.__codomain

    @property
    def Output(self):
        return self.__Output

    def __call__(self,*args):
        return self.__function(*args)

    def __matmul__(self,other):
        def function(*args):
            return self(*other.codomain(*args)) @ other(*args)
        return self.Output(function, self.codomain + other.codomain)

    @property
    def T(self):
        codomain = -self.codomain
        def function(*args):
            return self(*codomain(*args)).T
        return self.Output(function,codomain)

    @property
    def identity(self):
        def function(*args):
            return self(*args).identity
        return  self.Output(function,0*self.codomain)

    def __pow__(self,exponent):
        if exponent < 0:
            raise TypeError('exponent must be a non-negative integer.')
        if exponent == 0:
            return self.identity
        return self @ self**(exponent-1)

    def __add__(self,other):

        if other == 0: return self

        if not isinstance(other,Operator):
            other = other*self.identity

        codomain = self.codomain | other.codomain

        def function(*args):
            return self(*args) + other(*args)

        return self.Output(function, codomain)

    def __mul__(self,other):
        if isinstance(other,Operator):
            return self @ other - other @ self

        def function(*args):
            return other*self(*args)
        return self.Output(function,self.codomain)

     #   def function(*args):
     #       a,b = args[:len(self.codomain)], args[len(self.codomain):]
     #       return Kronecker(self(*a),other(*b))
     #   codomain = Codomain(self.codomain,other.codomain)
     #   return Operator(function,codomain)



    def __radd__(self,other):
        return self + other

    def __rmul__(self,other):
            return self*other

    def __truediv__(self,other):
        return self * (1/other)

    def __pos__(self):
        return self

    def __neg__(self):
        return (-1)*self

    def __sub__(self,other):
        return self + (-other)

    def __rsub__(self,other):
        return -self + other


class Codomain():
    """Base class for Codomain objects.


    Attributes
    ----------


    Methods
    -------


    """

    def __init__(self,*arrow,Output=None):
        if Output == None: Output = Codomain

        self.__arrow  = arrow
        self.__Output = Output

    @property
    def arrow(self):
        return self.__arrow

    @property
    def Output(self):
        return self.__Output

    def __getitem__(self,item):
        return self.__arrow[(item)]

    def __len__(self):
        return len(self[:])

    def __str__(self):
        return str(self.arrow)

    def __repr__(self):
        return str(self)

    def __add__(self,other):
        return self.Output(*tuple(a+b for a,b in zip(self[:],other[:])))

    def __call__(self,*args):
        return tuple(a+b for a,b in zip(self.arrow,args))

    def __eq__(self,other):
        return self[:] == other[:]

    def __or__(self,other):
        if self != other:
            raise TypeError('operators have incompatible codomains.')
        return self.Output(*tuple(a|b for a,b in zip(self[:],other[:])))

    def __neg__(self):
        return self.Output(*tuple(-a for a in self.arrow))

    def __mul__(self,other):
        if type(other) != int:
            raise TypeError('only integer multiplication defined.')

        if other == 0:
            return self.Output(*(len(self.arrow)*(0,)))

        if other < 0:
            return -self + (other+1)*self

    def __rmul__(self,other):
        return self*other

    def __sub__(self,other):
        return self + (-other)



class infinite_csr(csr_matrix):
    """
    Base class for extendable addition with csr_matrix types.

    If A.shape = (j,n), and B.shape = (k,n) we can add A+B by only summing rows i <= min(j,k).
    This is equivalent to padding the small array with rows of zeros.

    The class inherits from csr_matrix.

    Attributes
    ----------
    self.square:s
        returns square array given by number of columns.
    self.T: transpose.
        because csr_matrix.T returns csc_matrix.
    self.identity:
        returns square identity matrix with the number of columns of self.

    Methods
    -------
    self[item]: item = int(s), or slice(s).
        row-extendable slicing. Returns zero-padded array if sliced beyond self.shape[0]
    self + other:
        row-extendable addition.

    """

    def __init__(self,*args,**kwargs):
        csr_matrix.__init__(self,*args,**kwargs)

    def __repr__(self):
        s = csr_matrix(self).__repr__()
        i = s.find('sparse matrix')
        j = s.find('with')
        k = s.find('stored elements')
        return s[:i] + 'Infinite Compressed Sparse Row matrix; ' + s[j:k+15] + '>'

    @property
    def T(self):
        return infinite_csr(csr_matrix(self).T)

    @property
    def identity(self):
        return infinite_csr(id_matrix(self.shape[1]))

    @property
    def square(self):
        return csr_matrix(self[:self.shape[1]])

    def __getitem__(self,item):

        if type(item) == tuple:
            i, j = item[0], item[1:]
        else:
            i, j = item, ()

        if type(i) != slice:
            i = slice(i, i+1, 1)

        s = self.shape

        if i.stop == None or i.stop < s[0]:
            return infinite_csr(csr_matrix(self)[item])

        coo_self = coo_matrix(self)
        output = coo_matrix((coo_self.data, (coo_self.row, coo_self.col)), shape=(i.stop+1,s[1]), dtype=self.dtype)
        output = lil_matrix(output)
        return infinite_csr(output[(i,) + j])


    def __add__(self,other):

        ns, no = self.shape[0], other.shape[0]

        if ns == no:
            sum_ = csr_matrix(self) + csr_matrix(other)

        if ns > no:
            sum_ = lil_matrix(self)
            sum_[:no] += other

        if ns < no:
            sum_ = lil_matrix(other)
            sum_[:ns] += self


        return infinite_csr(sum_)

    def __radd__(self,other):
        return self + other
