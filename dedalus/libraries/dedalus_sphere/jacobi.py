"""Jacobi polynomial functions."""

import numpy as np
import xprec
from scipy.sparse import dia_matrix
from scipy.special import beta, betaln
from scipy.linalg import eigvalsh_tridiagonal
from .operators import Operator, Codomain, infinite_csr


# Default dtypes
DEFAULT_GRID_DTYPE = xprec.ddouble
DEFAULT_OPERATOR_DTYPE = np.float64


def coefficient_connection(n, ab, cd, init_ab=1, init_cd=1):
    """
    The connection matrix between any bases coefficients:
        Pab(z) = Pcd(z) @ Cab2cd    -->    Acd = Cab2cd @ Aab

    The output is always a dense matrix format.

    Parameters
    ----------
    n : int
        Number of polynomials (max degree + 1).
    ab, cd : tuple of floats
        Tuples of input and output Jacobi parameters.
    init_ab, init_cd : floats or arrays, optional.
        Initial envelopes for input  and output polynomials.

    Returns
    -------
    conn : array
        Coupling coefficient matrix from (a,b) to (c,d) polynomials.
    """
    a, b = ab
    c, d = cd
    zcd, wcd = quadrature(n, c, d)
    wcd /= np.sum(wcd)
    Pab = polynomials(n, a, b, zcd, init_ab)
    Pcd = polynomials(n, c, d, zcd, init_cd)
    return Pcd @ (wcd * Pab).T


def polynomials(n, a, b, z, init=None, Newton=False, normalized=True, dtype=None):
    """
    Jacobi polynomials of type (a,b) up to degree n-1.

    Parameters
    ----------
    n : int
        Number of polynomials to compute (max degree + 1).
    a, b : float
        Jacobi parameters.
    z : array
        Grid array.
    init : float or array, optional
        Initial envelope for polynomial recursion. Default determined by 'normalized' option.
    Newton : bool, optional
        Return cubicly converging update to grid as roots of P[n]. Default: False.
    normalized : bool, optional
        True to initialize with 1/sqrt(mass) if init not provided. Default: True.
    dtype : dtype, optional
        Data type. Default: module-level DEFAULT_GRID_DTYPE.

    Returns
    -------
    grid : array, optional
        Refined grid as roots of P[n]. Only returned if Newton is True.
    polynomials : array
        Array of polynomials evaluated at specified z points.
        First axis is polynomial degree.
    """
    if dtype is None:
        dtype = DEFAULT_GRID_DTYPE
    # Return empty array if n < 1
    if n < 1:
        return np.zeros((0, z.size), dtype=dtype)
    # Cast z to working dtype
    if np.isscalar(z):
        z = np.dtype(dtype).type(z)
        z_shape = tuple()
    else:
        z = z.astype(dtype)
        z_shape = z.shape
    # Setup initial envelope
    if init is None:
        init = 1 + 0 * z
        if normalized:
            init /= np.sqrt(mass(a, b, dtype=dtype))
    else:
        init = init + 0 * z  # Cast to working dtype
    # Get Jacobi operator
    Z = operator('Z', normalized=normalized)
    Z = dia_matrix(Z(n + 1, a, b).T).data.astype(dtype)
    # Build polynomials via recursion in degree
    shape = (n + 1,) + z_shape
    P = np.zeros(shape, dtype=dtype)
    P[0] = init
    if len(Z) == 2:
        P[1] = z * P[0] / Z[1, 1]
        for k in range(2, n + 1):
            P[k] = (z * P[k - 1] - Z[0, k - 2] * P[k - 2]) / Z[1, k]
    else:
        P[1] = (z - Z[1, 0]) * P[0] / Z[2, 1]
        for k in range(2, n + 1):
            P[k] = ((z - Z[1, k - 1]) * P[k - 1] - Z[0, k - 2] * P[k - 2]) / Z[2, k]
    # Newton update to grid
    if Newton:
        L = n + (a + b) / 2
        return z + (1 - z * z) * P[n - 1] / (L * Z[-1, n] * P[n] - (L - 1) * Z[0, n - 2] * P[n - 2]), P[: n - 1]
    return P[:n]


def quadrature(n, a, b, iterations=3, probability=False, dtype=None):
    """
    Gauss-Jacobi quadrature grid z and weights w.
    The grid points are the roots of P[n]: P(n, a, b, z[i]) = 0, len(z) = n.
    For all polynomials p up to degree 2*n - 1, the weights satisfy: sum(w[i]*p(z[i])) = integ(p).

    Parameters
    ----------
    n : int
        Number of quadrature nodes.
    a, b : float
        Jacobi parameters.
    iterations : int, optional
        Number of Newton updates. Default: 3.
    probability : bool, optional.
        True for sum(weights) = 1 or False for sum(weights) = mass(a,b). Default: True.
    dtype : dtype, optional
        Data type. Default: module-level DEFAULT_GRID_DTYPE.

    Returns
    -------
    z : array
        Quadrature grid.
    w : array
        Quadrature weights.
    """
    if dtype is None:
        dtype = DEFAULT_GRID_DTYPE
    # Initial guesses
    z = grid_guess(n, a, b, dtype=dtype)
    if probability:
        w = 1 / n + 0 * z
    else:
        m = mass(a, b, dtype=dtype)
        w = mass(a, b, dtype=dtype) / int(n) + 0 * z
    # Chebyshev T: grid guess exact, uniform weights
    if a == b == -1 / 2:
        return z, w
    # Chebyshev U: grid guess exact
    elif a == b == 1 / 2:
        P = polynomials(n, a, b, z, dtype=dtype)
    # Newton iterations for other Jacobi
    else:
        for _ in range(iterations):
            z, P = polynomials(n + 1, a, b, z, dtype=dtype, Newton=True)
    # Compute weights
    w *= n * P[0] * P[0] / np.sum(P * P, axis=0, initial=0.0)
    return z, w


def grid_guess(n, a, b, quick=False, dtype=None):
    """
    Approximate guesses for roots of P^(a,b)_n(z) using Golub-Welsch.

    Parameters
    ----------
    n : int
        Degree / number of roots in grid.
    a, b : ints or floats
        Jacobi parameters.
    quick : bool, optional
        Use a quick guess. Default: False.
    dtype : dtype
        Data type. Default: module-level DEFAULT_GRID_DTYPE.

    Returns
    -------
    grid : array
        Grid guess.
    """
    if dtype is None:
        dtype = DEFAULT_GRID_DTYPE
    # Use constant for n = 1
    if n == 1:
        return operator('Z')(n, a, b).A[0].astype(dtype)
    # Quick guess is exact for Chebyshev
    if a == b == -1 / 2:
        quick = True
    # Compute guess
    if quick:
        return np.cos(np.pi * (np.arange(4 * n - 1, 2, -4).astype(dtype) + 2 * a) / (4 * int(n) + 2 * (a + b + 1)))
    else:
        # Use Golub-Welsch
        Z = dia_matrix(operator('Z')(n, a, b))
        return eigvalsh_tridiagonal(Z.diagonal(0), Z.diagonal(1)).astype(dtype)


def measure(a, b, z, probability=True, log=False, dtype=None):
    """
    Jacobi measure evaluated on a grid. Classical/unnormalized measure:
        mu(a,b,z) = (1-z)^a (1+z)^b

    Parameters
    ----------
    a, b : ints or floats
        Jacobi parameters.
    z : float or array
        Points for evaluating measure.
    probability : bool, optional
        False for classical measure, True to renormalize to have unit integral. Default: True.
    log : bool, optional
        Return log of measure. Default: False.
    dtype : dtype, optional.
        Data type. Default: module-level DEFAULT_GRID_DTYPE.

    Returns
    -------
    measure : float or array
        Measure evaluated at a z.
    """
    if dtype is None:
        dtype = DEFAULT_GRID_DTYPE
    # Cast z to working dtype
    if np.isscalar(z):
        z = np.dtype(dtype).type(z)
    else:
        z = z.astype(dtype)
    # Compute measure
    if log:
        log_w = 0 * z
        if a:
            log_w += a * np.log(1 - z)
        if b:
            log_w += b * np.log(1 + z)
        if probability:
            log_w -= mass(a, b, log=True, dtype=dtype)
        if np.isscalar(z):
            if np.isnan(log_w):
                log_w = np.dtype(dtype).type(-np.inf)
        else:
            log_w[np.isnan(log_w)] = -np.inf  # log(0) gives nan instead of -inf
        return log_w
    else:
        w = 1 + 0 * z
        if a:
            w *= (1 - z) ** a
        if b:
            w *= (1 + z) ** b
        if probability:
            w /= mass(a, b, dtype=dtype)
        return w


def mass(a, b, log=False, dtype=None):
    """
    Integral of classical Jacobi measure:
        \integ_{-1,+1} (1-z)^a (1+z)^b dz = 2^(a+b+1) beta(a+1,b+1)

    Parameters
    __________
    a, b : ints or floats
        Jacobi parameters.
    log: bool, optional
        Return natural logarithm of mass. Default: False.
    dtype : dtype, optional
        Data type. Default: module-level DEFAULT_GRID_DTYPE.

    Returns
    -------
    mass : float
        Integral of Jacobi measure.
    """
    if dtype is None:
        dtype = DEFAULT_GRID_DTYPE
    # beta and betaln do not natively support xprec, so outputs must be casted
    type = np.dtype(dtype).type
    if log:
        return (a + b + 1) * type(np.log(2)) + type(betaln(a + 1, b + 1))
    else:
        return 2 ** (a + b + 1) * type(beta(a + 1, b + 1))


def norm_ratio(dn, da, db, n, a, b, squared=False):
    """
    Ratio of classical Jacobi normalization.

        N(n,a,b) = integrate_(-1,+1)( (1-z)**a (1+z)**b P(n,a,b,z)**2 )

                                   Gamma(n+a+1) * Gamma(n+b+1)
        N(n,a,b) = 2**(a+b+1) * ----------------------------------
                                 (2n+a+b+1) * Gamma(n+a+b+1) * n!


    The function returns: sqrt(N(n+dn,a+da,b+db)/N(n,a,b))

    This is used in rescaling the input and output of operators that increment n,a,b.

    Parameters
    ----------
    dn, da, db : int
        Increments in n, a, b
    n : int or array
        Starting polynomial degrees
    a, b : float
        Starting Jacobi parameters.
    squared : bool, optinoal
        True for direct norm ratio, False for square root of norm ratio. Default: False.

    Returns
    -------
    ratios : float or array
        Norm ratios for specified n.
    """
    if not all(type(d) == int for d in (dn, da, db)):
        raise TypeError('can only increment by integers.')

    def tricky(n, a, b):
        # 0/0 = 1
        if a + b != -1:
            return (2 * n + a + b + 1) / (n + a + b + 1)
        return 2 - (n == 0)

    def n_ratio(d, n, a, b):
        if d < 0:
            return 1 / n_ratio(-d, n + d, a, b)
        if d == 0:
            return 1 + 0 * n
        if d == 1:
            return ((n + a + 1) * (n + b + 1) / ((n + 1) * (2 * n + a + b + 3))) * tricky(n, a, b)
        return n_ratio(1, n + d - 1, a, b) * n_ratio(d - 1, n, a, b)

    def ab_ratio(d, n, a, b):
        if d < 0:
            return 1 / ab_ratio(-d, n, a + d, b)
        if d == 0:
            return 1 + 0 * n
        if d == 1:
            return (2 * (n + a + 1) / (2 * n + a + b + 2)) * tricky(n, a, b)
        return ab_ratio(1, n, a + d - 1, b) * ab_ratio(d - 1, n, a, b)

    ratio = n_ratio(dn, n, a + da, b + db) * ab_ratio(da, n, a, b + db) * ab_ratio(db, n, b, a)
    if squared:
        return ratio
    else:
        return np.sqrt(ratio)


def operator(name, **kw):
    """
    Build Jacobi operators by name.

    Parameters
    ----------
    name : str
        Jacobi operator name: 'A', 'B', 'C', 'D', 'Id', 'Pi', 'N', or 'Z'
    normalized : bool, optional
        True for unit-integral normalization, False for classical normalization. Default: True.
    **kw : dict, optional
        Other keywords passed to JacobiOperator.

    Returns
    -------
    operator : JacobiOperator
        Specified operator.
    """
    if name == 'Id':
        return JacobiOperator.identity(**kw)
    elif name == 'Pi':
        return JacobiOperator.parity(**kw)
    elif name == 'N':
        return JacobiOperator.number(**kw)
    elif name == 'Z':
        A = JacobiOperator('A', **kw)
        B = JacobiOperator('B', **kw)
        return (B(-1) @ B(+1) - A(-1) @ A(+1)) / 2
    else:
        return JacobiOperator(name, **kw)


class JacobiOperator:
    """
    Operators acting on finite row vectors of Jacobi polynomials.

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
    normalized: bool
        True gives operators on unit-integral polynomials, False on classical normalization.
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

    def __init__(self, name, normalized=True, dtype=None):
        self.__function = getattr(self, f'_JacobiOperator__{name}')
        self.__normalized = normalized
        if dtype is None:
            dtype = DEFAULT_OPERATOR_DTYPE
        self.dtype = dtype

    @property
    def normalized(self):
        return self.__normalized

    def __call__(self, p):
        return Operator(*self.__function(p))

    def __A(self, p):
        if p == 0:

            def A(n, a, b):
                N = a * np.ones(n, dtype=self.dtype)
                return infinite_csr(dia_matrix((N, [0]), (n, n)))

            return A, JacobiCodomain(0, 0, 0, 0)

        def A(n, a, b):
            N = np.arange(n, dtype=self.dtype)
            bands = np.array({+1: [N + (a + b + 1), -(N + b)], -1: [2 * (N + a), -2 * (N + 1)]}[p])
            bands[:, 0] = 1 if a + b == -1 else bands[:, 0] / (a + b + 1)
            bands[:, 1:] /= 2 * N[1:] + a + b + 1
            if self.normalized:
                bands[0] *= norm_ratio(0, p, 0, N, a, b)
                bands[1, (1 + p) // 2 :] *= norm_ratio(-p, p, 0, N[(1 + p) // 2 :], a, b)
            return infinite_csr(dia_matrix((bands, [0, p]), (max(n + (1 - p) // 2, 0), max(n, 0))))

        return A, JacobiCodomain((1 - p) // 2, p, 0, 0)

    def __B(self, p):
        def B(n, a, b):
            Pi = operator('Pi')
            return (Pi @ operator('A')(p) @ Pi)(n, a, b)

        return B, JacobiCodomain((1 - p) // 2, 0, p, 0)

    def __C(self, p):
        def C(n, a, b):
            N = np.arange(n, dtype=self.dtype)
            bands = np.array([N + {+1: b, -1: a}[p]])
            if self.normalized:
                bands[0] *= norm_ratio(0, p, -p, N, a, b)
            return infinite_csr(dia_matrix((bands, [0]), (max(n, 0), max(n, 0))))

        return C, JacobiCodomain(0, p, -p, 0)

    def __D(self, p):
        def D(n, a, b):
            N = np.arange(n, dtype=self.dtype)
            bands = np.array([(N + {+1: a + b + 1, -1: 1}[p]) * 2 ** (-p)])
            if self.normalized:
                bands[0, (1 + p) // 2 :] *= norm_ratio(-p, p, p, N[(1 + p) // 2 :], a, b)
            return infinite_csr(dia_matrix((bands, [p]), (max(n - p, 0), max(n, 0))))

        return D, JacobiCodomain(-p, p, p, 0)

    @staticmethod
    def identity(dtype=None):
        if dtype is None:
            dtype = DEFAULT_OPERATOR_DTYPE

        def I(n, a, b):
            N = np.ones(n, dtype=dtype)
            return infinite_csr(dia_matrix((N, [0]), (max(n, 0), max(n, 0))))

        return Operator(I, JacobiCodomain(0, 0, 0, 0))

    @staticmethod
    def parity(dtype=None):
        if dtype is None:
            dtype = DEFAULT_OPERATOR_DTYPE

        def P(n, a, b):
            N = np.arange(n, dtype=dtype)
            return infinite_csr(dia_matrix(((-1) ** N, [0]), (max(n, 0), max(n, 0))))

        return Operator(P, JacobiCodomain(0, 0, 0, 1))

    @staticmethod
    def number(dtype=None):
        if dtype is None:
            dtype = DEFAULT_OPERATOR_DTYPE

        def N(n, a, b):
            return infinite_csr(dia_matrix((np.arange(n, dtype=dtype), [0]), (max(n, 0), max(n, 0))))

        return Operator(N, JacobiCodomain(0, 0, 0, 0))


class JacobiCodomain(Codomain):
    """
    Jacobi codomain:
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

    def __init__(self, dn=0, da=0, db=0, pi=0, Output=None):
        if Output == None:
            Output = JacobiCodomain
        Codomain.__init__(self, *(dn, da, db, pi), Output=Output)

    def __len__(self):
        return 3

    def __str__(self):
        s = f'(n->n+{self[0]},a->a+{self[1]},b->b+{self[2]})'
        if self[3]:
            s = s.replace('a->a', 'a->b').replace('b->b', 'b->a')
        return s.replace('+0', '').replace('+-', '-')

    def __add__(self, other):
        return self.Output(*self(*other[:3], evaluate=False), self[3] ^ other[3])

    def __call__(self, *args, evaluate=True):
        n, a, b = args[:3]
        if self[3]:
            a, b = b, a
        n, a, b = self[0] + n, self[1] + a, self[2] + b
        if evaluate and (a <= -1 or b <= -1):
            raise ValueError('invalid Jacobi parameter.')
        return n, a, b

    def __neg__(self):
        a, b = -self[1], -self[2]
        if self[3]:
            a, b = b, a
        return self.Output(-self[0], a, b, self[3])

    def __eq__(self, other):
        return self[1:] == other[1:]

    def __or__(self, other):
        if self != other:
            raise TypeError('operators have incompatible codomains.')
        if self[0] >= other[0]:
            return self
        return other
