"""Spin-weighted spherical harmonic functions."""

import numpy as np
from scipy.sparse import dia_matrix
from . import jacobi
from .operators import Operator, infinite_csr


def quadrature(Lmax, **kw):
    """
    Gauss quadrature grid and weights for SWSH transform.
    Will exactly integrate polynomials up to degree 2*Lmax+1.

    Parameters
    ----------
    Lmax : int
        Maximum spherical harmonic degree.
    **kw : dict, optional
        Other keywords passed to jacobi.quadrature.

    Returns
    -------
    cos_theta : array
        Quadrature grid in z = cos(theta).
    w : array
        Quadrature weights.
    """
    return jacobi.quadrature(Lmax + 1, a=0, b=0, **kw)


def spin2Jacobi(Lmax, m, s, ds=None, dm=None):
    """Compute Jacobi parameters from SWSH parameters."""
    n = Lmax + 1 - max(abs(m), abs(s))
    a, b = abs(m + s), abs(m - s)
    if ds is None and dm is None:
        return n, a, b
    if ds is None:
        ds = 0
    if dm is None:
        dm = 0
    m += dm
    s += ds
    dn = Lmax + 1 - max(abs(m), abs(s)) - n
    da, db = abs(m + s) - a, abs(m - s) - b
    return n, a, b, dn, da, db


def harmonics(Lmax, m, s, cos_theta, **kw):
    """
    SWSH of type (m,s) up to degree Lmax.

    Parameters
    ----------
    Lmax : int
        Maximum spherical-harmonic degree.
    m, s : int
        SWSH parameters.
    cos_theta : float or array
        Grid array in z = cos(theta).
    **kw : dict, optional
        Other keywords passed to jacobi.polynomials.

    Returns
    -------
    polynomials : array
        Array of polynomials evaluated at specified z points.
        First axis is polynomial degree.
    """
    # Compute Jacobi parameters
    n, a, b = spin2Jacobi(Lmax, m, s)
    # Build envelope
    if np.isscalar(cos_theta):
        # Call non-log version to avoid issues with |z| = 1
        init = np.sqrt(jacobi.measure(a, b, cos_theta, dtype=np.float64))
    else:
        init = np.exp(0.5 * jacobi.measure(a, b, cos_theta, log=True))
    init *= (-1.0) ** max(m, -s)
    # Build polynomials
    return jacobi.polynomials(n, a, b, cos_theta, init, **kw)


def operator(name, **kw):
    """
    Build SWSH operators by name.

    Parameters
    ----------
    name : str
        Sphere operator name: 'D', 'Sin', 'Cos', 'Id', 'Pi', 'L', 'M', or 'S'
    **kw : dict, optional
        Other keywords passed to SphereOperator or JacobiOperator.

     Returns
    -------
    operator : SphereOperator or JacobiOperator
        Specified operator.
    """
    if name == 'Id':
        return SphereOperator.identity(**kw)
    if name == 'Pi':
        return SphereOperator.parity(**kw)
    if name == 'L':
        return SphereOperator.L(**kw)
    if name == 'M':
        return SphereOperator.M(**kw)
    if name == 'S':
        return SphereOperator.S(**kw)
    if name == 'Cos':

        def Cos(Lmax, m, s):
            return jacobi.operator('Z', **kw)(*spin2Jacobi(Lmax, m, s))
            #return Jacobi.operator('Z',dtype=dtype)(Lmax+1, abs(m+s), abs(m-s))

        return Operator(Cos, SphereCodomain(1, 0, 0, 0))
    return SphereOperator(name, **kw)


class SphereOperator:
    """Operators acting on finite row vectors of SWSH."""

    def __init__(self, name, radius=1, dtype=None):
        self.__function = getattr(self, f'_SphereOperator__{name}')
        self.__radius = radius
        if dtype is None:
            dtype = jacobi.DEFAULT_OPERATOR_DTYPE
        self.__dtype = dtype

    def __call__(self, ds):
        return Operator(*self.__function(ds))

    @property
    def radius(self):
        return self.__radius

    @property
    def dtype(self):
        return self.__dtype

    def __D(self, ds):
        def D(Lmax, m, s):
            n, a, b, dn, da, db = spin2Jacobi(Lmax, m, s, ds=ds)
            D = jacobi.operator('C' if da + db == 0 else 'D', dtype=self.dtype)(da)
            return (-ds * np.sqrt(0.5) / self.radius) * D(n, a, b)

        return D, SphereCodomain(0, 0, ds, 0)

    def __Sin(self, ds):
        def Sin(Lmax, m, s):
            n, a, b, dn, da, db = spin2Jacobi(Lmax, m, s, ds=ds)
            S = jacobi.operator('A', dtype=self.dtype)(da)
            S = S @ jacobi.operator('B', dtype=self.dtype)(db)
            return (da * ds) * S(n, a, b)

        return Sin, SphereCodomain(1, 0, ds, 0)

    @staticmethod
    def identity(dtype=None):
        if dtype is None:
            dtype = jacobi.DEFAULT_OPERATOR_DTYPE

        def I(Lmax, m, s):
            n = spin2Jacobi(Lmax, m, s)[0]
            N = np.ones(n, dtype=dtype)
            return infinite_csr(dia_matrix((N, [0]), (max(n, 0), max(n, 0))))

        return Operator(I, SphereCodomain(0, 0, 0, 0))

    @staticmethod
    def parity(dtype=None):
        if dtype is None:
            dtype = jacobi.DEFAULT_OPERATOR_DTYPE

        def Pi(Lmax, m, s):
            return jacobi.operator('Pi', dtype=dtype)(*spin2Jacobi(Lmax, m, s))

        return Operator(Pi, SphereCodomain(0, 0, 0, 1))

    @staticmethod
    def L(dtype=None):
        if dtype is None:
            dtype = jacobi.DEFAULT_OPERATOR_DTYPE

        def L(Lmax, m, s):
            n = spin2Jacobi(Lmax, m, s)[0]
            N = np.arange(Lmax + 1 - n, Lmax + 1, dtype=dtype)
            return infinite_csr(dia_matrix((N, [0]), (max(n, 0), max(n, 0))))

        return Operator(L, SphereCodomain(0, 0, 0, 0))

    @staticmethod
    def M(dtype=None):
        if dtype is None:
            dtype = jacobi.DEFAULT_OPERATOR_DTYPE

        def M(Lmax, m, s):
            n = spin2Jacobi(Lmax, m, s)[0]
            N = m * np.ones(n, dtype=dtype)
            return infinite_csr(dia_matrix((N, [0]), (max(n, 0), max(n, 0))))

        return Operator(M, SphereCodomain(0, 0, 0, 0))

    @staticmethod
    def S(dtype=None):
        if dtype is None:
            dtype = jacobi.DEFAULT_OPERATOR_DTYPE

        def S(Lmax, m, s):
            n = spin2Jacobi(Lmax, m, s)[0]
            N = s * np.ones(n, dtype=dtype)
            return infinite_csr(dia_matrix((N, [0]), (max(n, 0), max(n, 0))))

        def S(Lmax,m,s):
            n = spin2Jacobi(Lmax,m,s)[0]
            N = abs(s)*np.ones(n,dtype=dtype)
            return infinite_csr(banded((N,[0]),(max(n,0),max(n,0))))

        return Operator(S,SphereCodomain(0,0,0,0))


class SphereCodomain(jacobi.JacobiCodomain):
    """Sphere codomain."""

    def __init__(self, dL=0, dm=0, ds=0, pi=0):
        jacobi.JacobiCodomain.__init__(self, *(dL, dm, ds, pi), Output=SphereCodomain)

    def __str__(self):
        s = f'(L->L+{self[0]},m->m+{self[1]},s->s+{self[2]})'
        if self[3]:
            s = s.replace('s->s', 's->-s')
        return s.replace('+0', '').replace('+-', '-')

    def __call__(self, *args, evaluate=True):
        L, m, s = args[:3]
        if self[3]:
            s *= -1
        return self[0] + L, self[1] + m, self[2] + s

    def __neg__(self):
        m, s = -self[1], -self[2]
        if self[3]:
            s *= -1
        return SphereCodomain(-self[0], m, s, self[3])
