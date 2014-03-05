"""
ODE solvers for timestepping.

"""

from collections import deque
import numpy as np

from ..data.system import CoeffSystem, FieldSystem


class MultistepIMEX:
    """
    Base class for implicit-explicit multistep methods.

    Parameters
    ----------
    nfields : int
        Number of fields in problem
    domain : domain object
        Problem domain

    Notes
    -----
    These timesteppers discretize the system
        M.dt(X) + L.X = F
    into the general form
        aj M.X(n-j) + bj L.X(n-j) = cj F(n-j)
    where j runs from {0, 0, 1} to {amax, bmax, cmax}.

    The system is then solved as
        (a0 M + b0 L).X(n) = cj F(n-j) - aj M.X(n-j) - bj L.X(n-j)
    where j runs from {1, 1, 1} to {cmax, amax, bmax}.

    References
    ----------
    D. Wang and S. J. Ruuth, Journal of Computational Mathematics 26, (2008).*

    * Our coefficients are related to those used by Wang as:
        amax = bmax = cmax = s
        aj = α(s-j) / k(n+s-1)
        bj = γ(s-j)
        cj = β(s-j)

    """

    def __init__(self, nfields, domain):

        # Create deque for storing recent timesteps
        N = max(self.amax, self.bmax, self.cmax)
        self.dt = deque([0.]*N)

        # Create coefficient systems for multistep history
        self.MX = MX = deque()
        self.LX = LX = deque()
        self.F = F = deque()
        for j in range(self.amax):
            MX.append(CoeffSystem(nfields, domain))
        for j in range(self.bmax):
            LX.append(CoeffSystem(nfields, domain))
        for j in range(self.cmax):
            F.append(CoeffSystem(nfields, domain))

        # Attributes
        self._iteration = 0

    def update_pencils(self, pencils, state, RHS, Fe, Fb, dt):
        """Compute elements for the implicit solve, by pencil."""

        # References
        MX = self.MX
        LX = self.LX
        F = self.F

        # Cycle and compute timesteps
        self.dt.rotate()
        self.dt[0] = dt

        # Compute IMEX coefficients
        a, b, c = self.compute_coefficients(self.dt, self._iteration)
        self._iteration += 1

        # Update RHS components and LHS matrices
        MX.rotate()
        LX.rotate()
        F.rotate()

        MX0 = MX[0]
        LX0 = LX[0]
        F0 = F[0]
        a0 = a[0]
        b0 = b[0]
        for p in pencils:
            x = state.get_pencil(p)
            pFe = Fe.get_pencil(p)
            pFb = Fb.get_pencil(p)

            MX0.set_pencil(p, p.M*x)
            LX0.set_pencil(p, p.L*x)
            F0.set_pencil(p, p.G_eq*pFe + p.G_bc*pFb)

            np.copyto(p.LHS.data, a0*p.M.data + b0*p.L.data)

        # Build RHS
        RHS.data.fill(0)
        for j in range(1, len(c)):
            RHS.data += c[j] * F[j-1].data
        for j in range(1, len(a)):
            RHS.data -= a[j] * MX[j-1].data
        for j in range(1, len(b)):
            RHS.data -= b[j] * LX[j-1].data


class Euler(MultistepIMEX):
    """First-order forward/backward Euler [Wang 2008 eqn 2.6]"""

    amax = 1
    bmax = 1
    cmax = 1

    @classmethod
    def compute_coefficients(self, timesteps, iteration):

        a = np.zeros(self.amax+1)
        b = np.zeros(self.bmax+1)
        c = np.zeros(self.cmax+1)

        k1, *rest = timesteps

        a[0] = 1 / k1
        a[1] = -1 / k1
        b[0] = 1
        c[1] = 1

        return a, b, c


class SBDF2(MultistepIMEX):
    """Second-order semi-implicit BDF [Wang 2008 eqn 2.8]"""

    amax = 2
    bmax = 2
    cmax = 2

    @classmethod
    def compute_coefficients(self, timesteps, iteration):

        if iteration < 1:
            return Euler.compute_coefficients(timesteps, iteration)

        a = np.zeros(self.amax+1)
        b = np.zeros(self.bmax+1)
        c = np.zeros(self.cmax+1)

        k1, k2, *rest = timesteps
        w1 = k1 / k2

        a[0] = (1 + 2*w1) / (1 + w1) / k1
        a[1] = -(1 + w1) / k1
        a[2] = w1*w1 / (1 + w1) / k1
        b[0] = 1
        c[1] = 1 + w1
        c[2] = -w1

        return a, b, c


class CNAB(MultistepIMEX):
    """Second-order Crank-Nicolson-Adams-Bashforth [Wang 2008 eqn 2.9]"""

    amax = 2
    bmax = 2
    cmax = 2

    @classmethod
    def compute_coefficients(self, timesteps, iteration):

        if iteration < 1:
            return Euler.compute_coefficients(timesteps, iteration)

        a = np.zeros(self.amax+1)
        b = np.zeros(self.bmax+1)
        c = np.zeros(self.cmax+1)

        k1, k2, *rest = timesteps
        w1 = k1 / k2

        a[0] = 1 / k1
        a[1] = -1 / k1
        b[0] = 1 / 2
        b[1] = 1 / 2
        c[1] = 1 + w1 / 2
        c[2] = -w1 / 2

        return a, b, c


class MCNAB(MultistepIMEX):
    """Second-order modified Crank-Nicolson-Adams-Bashforth [Wang 2008 eqn 2.10]"""

    amax = 2
    bmax = 2
    cmax = 2

    @classmethod
    def compute_coefficients(self, timesteps, iteration):

        if iteration < 1:
            return Euler.compute_coefficients(timesteps, iteration)

        a = np.zeros(self.amax+1)
        b = np.zeros(self.bmax+1)
        c = np.zeros(self.cmax+1)

        k1, k2, *rest = timesteps
        w1 = k1 / k2

        a[0] = 1 / k1
        a[1] = -1 / k1
        b[0] = (8 + 1/w1) / 16
        b[1] = (7 - 1/w1) / 16
        b[2] = 1 / 16
        c[1] = 1 + w1 / 2
        c[2] = -w1 / 2

        return a, b, c


class CNLF(MultistepIMEX):
    """Second-order Crank-Nicolson-Leap-Frog [Wang 2008 eqn 2.11]"""

    amax = 2
    bmax = 2
    cmax = 2

    @classmethod
    def compute_coefficients(self, timesteps, iteration):

        if iteration < 1:
            return Euler.compute_coefficients(timesteps, iteration)

        a = np.zeros(self.amax+1)
        b = np.zeros(self.bmax+1)
        c = np.zeros(self.cmax+1)

        k1, k2, *rest = timesteps
        w1 = k1 / k2

        a[0] = 1 / (1 + w1) / k1
        a[1] = (w1 - 1) / k1
        a[2] = -w1*w1 / (1 + w1) / k1
        b[0] = 1 / w1 / 2
        b[1] = (1 - 1/w1) / 2
        b[2] = 1 / 2
        c[1] = 1

        return a, b, c


class CNAB3(MultistepIMEX):
    """Third order Crank-Nicolson-Adams-Bashforth"""

    amax = 1
    bmax = 1
    cmax = 3

    def compute_coefficients(self, iteration):

        a = [0.] * self.amax
        b = [0.] * self.bmax
        c = [0.] * self.cmax

        # References
        dt0 = self.dt[0]
        dt1 = self.dt[1]
        dt2 = self.dt[2]

        # LHS coefficients
        ai = 1.
        bi = dt0 / 2.

        # RHS coefficients
        a[0] = 1.
        b[0] = -dt0 / 2.

        if iteration == 0:
            c[0] = 1.
        elif iteration == 1:
            c[1] = -dt0 / (2. * dt1)
            c[0] = 1. - c[1]
        else:
            c[2] = dt0 * (2.*dt0 + 3.*dt1) / (6.*dt2*(dt1 + dt2))
            c[1] = -(dt0 + 2.*c[2]*(dt1 + dt2)) / (2. * dt1)
            c[0] = 1. - c[1] - c[2]

        c[0] *= dt0
        c[1] *= dt0
        c[2] *= dt0

        return ai, bi, a, b, c

