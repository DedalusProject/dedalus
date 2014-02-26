"""
ODE solvers for timestepping.

"""

from collections import deque
import numpy as np

from ..data.system import CoeffSystem, FieldSystem


class IMEXBase:
    """
    Base class for implicit-explicit timesteppers.

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
        ai M.X(n+1) + bi L.X(n+1) = aj M.X(n-j) + bj L.X(n-j) + cj F(n-j)
    where j runs from 0 to {amax-1, bmax-1, cmax-1} in each sum, respectively.

    """

    def __init__(self, nfields, domain):

        # Create deque for storing recent timesteps
        N = max(self.amax, self.bmax, self.cmax)
        self.dt = deque([0.]*N)

        # Create systems for multistep history
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
        ai, bi, a, b, c = self.compute_coefficients(self._iteration)
        self._iteration += 1

        # Update RHS components and LHS matrices
        MX.rotate()
        LX.rotate()
        F.rotate()

        MX0 = MX[0]
        LX0 = LX[0]
        F0 = F[0]
        for p in pencils:
            x = state.get_pencil(p)
            pFe = Fe.get_pencil(p)
            pFb = Fb.get_pencil(p)

            MX0.set_pencil(p, p.M*x)
            LX0.set_pencil(p, p.L*x)
            F0.set_pencil(p, p.G_eq*pFe + p.G_bc*pFb)

            np.copyto(p.LHS.data, ai*p.M.data + bi*p.L.data)

        # Build RHS
        RHS.data.fill(0)
        for j in range(self.amax):
            RHS.data += a[j] * MX[j].data
        for j in range(self.bmax):
            RHS.data += b[j] * LX[j].data
        for j in range(self.cmax):
            RHS.data += c[j] * F[j].data


class Euler(IMEXBase):
    """Backward-Euler/Forward-Euler"""

    amax = 1
    bmax = 1
    cmax = 1

    def compute_coefficients(self, iteration):

        a = [0.] * self.amax
        b = [0.] * self.bmax
        c = [0.] * self.cmax

        # References
        dt = self.dt[0]

        # LHS coefficients
        ai = 1.
        bi = dt

        # RHS coefficients
        a[0] = 1.
        b[0] = 0.
        c[0] = dt

        return ai, bi, a, b, c


class CNAB3(IMEXBase):
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


class MCNAB2(IMEXBase):
    """Second order Modified Crank-Nicolson-Adams-Bashforth with variable timestep size (VSIMEX)"""

    amax = 1
    bmax = 2
    cmax = 2

    def compute_coefficients(self, iteration):

        a = [0.] * self.amax
        b = [0.] * self.bmax
        c = [0.] * self.cmax

        # References
        dt0 = self.dt[0]
        dt1 = self.dt[1]

        # LHS coefficients
        ai = 1.
        bi = dt0 * (1./16.)*(8. + (dt1/dt0))

        # RHS coefficients
        a[0] = 1.

        if iteration == 0:
            b[0] = -dt0 / 2.
            c[0] = 1.

        else:
            b[0] = -dt0 * (1./16.)*(7. - (dt1/dt0))
            b[1] = -dt0 * (1./16.)
            c[1] = -1./2. * (dt0 / dt1)
            c[0] = 1. + 1./2. *(dt0 / dt1)

        c[0] *= dt0
        c[1] *= dt0

        return ai, bi, a, b, c

