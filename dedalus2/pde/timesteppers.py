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

    """

    def __init__(self, nfields, domain):

        # Create deque for storing recent timesteps
        N = max(self.qmax, self.pmax)
        self.dt = deque([0.]*N)

        # Create systems for multistep history
        self.MX = MX = deque()
        self.LX = LX = deque()
        self.F = F = deque()
        for q in range(self.qmax):
            MX.append(CoeffSystem(nfields, domain))
            LX.append(CoeffSystem(nfields, domain))
        for p in range(self.pmax):
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
        a, b, c, d = self.compute_coefficients(self._iteration)
        self._iteration += 1

        # Update RHS components and LHS matrices
        MX.rotate()
        LX.rotate()
        F.rotate()

        MX0 = MX[0]
        LX0 = LX[0]
        F0 = F[0]
        d0, d1 = d
        for p in pencils:
            x = state.get_pencil(p)
            pFe = Fe.get_pencil(p)
            pFb = Fb.get_pencil(p)

            MX0.set_pencil(p, p.M*x)
            LX0.set_pencil(p, p.L*x)
            F0.set_pencil(p, p.G_eq*pFe + p.G_bc*pFb)

            np.copyto(p.LHS.data, d0*p.M.data + d1*p.L.data)

        # Build RHS
        RHS.data.fill(0)
        for q in range(self.qmax):
            RHS.data += a[q] * MX[q].data
            RHS.data += b[q] * LX[q].data
        for p in range(self.pmax):
            RHS.data += c[p] * F[p].data


class Euler(IMEXBase):
    """Backward-Euler/Forward-Euler"""

    qmax = 1
    pmax = 1

    def compute_coefficients(self, iteration):

        a = [0.] * self.qmax
        b = [0.] * self.qmax
        c = [0.] * self.pmax
        d = [0.] * 2

        # References
        dt = self.dt[0]

        # LHS coefficients
        d[0] = 1.
        d[1] = dt

        # RHS coefficients
        a[0] = 1.
        b[0] = 0.
        c[0] = dt

        return a, b, c, d


class CNAB3(IMEXBase):
    """Third order Crank-Nicolson-Adams-Bashforth"""

    qmax = 1
    pmax = 3

    def compute_coefficients(self, iteration):

        a = [0.] * self.qmax
        b = [0.] * self.qmax
        c = [0.] * self.pmax
        d = [0.] * 2

        # References
        dt0 = self.dt[0]
        dt1 = self.dt[1]
        dt2 = self.dt[2]

        # LHS coefficients
        d[0] = 1.
        d[1] = dt0 / 2.

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

        return a, b, c, d


class MCNAB2(IMEXBase):
    """Second order Modified Crank-Nicolson-Adams-Bashforth with variable timestep size (VSIMEX)"""

    qmax = 2   # 2nd order in implicit operator
    pmax = 2   # 2nd order in explicit operator

    def compute_coefficients(self, iteration):

        a = [0.] * self.qmax
        b = [0.] * self.qmax
        c = [0.] * self.pmax
        d = [0.] * 2

        # References
        dt0 = self.dt[0]
        dt1 = self.dt[1]

        # LHS coefficients
        d[0] = 1.
        d[1] = dt0 * (1./16.)*(8. + (dt1/dt0))

        # RHS coefficients
        a[0] = 1.    # u_n
        a[1] = 0.    # no u_n-1

        # the c coefs are the "explicit" part
        # same as in CNAB2
        if iteration == 0:
            # do CNAB2 for first step
            c[0] = 1.
            c[1] = 0.
            b[0] = -dt0/2.
            b[1] = 0.
        else:
            c[1] = -1./2. * (dt0 / dt1)
            c[0] = 1. + 1./2. *(dt0 / dt1)
            b[0] = -dt0 * (1./16.)*(7. - (dt1/dt0)) # This is L_{n}
            b[1] = -dt0 * (1./16.)                  # This is L_{n-1}


        c[0] *= dt0
        c[1] *= dt0

        return a, b, c, d

