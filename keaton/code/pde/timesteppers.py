

import numpy as np
from collections import deque

from ..data.system import System


class IMEXBase(object):
    """Base class for implicit-explicit timesteppers."""

    def __init__(self, pencils, state, rhs):

        # Store inputs
        self.pencils = pencils
        self.state = state
        self.rhs = rhs

        # Create timestep deque
        N = max(self.qmax, self.pmax)
        self.dt = deque([0.]*N)

        # Create RHS components
        field_names = state.field_names
        domain = state.domain

        self.MX = deque()
        self.LX = deque()
        self.F = deque()

        for q in xrange(self.qmax):
            self.MX.append(System(field_names, domain))
            self.LX.append(System(field_names, domain))
        for p in xrange(self.pmax):
            self.F.append(System(field_names, domain))

    def update_pencils(self, dt, iteration):

        # References
        state = self.state
        rhs = self.rhs
        MX = self.MX
        LX = self.LX
        F = self.F

        # Cycle and compute timesteps
        self.dt.rotate()
        self.dt[0] = dt

        # Cycle and compute RHS components
        MX.rotate()
        LX.rotate()
        F.rotate()

        for pencil in self.pencils:

            # (Assuming no coupling between pencils)
            mx = pencil.M.dot(state[pencil])
            lx = pencil.L.dot(state[pencil])
            # Need to add RHS (for eqns and BCs)
            f = state[pencil] * 0. + pencil.b###################################

            MX[0][pencil] = mx
            LX[0][pencil] = lx
            F[0][pencil] = f

        # Compute IMEX coefficients
        a, b, c, d = self.compute_coefficients(iteration)

        # Construct pencil LHS matrix
        for pencil in self.pencils:
            pencil.LHS = d[0] * pencil.M + d[1] * pencil.L

        # Construct RHS field
        for fn in rhs.field_names:
            rhs[fn]['k'] = 0.
            for q in xrange(self.qmax):
                rhs[fn]['k'] += a[q] * MX[q][fn]['k']
                rhs[fn]['k'] += b[q] * LX[q][fn]['k']
            for p in xrange(self.pmax):
                rhs[fn]['k'] += c[p] * F[p][fn]['k']


class CNAB3(IMEXBase):
    """Third order Crank-Nicolson-Adams-Bashforth"""

    qmax = 1
    pmax = 3

    def compute_coefficients(self, iteration):

        a = [0.] * (self.qmax+1)
        b = [0.] * (self.qmax+1)
        c = [0.] * self.pmax
        d = [0.] * 2

        # References
        dt0 = self.dt[0]
        dt1 = self.dt[1]
        dt2 = self.dt[2]

        # LHS coefficients
        d[0] = 1.
        d[d] = dt0 / 2.

        # RHS coefficients
        a[0] = 1.
        b[0] = -dt0 / 2.

        print self.dt

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


class SimpleSolve(IMEXBase):
    """Simple BVP solve"""

    qmax = 1
    pmax = 1

    def compute_coefficients(self, iteration):

        a = [0.]
        b = [0.]
        c = [1.]
        d = [0., 1.]

        return a, b, c, d
