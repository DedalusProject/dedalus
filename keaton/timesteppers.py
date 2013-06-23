

import numpy as np
from collections import deque

from system import System


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
            mx = pencil.M.dot(pencil.get(state))
            lx = pencil.L.dot(pencil.get(state))
            f = pencil.get(state) * 0.##########################################

            pencil.set(MX[0], mx)
            pencil.set(LX[0], lx)
            pencil.set(F[0], f)

        # Compute IMEX coefficients
        a, b, c = self.compute_coefficients(iteration)

        # Construct pencil LHS matrix
        for pencil in self.pencils:
            pencil.LHS = a[-1] * pencil.M + b[-1] * pencil.L

        # Construct RHS
        for fn in rhs.field_names:
            rhs[fn]['kspace'] = 0.
            for q in xrange(self.qmax):
                rhs[fn]['kspace'] += a[q] * MX[q][fn]['kspace']
                rhs[fn]['kspace'] += b[q] * LX[q][fn]['kspace']
            for p in xrange(self.pmax):
                rhs[fn]['kspace'] += c[p] * F[p][fn]['kspace']


class CNAB3(IMEXBase):
    """Third order Crank-Nicolson-Adams-Bashforth"""

    qmax = 1
    pmax = 3

    def compute_coefficients(self, iteration):

        a = [0.] * (self.qmax+1)
        b = [0.] * (self.qmax+1)
        c = [0.] * self.pmax

        # References
        dt = self.dt

        # LHS coefficients
        a[-1] = 1.
        b[-1] = dt[0] / 2.

        # RHS coefficients
        a[0] = 1.
        b[0] = -dt[0] / 2.

        if iteration == 0:
            c[0] = 1.
        elif iteration == 1:
            c[1] = -dt[0] / (2. * dt[1])
            c[0] = 1. - c[1]
        else:
            c[2] = dt[0] * (2.*dt[0] + 3*dt[1]) / (6. * dt[2] * (dt[1] + dt[2]))
            c[1] = -(dt[0] + 2*c[2]*(dt[1] + dt[2])) / (2. * dt[1])
            c[0] = 1. - c[1] - c[2]

        c[0] *= dt[0]
        c[1] *= dt[0]
        c[2] *= dt[0]

        return a, b, c

