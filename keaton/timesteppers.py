

import numpy as np
from scipy import sparse

from system import System


class TimeStepper(object):
    """Base class for timesteppers"""

    def __init__(self, pencils, state, rhs):

        # Store inputs
        self.pencils = pencils
        self.state = state
        self.rhs = rhs

        # Create derivative system
        field_names = state.field_names
        domain = state.domain
        self.deriv = System(field_names, domain)


class CNAB3(TimeStepper):
    """Third order Crank-Nicolson-Adams-Bashforth"""

    def __init__(self, pencils, state, rhs):

        # Inherited initialization
        TimeStepper.__init__(self, pencils, state, rhs)

        # Create F systems
        field_names = state.field_names
        domain = state.domain
        self.F0 = System(field_names, domain)
        self.F1 = System(field_names, domain)
        self.F2 = System(field_names, domain)

        self.dt0 = self.dt1 = self.dt2 = 0

    def update_pencils(self, dt, iteration):

        # References
        state = self.state
        deriv = self.deriv
        rhs = self.rhs

        # Compute derivative
        for field_name in state.field_names:
            deriv[field_name]['kspace'] = state[field_name].differentiate()

        # Cycle F references
        F0, F1, F2 = self.F2, self.F0, self.F1
        self.F0, self.F1, self.F2 = F0, F1, F2

        # Compute F0
        ########################################################################

        # Cycle timesteps
        dt0, dt1, dt2 = dt, self.dt0, self.dt1
        self.dt0, self.dt1, self.dt2 = dt0, dt1, dt2

        # Calculate Adams-Bashforth coefficients
        if iteration < 2:
            alpha2 = alpha1 = 0.
            alpha0 = 1.
        else:
            alpha2 = dt0 * (2.*dt0 + 3*dt1) / 6. / dt2 / (dt1 + dt2)
            alpha1 = (dt0 + 2*alpha2*(dt1 + dt2)) / 2. / dt1
            alpha0 = 1. + alpha1 - alpha2

        I = sparse.identity(state.domain.primary_basis.size)

        # Update pencil arrays
        for pencil in self.pencils:

            pencil.A = pencil.M0 + dt / 2. * pencil.L0
            pencil.B = pencil.M1 + dt / 2. * pencil.L1

            C1 = pencil.M0 - dt / 2. * pencil.L0
            C2 = pencil.M1 - dt / 2. * pencil.L1

            f = sparse.kron(C1, I).dot(pencil.get(state))
            f += sparse.kron(C2, I).dot(pencil.get(deriv))
            f += dt * (alpha0 * pencil.get(F0) +
                       alpha1 * pencil.get(F1) +
                       alpha2 * pencil.get(F2))

            pencil.set(rhs, f)

