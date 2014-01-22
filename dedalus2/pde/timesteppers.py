"""
ODE solvers for timestepping.

"""

import numpy as np
from collections import deque

from .nonlinear import compute_expressions
from ..data.system import System


class IMEXBase:
    """
    Base class for implicit-explicit timesteppers.

    Parameters
    ----------
    problem : problem object
        Problem describing system of differential equations and constraints
    pencilset : pencilset object
        Pencilset for problem domain
    state : system object
        System containing current solution fields
    rhs : system object
        System for storing the RHS fields

    """

    def __init__(self, problem, pencilset, state, rhs):

        # Initial attributes
        self.pencilset = pencilset
        self.state = state
        self.rhs = rhs

        # Create deque for storing recent timesteps
        N = max(self.qmax, self.pmax)
        self.dt = deque([0.]*N)

        # Create systems needed for multistep history
        field_names = state.field_names
        domain = state.domain

        self.MX = deque()
        self.LX = deque()
        self.F = deque()

        for q in range(self.qmax):
            self.MX.append(System(field_names, domain))
            self.LX.append(System(field_names, domain))
        for p in range(self.pmax):
            self.F.append(System(field_names, domain))

        # Create F operator trees for string representations
        self.F_expressions = []
        namespace = dict()
        namespace.update(state.fields)
        namespace.update(problem.parameters)
        for f in problem.F:
            if f is None:
                self.F_expressions.append(None)
            else:
                self.F_expressions.append(eval(f, namespace))

    def update_pencils(self, dt, iteration):
        """Compute elements for the implicit solve, by pencil."""

        # References
        pencilset = self.pencilset
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

        pencilset.get_system(state)
        for pencil in pencilset.pencils:
            np.copyto(pencil.data, pencil.M.dot(pencil.data))
        pencilset.set_system(MX[0])

        pencilset.get_system(state)
        for pencil in pencilset.pencils:
            np.copyto(pencil.data, pencil.L.dot(pencil.data))
        pencilset.set_system(LX[0])

        # Compute nonlinear component
        compute_expressions(self.F_expressions, F[0])

        pencilset.get_system(F[0])
        for pencil in pencilset.pencils:
            np.copyto(pencil.data, pencil.F_eval.dot(pencil.data))
            for i, r in enumerate(pencil.bc_rows):
                pencil.data[r] = pencil.bc_f[i]
        pencilset.set_system(F[0])

        # Compute IMEX coefficients
        a, b, c, d = self.compute_coefficients(iteration)

        # Construct pencil LHS matrix
        for pencil in pencilset.pencils:
            np.copyto(pencil.LHS.data, d[0]*pencil.M.data + d[1]*pencil.L.data)

        # Construct RHS field
        for fn in rhs.field_names:
            rhs[fn]['c'] = 0.
            for q in range(self.qmax):
                rhs[fn]['c'] += a[q] * MX[q][fn]['c']
                rhs[fn]['c'] += b[q] * LX[q][fn]['c']
            for p in range(self.pmax):
                rhs[fn]['c'] += c[p] * F[p][fn]['c']


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

