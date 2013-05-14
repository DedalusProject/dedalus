

import numpy as np
import time
from scipy import sparse
from scipy.sparse import linalg

from systems import System
from pencils import Pencil


class Integrator(object):
    """Tau method"""

    def __init__(self, problem, domain, timestepper):

        # Input parameters
        self.problem = problem
        self.domain = domain

        # Build systems
        self.state = System(problem.field_names, domain)
        self.rhs = System(problem.field_names, domain)

        # Build pencils
        self.pencils = []
        for _slice in domain.slices:
            pencil = Pencil(_slice)
            pencil.M0 = problem.M0
            pencil.M1 = problem.M1
            pencil.L0 = problem.L0
            pencil.L1 = problem.L1
            pencil.CL = problem.CL
            pencil.CR = problem.CR
            pencil.b = problem.b
            self.pencils.append(pencil)

        # Initialize timestepper
        self.timestepper = timestepper(self.pencils, self.state, self.rhs)

        # Integration parameters
        self.dt = 0.01
        self.sim_stop_time = 1.
        self.wall_stop_time = 60.
        self.stop_iteration = 100.

        # Instantiation time
        self.start_time = time.time()
        self.time = 0.
        self.iteration = 0

    @property
    def ok(self):

        if self.time >= self.sim_stop_time:
            ok_flag = False
            print 'Simulation stop time reached.'
        elif (time.time() - self.start_time) >= self.wall_stop_time:
            ok_flag = False
            print 'Wall stop time reached.'
        elif self.iteration >= self.stop_iteration:
            ok_flag = False
            print 'Stop iteration reached.'
        else:
            ok_flag = True

        return ok_flag

    def advance(self, dt=None):

        if dt is None:
            dt = self.dt

        # Update pencil arrays
        self.timestepper.update_pencils(dt, self.iteration)

        primary_basis = self.domain.primary_basis
        I = sparse.identity(self.problem.size)

        for pencil in self.pencils:

            # Compute Kronecker products
            A_Eval = sparse.kron(pencil.A, primary_basis.Eval)
            B_Deriv = sparse.kron(pencil.B, primary_basis.Deriv)
            CL_Left = sparse.kron(pencil.CL, primary_basis.Left)
            CR_Right = sparse.kron(pencil.CR, primary_basis.Right)

            I_E = sparse.kron(I, primary_basis.Eval)
            rhs = pencil.get(self.rhs)
            I_E_rhs = I_E.dot(rhs)
            b_last = np.kron(pencil.b, primary_basis.last)

            # Construct Tau system
            LHS = A_Eval + B_Deriv + CL_Left + CR_Right
            RHS = I_E_rhs + b_last

            # Solve Tau system
            X = linalg.spsolve(LHS, RHS)

            # Update state
            pencil.set(self.state, X)

        self.time += dt
        self.iteration += 1

        # Aim for final time
        if self.time + self.dt > self.sim_stop_time:
            self.dt = self.sim_stop_time - self.time

