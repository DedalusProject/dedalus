

import numpy as np
import time
from scipy import sparse
from scipy.sparse import linalg

from system import System
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
        primary_basis = domain.primary_basis
        for _slice in domain.slices:
            pencil = Pencil(_slice)

            pencil.M = (sparse.kron(problem.M0, domain.primary_basis.Eval) +
                        sparse.kron(problem.M1, domain.primary_basis.Deriv))
            pencil.L = (sparse.kron(problem.L0, domain.primary_basis.Eval) +
                        sparse.kron(problem.L1, domain.primary_basis.Deriv))
            pencil.CL = sparse.kron(problem.CL, domain.primary_basis.Left)
            pencil.CR = sparse.kron(problem.CR, domain.primary_basis.Right)
            pencil.b = np.kron(problem.b, domain.primary_basis.last)

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

            # Add boundary conditions
            LHS = pencil.LHS + pencil.CL + pencil.CR
            RHS = pencil.get(self.rhs) + pencil.b

            # Solve Tau system
            X = linalg.spsolve(LHS, RHS)

            # Update state
            pencil.set(self.state, X)

        self.time += dt
        self.iteration += 1

        # Aim for final time
        if self.time + self.dt > self.sim_stop_time:
            self.dt = self.sim_stop_time - self.time

