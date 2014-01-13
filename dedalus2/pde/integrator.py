

import numpy as np
import time
from scipy.sparse import linalg

from ..data.system import System
from ..data.pencil import PencilSet


class Integrator:
    """Initial value problem solver."""

    def __init__(self, problem, domain, timestepper):

        # Input parameters
        self.problem = problem
        self.domain = domain

        # Build systems
        self.state = System(problem.field_names, domain)
        self.rhs = System(problem.field_names, domain)

        # Build pencilset and pencil matrices
        self.pencilset = PencilSet(domain, problem.size)
        primary_basis = domain.bases[-1]
        for pencil in self.pencilset.pencils:
            pencil.build_matrices(problem, primary_basis)

        # Initialize timestepper
        self.timestepper = timestepper(problem, self.pencilset, self.state, self.rhs)

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
            print('Simulation stop time reached.')
        elif (time.time() - self.start_time) >= self.wall_stop_time:
            ok_flag = False
            print('Wall stop time reached.')
        elif self.iteration >= self.stop_iteration:
            ok_flag = False
            print('Stop iteration reached.')
        else:
            ok_flag = True

        return ok_flag

    def advance(self, dt=None):

        if dt is None:
            dt = self.dt

        # Update pencil arrays
        self.timestepper.update_pencils(dt, self.iteration)

        self.pencilset.get_system(self.rhs)
        for pencil in self.pencilset.pencils:
            # Solve Tau system
            LHS = pencil.LHS
            RHS = pencil.data
            pencil.data = linalg.spsolve(LHS, RHS)

        # Update state
        self.pencilset.set_system(self.state)

        self.time += dt
        self.iteration += 1

        # Aim for final time
        if self.time + self.dt > self.sim_stop_time:
            self.dt = self.sim_stop_time - self.time

