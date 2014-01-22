"""
Classes for solving differential equations.

"""

import numpy as np
import time
from scipy.sparse import linalg

from ..data.system import System
from ..data.pencil import PencilSet
from ..tools.logging import logger


class Integrator:
    """
    Initial value problem solver.

    Parameters
    ----------
    problem : problem object
        Problem describing system of differential equations and constraints
    domain : domain object
        Problem domain
    timestepper : timestepper class
        Timestepper to use in evolving initial conditions

    Attributes
    ----------
    state : system object
        System containing current solution fields
    dt : float
        Timestep
    sim_stop_time : float
        Simulation stop time, in simulation units
    wall_stop_time : float
        Wall stop time, in seconds from instantiation
    stop_iteration : int
        Stop iteration
    time : float
        Current simulation time
    iteration : int
        Current iteration

    """

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
        """Check that current time and iteration pass stop conditions."""

        if self.time >= self.sim_stop_time:
            ok_flag = False
            logger.info('Simulation stop time reached.')
        elif (time.time() - self.start_time) >= self.wall_stop_time:
            ok_flag = False
            logger.info('Wall stop time reached.')
        elif self.iteration >= self.stop_iteration:
            ok_flag = False
            logger.info('Stop iteration reached.')
        else:
            ok_flag = True

        return ok_flag

    def advance(self, dt=None):
        """Advance system by one iteration/timestep."""

        # Default to attribute-specified timestep
        if dt is None:
            dt = self.dt

        # Update pencil LHS matrices and compute RHS fields
        self.timestepper.update_pencils(dt, self.iteration)
        self.pencilset.get_system(self.rhs)

        # Solve system for each pencil
        for pencil in self.pencilset.pencils:
            LHS = pencil.LHS
            RHS = pencil.data
            np.copyto(RHS, linalg.spsolve(LHS, RHS, use_umfpack=False, permc_spec='NATURAL'))

        # Update state
        self.pencilset.set_system(self.state)

        # Update solver statistics
        self.time += dt
        self.iteration += 1

        # Aim for final time
        if self.time + self.dt > self.sim_stop_time:
            self.dt = self.sim_stop_time - self.time

