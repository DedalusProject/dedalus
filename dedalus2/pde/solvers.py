"""
Classes for solving differential equations.

"""

import numpy as np
import time
from scipy.sparse import linalg

from .nonlinear import compute_expressions
from ..data.system import CoeffSystem, FieldSystem
from ..data.pencil import build_pencils
from ..tools.logging import logger


class LinearBVP:
    """
    Linear boundary value problem solver.

    Parameters
    ----------
    problem : problem object
        Problem describing system of differential equations and constraints
    domain : domain object
        Problem domain

    Attributes
    ----------
    state : system object
        System containing solution fields (after solve method is called)

    Notes
    -----
    Any problem terms with time derivatives will be dropped.

    """

    def __init__(self, problem, domain):

        # Build pencils and pencil matrices
        self.pencils = build_pencils(domain)
        primary_basis = domain.bases[-1]
        for pencil in self.pencils:
            pencil.build_matrices(problem, primary_basis)

        # Build systems
        self.state = FieldSystem(problem.field_names, domain)
        self.Fe = FieldSystem(range(problem.neqns), domain)
        self.Fb = FieldSystem(range(problem.nbc), domain)

        # Create F operator trees
        # Linear BVP: available terms are axes, parameters, and diff ops
        vars = dict()
        vars.update(zip(problem.axis_names, domain.grids()))
        vars.update(zip(problem.diff_names, domain.diff_ops))
        vars.update(problem.parameters)
        self.F_eq = [eval(f, vars) for f in problem.eqn_set['F']]
        self.F_bc = [eval(f, vars) for f in problem.bc_set['F']]

    def solve(self):
        """Solve BVP."""

        # Compute RHS
        compute_expressions(self.F_eq, self.Fe)
        compute_expressions(self.F_bc, self.Fb)

        # Solve system for each pencil, updating state
        self.Fe.gather()
        self.Fb.gather()
        for p in self.pencils:
            pFe = self.Fe.get_pencil(p)
            pFb = self.Fb.get_pencil(p)
            A = p.L
            b = p.G_eq * pFe + p.G_bc * pFb
            x = linalg.spsolve(A, b, use_umfpack=False, permc_spec='NATURAL')
            self.state.set_pencil(p, x)
        self.state.scatter()


class IVP:
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

        # Build pencils and pencil matrices
        self.pencils = pencils = build_pencils(domain)
        primary_basis = domain.bases[-1]
        for p in pencils:
            p.build_matrices(problem, primary_basis)

        # Build systems
        self.state = state = FieldSystem(problem.field_names, domain)
        self.RHS = RHS = CoeffSystem(problem.nfields, domain)
        self.Fe = Fe = FieldSystem(range(problem.neqns), domain)
        self.Fb = Fb = FieldSystem(range(problem.nbc), domain)

        # Create F operator trees
        # IVP: available terms are axes, parameters, and diff ops
        vars = dict()
        vars.update(zip(problem.axis_names, domain.grids()))
        vars.update(zip(problem.diff_names, domain.diff_ops))
        vars.update(state.field_dict)
        vars.update(problem.parameters)
        self.F_eq = [eval(f, vars) for f in problem.eqn_set['F']]
        self.F_bc = [eval(f, vars) for f in problem.bc_set['F']]

        # Initialize timestepper
        self.timestepper = timestepper(problem.nfields, domain)

        # Attributes
        self.start_time = time.time()
        self.time = 0.
        self.iteration = 0

        # Default integration parameters
        self.dt = 1.
        self.sim_stop_time = 10.
        self.wall_stop_time = 10.
        self.stop_iteration = 10.

    @property
    def ok(self):
        """Check that current time and iteration pass stop conditions."""

        if self.time >= self.sim_stop_time:
            logger.info('Simulation stop time reached.')
            return False
        elif (time.time() - self.start_time) >= self.wall_stop_time:
            logger.info('Wall stop time reached.')
            return False
        elif self.iteration >= self.stop_iteration:
            logger.info('Stop iteration reached.')
            return False
        else:
            return True

    def step(self, dt):
        """Advance system by one iteration/timestep."""

        # References
        pencils = self.pencils
        state = self.state
        RHS = self.RHS
        Fe = self.Fe
        Fb = self.Fb

        # Compute RHS
        state.gather()
        compute_expressions(self.F_eq, Fe)
        compute_expressions(self.F_bc, Fb)
        Fe.gather()
        Fb.gather()

        # Update pencil matrices
        self.timestepper.update_pencils(pencils, state, RHS, Fe, Fb, dt)

        # Solve system for each pencil, updating state
        for p in self.pencils:
            A = p.LHS
            b = RHS.get_pencil(p)
            x = linalg.spsolve(A, b, use_umfpack=False, permc_spec='NATURAL')
            state.set_pencil(p, x)
        state.scatter()

        # Update solver statistics
        self.time += dt
        self.iteration += 1

    def evolve(self, timestep_function):
        """Advance system until stopping criterion is reached."""

        # Check for a stopping criterion
        if np.isinf(self.sim_stop_time):
            if np.isinf(self.wall_stop_time):
                if np.isinf(self.stop_iteration):
                    raise ValueError("No stopping criterion specified.")

        # Evolve
        while self.ok:
            dt = timestep_function()
            if self.time + dt > self.sim_stop_time:
                dt = self.sim_stop_time - self.time
            self.step(dt)

