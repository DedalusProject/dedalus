"""
Classes for solving differential equations.

"""

from mpi4py import MPI
import numpy as np
import time
from scipy.sparse import linalg

from ..data.operators import parsable_ops
from ..data.evaluator import Evaluator
from ..data.system import CoeffSystem, FieldSystem
from ..data.pencil import build_pencils
from ..data.field import Field
from ..tools.logging import logger
from ..tools.progressbar import progress


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

        # Assign axis names to bases
        for i, b in enumerate(domain.bases):
            b.name = problem.axis_names[i]

        # Build pencils and pencil matrices
        self.pencils = build_pencils(domain)
        primary_basis = domain.bases[-1]
        for pencil in self.pencils:
            pencil.build_matrices(problem, primary_basis)

        # Build systems
        self.state = FieldSystem(problem.field_names, domain)

        # Create F operator trees
        # Linear BVP: available terms are parse ops, diff ops, axes, and parameters
        vars = dict()
        vars.update(parsable_ops)
        vars.update(zip(problem.diff_names, domain.diff_ops))
        vars.update(zip(problem.axis_names, domain.grids()))
        vars.update(problem.parameters)

        self.evaluator = Evaluator(domain, vars)
        Fe_handler = self.evaluator.add_system_handler(iter=1, group='F')
        Fb_handler = self.evaluator.add_system_handler(iter=1, group='F')
        Fe_handler.add_tasks(problem.eqn_set['F'])
        Fb_handler.add_tasks(problem.bc_set['F'])
        self.Fe = Fe_handler.build_system()
        self.Fb = Fb_handler.build_system()

    def solve(self):
        """Solve BVP."""

        # Compute RHS
        self.evaluator.evaluate_group('F', 0, 0, 0)

        # Solve system for each pencil, updating state
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
    stop_sim_time : float
        Simulation stop time, in simulation units
    stop_wall_time : float
        Wall stop time, in seconds from instantiation
    stop_iteration : int
        Stop iteration
    time : float
        Current simulation time
    iteration : int
        Current iteration

    """

    def __init__(self, problem, domain, timestepper):

        # Assign axis names to bases
        for i, b in enumerate(domain.bases):
            b.name = problem.axis_names[i]

        # Build pencils and pencil matrices
        self.pencils = pencils = build_pencils(domain)
        primary_basis = domain.bases[-1]
        write_progress = (domain.distributor.rank == 0)
        for p in progress(pencils, desc='Building pencil matrices (rank 0):', write=write_progress, bar=False):
            p.build_matrices(problem, primary_basis)

        # Build systems
        self.state = state = FieldSystem(problem.field_names, domain)

        # Create F operator trees
        # IVP: available terms are parse ops, diff ops, axes, parameters, and state
        vars = dict()
        vars.update(parsable_ops)
        vars.update(zip(problem.diff_names, domain.diff_ops))
        vars.update(zip(problem.axis_names, domain.grids()))
        vars.update(problem.parameters)
        vars.update(state.field_dict)

        self._sim_time_field = Field(domain, name='sim_time')
        self._sim_time_field.constant[:] = True
        vars['t'] = self._sim_time_field

        self.evaluator = Evaluator(domain, vars)
        Fe_handler = self.evaluator.add_system_handler(iter=1, group='F')
        Fb_handler = self.evaluator.add_system_handler(iter=1, group='F')
        Fe_handler.add_tasks(problem.eqn_set['F'])
        Fb_handler.add_tasks(problem.bc_set['F'])
        self.Fe = Fe_handler.build_system()
        self.Fb = Fb_handler.build_system()

        # Initialize timestepper
        self.timestepper = timestepper(problem.nfields, domain)

        # Attributes
        self.problem = problem
        self.domain = domain
        self._wall_time_array = np.zeros(1, dtype=float)
        self.start_time = self.get_wall_time()
        self.sim_time = 0.
        self.iteration = 0

        # Default integration parameters
        self.stop_sim_time = 10.
        self.stop_wall_time = 10.
        self.stop_iteration = 10.

    @property
    def sim_time(self):
        return self._sim_time

    @sim_time.setter
    def sim_time(self, t):
        self._sim_time = t
        self._sim_time_field['g'] = t

    def get_wall_time(self):
        self._wall_time_array[0] = time.time()
        comm = self.domain.dist.comm_cart
        comm.Allreduce(MPI.IN_PLACE, self._wall_time_array, op=MPI.MAX)
        return self._wall_time_array[0]

    @property
    def ok(self):
        """Check that current time and iteration pass stop conditions."""

        if self.sim_time >= self.stop_sim_time:
            logger.info('Simulation stop time reached.')
            return False
        elif (self.get_wall_time() - self.start_time) >= self.stop_wall_time:
            logger.info('Wall stop time reached.')
            return False
        elif self.iteration >= self.stop_iteration:
            logger.info('Stop iteration reached.')
            return False
        else:
            return True

    def step(self, dt):
        """Advance system by one iteration/timestep."""

        if not np.isfinite(dt):
            raise ValueError("Invalid timestep")

        # References
        state = self.state

        # (Safety gather)
        state.gather()

        # Advance using timestepper
        wall_time = self.get_wall_time() - self.start_time
        self.timestepper.step(self, dt, wall_time)

        # (Safety scatter)
        state.scatter()

        # Update iteration
        self.iteration += 1

    def evolve(self, timestep_function):
        """Advance system until stopping criterion is reached."""

        # Check for a stopping criterion
        if np.isinf(self.stop_sim_time):
            if np.isinf(self.stop_wall_time):
                if np.isinf(self.stop_iteration):
                    raise ValueError("No stopping criterion specified.")

        # Evolve
        while self.ok:
            dt = timestep_function()
            if self.sim_time + dt > self.stop_sim_time:
                dt = self.stop_sim_time - self.sim_time
            self.step(dt)

