"""
Classes for solving differential equations.

"""

from mpi4py import MPI
import numpy as np
import time
from scipy.sparse import linalg
from scipy.linalg import eig

#from ..data.operators import parsable_ops
from . import operators
from . import subsystems
from .evaluator import Evaluator
from .system import CoeffSystem, FieldSystem
from .field import Field
from ..tools.progress import log_progress

from ..tools.config import config
PERMC_SPEC = config['linear algebra']['permc_spec']
USE_UMFPACK = config['linear algebra'].getboolean('use_umfpack')

import logging
logger = logging.getLogger(__name__.split('.')[-1])


class EigenvalueSolver:
    """
    Solves linear eigenvalue problems for oscillation frequency omega, (d_t -> -i omega).
    First converts to dense matrices, then solves the eigenvalue problem for a given pencil,
    and stored the eigenvalues and eigenvectors.  The set_state method can be used to set
    the state to the ith eigenvector.

    Parameters
    ----------
    problem : problem object
        Problem describing system of differential equations and constraints

    Attributes
    ----------
    state : system object
        System containing solution fields (after solve method is called)
    eigenvalues : numpy array
        Contains a list of eigenvalues omega
    eigenvectors : numpy array
        Contains a list of eigenvectors.  The eigenvector corresponding to the ith
        eigenvalue is in eigenvectors[..., i]
    eigenvalue_pencil : pencil
        The pencil for which the eigenvalue problem has been solved.

    """

    def __init__(self, problem):

        logger.debug('Beginning EVP instantiation')

        self.problem = problem
        self.domain = domain = problem.domain

        # Build subproblems and subproblem matrices
        self.subproblems = subsystems.build_local_subproblems(problem)

        # Build systems
        namespace = problem.namespace
        #vars = [namespace[var] for var in problem.variables]
        #self.state = FieldSystem.from_fields(vars)
        self.state = problem.variables

        # Create F operator trees
        self.evaluator = Evaluator(domain, namespace)

        logger.debug('Finished EVP instantiation')

    def solve(self, subproblem):
        """Solve EVP."""
        self.eigenvalue_subproblem = subproblem
        subsystems.build_matrices([subproblem], ['L','M'])
        L = subproblem.L_min.todense()
        M = subproblem.M_min.todense()
        self.eigenvalues, self.eigenvectors = eig(L, b=-M)

    def set_state(self, num):
        """Set state vector to the num-th eigenvector"""

        for p in self.pencils:
            if p == self.eigenvalue_pencil:
                self.state.set_pencil(p, self.eigenvectors[:,num])
            else:
                self.state.set_pencil(p, 0.*self.eigenvectors[:,num])
        self.state.scatter()


class LinearBoundaryValueSolver:
    """
    Linear boundary value problem solver.

    Parameters
    ----------
    problem : problem object
        Problem describing system of differential equations and constraints

    Attributes
    ----------
    state : system object
        System containing solution fields (after solve method is called)

    """

    def __init__(self, problem, matrix_coupling=None):

        logger.debug('Beginning LBVP instantiation')

        self.problem = problem
        self.dist = problem.dist

        # Build subsystems and subproblem matrices
        self.subsystems = subsystems.build_subsystems(problem, matrix_coupling=matrix_coupling)
        self.subproblems = subsystems.build_subproblems(problem, self.subsystems, ['L'])

        self.state = problem.variables

        # Create F operator trees
        namespace = {}
        self.evaluator = Evaluator(self.dist, namespace)
        F_handler = self.evaluator.add_system_handler(iter=1, group='F')
        for eq in problem.eqs:
            F_handler.add_task(eq['F'])
        F_handler.build_system()
        self.F = F_handler.fields

        logger.debug('Finished LBVP instantiation')

    def solve(self):
        """Solve BVP."""

        # Compute RHS
        self.evaluator.evaluate_group('F', sim_time=0, wall_time=0, iteration=0)
        # Solve system for each subproblem, updating state
        for sp in self.subproblems:
            LHS = sp.L_exp
            for ss in sp.subsystems:
                RHS = sp.rhs_map * ss.gather(self.F)
                X = linalg.spsolve(LHS, RHS, use_umfpack=USE_UMFPACK, permc_spec=PERMC_SPEC)
                ss.scatter(X, self.state)
        #self.state.scatter()


class NonlinearBoundaryValueSolver:
    """
    Nonlinear boundary value problem solver.

    Parameters
    ----------
    problem : problem object
        Problem describing system of differential equations and constraints

    Attributes
    ----------
    state : system object
        System containing solution fields (after solve method is called)

    """

    def __init__(self, problem):

        logger.debug('Beginning NLBVP instantiation')

        self.problem = problem
        self.domain = domain = problem.domain
        self.iteration = 0

        # Build pencils and pencil matrices
        self.pencils = pencil.build_pencils(domain)
        pencil.build_matrices(self.pencils, ['L', 'dF'])

        # Build systems
        namespace = problem.namespace
        vars = [namespace[var] for var in problem.variables]
        perts = [namespace['δ'+var] for var in problem.variables]
        self.state = FieldSystem.from_fields(vars)
        self.perturbations = FieldSystem.from_fields(perts)

        # Create F operator trees
        self.evaluator = Evaluator(domain, namespace)
        Fe_handler = self.evaluator.add_system_handler(iter=1, group='F')
        Fb_handler = self.evaluator.add_system_handler(iter=1, group='F')
        for eqn in problem.eqs:
            Fe_handler.add_task(eqn['F-L'])
        for bc in problem.bcs:
            Fb_handler.add_task(bc['F-L'])
        self.Fe = Fe_handler.build_system()
        self.Fb = Fb_handler.build_system()

        logger.debug('Finished NLBVP instantiation')

    def newton_iteration(self):
        """Update solution with a Newton iteration."""
        # Compute RHS
        self.evaluator.evaluate_group('F', sim_time=0, wall_time=0, iteration=self.iteration)
        # Recompute Jacobian
        pencil.build_matrices(self.pencils, ['dF'])
        # Solve system for each pencil, updating perturbations
        for p in self.pencils:
            pFe = self.Fe.get_pencil(p)
            pFb = self.Fb.get_pencil(p)
            A = p.L - p.dF
            b = p.G_eq * pFe + p.G_bc * pFb
            x = linalg.spsolve(A, b, use_umfpack=USE_UMFPACK, permc_spec=PERMC_SPEC)
            self.perturbations.set_pencil(p, x)
        self.perturbations.scatter()
        # Update state
        self.state.gather()
        self.state.data += self.perturbations.data
        self.state.scatter()
        self.iteration += 1


class InitialValueSolver:
    """
    Initial value problem solver.

    Parameters
    ----------
    problem : problem object
        Problem describing system of differential equations and constraints
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

    def __init__(self, problem, timestepper, matrix_coupling=None):

        logger.debug('Beginning IVP instantiation')

        self.problem = problem
        #self.domain = domain = problem.domain
        self.dist = problem.dist
        self.dtype = problem.dtype
        self._wall_time_array = np.zeros(1, dtype=float)
        self.start_time = self.get_wall_time()

        # Build subproblems and subproblem matrices
        self.subsystems = subsystems.build_subsystems(problem, matrix_coupling=matrix_coupling)
        self.subproblems = subsystems.build_subproblems(problem, self.subsystems, ['M', 'L'])

        # Build systems
        # namespace = problem.namespace
        # #vars = [namespace[var] for var in problem.variables]
        # #self.state = FieldSystem.from_fields(vars)
        self.state = problem.variables
        # self._sim_time = namespace[problem.time]

        # Create F operator trees
        namespace = {}
        self.evaluator = Evaluator(self.dist, namespace)
        F_handler = self.evaluator.add_system_handler(iter=1, group='F')
        for eq in problem.eqs:
            F_handler.add_task(eq['F'])
        F_handler.build_system()
        self.F = F_handler.fields

        # Initialize timestepper
        self.timestepper = timestepper(self)

        # Attributes
        self.sim_time = 0.
        self.iteration = 0

        # Default integration parameters
        self.stop_sim_time = np.inf
        self.stop_wall_time = np.inf
        self.stop_iteration = np.inf

        logger.debug('Finished IVP instantiation')

    # @property
    # def sim_time(self):
    #     return self._sim_time.value

    # @sim_time.setter
    # def sim_time(self, t):
    #     self._sim_time.value = t

    def get_wall_time(self):
        self._wall_time_array[0] = time.time()
        comm = self.dist.comm_cart
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

    def euler_step(self, dt):
        """
        M.dt(X) + L.X = F
        M.X1 - M.X0 + h L.X1 = h F
        (M + h L).X1 = M.X0 + h F
        """
        # Compute RHS
        self.evaluator.evaluate_group('F', sim_time=0, wall_time=0, iteration=self.iteration)
        # Solve system for each subproblem, updating state
        for sp in self.subproblems:
            LHS = sp.M_min + dt*sp.L_min
            for ss in sp.subsystems:
                X0 = ss.gather(self.state)
                F0 = ss.gather(self.F)
                RHS = sp.M_min*X0 + dt*sp.rhs_map*F0
                X1 = linalg.spsolve(LHS, RHS, permc_spec=PERMC_SPEC)
                ss.scatter(X1, self.state)
        self.iteration += 1
        self.sim_time += dt

    def step(self, dt):
        """Advance system by one iteration/timestep."""
        if not np.isfinite(dt):
            raise ValueError("Invalid timestep")
        # References
        #state = self.state
        # (Safety gather)
        #state.gather()
        # Advance using timestepper
        wall_time = self.get_wall_time() - self.start_time
        self.timestepper.step(dt, wall_time)
        # (Safety scatter)
        #state.scatter()
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


