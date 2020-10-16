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
from ..libraries.matsolvers import matsolvers
from ..tools.progress import log_progress
from ..tools.config import config

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

    def __init__(self, problem, matsolver=None):
        logger.debug('Beginning EVP instantiation')
        if matsolver is None:
            # Default to factorizer to speed up solves within the Arnoldi iteration
            matsolver = matsolvers[config['linear algebra']['MATRIX_FACTORIZER'].lower()]
        elif isinstance(matsolver, str):
            matsolver = matsolvers[matsolver.lower()]
        self.problem = problem
        self.domain = domain = problem.domain
        self.matsolver = matsolver
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

    def __init__(self, problem, matrix_coupling=None, matsolver=None):

        logger.debug('Beginning LBVP instantiation')

        if matsolver is None:
            # Default to factorizer to speed up repeated solves
            matsolver = matsolvers[config['linear algebra']['MATRIX_FACTORIZER'].lower()]
        elif isinstance(matsolver, str):
            matsolver = matsolvers[matsolver.lower()]
        self.problem = problem
        self.dist = problem.dist
        self.matsolver = matsolver
        self.dtype = problem.dtype

        # Build subsystems and subproblem matrices
        self.subsystems = subsystems.build_subsystems(problem, matrix_coupling=matrix_coupling)
        self.subproblems = subsystems.build_subproblems(problem, self.subsystems, ['L'])
        self._build_subproblem_matsolvers()

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

    def _build_subproblem_matsolvers(self):
        """Build matsolvers for each pencil LHS."""
        self.subproblem_matsolvers = {}
        if self.problem.STORE_EXPANDED_MATRICES:
            for sp in self.subproblems:
                L = sp.L_exp
                self.subproblem_matsolvers[sp] = self.matsolver(L, self)
        else:
            for sp in self.subproblems:
                L = sp.L_min @ sp.pre_right
                self.subproblem_matsolvers[sp] = self.matsolver(L, self)

    def solve(self):
        """Solve BVP."""
        # Compute RHS
        self.evaluator.evaluate_group('F', sim_time=0, wall_time=0, iteration=0)
        # Solve system for each subproblem, updating state
        for sp in self.subproblems:
            sp_matsolver = self.subproblem_matsolvers[sp]
            for ss in sp.subsystems:
                RHS = sp.pre_left @ ss.gather(self.F)
                X = sp_matsolver.solve(RHS)
                X = sp.pre_right @ X
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

    def __init__(self, problem, matrix_coupling=None, matsolver=None):

        logger.debug('Beginning NLBVP instantiation')

        if matsolver is None:
            # Default to solver since every iteration sees a new matrix
            matsolver = matsolvers[config['linear algebra']['MATRIX_SOLVER'].lower()]
        elif isinstance(matsolver, str):
            matsolver = matsolvers[matsolver.lower()]
        self.matsolver = matsolver
        self.problem = problem
        self.dist = problem.dist
        self.dtype = problem.dtype

        self.iteration = 0

        # Build subsystems and subproblem matrices
        self.subsystems = subsystems.build_subsystems(problem, matrix_coupling=matrix_coupling)

        self.state = problem.variables
        self.perturbations = problem.perturbations

        # Create F operator trees
        namespace = {}
        self.evaluator = Evaluator(self.dist, namespace)
        F_handler = self.evaluator.add_system_handler(iter=1, group='F')
        for eq in problem.eqs:
            F_handler.add_task(eq['-H'])
        F_handler.build_system()
        self.F = F_handler.fields

        logger.debug('Finished NLBVP instantiation')

    def _build_subproblem_matsolver(self, sp):
        """Build matsolvers for each subproblem LHS."""
        if self.problem.STORE_EXPANDED_MATRICES:
            L = sp.dH_exp
            matsolver = self.matsolver(L, self)
        else:
            L = sp.dH_min @ sp.pre_right
            matsolver = self.matsolver(L, self)
        return matsolver

    def newton_iteration(self):
        """Update solution with a Newton iteration."""
        # Compute RHS
        self.evaluator.evaluate_group('F', sim_time=0, wall_time=0, iteration=self.iteration)
        # Recompute Jacobian
        self.subproblems = subsystems.build_subproblems(self.problem, self.subsystems, ['dH'])
        # Solve system for each subproblem, updating state
        for sp in self.subproblems:
            sp_matsolver = self._build_subproblem_matsolver(sp)
            for ss in sp.subsystems:
                RHS = sp.pre_left @ ss.gather(self.F)
                X = sp_matsolver.solve(RHS)
                X = sp.pre_right @ X
                ss.scatter(X, self.perturbations)

        # Update state
        for var, pert in zip(self.state, self.perturbations):
            var['c'] += pert['c']
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

    def __init__(self, problem, timestepper, matrix_coupling=None, matsolver=None):

        logger.debug('Beginning IVP instantiation')

        if matsolver is None:
            # Default to factorizer to speed up repeated solves
            matsolver = matsolvers[config['linear algebra']['MATRIX_FACTORIZER'].lower()]
        elif isinstance(matsolver, str):
            matsolver = matsolvers[matsolver.lower()]
        self.problem = problem
        #self.domain = domain = problem.domain
        self.matsolver = matsolver
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

    # TO-DO: remove this
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
                X1 = self.matsolver(LHS, self).solve(RHS)
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
