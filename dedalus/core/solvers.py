"""
Classes for solving differential equations.

"""

from mpi4py import MPI
import numpy as np
import time
import pathlib
import h5py
import uuid
from scipy.sparse import linalg
from scipy.linalg import eig

#from ..data.operators import parsable_ops
from . import operators
from . import pencil
from . import timesteppers
from .evaluator import Evaluator
from .system import FieldSystem
from .field import Scalar, Field
from ..libraries.matsolvers import matsolvers
from ..tools.cache import CachedAttribute
from ..tools.progress import log_progress
from ..tools.sparse import scipy_sparse_eigs
from ..tools.config import config

import logging
logger = logging.getLogger(__name__.split('.')[-1])


class EigenvalueSolver:
    """
    Solves linear eigenvalue problems for oscillation frequency omega, (d_t -> -i omega)
    for a given pencil, and stores the eigenvalues and eigenvectors. The set_state method
    can be used to set solver.state to the specified eigenmode.

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
        eigenvalue is in eigenvectors[:,i]
    eigenvalue_pencil : pencil
        The pencil for which the eigenvalue problem has been solved.

    """

    def __init__(self, problem, matsolver=None):
        logger.debug('Beginning EVP instantiation')
        if matsolver is None:
            # Default to factorizer to speed up solves within the Arnoldi iteration
            matsolver = matsolvers[config['linear algebra']['MATRIX_FACTORIZER'].lower()]
        self.problem = problem
        self.domain = domain = problem.domain
        self.matsolver = matsolver
        # Build pencils and pencil matrices
        self.pencils = pencil.build_pencils(domain)
        # Build systems
        namespace = problem.namespace
        vars = [namespace[var] for var in problem.variables]
        self.state = FieldSystem(vars)
        # Create F operator trees
        self.evaluator = Evaluator(domain, namespace)
        logger.debug('Finished EVP instantiation')

    def solve_dense(self, pencil, rebuild_coeffs=False, normalize_left=False, **kw):
        """
        Solve EVP for selected pencil.

        Parameters
        ----------
        pencil : pencil object
            Pencil for which to solve the EVP
        rebuild_coeffs : bool, optional
            Flag to rebuild cached coefficient matrices (default: False)
        normalize_left : bool, optional
            Flag to normalize the left eigenvectors (if left=True) such that the
            modified left eigenvectors form a biorthonormal (not just biorthogonal)
            set with respect to the right eigenvectors.
            (default: True)

        Other keyword options passed to scipy.linalg.eig.

        """
        # Build matrices
        if rebuild_coeffs:
            # Generate unique cache
            cacheid = uuid.uuid4()
        else:
            cacheid = None
        pencil.build_matrices(self.problem, ['M', 'L'], cacheid=cacheid)
        # Solve as dense general eigenvalue problem
        eig_output = eig(pencil.L_exp.A, b=-pencil.M_exp.A, **kw)
        # Unpack output
        if len(eig_output) == 2:
            self.eigenvalues, self.eigenvectors = eig_output
        elif len(eig_output) == 3:
            self.eigenvalues, self.left_eigenvectors, self.eigenvectors = eig_output
            self.modified_left_eigenvectors = np.conjugate(np.transpose(np.conjugate(self.left_eigenvectors.T) * -pencil.M))
        if pencil.pre_right is not None:
            self.eigenvectors = pencil.pre_right @ self.eigenvectors
        self.eigenvalue_pencil = pencil
        if len(eig_output) == 3 and normalize_left:
            norms = np.diag(self.modified_left_eigenvectors.T.conj() @ self.eigenvectors)
            self.left_eigenvectors /= norms
            self.modified_left_eigenvectors = np.conjugate(np.transpose(np.conjugate(self.left_eigenvectors.T) * -pencil.M))

    def solve_sparse(self, pencil, N, target, rebuild_coeffs=False, left=False, normalize_left=True, raise_on_mismatch=True, **kw):
        """
        Perform targeted sparse eigenvalue search for selected pencil.

        Parameters
        ----------
        pencil : pencil object
            Pencil for which to solve the EVP
        N : int
            Number of eigenmodes to solver for.  Note: the dense method may
            be more efficient for finding large numbers of eigenmodes.
        target : complex
            Target eigenvalue for search.
        rebuild_coeffs : bool, optional
            Flag to rebuild cached coefficient matrices (default: False)
        left : bool, optional
            Flag to solve for the left eigenvectors of the system
            (IN ADDITION TO the right eigenvectors), defined as the right
            eigenvectors of the conjugate-transposed problem. Follows same
            definition described in scipy.linalg.eig documentation.
            (default: False)
        normalize_left : bool, optional
            Flag to normalize the left eigenvectors such that the modified
            left eigenvectors form a biorthonormal (not just biorthogonal)
            set with respect to the right eigenvectors.
            (default: True)
        raise_on_mismatch : bool, optional
            Flag to raise a RuntimeError if the eigenvalues of the conjugate-
            transposed problem (i.e. the left eigenvalues) do not match
            the original (or "right") eigenvalues.
            (default: True)

        Other keyword options passed to scipy.sparse.linalg.eigs.

        """
        # Build matrices
        if rebuild_coeffs:
            # Generate unique cache
            cacheid = uuid.uuid4()
        else:
            cacheid = None
        pencil.build_matrices(self.problem, ['M', 'L'], cacheid=cacheid)
        # Solve as sparse general eigenvalue problem
        A = pencil.L_exp
        B = -pencil.M_exp
        # Solve for the right eigenvectors
        self.eigenvalues, self.eigenvectors = scipy_sparse_eigs(A=A, B=B, N=N, target=target, matsolver=self.matsolver, **kw)
        if pencil.pre_right is not None:
            self.eigenvectors = pencil.pre_right @ self.eigenvectors
        if left:
            # Solve for the left eigenvectors
            # Note: this definition of "left eigenvectors" is consistent with the documentation for scipy.linalg.eig
            self.left_eigenvalues, self.left_eigenvectors = scipy_sparse_eigs(A=A.getH(),
                                                                              B=B.getH(),
                                                                              N=N, target=np.conjugate(target),
                                                                              matsolver=self.matsolver, **kw)
            if not np.allclose(self.eigenvalues, np.conjugate(self.left_eigenvalues)):
                if raise_on_mismatch:
                    raise RuntimeError("Conjugate of left eigenvalues does not match right eigenvalues. "
                                       "The full sets of left and right vectors won't form a biorthogonal set. "
                                       "This error can be disabled by passing raise_on_mismatch=False to "
                                       "solve_sparse().")
                else:
                    logger.warning("Conjugate of left eigenvalues does not match right eigenvalues.")
            # In absence of above warning, modified_left_eigenvectors forms a biorthogonal set with the right
            # eigenvectors.
            if normalize_left:
                unnormalized_modified_left_eigenvectors = np.conjugate(np.transpose(np.conjugate(self.left_eigenvectors.T) * -pencil.M))
                norms = np.diag(unnormalized_modified_left_eigenvectors.T.conj() @ self.eigenvectors)
                self.left_eigenvectors /= norms
            self.modified_left_eigenvectors = np.conjugate(np.transpose(np.conjugate(self.left_eigenvectors.T) * -pencil.M))
        self.eigenvalue_pencil = pencil

    def set_state(self, index, left=False, modified_left=False):
        """
        Set state vector to the specified eigenmode.

        Parameters
        ----------
        index : int
            Index of desired eigenmode
        left : bool, optional
            If true, sets state vector to a left (or adjoint) eigenmode
            instead of right eigenmode unless modified_left=True
            (default: False)
        modified_left : bool, optional
            If true, sets state vector to a modified left eigenmode,
            which is dual (under the standard complex dot product in
            coefficient space) to the corresponding right eigenmode.
            Supersedes left=True (default: False)
        """
        self.state.data[:] = 0
        if left or modified_left:
            if modified_left:
                self.state.set_pencil(self.eigenvalue_pencil, self.modified_left_eigenvectors[:,index])
            else:
                self.state.set_pencil(self.eigenvalue_pencil, self.left_eigenvectors[:, index])
        else:
            self.state.set_pencil(self.eigenvalue_pencil, self.eigenvectors[:,index])
        self.state.scatter()

    def solve(self, *args, **kw):
        """Deprecated.  Use solve_dense instead."""
        logger.warning("The 'EigenvalueSolver.solve' method is being deprecated. Use 'EigenvalueSolver.solve_dense' instead.")
        return self.solve_dense(*args, **kw)


class LinearBoundaryValueSolver:
    """
    Linear boundary value problem solver.

    Parameters
    ----------
    problem : problem object
        Problem describing system of differential equations and constraints.
    matsolver : matsolver class or name, optional
        Matrix solver routine (default set by config file).

    Attributes
    ----------
    state : system object
        System containing solution fields (after solve method is called).

    """

    def __init__(self, problem, matsolver=None):

        logger.debug('Beginning LBVP instantiation')

        if matsolver is None:
            # Default to factorizer to speed up repeated solves
            matsolver = matsolvers[config['linear algebra']['MATRIX_FACTORIZER'].lower()]
        self.problem = problem
        self.domain = domain = problem.domain
        self.matsolver = matsolver

        # Build pencils and pencil matrices
        self.pencils = pencil.build_pencils(domain)
        pencil.build_matrices(self.pencils, problem, ['L'])
        self._build_pencil_matsolvers()

        # Build systems
        namespace = problem.namespace
        vars = [namespace[var] for var in problem.variables]
        self.state = FieldSystem(vars)

        # Create F operator trees
        self.evaluator = Evaluator(domain, namespace)
        F_handler = self.evaluator.add_system_handler(iter=1, group='F')
        for eqn in problem.eqs:
            F_handler.add_task(eqn['F'])
        self.F = F_handler.build_system()

        logger.debug('Finished LBVP instantiation')

    def _build_pencil_matsolvers(self):
        """Build matsolvers for each pencil LHS."""
        self.pencil_matsolvers = {}
        for p in self.pencils:
            self.pencil_matsolvers[p] = self.matsolver(p.L_exp, self)

    def solve(self, rebuild_coeffs=False):
        """Solve BVP."""

        # Compute RHS
        self.evaluator.evaluate_group('F')

        # Rebuild matrices
        if rebuild_coeffs:
            pencil.build_matrices(self.pencils, self.problem, ['L'])
            self._build_pencil_matsolvers()

        # Solve system for each pencil, updating state
        for p in self.pencils:
            b = p.pre_left @ self.F.get_pencil(p)
            x = self.pencil_matsolvers[p].solve(b)
            if p.pre_right is not None:
                x = p.pre_right @ x
            self.state.set_pencil(p, x)
        self.state.scatter()


class NonlinearBoundaryValueSolver:
    """
    Nonlinear boundary value problem solver.

    Parameters
    ----------
    problem : problem object
        Problem describing system of differential equations and constraints
    matsolver : matsolver class or name, optional
        Matrix solver routine (default set by config file).

    Attributes
    ----------
    state : system object
        System containing solution fields (after solve method is called)

    """

    def __init__(self, problem, matsolver=None):

        logger.debug('Beginning NLBVP instantiation')

        if matsolver is None:
            # Default to solver since every iteration sees a new matrix
            matsolver = matsolvers[config['linear algebra']['MATRIX_SOLVER'].lower()]
        self.problem = problem
        self.domain = domain = problem.domain
        self.matsolver = matsolver
        self.iteration = 0

        # Build pencils and pencil matrices
        self.pencils = pencil.build_pencils(domain)
        pencil.build_matrices(self.pencils, problem, ['L'])

        # Build systems
        namespace = problem.namespace
        vars = [namespace[var] for var in problem.variables]
        perts = [namespace['δ'+var] for var in problem.variables]
        self.state = FieldSystem(vars)
        self.perturbations = FieldSystem(perts)

        # Set variable scales back to 1 for initialization
        for field in self.state.fields + self.perturbations.fields:
            field.set_scales(1)

        # Create F operator trees
        self.evaluator = Evaluator(domain, namespace)
        F_handler = self.evaluator.add_system_handler(iter=1, group='F')
        for eqn in problem.eqs:
            F_handler.add_task(eqn['F-L'])
        self.F = F_handler.build_system()

        logger.debug('Finished NLBVP instantiation')

    def newton_iteration(self, damping=1):
        """Update solution with a Newton iteration."""
        # Compute RHS
        self.evaluator.evaluate_group('F', iteration=self.iteration)
        # Recompute Jacobian
        pencil.build_matrices(self.pencils, self.problem, ['dF'])
        # Solve system for each pencil, updating perturbations
        for p in self.pencils:
            A = p.L_exp - p.dF_exp
            b = p.pre_left @ self.F.get_pencil(p)
            x = self.matsolver(A, self).solve(b)
            if p.pre_right is not None:
                x = p.pre_right @ x
            self.perturbations.set_pencil(p, x)
        self.perturbations.scatter()
        # Update state
        self.state.gather()
        self.state.data += damping * self.perturbations.data
        self.state.scatter()
        self.iteration += 1


class InitialValueSolver:
    """
    Initial value problem solver.

    Parameters
    ----------
    problem : problem object
        Problem describing system of differential equations and constraints
    timestepper : timestepper class or name
        Timestepper to use in evolving initial conditions
    matsolver : matsolver class or name, optional
        Matrix solver routine (default set by config file).
    enforce_real_cadence : int, optional
        Iteration cadence for enforcing Hermitian symmetry on real variables (default: 100).

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

    def __init__(self, problem, timestepper, matsolver=None, enforce_real_cadence=100, warmup_iterations=10):

        logger.debug('Beginning IVP instantiation')

        if matsolver is None:
            # Default to factorizer to speed up repeated solves
            matsolver = matsolvers[config['linear algebra']['MATRIX_FACTORIZER'].lower()]
        self.problem = problem
        self.domain = domain = problem.domain
        self.matsolver = matsolver
        self.enforce_real_cadence = enforce_real_cadence
        self._float_array = np.zeros(1, dtype=float)
        self.init_time = self.get_world_time()
        self._wall_time_array = np.zeros(1, dtype=float)

        # Build pencils and pencil matrices
        self.pencils = pencil.build_pencils(domain)
        pencil.build_matrices(self.pencils, problem, ['M', 'L'])
        local_modes = sum(p.pre_right.shape[1] for p in self.pencils)
        self.total_modes = self.domain.dist.comm.allreduce(local_modes, op=MPI.SUM)

        # Build systems
        namespace = problem.namespace
        vars = [namespace[var] for var in problem.variables]
        self.state = FieldSystem(vars)
        self._sim_time = namespace[problem.time]

        # Create F operator trees
        self.evaluator = Evaluator(domain, namespace)
        F_handler = self.evaluator.add_system_handler(iter=1, group='F')
        for eqn in problem.eqs:
            F_handler.add_task(eqn['F'])
        self.F = F_handler.build_system()

        # Initialize timestepper
        # Dereference strings
        if isinstance(timestepper, str):
            timestepper = timesteppers.schemes[timestepper]
        pencil_length = problem.nvars_nonconst * domain.local_coeff_shape[-1] + problem.nvars_const
        self.timestepper = timestepper(pencil_length, domain)

        # Attributes
        self.sim_time = self.initial_sim_time = 0.
        self.iteration = self.initial_iteration = 0
        self.warmup_iterations = warmup_iterations

        # Default integration parameters
        self.stop_sim_time = np.inf
        self.stop_wall_time = np.inf
        self.stop_iteration = np.inf

        logger.debug('Finished IVP instantiation')

    @property
    def sim_time(self):
        return self._sim_time.value

    @sim_time.setter
    def sim_time(self, t):
        self._sim_time.value = t

    def get_world_time(self):
        self._float_array[0] = time.time()
        comm = self.domain.dist.comm_cart
        comm.Allreduce(MPI.IN_PLACE, self._float_array, op=MPI.MAX)
        return self._float_array[0]

    def load_state(self, path, index=-1):
        """
        Load state from HDF5 file.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to Dedalus HDF5 savefile
        index : int, optional
            Local write index (within file) to load (default: -1)

        Returns
        -------
        write : int
            Global write number of loaded write
        dt : float
            Timestep at loaded write
        """
        path = pathlib.Path(path)
        logger.info("Loading solver state from: {}".format(path))
        with h5py.File(str(path), mode='r') as file:
            # Load solver attributes
            write = file['scales']['write_number'][index]
            try:
                dt = file['scales']['timestep'][index]
            except KeyError:
                dt = None
            self.iteration = self.initial_iteration = file['scales']['iteration'][index]
            self.sim_time = self.initial_sim_time = file['scales']['sim_time'][index]
            # Log restart info
            logger.info("Loading iteration: {}".format(self.iteration))
            logger.info("Loading write: {}".format(write))
            logger.info("Loading sim time: {}".format(self.sim_time))
            logger.info("Loading timestep: {}".format(dt))
            # Load fields
            for field in self.state.fields:
                dset = file['tasks'][field.name]
                # Find matching layout
                for layout in self.domain.dist.layouts:
                    if np.allclose(layout.grid_space, dset.attrs['grid_space']):
                        break
                else:
                    raise ValueError("No matching layout")
                # Set scales to match saved data
                scales = dset.shape[1:] / layout.global_shape(scales=1)
                scales[~layout.grid_space] = 1
                # Extract local data from global dset
                dset_slices = (index,) + layout.slices(tuple(scales))
                local_dset = dset[dset_slices]
                # Copy to field
                field_slices = tuple(slice(n) for n in local_dset.shape)
                field.set_scales(scales, keep_data=False)
                field[layout][field_slices] = local_dset
                field.set_scales(self.domain.dealias, keep_data=True)
        return write, dt

    @property
    def proceed(self):
        """Check that current time and iteration pass stop conditions."""
        if self.sim_time >= self.stop_sim_time:
            logger.info('Simulation stop time reached.')
            return False
        elif (self.get_world_time() - self.init_time) >= self.stop_wall_time:
            logger.info('Wall stop time reached.')
            return False
        elif self.iteration >= self.stop_iteration:
            logger.info('Stop iteration reached.')
            return False
        else:
            return True

    @property
    def ok(self):
        """Deprecated. Use 'solver.proceed'."""
        return self.proceed

    @CachedAttribute
    def sim_dt_cadences(self):
        """Build array of finite handler sim_dt cadences."""
        cadences = [h.sim_dt for h in self.evaluator.handlers]
        cadences = list(filter(np.isfinite, cadences))
        return np.array(cadences)

    def step(self, dt, trim=False):
        """Advance system by one iteration/timestep."""
        # Assert finite timestep
        if not np.isfinite(dt):
            raise ValueError("Invalid timestep")
        # Trim timestep to hit handler sim_dt cadences
        if trim:
            t = self.sim_time
            cadences = self.sim_dt_cadences
            # Compute next scheduled evaluation
            schedule = min(cadences * (t//cadences + 1))
            # Modify timestep if necessary
            dt = min(dt, schedule - t)
        # (Safety gather)
        self.state.gather()
        # Record times
        wall_time = self.get_world_time()
        if self.iteration == self.initial_iteration:
            self.start_time = wall_time
        if self.iteration == self.initial_iteration + self.warmup_iterations:
            self.warmup_time = wall_time
        # Advance using timestepper
        self.timestepper.step(self, dt)
        # (Safety scatter)
        self.state.scatter()
        # Update iteration
        self.iteration += 1
        # Enforce Hermitian symmetry for real variables
        if self.domain.grid_dtype == np.float64:
            # Enforce for as many iterations as timestepper uses internally
            if self.iteration % self.enforce_real_cadence <= self.timestepper._history:
                # Transform state variables to grid and back
                for field in self.state.fields:
                    field.set_scales(self.domain.dealias)
                for path in self.domain.dist.paths:
                    path.increment(self.state.fields)
                for path in self.domain.dist.paths[::-1]:
                    path.decrement(self.state.fields)
        return dt

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

    def evaluate_handlers_now(self, dt, handlers=None):
        """Evaluate all handlers right now. Useful for writing final outputs.

        by default, all handlers are evaluated; if a list is given
        only those will be evaluated.

        """
        end_world_time = self.get_world_time()
        end_wall_time = end_world_time - self.start_time
        if handlers is None:
            handlers = self.evaluator.handlers
        self.evaluator.evaluate_handlers(handlers, timestep=dt, sim_time=self.sim_time, world_time=end_world_time, wall_time=end_wall_time, iteration=self.iteration)

    def log_stats(self, format=".4g"):
        """Log timing statistics with specified string formatting (optional)."""
        log_time = self.get_world_time()
        logger.info(f"Final iteration: {self.iteration}")
        logger.info(f"Final sim time: {self.sim_time}")
        setup_time = self.start_time - self.init_time
        logger.info(f"Setup time (init - iter 0): {setup_time:{format}} sec")
        if self.iteration >= self.initial_iteration + self.warmup_iterations:
            warmup_time = self.warmup_time - self.start_time
            run_time = log_time - self.warmup_time
            cpus = self.domain.dist.comm.size
            modes = self.total_modes
            stages = (self.iteration - self.warmup_iterations - self.initial_iteration) * self.timestepper.stages
            logger.info(f"Warmup time (iter 0-{self.warmup_iterations}): {warmup_time:{format}} sec")
            logger.info(f"Run time (iter {self.warmup_iterations}-end): {run_time:{format}} sec")
            logger.info(f"CPU time (iter {self.warmup_iterations}-end): {run_time*cpus/3600:{format}} cpu-hr")
            logger.info(f"Speed: {(modes*stages/cpus/run_time):{format}} mode-stages/cpu-sec")
        else:
            logger.info(f"Timings unavailable due because warmup did not complete.")
