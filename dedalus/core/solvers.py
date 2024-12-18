"""Solver classes."""

from mpi4py import MPI
import numpy as np
import time
import h5py
import pathlib
import scipy.linalg
import cProfile
import pstats
from math import prod
from collections import defaultdict
import pickle

from . import subsystems
from . import timesteppers
from .evaluator import Evaluator
from ..libraries.matsolvers import matsolvers
from ..tools.config import config
from ..tools.array import scipy_sparse_eigs
from ..tools.parallel import ProfileWrapper, parallel_mkdir

PROFILE_DEFAULT = config['profiling'].getboolean('PROFILE_DEFAULT')
PARALLEL_PROFILE_DEFAULT = config['profiling'].getboolean('PARALLEL_PROFILE_DEFAULT')
PROFILE_DIRECTORY = pathlib.Path(config['profiling'].get('PROFILE_DIRECTORY'))

import logging
logger = logging.getLogger(__name__.split('.')[-1])


class SolverBase:
    """
    Base class for PDE solvers.

    Parameters
    ----------
    problem : Problem object
        Dedalus problem.
    ncc_cutoff : float, optional
        Mode amplitude cutoff for LHS NCC expansions (default: 1e-6)
    max_ncc_terms : int, optional
        Maximum terms to include in LHS NCC expansions (default: None (no limit))
    entry_cutoff : float, optional
        Matrix entry cutoff to avoid fill-in from cancellation errors (default: 1e-12)
    matrix_coupling : tuple of bool, optional
        Matrix coupling override.
    matsolver : str or Matsolver class, optional
        Matrix solver. Default taken from config.
    bc_top : bool, optional
        Whether to place boundary conditions at top of matrices. Default taken from config.
    tau_left : bool, optional
        Whether to place tau columns at left of matrices. Default taken from config.
    interleave_components : bool, optional
        Whether to interleave components before variables. Default taken from config.
    store_expanded_matrices : bool, optional
        Whether to store right-preconditioned matrices. Default taken from config.
    """

    def __init__(self, problem, ncc_cutoff=1e-6, max_ncc_terms=None, entry_cutoff=1e-12, matrix_coupling=None, matsolver=None,
                 bc_top=None, tau_left=None, interleave_components=None, store_expanded_matrices=None):
        # Take attributes from problem
        self.problem = problem
        self.dist = problem.dist
        self.dtype = problem.dtype
        self.state = problem.variables
        # Process options
        self.ncc_cutoff = ncc_cutoff
        self.max_ncc_terms = max_ncc_terms
        self.entry_cutoff = entry_cutoff
        if matrix_coupling is None:
            matrix_coupling = np.array(problem.matrix_coupling)
            # Couple fully separable problems along last axis by default for efficiency
            if not np.any(matrix_coupling):
                matrix_coupling[-1] = True
        else:
            # Check specified coupling for compatibility
            problem_coupling = np.array(problem.matrix_coupling)
            matrix_coupling = np.array(matrix_coupling)
            if np.any(~matrix_coupling & problem_coupling):
                raise ValueError(f"Specified solver coupling is incompatible with problem coupling: {problem_coupling}")
        # Check that coupled dimensions are local
        coeff_layout = self.dist.coeff_layout
        coupled_nonlocal = matrix_coupling & ~coeff_layout.local
        if np.any(coupled_nonlocal):
            raise ValueError(f"Problem is coupled along distributed dimensions: {tuple(np.where(coupled_nonlocal)[0])}")
        self.matrix_coupling = matrix_coupling
        # Determine matrix dependence based on specified coupling
        self.matrix_dependence = np.array(problem.matrix_dependence)
        for eq in problem.eqs:
            for basis in eq['domain'].bases:
                first_axis = self.dist.get_basis_axis(basis)
                slices = slice(first_axis, first_axis+basis.dim)
                self.matrix_dependence[slices] = self.matrix_dependence[slices] | basis.matrix_dependence(matrix_coupling[slices])
        # Process config options
        if matsolver is None:
            matsolver = config['linear algebra'][self.matsolver_default]
        if isinstance(matsolver, str):
            matsolver = matsolvers[matsolver.lower()]
        self.matsolver = matsolver
        if bc_top is None:
            bc_top = config['matrix construction'].getboolean('BC_TOP')
        self.bc_top = bc_top
        if tau_left is None:
            tau_left = config['matrix construction'].getboolean('TAU_LEFT')
        self.tau_left = tau_left
        if interleave_components is None:
            interleave_components = config['matrix construction'].getboolean('INTERLEAVE_COMPONENTS')
        self.interleave_components = interleave_components
        if store_expanded_matrices is None:
            store_expanded_matrices = config['matrix construction'].getboolean('STORE_EXPANDED_MATRICES')
        self.store_expanded_matrices = store_expanded_matrices
        # Process option overrides from matsolver
        for key, value in matsolver.config.items():
            if getattr(self, key, None) is not value:
                logger.info("matsolver overriding solver option '%i' with '%s'" %(key, value))
                setattr(self, key, value)
        # Setup subsystems and subproblems without building matrices
        self.subsystems = subsystems.build_subsystems(self)
        self.subproblems = subsystems.build_subproblems(self, self.subsystems)
        self.subproblems_by_group = {sp.group: sp for sp in self.subproblems}
        # Build evaluator
        namespace = {}
        self.evaluator = Evaluator(self.dist, namespace)

    def build_matrices(self, subproblems=None, matrices=None):
        """Build matrices for selected subproblems."""
        if subproblems is None:
            subproblems = self.subproblems
        if matrices is None:
            matrices = self.matrices
        subsystems.build_subproblem_matrices(self, subproblems, matrices)


class EigenvalueSolver(SolverBase):
    """
    Linear eigenvalue problem solver.

    Parameters
    ----------
    problem : Problem object
        Dedalus eigenvalue problem.
    **kw :
        Other options passed to ProblemBase.

    Attributes
    ----------
    state : list of Field objects
        Problem variables containing solution (after set_state method is called).
    eigenvalues : ndarray
        Vector of eigenvalues.
    eigenvectors : ndarray
        Array of eigenvectors. The eigenvector corresponding to the i-th
        eigenvalues is eigenvectors[:, i].
    eigenvalue_subproblem : Subproblem object
        The subproblem for which the EVP has een solved.
    """

    # Default to factorizer to speed up repeated solves
    matsolver_default = 'MATRIX_FACTORIZER'
    matrices = ['M', 'L']

    def __init__(self, problem, **kw):
        logger.debug('Beginning EVP instantiation')
        super().__init__(problem, **kw)
        logger.debug('Finished EVP instantiation')

    def print_subproblem_ranks(self, subproblems=None, target=0):
        """Print rank of each subproblem LHS."""
        if subproblems is None:
            subproblems = self.subproblems
        # Check matrix rank
        for i, sp in enumerate(subproblems):
            if not hasattr(sp, 'L_min'):
                continue
            L = sp.L_min.toarray()
            M = sp.M_min.toarray()
            A = L + target * M
            print(f"MPI rank: {self.dist.comm.rank}, subproblem: {i}, group: {sp.group}, matrix rank: {np.linalg.matrix_rank(A)}/{A.shape[0]}, cond: {np.linalg.cond(A):.1e}")

    def solve_dense(self, subproblem, rebuild_matrices=False, left=False, normalize_left=True, **kw):
        """
        Perform dense eigenvector search for selected subproblem.
        This routine finds all eigenvectors but is computationally expensive.

        Parameters
        ----------
        subproblem : Subproblem object
            Subproblem for which to solve the EVP.
        rebuild_matrices : bool, optional
            Rebuild LHS matrices if coefficients have changed (default: False).
        left : bool, optional
            Solve for the left eigenvectors of the system in addition to the
            right eigenvectors. The left eigenvectors are the right eigenvectors
            of the conjugate-transposed problem. Follows same definition described
            in scipy.linalg.eig documentation (default: False).
        normalize_left : bool, optional
            Normalize the left eigenvectors such that the modified left eigenvectors
            form a biorthonormal (not just biorthogonal) set with respect to the right
            eigenvectors (default: True).
        **kw :
            Other keyword options passed to scipy.linalg.eig.
        """
        self.eigenvalue_subproblem = sp = subproblem
        # Build matrices if directed or not yet built
        if rebuild_matrices or not hasattr(sp, 'L_min'):
            self.build_matrices([sp], ['M', 'L'])
        # Solve as dense general eigenvalue problem
        A = sp.L_min.toarray()
        B = - sp.M_min.toarray()
        eig_output = scipy.linalg.eig(A, b=B, left=left, **kw)
        # Unpack output
        if left:
            self.eigenvalues, pre_left_evecs, pre_right_evecs = eig_output
            self.right_eigenvectors = self.eigenvectors = sp.pre_right @ pre_right_evecs
            self.left_eigenvectors = sp.pre_left.conj().T @ pre_left_evecs
            self.modified_left_eigenvectors = (sp.M_min @ sp.pre_right_pinv).conj().T @ pre_left_evecs
            if normalize_left:
                norms = np.diag(pre_left_evecs.T.conj() @ sp.M_min @ pre_right_evecs)
                self.left_eigenvectors /= np.conj(norms)
                self.modified_left_eigenvectors /= np.conj(norms)
        else:
            self.eigenvalues, pre_eigenvectors = eig_output
            self.eigenvectors = sp.pre_right @ pre_eigenvectors

    def solve_sparse(self, subproblem, N, target, rebuild_matrices=False, left=False, normalize_left=True, raise_on_mismatch=True, v0=None, **kw):
        """
        Perform targeted sparse eigenvector search for selected subproblem.
        This routine finds a subset of eigenvectors near the specified target.

        Parameters
        ----------
        subproblem : Subproblem object
            Subproblem for which to solve the EVP.
        N : int
            Number of eigenvectors to solve for. Note: the dense method may
            be more efficient for finding large numbers of eigenvectors.
        target : complex
            Target eigenvalue for search.
        rebuild_matrices : bool, optional
            Rebuild LHS matrices if coefficients have changed (default: False).
        left : bool, optional
            Solve for the left eigenvectors of the system in addition to the
            right eigenvectors. The left eigenvectors are the right eigenvectors
            of the conjugate-transposed problem. Follows same definition described
            in scipy.linalg.eig documentation (default: False).
        normalize_left : bool, optional
            Normalize the left eigenvectors such that the modified left eigenvectors
            form a biorthonormal (not just biorthogonal) set with respect to the right
            eigenvectors (default: True).
        raise_on_mismatch : bool, optional
            Raise a RuntimeError if the left and right eigenvalues do not match (default: True).
        v0 : ndarray, optional
            Initial guess for eigenvector, e.g. from subsystem.gather (default: None).
        **kw :
            Other keyword options passed to scipy.sparse.linalg.eig.
        """
        self.eigenvalue_subproblem = sp = subproblem
        # Build matrices if directed or not yet built
        if rebuild_matrices or not hasattr(sp, 'L_min'):
            self.build_matrices([sp], ['M', 'L'])
        # Solve as sparse general eigenvalue problem
        A = sp.L_min
        B = - sp.M_min
        # Precondition starting guess if provided
        if v0 is not None:
            v0 = sp.pre_right_pinv @ v0
        # Solve for the right (and optionally left) eigenvectors
        eig_output = scipy_sparse_eigs(A=A, B=B, left=left, N=N, target=target, matsolver=self.matsolver, v0=v0, **kw)
        if left:
            # Note: this definition of "left eigenvectors" is consistent with the documentation for scipy.linalg.eig
            self.eigenvalues, pre_right_evecs, self.left_eigenvalues, pre_left_evecs = eig_output
            self.right_eigenvectors = self.eigenvectors = sp.pre_right @ pre_right_evecs
            self.left_eigenvectors = sp.pre_left.conj().T @ pre_left_evecs
            self.modified_left_eigenvectors = (sp.M_min @ sp.pre_right_pinv).conj().T @ pre_left_evecs
            # Check that eigenvalues match
            if not np.allclose(self.eigenvalues, np.conjugate(self.left_eigenvalues)):
                if raise_on_mismatch:
                    raise RuntimeError("Conjugate of left eigenvalues does not match right eigenvalues. "
                                       "The full sets of left and right vectors won't form a biorthogonal set. "
                                       "This error can be disabled by passing raise_on_mismatch=False to "
                                       "solve_sparse().")
                else:
                    logger.warning("Conjugate of left eigenvalues does not match right eigenvalues.")
                    if normalize_left:
                        logger.warning("Cannot normalize left eigenvectors.")
                        normalize_left = False
            # Normalize
            if normalize_left:
                norms = np.diag(pre_left_evecs.T.conj() @ sp.M_min @ pre_right_evecs)
                self.left_eigenvectors /= np.conj(norms)
                self.modified_left_eigenvectors /= np.conj(norms)
        else:
            self.eigenvalues, pre_right_evecs = eig_output
            self.right_eigenvectors = self.eigenvectors = sp.pre_right @ pre_right_evecs

    def set_state(self, index, subsystem=0):
        """
        Set state vector to the specified eigenmode.

        Parameters
        ----------
        index : int
            Index of desired eigenmode.
        subsystem : Subsystem object or int, optional
            Subsystem that will be set to the corresponding eigenmode.
            If an integer, the corresponding subsystem of the last specified
            eigenvalue_subproblem will be used. Default: 0.
        """
        # TODO: allow setting left modified eigenvectors?
        subproblem = self.eigenvalue_subproblem
        if isinstance(subsystem, int):
            subsystem = subproblem.subsystems[subsystem]
        # Check selection
        if subsystem not in subproblem.subsystems:
            raise ValueError("subsystem must be in eigenvalue_subproblem")
        # Set coefficients
        for var in self.state:
            var['c'] = 0
        subsystem.scatter(self.eigenvectors[:, index], self.state)
        # Set eigenvalue
        self.problem.eigenvalue['g'] = self.eigenvalues[index]


class LinearBoundaryValueSolver(SolverBase):
    """
    Linear boundary value problem solver.

    Parameters
    ----------
    problem : Problem object
        Dedalus problem.
    **kw :
        Other options passed to ProblemBase.

    Attributes
    ----------
    state : list of Field objects
        Problem variables containing solution (after solve method is called).
    """

    # Default to factorizer to speed up repeated solves
    matsolver_default = 'MATRIX_FACTORIZER'
    matrices = ['L']

    def __init__(self, problem, **kw):
        logger.debug('Beginning LBVP instantiation')
        super().__init__(problem, **kw)
        self.subproblem_matsolvers = {}
        self.iteration = 0
        # Create RHS handler
        F_handler = self.evaluator.add_system_handler(iter=1, group='F')
        for eq in problem.eqs:
            F_handler.add_task(eq['F'])
        F_handler.build_system()
        self.F = F_handler.fields
        logger.debug('Finished LBVP instantiation')

    def print_subproblem_ranks(self, subproblems=None):
        """Print rank of each subproblem LHS."""
        if subproblems is None:
            subproblems = self.subproblems
        # Check matrix rank
        for i, sp in enumerate(subproblems):
            if not hasattr(sp, 'L_min'):
                continue
            L = sp.L_min.toarray()
            print(f"MPI rank: {self.dist.comm.rank}, subproblem: {i}, group: {sp.group}, matrix rank: {np.linalg.matrix_rank(L)}/{L.shape[0]}, cond: {np.linalg.cond(L):.1e}")

    def solve(self, subproblems=None, rebuild_matrices=False):
        """
        Solve BVP over selected subproblems.

        Parameters
        ----------
        subproblems : Subproblem object or list of Subproblem objects, optional
            Subproblems for which to solve the BVP (default: None (all)).
        rebuild_matrices : bool, optional
            Rebuild LHS matrices if coefficients have changed (default: False).
        """
        # Resolve subproblems
        if subproblems is None:
            subproblems = self.subproblems
        if isinstance(subproblems, subsystems.Subproblem):
            subproblems = [subproblems]
        # Build matrices and matsolvers if directed or not yet built
        if rebuild_matrices:
            sp_to_build = subproblems
        else:
            sp_to_build = [sp for sp in subproblems if sp not in self.subproblem_matsolvers]
        if sp_to_build:
            self.build_matrices(sp_to_build, ['L'])
            for sp in sp_to_build:
                self.subproblem_matsolvers[sp] = self.matsolver(sp.L_min, self)
        # Compute RHS
        self.evaluator.evaluate_scheduled(iteration=self.iteration)
        # Ensure coeff space before subsystem gathers/scatters
        for field in self.F:
            field.change_layout('c')
        for field in self.state:
            field.preset_layout('c')
        # Solve system for each subproblem
        for sp in subproblems:
            # Gather RHS
            spF = sp.gather_outputs(self.F)
            # Solve
            spX = self.subproblem_matsolvers[sp].solve(spF)  # CREATES TEMPORARY
            # Scatter solution
            sp.scatter_inputs(spX, self.state)
        self.iteration += 1

    def evaluate_handlers(self, handlers=None):
        """Evaluate specified list of handlers (all by default)."""
        if handlers is None:
            handlers = self.evaluator.handlers
        self.evaluator.evaluate_handlers(handlers, iteration=self.iteration)


class NonlinearBoundaryValueSolver(SolverBase):
    """
    Nonlinear boundary value problem solver.

    Parameters
    ----------
    problem : Problem object
        Dedalus problem.
    **kw :
        Other options passed to ProblemBase.

    Attributes
    ----------
    state : list of Field objects
        Problem variables containing solution.
    perturbations : list of Field objects
        Perturbations to problem variables from each Newton iteration.
    iteration : int
        Current iteration.
    """

    # Default to solver since every iteration sees a new matrix
    matsolver_default = 'MATRIX_SOLVER'
    matrices = ['dF']

    def __init__(self, problem, **kw):
        logger.debug('Beginning NLBVP instantiation')
        super().__init__(problem, **kw)
        self.perturbations = problem.perturbations
        # Copy valid modes from variables to perturbations (may have been changed after problem instantiation)
        for pert, var in zip(problem.perturbations, problem.variables):
            pert.valid_modes[:] = var.valid_modes
        self.iteration = 0
        # Create RHS handler
        F_handler = self.evaluator.add_system_handler(iter=1, group='F')
        for eq in problem.eqs:
            F_handler.add_task(eq['F'])
        F_handler.build_system()
        self.F = F_handler.fields
        logger.debug('Finished NLBVP instantiation')

    def print_subproblem_ranks(self, subproblems=None):
        """Print rank of each subproblem LHS."""
        if subproblems is None:
            subproblems = self.subproblems
        # Check matrix rank
        for i, sp in enumerate(subproblems):
            if not hasattr(sp, 'dF_min'):
                continue
            dF = sp.dF_min.toarray()
            print(f"MPI rank: {self.dist.comm.rank}, subproblem: {i}, group: {sp.group}, matrix rank: {np.linalg.matrix_rank(dF)}/{dF.shape[0]}, cond: {np.linalg.cond(dF):.1e}")

    def newton_iteration(self, damping=1):
        """Update solution with a Newton iteration."""
        # Compute RHS
        self.evaluator.evaluate_scheduled(iteration=self.iteration)
        # Recompute Jacobian
        # TODO: split out linear part for faster recomputation?
        self.build_matrices(self.subproblems, ['dF'])
        # Ensure coeff space before subsystem gathers/scatters
        for field in self.F:
            field.change_layout('c')
        for field in self.perturbations:
            field.preset_layout('c')
        # Solve system for each subproblem, updating perturbations
        for sp in self.subproblems:
            # Gather RHS
            spF = sp.gather_outputs(self.F)
            # Solve
            sp_matsolver = self.matsolver(sp.dF_min, self)
            spX = sp_matsolver.solve(spF)  # CREATES TEMPORARY
            # Scatter solution to perturbations
            sp.scatter_inputs(spX, self.perturbations)
        # Update state
        for var, pert in zip(self.state, self.perturbations):
            var['c'] -= damping * pert['c']
        self.iteration += 1

    def evaluate_handlers(self, handlers=None):
        """Evaluate specified list of handlers (all by default)."""
        if handlers is None:
            handlers = self.evaluator.handlers
        self.evaluator.evaluate_handlers(handlers, iteration=self.iteration)


class InitialValueSolver(SolverBase):
    """
    Initial value problem solver.

    Parameters
    ----------
    problem : Problem object
        Dedalus problem.
    timestepper : Timestepper class or str
        Timestepper to use in evolving initial conditions.
    enforce_real_cadence : int, optional
        Iteration cadence for enforcing Hermitian symmetry on real variables (default: 100).
    warmup_iterations : int, optional
        Number of warmup iterations to disregard when computing runtime statistics (default: 10).
    profile : bool, optional
        Save accumulated profiles with cProfile (default: False).
    parallel_profile : bool, optional
        Save per-process and accumulated profiles with cProfile (default: False).
    **kw :
        Other options passed to ProblemBase.

    Attributes
    ----------
    state : list of Field objects
        Problem variables containing solution.
    stop_sim_time : float
        Simulation stop time, in simulation units.
    stop_wall_time : float
        Wall stop time, in seconds from instantiation.
    stop_iteration : int
        Stop iteration.
    sim_time : float
        Current simulation time.
    iteration : int
        Current iteration.
    dt : float
        Last timestep.
    """

    # Default to factorizer to speed up repeated solves
    matsolver_default = 'MATRIX_FACTORIZER'
    matrices = ['M', 'L']

    def __init__(self, problem, timestepper, enforce_real_cadence=100, warmup_iterations=10, profile=PROFILE_DEFAULT, parallel_profile=PARALLEL_PROFILE_DEFAULT, **kw):
        logger.debug('Beginning IVP instantiation')
        # Setup timing and profiling
        self.dist = problem.dist
        self._bcast_array = np.zeros(1, dtype=float)
        self.init_time = self.world_time
        if profile or parallel_profile:
            parallel_mkdir(PROFILE_DIRECTORY, comm=self.dist.comm)
            self.profile = True
            self.parallel_profile = parallel_profile
            self.setup_profiler = cProfile.Profile()
            self.warmup_profiler = cProfile.Profile()
            self.run_profiler = cProfile.Profile()
            self.setup_profiler.enable()
        else:
            self.profile = False
        # Build subsystems and subproblems
        super().__init__(problem, **kw)
        # Build LHS matrices
        self.build_matrices(self.subproblems, ['M', 'L'])
        # Compute total modes
        local_modes = sum([prod(sp.shape) for sp in self.subproblems])
        self.total_modes = self.dist.comm.allreduce(local_modes, op=MPI.SUM)
        # Create RHS handler
        F_handler = self.evaluator.add_system_handler(iter=1, group='F')
        for eq in problem.eqs:
            F_handler.add_task(eq['F'])
        F_handler.build_system()
        self.F = F_handler.fields
        # Initialize timestepper
        if isinstance(timestepper, str):
            timestepper = timesteppers.schemes[timestepper]
        self.timestepper = timestepper(self)
        # Attributes
        self.sim_time = self.initial_sim_time = problem.time.allreduce_data_max(layout='g')
        self.iteration = self.initial_iteration = 0
        self.warmup_iterations = warmup_iterations
        if np.isrealobj(self.dtype.type()):
            self.enforce_real_cadence = enforce_real_cadence
        else:
            self.enforce_real_cadence = None
        # Default integration parameters
        self.stop_sim_time = np.inf
        self.stop_wall_time = np.inf
        self.stop_iteration = np.inf
        logger.debug('Finished IVP instantiation')

    @property
    def sim_time(self):
        return self._sim_time

    @sim_time.setter
    def sim_time(self, t):
        self._sim_time = t
        self.problem.time['g'] = t

    @property
    def world_time(self):
        if self.dist.comm.size == 1:
            return time.time()
        else:
            # Broadcast time from root process
            self._bcast_array[0] = time.time()
            self.dist.comm_cart.Bcast(self._bcast_array, root=0)
            return self._bcast_array[0]

    @property
    def wall_time(self):
        """Seconds ellapsed since instantiation."""
        return self.world_time - self.init_time

    @property
    def proceed(self):
        """Check that current time and iteration pass stop conditions."""
        if self.sim_time >= self.stop_sim_time:
            logger.info('Simulation stop time reached.')
            return False
        elif self.wall_time >= self.stop_wall_time:
            logger.info('Wall stop time reached.')
            return False
        elif self.iteration >= self.stop_iteration:
            logger.info('Stop iteration reached.')
            return False
        else:
            return True

    def load_state(self, path, index=-1, allow_missing=False):
        """
        Load state from HDF5 file. Currently can only load grid space data.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to Dedalus HDF5 savefile
        index : int, optional
            Local write index (within file) to load (default: -1)
        allow_missing : bool, optional
            Do not raise an error if state variables are missing from the savefile (default: False).

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
            dt = file['scales']['timestep'][index]
            self.iteration = self.initial_iteration = file['scales']['iteration'][index]
            self.sim_time = self.initial_sim_time = file['scales']['sim_time'][index]
            # Log restart info
            logger.info("Loading iteration: {}".format(self.iteration))
            logger.info("Loading write: {}".format(write))
            logger.info("Loading sim time: {}".format(self.sim_time))
            logger.info("Loading timestep: {}".format(dt))
            # Load fields
            for field in self.state:
                if field.name in file['tasks']:
                    field.load_from_hdf5(file, index)
                elif allow_missing:
                    logger.warning(f"Field '{field.name}' not found in savefile.")
                else:
                    raise IOError(f"Field '{field.name}' not found in savefile. Set allow_missing=True to ignore this error.")
        return write, dt

    def enforce_hermitian_symmetry(self, fields):
        """Transform fields to grid and back."""
        # TODO: maybe this should be on scales=1?
        for f in fields:
            f.change_scales(f.domain.dealias)
        self.evaluator.require_grid_space(fields)
        self.evaluator.require_coeff_space(fields)

    def step(self, dt):
        """Advance system by one iteration/timestep."""
        # Assert finite timestep
        if not np.isfinite(dt):
            raise ValueError("Invalid timestep")
        # Record times
        wall_time = self.wall_time
        if self.iteration == self.initial_iteration:
            self.start_time_end = wall_time
            if self.profile:
                self.dump_profiles(self.setup_profiler, "setup")
                self.warmup_profiler.enable()
            self.warmup_time_start = self.wall_time
        if self.iteration == self.initial_iteration + self.warmup_iterations:
            self.warmup_time_end = self.wall_time
            if self.profile:
                self.dump_profiles(self.warmup_profiler, "warmup")
                self.run_profiler.enable()
            self.run_time_start = self.wall_time
        # Advance using timestepper
        self.timestepper.step(dt, wall_time)
        # Enforce Hermitian symmetry for real variables
        if self.enforce_real_cadence:
            # Enforce for as many iterations as timestepper uses internally
            if self.iteration % self.enforce_real_cadence < self.timestepper.steps:
                self.enforce_hermitian_symmetry(self.state)
        # Update iteration
        self.iteration += 1
        self.dt = dt

    def evolve(self, timestep_function, log_cadence=100):
        """Advance system until stopping criterion is reached."""
        # Check for a stopping criterion
        if np.isinf(self.stop_sim_time):
            if np.isinf(self.stop_wall_time):
                if np.isinf(self.stop_iteration):
                    raise ValueError("No stopping criterion specified.")
        # Evolve
        try:
            logger.info("Starting main loop")
            while self.proceed:
                timestep = timestep_function()
                self.step(timestep)
                if (self .iteration-1) % log_cadence == 0:
                    logger.info(f"Iteration={self.iteration}, Time={self.sim_time:e}, Step={timestep:e}")
        except:
            logger.error('Exception raised, triggering end of main loop.')
            raise
        finally:
            self.log_stats()

    def print_subproblem_ranks(self, subproblems=None, dt=1):
        """Print rank of each subproblem LHS."""
        if subproblems is None:
            subproblems = self.subproblems
        # Check matrix rank
        for i, sp in enumerate(subproblems):
            M = sp.M_min
            L = sp.L_min
            A = (M + dt*L).toarray()
            print(f"MPI rank: {self.dist.comm.rank}, subproblem: {i}, group: {sp.group}, matrix rank: {np.linalg.matrix_rank(A)}/{A.shape[0]}, cond: {np.linalg.cond(A):.1e}")

    def evaluate_handlers_now(self, dt, handlers=None):
        logger.warning("Deprecation warning: evaluate_handlers_now -> evaluate_handlers")
        self.evaluate_handlers(handlers=handlers, dt=dt)

    def evaluate_handlers(self, handlers=None, dt=0):
        """Evaluate specified list of handlers (all by default)."""
        if handlers is None:
            handlers = self.evaluator.handlers
        self.evaluator.evaluate_handlers(handlers, iteration=self.iteration, wall_time=self.wall_time, sim_time=self.sim_time, timestep=dt)

    def log_stats(self, format=".4g"):
        """Log timing statistics with specified string formatting (optional)."""
        self.run_time_end = self.wall_time
        start_time = self.start_time_end
        logger.info(f"Final iteration: {self.iteration}")
        logger.info(f"Final sim time: {self.sim_time}")
        logger.info(f"Setup time (init - iter 0): {start_time:{format}} sec")
        if self.profile:
            self.dump_profiles(self.run_profiler, "runtime")
        if self.iteration >= self.initial_iteration + self.warmup_iterations:
            warmup_time = self.warmup_time_end - self.warmup_time_start
            run_time = self.run_time_end - self.run_time_start
            profile_output_time = (self.warmup_time_start-self.start_time_end) + (self.run_time_start-self.warmup_time_end)
            cpus = self.dist.comm.size
            modes = self.total_modes
            stages = (self.iteration - self.warmup_iterations - self.initial_iteration) * self.timestepper.stages
            logger.info(f"Warmup time (iter 0-{self.warmup_iterations}): {warmup_time:{format}} sec")
            if self.profile:
                logger.info(f"Profile output time: {profile_output_time:{format}} sec")
            logger.info(f"Run time (iter {self.warmup_iterations}-end): {run_time:{format}} sec")
            logger.info(f"CPU time (iter {self.warmup_iterations}-end): {run_time*cpus/3600:{format}} cpu-hr")
            logger.info(f"Speed: {(modes*stages/cpus/run_time):{format}} mode-stages/cpu-sec")
        else:
            logger.info(f"Timings unavailable because warmup did not complete.")

    def dump_profiles(self, profiler, name):
        "Save profiling data to disk."
        comm = self.dist.comm
        # Disable and create stats on each process
        profiler.create_stats()
        p = pstats.Stats(profiler)
        p.strip_dirs()
        # Gather using wrapper class to avoid pickling issues
        profiles = comm.gather(ProfileWrapper(p.stats), root=0)
        # Sum stats on root process
        if comm.rank == 0:
            if self.parallel_profile:
                stats = {'primcalls': defaultdict(list),
                         'totcalls': defaultdict(list),
                         'tottime': defaultdict(list),
                         'cumtime': defaultdict(list)}
                for profile in profiles:
                    for func, (primcalls, totcalls, tottime, cumtime, callers) in profile.stats.items():
                        stats['primcalls'][func].append(primcalls)
                        stats['totcalls'][func].append(totcalls)
                        stats['tottime'][func].append(tottime)
                        stats['cumtime'][func].append(cumtime)
                pickle.dump(stats, open(PROFILE_DIRECTORY / f"{name}_parallel.pickle", 'wb'))
            # Creation of joint_stats destroys profiles, so do this second
            joint_stats = pstats.Stats(*profiles)
            joint_stats.dump_stats(PROFILE_DIRECTORY / f"{name}.prof")
        comm.Barrier()
