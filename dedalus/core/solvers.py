"""
Classes for solving differential equations.

"""

from mpi4py import MPI
import numpy as np
import time
import h5py
import pathlib
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
from ..tools.array import csr_matvec

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
                raise ValueError(f"Specified solver coupling incompatible with problem coupling: {problem_coupling}")
        # Check that coupled dimensions are local
        coeff_layout = self.dist.coeff_layout
        coupled_nonlocal = matrix_coupling & ~coeff_layout.local
        if np.any(coupled_nonlocal):
            raise ValueError(f"Problem is coupled along distributed dimensions: {tuple(np.where(coupled_nonlocal)[0])}")
        self.matrix_coupling = matrix_coupling
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


class EigenvalueSolver(SolverBase):
    """
    Solves linear eigenvalue problems for oscillation frequency omega, (d_t -> -i omega).
    First converts to dense matrices, then solves the eigenvalue problem for a given pencil,
    and stored the eigenvalues and eigenvectors.  The set_state method can be used to set
    the state to the ith eigenvector.

    Parameters
    ----------
    problem : Problem object
        Dedalus problem.
    **kw :
        Other options passed to ProblemBase.

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

    # Default to factorizer to speed up repeated solves
    matsolver_default = 'MATRIX_FACTORIZER'

    def __init__(self, problem, **kw):
        logger.debug('Beginning EVP instantiation')
        super().__init__(problem, **kw)
        # Build subsystems and subproblem matrices
        self.subsystems = subsystems.build_subsystems(self)
        self.subproblems = subsystems.build_subproblems(self, self.subsystems, ['M','L'])
        self.state = problem.variables
        logger.debug('Finished EVP instantiation')

    def solve_dense(self, subproblem, rebuild_coeffs=False, **kw):
        """
        Solve EVP for selected pencil.

        Parameters
        ----------
        subproblem : subproblem object
            Subproblem for which to solve the EVP
        #rebuild_coeffs : bool, optional
        #    Flag to rebuild cached coefficient matrices (default: False)

        Other keyword options passed to scipy.linalg.eig.

        """
        # TODO caching of matrices for loops over parameter values
        # # Build matrices
        # if rebuild_coeffs:
        #     # Generate unique cache
        #     cacheid = uuid.uuid4()
        # else:
        #     cacheid = None
        #pencil.build_matrices(self.problem, ['M', 'L'], cacheid=cacheid)

        # Solve as dense general eigenvalue problem
        eig_output = eig(subproblem.L_min.A, b=-subproblem.M_min.A, **kw)
        # Unpack output
        if len(eig_output) == 2:
            self.eigenvalues, self.eigenvectors = eig_output
        elif len(eig_output) == 3:
            self.eigenvalues, self.left_eigenvectors, self.eigenvectors = eig_output
        self.eigenvalue_subproblem = subproblem

    def solve_sparse(self, pencil, N, target, rebuild_coeffs=False, **kw):
        raise NotImplementedError()

    def set_state(self, index, subsystem):
        """
        Set state vector to the specified eigenmode.

        Parameters
        ----------
        index : int
            Index of desired eigenmode
        subsystem : Subsystem object
            subsystem that eigenvalue data will be put into
        """
        if subsystem not in self.eigenvalue_subproblem.subsystems:
            raise ValueError("subsystem must be in eigenvalue_subproblem")
        for var in self.state:
            var['c'] = 0
        X = self.eigenvectors[:,index]
        subsystem.scatter(X, self.state)


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
    state : system object
        System containing solution fields (after solve method is called)

    """

    # Default to factorizer to speed up repeated solves
    matsolver_default = 'MATRIX_FACTORIZER'

    def __init__(self, problem, **kw):
        logger.debug('Beginning LBVP instantiation')
        super().__init__(problem, **kw)
        # Build subsystems and subproblem matrices
        self.subsystems = subsystems.build_subsystems(self)
        self.subproblems = subsystems.build_subproblems(self, self.subsystems, ['L'])
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
        if self.store_expanded_matrices:
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
        # Build matsolvers on demand
        if not hasattr(self, "subproblem_matsolvers"):
            self._build_subproblem_matsolvers()
        # Ensure coeff space before subsystem gathers/scatters
        for field in self.F:
            field.require_layout('c')
        for field in self.state:
            field.set_layout('c')
        # Solve system for each subproblem, updating state
        for sp in self.subproblems:
            sp_matsolver = self.subproblem_matsolvers[sp]
            F = X = np.empty(sp.L_min.shape[0], dtype=self.dtype)
            for ss in sp.subsystems:
                F.fill(0)  # Must zero before csr_matvec
                csr_matvec(sp.pre_left, ss.gather(self.F), F)
                x = sp_matsolver.solve(F)
                X.fill(0)  # Must zero before csr_matvec
                csr_matvec(sp.pre_right, x, X)
                ss.scatter(X, self.state)
        #self.state.scatter()


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
    state : system object
        System containing solution fields (after solve method is called)

    """

    # Default to solver since every iteration sees a new matrix
    matsolver_default = 'MATRIX_SOLVER'

    def __init__(self, problem, **kw):
        logger.debug('Beginning NLBVP instantiation')
        super().__init__(problem, **kw)
        self.iteration = 0
        # Build subsystems and subproblem matrices
        self.subsystems = subsystems.build_subsystems(self)
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
        if self.store_expanded_matrices:
            L = sp.dH_exp
            matsolver = self.matsolver(L, self)
        else:
            L = sp.dH_min @ sp.pre_right
            matsolver = self.matsolver(L, self)
        return matsolver

    def newton_iteration(self, damping=1):
        """Update solution with a Newton iteration."""
        # Compute RHS
        self.evaluator.evaluate_group('F', sim_time=0, wall_time=0, iteration=self.iteration)
        # Recompute Jacobian
        self.subproblems = subsystems.build_subproblems(self, self.subsystems, ['dH'])
        # Ensure coeff space before subsystem gathers/scatters
        for field in self.F:
            field.require_layout('c')
        for field in self.perturbations:
            field.set_layout('c')
        # Solve system for each subproblem, updating state
        for sp in self.subproblems:
            sp_matsolver = self._build_subproblem_matsolver(sp)
            F = X = np.empty(sp.dH_min.shape[0], dtype=self.dtype)
            for ss in sp.subsystems:
                F.fill(0)  # Must zero before csr_matvec
                csr_matvec(sp.pre_left, ss.gather(self.F), F)
                x = sp_matsolver.solve(F)
                X.fill(0)  # Must zero before csr_matvec
                csr_matvec(sp.pre_right, x, X)
                ss.scatter(X, self.perturbations)
        # Update state
        for var, pert in zip(self.state, self.perturbations):
            var['c'] += damping * pert['c']
        self.iteration += 1


class InitialValueSolver(SolverBase):
    """
    Initial value problem solver.

    Parameters
    ----------
    problem : Problem object
        Dedalus problem.
    timestepper : timestepper class
        Timestepper to use in evolving initial conditions
    **kw :
        Other options passed to ProblemBase.

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

    # Default to factorizer to speed up repeated solves
    matsolver_default = 'MATRIX_FACTORIZER'

    def __init__(self, problem, timestepper, **kw):
        logger.debug('Beginning IVP instantiation')
        super().__init__(problem, **kw)
        self._wall_time_array = np.zeros(1, dtype=float)
        self.start_time = self.get_wall_time()
        # Build subproblems and subproblem matrices
        self.subsystems = subsystems.build_subsystems(self)
        self.subproblems = subsystems.build_subproblems(self, self.subsystems, ['M', 'L'])
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
        self.sim_time = self.initial_sim_time = 0  # TODO: allow picking up from current problem sim time?
        self.iteration = self.initial_iteration = 0
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
        self.problem.sim_time_field['g'] = t

    def get_wall_time(self):
        self._wall_time_array[0] = time.time()
        comm = self.dist.comm_cart
        comm.Allreduce(MPI.IN_PLACE, self._wall_time_array, op=MPI.MAX)
        return self._wall_time_array[0]

    @property
    def ok(self):
        import warnings
        warnings.warn("solver.ok is deprecated, use solver.proceed instead")
        return self.proceed

    @property
    def proceed(self):
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

    def load_state(self, path, index=-1):
        """
        Load state from HDF5 file. Currently can only load grid space data.

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
                field.load_from_hdf5(file, index)
        return write, dt

    # TO-DO: remove this
    def euler_step(self, dt):
        """
        M.dt(X) + L.X = F
        M.X1 - M.X0 + h L.X1 = h F
        (M + h L).X1 = M.X0 + h F
        """
        # Compute RHS
        self.evaluator.evaluate_group('F', sim_time=0, wall_time=0, iteration=self.iteration)
        # Ensure coeff space before subsystem gathers/scatters
        for field in self.F:
            field.require_layout('c')
        for field in self.state:
            field.require_layout('c')
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

    def print_subproblem_ranks(self, dt=1):
        # Check matrix rank
        for i, subproblem in enumerate(self.subproblems):
            M = subproblem.M_min @ subproblem.pre_right
            L = subproblem.L_min @ subproblem.pre_right
            A = (M + dt*L).A
            print(f"MPI rank: {self.dist.comm.rank}, subproblem: {i}, group: {subproblem.group}, matrix rank: {np.linalg.matrix_rank(A)}/{A.shape[0]}, cond: {np.linalg.cond(A):.1e}")

