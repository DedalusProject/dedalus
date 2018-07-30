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
from .evaluator import Evaluator
from .system import CoeffSystem, FieldSystem
from .field import Scalar, Field
from ..tools.cache import CachedAttribute
from ..tools.progress import log_progress
from ..tools.sparse import scipy_sparse_eigs

from ..tools.config import config
PERMC_SPEC = config['linear algebra']['permc_spec']
USE_UMFPACK = config['linear algebra'].getboolean('use_umfpack')

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

    def __init__(self, problem):
        logger.debug('Beginning EVP instantiation')
        self.problem = problem
        self.domain = domain = problem.domain
        # Build pencils and pencil matrices
        self.pencils = pencil.build_pencils(domain)
        # Build systems
        namespace = problem.namespace
        vars = [namespace[var] for var in problem.variables]
        self.state = FieldSystem.from_fields(vars)
        # Create F operator trees
        self.evaluator = Evaluator(domain, namespace)
        logger.debug('Finished EVP instantiation')

    def solve_dense(self, pencil, rebuild_coeffs=False, **kw):
        """
        Solve EVP for selected pencil.

        Parameters
        ----------
        pencil : pencil object
            Pencil for which to solve the EVP
        rebuild_coeffs : bool, optional
            Flag to rebuild cached coefficient matrices (default: False)

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
        eig_output = eig(pencil.L.A, b=-pencil.M.A, **kw)
        # Unpack output
        if len(eig_output) == 2:
            self.eigenvalues, self.eigenvectors = eig_output
        elif len(eig_output) == 3:
            self.eigenvalues, self.left_eigenvectors, self.eigenvectors = eig_output
        self.eigenvalue_pencil = pencil

    def solve_sparse(self, pencil, N, target, rebuild_coeffs=False, **kw):
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
        self.eigenvalues, self.eigenvectors = scipy_sparse_eigs(A=pencil.L_exp, B=-pencil.M_exp, N=N, target=target, **kw)
        if pencil.dirichlet:
            self.eigenvectors = pencil.JD * self.eigenvectors
        self.eigenvalue_pencil = pencil

    def set_state(self, index):
        """
        Set state vector to the specified eigenmode.

        Parameters
        ----------
        index : int
            Index of desired eigenmode
        """
        self.state.data[:] = 0
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
        Problem describing system of differential equations and constraints

    Attributes
    ----------
    state : system object
        System containing solution fields (after solve method is called)

    """

    def __init__(self, problem):

        logger.debug('Beginning LBVP instantiation')

        self.problem = problem
        self.domain = domain = problem.domain

        # Build pencils and pencil matrices
        self.pencils = pencil.build_pencils(domain)
        pencil.build_matrices(self.pencils, problem, ['L'])

        # Build systems
        namespace = problem.namespace
        vars = [namespace[var] for var in problem.variables]
        self.state = FieldSystem.from_fields(vars)

        # Create F operator trees
        self.evaluator = Evaluator(domain, namespace)
        Fe_handler = self.evaluator.add_system_handler(iter=1, group='F')
        Fb_handler = self.evaluator.add_system_handler(iter=1, group='F')
        for eqn in problem.eqs:
            Fe_handler.add_task(eqn['F'])
        for bc in problem.bcs:
            Fb_handler.add_task(bc['F'])
        self.Fe = Fe_handler.build_system()
        self.Fb = Fb_handler.build_system()

        logger.debug('Finished LBVP instantiation')

    def solve(self, rebuild_coeffs=False):
        """Solve BVP."""

        # Compute RHS
        self.evaluator.evaluate_group('F')

        # Rebuild matrices
        if rebuild_coeffs:
            pencil.build_matrices(self.pencils, self.problem, ['L'])

        # Solve system for each pencil, updating state
        for p in self.pencils:
            pFe = self.Fe.get_pencil(p)
            pFb = self.Fb.get_pencil(p)
            A = p.L_exp
            if p.G_bc is None:
                b = p.G_eq * pFe
            else:
                b = p.G_eq * pFe + p.G_bc * pFb
            x = linalg.spsolve(A, b, use_umfpack=USE_UMFPACK, permc_spec=PERMC_SPEC)
            if p.dirichlet:
                x = p.JD * x
            self.state.set_pencil(p, x)
        self.state.scatter()


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
        pencil.build_matrices(self.pencils, problem, ['L'])

        # Build systems
        namespace = problem.namespace
        vars = [namespace[var] for var in problem.variables]
        perts = [namespace['δ'+var] for var in problem.variables]
        self.state = FieldSystem.from_fields(vars)
        self.perturbations = FieldSystem.from_fields(perts)

        # Set variable scales back to 1 for initialization
        for field in self.state.fields + self.perturbations.fields:
            field.set_scales(1)

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

    def newton_iteration(self, damping=1):
        """Update solution with a Newton iteration."""
        # Compute RHS
        self.evaluator.evaluate_group('F', iteration=self.iteration)
        # Recompute Jacobian
        pencil.build_matrices(self.pencils, self.problem, ['dF'])
        # Solve system for each pencil, updating perturbations
        for p in self.pencils:
            pFe = self.Fe.get_pencil(p)
            pFb = self.Fb.get_pencil(p)
            A = p.L_exp - p.dF_exp
            b = p.G_eq * pFe + p.G_bc * pFb
            x = linalg.spsolve(A, b, use_umfpack=USE_UMFPACK, permc_spec=PERMC_SPEC)
            if p.dirichlet:
                x = p.JD * x
            self.perturbations.set_pencil(p, x)
        self.perturbations.scatter()
        # Update state
        self.state.gather()
        self.state.data += damping*self.perturbations.data
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

    def __init__(self, problem, timestepper):

        logger.debug('Beginning IVP instantiation')

        self.problem = problem
        self.domain = domain = problem.domain
        self._float_array = np.zeros(1, dtype=float)
        self.start_time = self.get_world_time()

        # Build pencils and pencil matrices
        self.pencils = pencil.build_pencils(domain)
        pencil.build_matrices(self.pencils, problem, ['M', 'L'])

        # Build systems
        namespace = problem.namespace
        vars = [namespace[var] for var in problem.variables]
        self.state = FieldSystem.from_fields(vars)
        self._sim_time = namespace[problem.time]

        # Create F operator trees
        self.evaluator = Evaluator(domain, namespace)
        Fe_handler = self.evaluator.add_system_handler(iter=1, group='F')
        Fb_handler = self.evaluator.add_system_handler(iter=1, group='F')
        for eqn in problem.eqs:
            Fe_handler.add_task(eqn['F'])
        for bc in problem.bcs:
            Fb_handler.add_task(bc['F'])
        self.Fe = Fe_handler.build_system()
        self.Fb = Fb_handler.build_system()

        # Initialize timestepper
        self.timestepper = timestepper(problem.nvars, domain)

        # Attributes
        self.sim_time = self.initial_sim_time = 0.
        self.iteration = self.initial_iteration = 0

        # Default integration parameters
        self.stop_sim_time = 10.
        self.stop_wall_time = 10.
        self.stop_iteration = 10.

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
    def ok(self):
        """Check that current time and iteration pass stop conditions."""

        if self.sim_time >= self.stop_sim_time:
            logger.info('Simulation stop time reached.')
            return False
        elif (self.get_world_time() - self.start_time) >= self.stop_wall_time:
            logger.info('Wall stop time reached.')
            return False
        elif self.iteration >= self.stop_iteration:
            logger.info('Stop iteration reached.')
            return False
        else:
            return True

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
        # Advance using timestepper
        self.timestepper.step(self, dt)
        # (Safety scatter)
        self.state.scatter()
        # Update iteration
        self.iteration += 1
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


