"""
Class for data fields.

"""

from functools import partial
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg
import weakref

from .future import Future
from ..tools.config import config

# Load config options
permc_spec = config['linear algebra']['permc_spec']
use_umfpack = config['linear algebra'].getboolean('use_umfpack')


class Field(Future):
    """
    Scalar field over a domain.

    Parameters
    ----------
    domain : domain object
        Problem domain
    name : str, optional
        Field name (default: Python object id)

    Attributes
    ----------
    layout : layout object
        Current layout of field
    data : ndarray
        View of internal buffer in current layout

    """

    def __init__(self, domain, name=None, constant=None):

        # Initial attributes
        self.name = name

        # Weak-reference domain to allow cyclic garbage collection
        self._domain_weak_ref = weakref.ref(domain)

        # Increment domain field count
        domain._field_count += 1

        # Allocate data buffer
        self._buffer = domain.distributor.create_buffer()

        # Set initial layout (property hook sets data view)
        self.layout = self.domain.distributor.coeff_layout

        # Default to non-constant
        if constant is None:
            constant = np.array([False] * domain.dim)
        self.constant = constant

    def clean(self):
        """Revert field to state at instantiation."""

        self.layout = self.domain.distributor.coeff_layout
        self.constant[:] = False
        self.data.fill(0.)
        self.name = None

    @property
    def layout(self):
        return self._layout

    @layout.setter
    def layout(self, layout):
        self._layout = layout
        self.data = layout.view_data(self._buffer)

    @property
    def domain(self):
        return self._domain_weak_ref()

    def __del__(self):
        """Intercept deallocation to cache unused fields in domain."""

        # Check that domain is still instantiated
        if self.domain:
            self.domain._collect_field(self)

    def __repr__(self):
        return '<Field %i>' %id(self)

    def __str__(self):
        if self.name:
            return self.name
        else:
            return self.__repr__()

    def __getitem__(self, layout):
        """Return data viewed in specified layout."""

        self.require_layout(layout)

        return self.data

    def __setitem__(self, layout, data):
        """Set data viewed in a specified layout."""

        layout = self.domain.distributor.get_layout_object(layout)
        self.layout = layout
        np.copyto(self.data, data)

    def require_layout(self, layout):
        """Change to specified layout."""

        layout = self.domain.distributor.get_layout_object(layout)

        # Transform to specified layout
        if self.layout.index < layout.index:
            while self.layout.index < layout.index:
                self.towards_grid_space()
        elif self.layout.index > layout.index:
            while self.layout.index > layout.index:
                self.towards_coeff_space()

    def towards_grid_space(self):
        """Change to next layout towards grid space."""

        self.domain.distributor.increment_layout(self)

    def towards_coeff_space(self):
        """Change to next layout towards coefficient space."""

        self.domain.distributor.decrement_layout(self)

    def require_grid_space(self, axis=None):
        """Require one axis (default: all axes) to be in grid space."""

        if axis is None:
            while not all(self.layout.grid_space):
                self.towards_grid_space()
        else:
            while not self.layout.grid_space[axis]:
                self.towards_grid_space()

    def require_coeff_space(self, axis=None):
        """Require one axis (default: all axes) to be in coefficient space."""

        if axis is None:
            while any(self.layout.grid_space):
                self.towards_coeff_space()
        else:
            while self.layout.grid_space[axis]:
                self.towards_coeff_space()

    def require_local(self, axis):
        """Require an axis to be local."""

        # Move towards transform path, since the surrounding layouts are local
        if self.layout.grid_space[axis]:
            while not self.layout.local[axis]:
                self.towards_coeff_space()
        else:
            while not self.layout.local[axis]:
                self.towards_grid_space()

    def differentiate(self, basis, out=None):
        """Differentiate field along one basis."""

        # Use differentiation operator
        basis = self.domain.get_basis_object(basis)
        axis = self.domain.bases.index(basis)
        diff_op = self.domain.diff_ops[axis]
        return diff_op(self, out=out).evaluate()

    def integrate(self, *bases, out=None):
        """Integrate field over bases."""

        # Use integration operator
        from .operators import Integrate
        return Integrate(self, *bases, out=out).evaluate()


    def antidifferentiate(self, basis, bc=None, parameters={}, out=None):
        """
        Antidifferentiate field using a LinearBVP solver, with parsed boundary
        condition where "AD" is the name of the antiderivative field.

        Parameters
        ----------
        basis : basis-like
            Basis to antidifferentiate along
        bc : tuple of str, optional
            Boundary conditions as (bc_str, functional) tuple.
            e.g. ("AD = 10", "int")
        parameters : dict, optional
            Any parameters needed for BC
            e.g. fields for inhomogeneous BCs
        out : field, optional
            Output field

        """

        # Import in method to avoid circular dependencies
        from ..pde.solvers import LinearBVP
        from ..pde.problems import ParsedProblem

        # Setup axis names and parameters for problem
        basis = self.domain.get_basis_object(basis)
        axis = self.domain.bases.index(basis)
        axis_names = ["x%i"%i for i in range(self.domain.dim)]
        parameters['__RHS'] = self

        # Solve simple problem for the antiderivative
        problem = ParsedProblem(axis_names=axis_names,
                                field_names=['AD'],
                                param_names=list(parameters.keys()))
        problem.add_equation("dx%s(AD) = __RHS" %axis)
        if bc:
            problem.add_bc(*bc)
        problem.parameters.update(parameters)
        problem.expand(self.domain)

        solver = LinearBVP(problem, self.domain)
        solver.solve()

        if not out:
            out = self.domain.new_field()
        out['c'] = solver.state['AD']['c']

        return out

    def antidifferentiate2(self, basis, bc, out=None):
        """
        Antidifferentiate field using a LinearBVP solver, with parsed boundary
        condition where "AD" is the name of the antiderivative field.

        Parameters
        ----------
        basis : basis-like
            Basis to antidifferentiate along
        bc : tuple of str
            Boundary conditions as (functional, value) tuple.
            `functional` is a string, e.g. "Left", "Right", "Int"
            `value` is a field or scalar
        out : field, optional
            Output field

        """

        # References
        basis = self.domain.get_basis_object(basis)
        domain = self.domain
        bc_type, bc_val = bc

        # Only solve along last basis
        if basis is not domain.bases[-1]:
            raise NotImplementedError()

        # Convert BC value to field
        if np.isscalar(bc_val):
            bc_val = domain.new_field()
            bc_val['g'] = bc[1]
        elif not isinstance(bc_val, Field):
            raise TypeError("bc_val must be field or scalar")

        # Build LHS matrix
        Pre = basis.Pre
        Diff = basis.Diff
        F = sparse.eye(basis.coeff_size) - sparse.diags(basis.bc_vector.flatten(), 0)
        G = F*Pre
        BC = getattr(basis, bc_type.capitalize())
        LHS = Pre*Diff + BC

        if not out:
            out = self.domain.new_field()
        out_c = out['c']
        f_c = self['c']
        bc_c = bc_val['c']

        for p in np.ndindex(out_c.shape[:-1]):
            rhs = G*f_c[p] + BC*bc_c[p]
            out_c[p] = splinalg.spsolve(LHS, rhs, use_umfpack=use_umfpack, permc_spec=permc_spec)

        return out
