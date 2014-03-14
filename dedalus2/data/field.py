"""
Class for data fields.

"""

from functools import partial
import numpy as np
import weakref

from .future import Future


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

    def __init__(self, domain, name=None):

        # Initial attributes
        self.name = name

        # Weak-reference domain to allow cyclic garbage collection
        self._domain_weak_ref = weakref.ref(domain)

        # Increment domain field count
        domain._field_count += 1

        # Allocate buffer
        self._buffer = domain.distributor.create_buffer()

        # Set initial layout (property hook sets data view)
        self.layout = self.domain.distributor.coeff_layout
        self.constant = np.array([False] * domain.dim)

    def clean(self):
        """Revert field to state at instantiation."""

        self.layout = self.domain.distributor.coeff_layout
        self.constant[:] = False
        self.data.fill(0.)

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

        # Resolve layout strings to corresponding layout objects
        if isinstance(layout, str):
            layout = self.domain.distributor.string_layouts[layout]

        self.layout = layout
        np.copyto(self.data, data)

    def require_layout(self, layout):
        """Change to specified layout."""

        # Resolve layout strings to corresponding layout objects
        if isinstance(layout, str):
            layout = self.domain.distributor.string_layouts[layout]

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

