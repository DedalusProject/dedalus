"""
Class for data fields.

"""

import numpy as np
import weakref

# Bottom-import to resolve cyclic dependencies:
# from .operators import Negation, Addition, Subtraction, Multiplication


class Field:
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
        if name is not None:
            self.name = name
        else:
            self.name = 'F' + str(id(self))

        # Weak-reference domain to allow cyclic garbage collection
        self._domain_weak_ref = weakref.ref(domain)

        # Increment domain field count
        domain._field_count += 1

        # Allocate buffer
        self._buffer = domain.distributor.create_buffer()

        # Set initial layout (property hook sets data view)
        self.layout = domain.distributor.coeff_layout

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
        return self.name

    # Use operators to define arithmetic on field objects

    def __neg__(self):
        return Negation(self)

    def __add__(self, other):
        return Addition(self, other)

    def __radd__(self, other):
        return Addition(other, self)

    def __sub__(self, other):
        return Subtraction(self, other)

    def __rsub__(self, other):
        return Subtraction(other, self)

    def __mul__(self, other):
        return Multiplication(self, other)

    def __rmul__(self, other):
        return Multiplication(other, self)

    def __truediv__(self, other):
        return Division(self, other)

    def __rtruediv__(self, other):
        return Division(other, self)

    def __getitem__(self, layout):
        """Return data viewed in specified layout."""

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

        return self.data

    def __setitem__(self, layout, data):
        """Set data viewed in a specified layout."""

        # Resolve layout strings to corresponding layout objects
        if isinstance(layout, str):
            layout = self.domain.distributor.string_layouts[layout]

        self.layout = layout
        np.copyto(self.data, data)

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

    def differentiate(self, axis, out):
        """Differentiate field along one axis."""

        # Require axis to be local and in coefficient space
        # UPGRADE: non-transverse bases don't strictly require locality
        self.require_local(axis)
        self.require_coeff_space(axis)

        # Call basis differentiation
        out.layout = self.layout
        self.domain.bases[axis].differentiate(self.data, out.data, axis=axis)

    def integrate(self, axes=None):
        """Integrate field over domain."""

        # Require full coefficient space
        # UPGRADE: implement distributed integral
        self.require_coeff_space()
        if not all(self.layout.local):
            raise NotImplementedError("Distributed integral not implemented.")

        # Integrate over all axes by default
        if axes is None:
            axes = range(self.domain.dim)
        else:
            axes = tuple(axes)

        # Integrate by coefficients
        data = self.data
        for i in reversed(sorted(axes)):
            b = self.domain.bases[i]
            data = b.integrate(data, i)
            data = b.grid_dtype(data)

        return data


# Bottom-import to resolve cyclic dependencies:
from .operators import Negation, Addition, Subtraction, Multiplication, Division

