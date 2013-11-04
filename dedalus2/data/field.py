

import numpy as np
import weakref

from ..tools.dist import distributor

# Bottom of module:
# # Import after definitions to resolve cyclic dependencies
# from .operators import Negation, Addition, Subtraction, Multiplication


class Field:
    """Scalar field defined over the distributed domain."""

    def __init__(self, domain, name=None):

        # Initial attributes
        if name is not None:
            self.name = name
        else:
            self.name = 'F' + str(id(self))

        # Weak reference to domain to allow cyclic garbage collection
        self.domain = weakref.ref(domain)

        # Increment domain field count
        domain._field_count += 1

        # Allocate buffer
        #self._buffer = np.zeros(distributor.buffer_size[domain], dtype=np.byte)

        # Set initial layout
        #self.layout = distributor.layouts[domain][0]

    @property
    def layout(self):
        return self._layout

    @layout.setter
    def layout(self, layout):
        self._layout = layout
        self.data = layout.view_data(self._buffer)

    def __del__(self):

        # Add self to domain manager
        if self.domain():
            self.domain()._collect_field(self)

    def __repr__(self):
        return self.name

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

    def require_global_space(self, space):

        # Expand full-space shortcuts
        if space == 'K':
            space = 'k' * self.domain().dim
        elif space == 'X':
            space = 'x' * self.domain().dim

        # Check each space
        for i, s in enumerate(space):
            self.require_space(i, s)

    def require_space(self, index, space):

        # Transform if necessary
        if self.space[index] != space:
            self.transform(index)

    def require_local(self, index):

        # Transpose if necessary
        if not self.local[index]:
            self.transpose(index)

    def __getitem__(self, space):

        # Check space
        self.require_global_space(space)

        return self.data

    def __setitem__(self, space, data):

        # Expand full-space shortcuts
        if space == 'K':
            space = 'k' * self.domain().dim
        elif space == 'X':
            space = 'x' * self.domain().dim

        # Set space and data
        self.space = list(space)
        self.data[:] = data

    def transform(self, i):

        # All transforms are performed locally
        self.require_local(i)

        # Call basis transform
        if self.space[i] == 'x':
            self.domain().bases[i].forward(self.data, self.data, axis=i)
            self.space[i] = 'k'

        elif self.space[i] == 'k':
            self.domain().bases[i].backward(self.data, self.data, axis=i)
            self.space[i] = 'x'

    def transpose(self, i):

        # NOT IMPLEMENTED
        raise NotImplementedError()

    def differentiate(self, i):

        # Check differentation space
        self.require_space(i, 'k')

        # Call basis differentiation
        self.domain().bases[i].differentiate(self.data, self._temp, axis=i)

        return self._temp


# Import after definitions to resolve cyclic dependencies
from .operators import Negation, Addition, Subtraction, Multiplication

