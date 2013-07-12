

import numpy as np


class Field(object):
    """Scalar field defined over the domain."""

    def __init__(self, domain):

        # Inputs
        self.domain = domain

        # Allocate data
        self.data = np.zeros(domain.shape, dtype=np.complex128)
        self._temp = np.zeros(domain.shape, dtype=np.complex128)

        # Initial space and distribution
        self.space = ['x'] * domain.dim
        self.local = [True] * domain.dim

    def require_global_space(self, space):

        # Expand full-space shortcuts
        if space == 'K':
            space = 'k' * self.domain.dim
        elif space == 'X':
            space = 'x' * self.domain.dim

        # Check each space
        for i in xrange(self.domain.dim):
            self.require_space(i, space[i])

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
            space = 'k' * self.domain.dim
        elif space == 'X':
            space = 'x' * self.domain.dim

        # Set space and data
        self.space = list(space)
        self.data[:] = data

    def transform(self, i):

        # All transforms are performed locally
        self.require_local(i)

        # Call basis transform
        if self.space[i] == 'x':
            self.domain.bases[i].forward(self.data, self.data)
            self.space[i] = 'k'

        elif self.space[i] == 'k':
            self.domain.bases[i].backward(self.data, self.data)
            self.space[i] = 'x'

    def transpose(self, i):

        # NOT IMPLEMENTED
        raise NotImplementedError()

    def differentiate(self, i):

        # Check differentation space
        self.require_space(i, self.domain.bases[i].diff_space)

        # Call basis differentiation
        self.domain.bases[i].differentiate(self.data, self._temp)

        return self._temp

