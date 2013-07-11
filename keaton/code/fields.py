

import numpy as np


class Field(object):

    def __init__(self, bases):

        # Inputs
        self.bases = bases

        # Build shape
        self.shape = tuple([b.size for b in bases])
        self.dim = len(self.shape)

        # Allocate data
        self.data = np.zeros(self.shape, dtype=np.complex128)
        self._temp = np.zeros(self.shape, dtype=np.complex128)

        # Initial space and distribution
        self.space = ['x'] * self.dim
        self.local = [True] * self.dim

    def require_global_space(self, space):

        # Expand full-space shortcuts
        if space == 'K':
            space = 'k' * self.dim
        elif space == 'X':
            space = 'x' * self.dim

        # Check each space
        for i in xrange(self.dim):
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
            space = 'k' * self.dim
        elif space == 'X':
            space = 'x' * self.dim

        # Set space and data
        self.space = list(space)
        self.data[:] = data

    def transform(self, i):

        # All transforms are performed locally
        self.require_local(i)

        # Call basis transform
        if self.space[i] == 'x':
            self.bases[i].forward(self.data, self.data)
            self.space[i] = 'k'

        elif self.space[i] == 'k':
            self.bases[i].backward(self.data, self.data)
            self.space[i] = 'x'

    def transpose(self, i):

        pass

    def differentiate(self, i):

        # Check differentation space
        self.require_space(i, self.bases[i].diff_space)

        # Call basis differentiation
        self.bases[i].differentiate(self.data, self._temp)

        return self._temp

