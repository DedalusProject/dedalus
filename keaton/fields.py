

import numpy as np


class Field(object):

    def __init__(self, primary_basis, transverse_basis=None):

        # Inputs
        self.primary_basis = primary_basis
        self.transverse_basis = transverse_basis

        # Setup data containers
        if transverse_basis:
            shape = transverse_basis.shape + primary_basis.shape
        else:
            shape = primary_basis.shape

        self.dim = len(shape)
        self.data = np.zeros(shape, dtype=np.complex128)
        self._temp = np.zeros(shape, dtype=np.complex128)

        # Initial space
        self.current_space = ['x']

    def require_space(self, space):

        # Expand full-space shortcuts
        if space == 'K':
            space = 'k' * self.dim
        elif space == 'X':
            space = 'x' * self.dim

        # Perform necessary transforms
        for i in xrange(self.dim):
            if self.current_space[i] != space[i]:
                if space[i] == 'x':
                    self.backward(i)
                elif space[i] == 'k':
                    self.forward(i)
                else:
                    raise KeyError("'space' must be 'x' or 'k'")

    def __getitem__(self, space):

        self.require_space(space)

        return self.data

    def __setitem__(self, space, data):

        # Expand full-space shortcuts
        if space == 'K':
            space = 'k' * self.dim
        elif space == 'X':
            space = 'x' * self.dim

        # Set space and data
        self.current_space = list(space)
        self.data[:] = data

    def forward(self, i):

        if self.current_space[i] == 'k':
            raise ValueError('Cannot perform forward transform from k.')

        if i == (self.dim - 1):
            self.primary_basis.forward(self.data, self.data)
        else:
            self.transverse_basis.forward(self.data, self.data, i)

        self.current_space[i] = 'k'

    def backward(self, i):

        if self.current_space[i] == 'x':
            raise ValueError('Cannot perform backward transform from x.')

        if i == (self.dim - 1):
            self.primary_basis.backward(self.data, self.data)
        else:
            self.transverse_basis.backward(self.data, self.data, i)

        self.current_space[i] = 'x'

    def differentiate(self, i):

        self.require_space('K')

        if i == (self.dim - 1):
            self.primary_basis.differentiate(self.data, self._temp)
        else:
            self.transverse_basis.differentaite(self.data, self._temp, i)

        return self._temp

