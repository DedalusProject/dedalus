

import numpy as np


class FieldBase(object):
    """Base class for fields."""

    pass


class OneDimensionalField(FieldBase):
    """One dimensional fields"""

    def __init__(self, primary_basis):

        # Input parameters
        self.primary_basis = primary_basis

        # Setup data containers
        shape = (primary_basis.size,)
        self._xdata = np.zeros(shape, dtype=np.complex128)
        self._kdata = np.zeros(shape, dtype=np.complex128)
        self._kderiv = np.zeros(shape, dtype=np.complex128)

        # Initial space
        self.current_space = 'xspace'
        self.data = self._xdata

    def __getitem__(self, space):

        self.require_space(space)

        return self.data

    def __setitem__(self, space, data):

        if space == 'xspace':
            self.data = self._xdata
        elif space == 'kspace':
            self.data = self._kdata
        else:
            raise KeyError("'space' must be 'xspace' or 'kspace'")

        self.data[:] = data
        self.current_space = space

    def require_space(self, space):

        if self.current_space != space:
            if space == 'xspace':
                self.backward()
            elif space == 'kspace':
                self.forward()
            else:
                raise ValueError("'space' must be 'kspace' or 'xspace'")

    def forward(self):

        if self.current_space == 'kspace':
            raise ValueError('Cannot perform forward transform from kspace.')

        self.primary_basis.forward(self._xdata, self._kdata)
        self.data = self._kdata
        self.current_space = 'kspace'

    def backward(self):

        if self.current_space == 'xspace':
            raise ValueError('Cannot perform backward transform from xspace.')

        self.primary_basis.backward(self._kdata, self._xdata)
        self.data = self._xdata
        self.current_space = 'xspace'

    def differentiate(self):

        self.require_space('kspace')
        self.primary_basis.differentiate(self._kdata, self._kderiv)

        return self._kderiv

