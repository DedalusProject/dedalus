

import numpy as np

from primary import PrimaryBasis
from fields import Field


class DomainBase(object):
    """Base class for domains"""

    pass


class OneDimensionalDomain(DomainBase):
    """1D domains: primary basis"""

    def __init__(self, primary_basis):

        # Input parameters
        self.primary_basis = primary_basis

        # Grid
        self.grid = primary_basis.grid

        # Pencil slices
        self.slices = [slice(None)]

    def create_field(self):

        field = Field(self.primary_basis)

        return field


class TwoDimensionalDomain(DomainBase):
    """2D domains: primary basis  + 1D transverse basis"""

    def __init__(self, transverse_basis, primary_basis):

        # Inputs
        self.transverse_basis = transverse_basis
        self.primary_basis = primary_basis


        self.shape = transverse_basis.shape + primary_basis.shape

        # Pencil slices
        self.slices = []
        for i in xrange(transverse_basis.size):
            self.slices.append((i, slice(None)))

    def create_field(self):

        field = Field(self.primary_basis, self.transverse_basis)

        return field

class ThreeDimensionalDomain(DomainBase):
    """3D domains: primary basis + 2D transverse basis"""

    def __init__(self, transverse_basis, primary_basis):

        pass

