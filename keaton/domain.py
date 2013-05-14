

import numpy as np

from primary import PrimaryBasis
from fields import OneDimensionalField


class DomainBase(object):
    """Base class for domains"""

    pass


class OneDimensionalDomain(DomainBase):
    """1D domains: primary basis"""

    def __init__(self, primary_basis):

        # Check basis type
        if not isinstance(primary_basis, PrimaryBasis):
            raise ValueError("'primary_basis' must be a PrimaryBasis instance")

        # Input parameters
        self.primary_basis = primary_basis

        # Grid
        self.grid = primary_basis.grid

        # Pencil slices
        self.slices = [slice(None)]

    def create_field(self):

        field = OneDimensionalField(self.primary_basis)

        return field


class TwoDimensionalDomain(DomainBase):
    """2D domains: primary basis  + 1D transverse basis"""

    def __init__(self, transverse_basis, primary_basis):

        pass


class ThreeDimensionalDomain(DomainBase):
    """3D domains: primary basis + 2D transverse basis"""

    def __init__(self, transverse_basis, primary_basis):

        pass

