"""
Classes for systems of coefficients/fields.

"""

import numpy as np

from ..tools.cache import CachedMethod
from ..tools.general import unify


class CoeffSystem:
    """
    Representation of a collection of fields that don't need to be transformed,
    and are therefore stored as a contigous set of coefficient data for
    efficient pencil and group manipulation.

    Parameters
    ----------
    nfields : int
        Number of fields to represent
    domain : domain object
        Problem domain

    Attributes
    ----------
    data : ndarray
        Contiguous buffer for field coefficients

    """

    def __init__(self, nfields, domain):

        # Allocate data for joined coefficients
        # Extend along last axis
        shape = domain.local_coeff_shape.copy()
        shape[-1] *= nfields
        dtype = domain.dist.coeff_layout.dtype
        self.data = np.zeros(shape, dtype=dtype)

    def get_pencil(self, pencil):
        """Return pencil view from system buffer."""
        return self.data[pencil.local_index]

    def set_pencil(self, pencil, data):
        """Set pencil data in system buffer."""
        np.copyto(self.data[pencil.local_index], data)


class FieldSystem(CoeffSystem):
    """
    Collection of fields alongside a CoeffSystem buffer for efficient pencil
    and group manipulation.

    Parameters
    ----------
    field_names : list of strings
        Names of fields to build
    domain : domain object
        Problem domain

    Attributes
    ----------
    data : ndarray
        Contiguous buffer for field coefficients
    fields : list
        Field objects
    nfields : int
        Number of fields in system
    field_dict : dict
        Dictionary of fields

    """

    def __init__(self, field_names, domain):

        # Build fields
        fields = [domain.new_field(name=fn) for fn in field_names]
        nfields = len(fields)

        # Allocate data for joined coefficients
        super().__init__(nfields, domain)

        # Attributes
        self.domain = domain
        self.field_names = field_names
        self.fields = fields
        self.nfields = nfields
        self.field_dict = dict(zip(field_names, fields))

    def __getitem__(self, name):
        """Return field corresponding to specified name."""
        return self.field_dict[name]

    @classmethod
    def from_fields(cls, fields):
        names = [field.name for field in fields]
        domain = unify(field.domain for field in fields)
        sys = FieldSystem(names, domain)
        sys.fields = fields
        sys.field_dict = dict(zip(names, fields))
        return sys

    def gather(self):
        """Copy fields into system buffer."""
        stride = self.nfields
        for start, field in enumerate(self.fields):
            field.require_coeff_space()
            np.copyto(self.data[..., start::stride], field.data)

    def scatter(self):
        """Extract fields from system buffer."""
        stride = self.nfields
        coeff_layout = self.domain.dist.coeff_layout
        for start, field in enumerate(self.fields):
            field.layout = coeff_layout
            np.copyto(field.data, self.data[..., start::stride])

