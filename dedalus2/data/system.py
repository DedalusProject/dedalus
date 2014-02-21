"""
Classes for systems of coefficients/fields.

"""

import numpy as np

from ..tools.cache import CachedMethod


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

        # Allocate data with extended coeff data shape
        coeff_layout = domain.distributor.coeff_layout
        shape = np.copy(coeff_layout.shape)
        shape[-1] *= nfields
        self.data = np.zeros(shape, dtype=coeff_layout.dtype)

    @CachedMethod
    def get_pencil(self, pencil):
        """Return pencil from system buffer."""

        return self.data[pencil.index]

    def set_pencil(self, pencil, data):
        """Set pencil in system buffer."""

        np.copyto(self.get_pencil(pencil), data)


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
        fields = [domain.new_field() for fn in field_names]
        for i, f in enumerate(fields):
            f.name = field_names[i]
        field_dict = dict(zip(field_names, fields))
        nfields = len(field_names)

        # Allocate data with extended coeff data shape
        coeff_layout = domain.distributor.coeff_layout
        shape = np.copy(coeff_layout.shape)
        shape[-1] *= nfields
        self.data = np.zeros(shape, dtype=coeff_layout.dtype)

        # References
        self.domain = domain
        self.field_names = field_names
        self.fields = fields
        self.nfields = nfields
        self.field_dict = field_dict
        self._coeff_layout = coeff_layout

    def __getitem__(self, name):
        """Return field corresponding to specified name."""

        return self.field_dict[name]

    def gather(self):
        """Copy fields into system buffer."""

        stride = self.nfields
        for start, field in enumerate(self.fields):
            field.require_coeff_space()
            np.copyto(self.data[..., start::stride], field.data)

    def scatter(self):
        """Extract fields from system buffer."""

        stride = self.nfields
        for start, field in enumerate(self.fields):
            field.layout = self._coeff_layout
            np.copyto(field.data, self.data[..., start::stride])

