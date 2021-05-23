"""
Classes for systems of coefficients/fields.

"""

import numpy as np

from ..tools.general import unify


class CoeffSystem:
    """
    Representation of a collection of fields that don't need to be transformed,
    and are therefore stored as a contigous set of coefficient data, joined
    along the last axis, for efficient pencil and group manipulation.

    Parameters
    ----------
    pencil_length : int
        Number of coefficients in a single pencil
    domain : domain object
        Problem domain

    Attributes
    ----------
    data : ndarray
        Contiguous buffer for field coefficients

    """

    def __init__(self, pencil_length, domain):
        self.pencil_length = pencil_length
        self.domain = domain
        # Allocate data for joined coefficients
        shape = domain.local_coeff_shape.copy()
        shape[-1] = pencil_length
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
    fields : list of field objets
        Fields to join into system

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
    slices : dict
        Dictionary of last-axis slice objects connecting field and system data

    """

    def __init__(self, fields):
        domain = unify(field.domain for field in fields)
        zbasis = domain.bases[-1]
        # Allocate data for joined coefficients
        pencil_length = len(fields) * zbasis.coeff_size
        super().__init__(pencil_length, domain)
        # Create views for each field's data
        self.views = {}
        stride = zbasis.coeff_size
        for i, f in enumerate(fields):
            self.views[f] = self.data[..., i*stride:(i+1)*stride]
        # Attributes
        self.domain = domain
        self.fields = fields
        self.field_names = [f.name for f in self.fields]
        self.nfields = len(self.fields)
        self.field_dict = dict(zip(self.field_names, self.fields))

    def __getitem__(self, name):
        """Return field corresponding to specified name."""
        return self.field_dict[name]

    def gather(self):
        """Copy fields into system buffer."""
        views = self.views
        for field in self.fields:
            field.require_coeff_space()
            np.copyto(views[field], field.data)

    def scatter(self):
        """Extract fields from system buffer."""
        views = self.views
        coeff_layout = self.domain.dist.coeff_layout
        for field in self.fields:
            field.layout = coeff_layout
            np.copyto(field.data, views[field])

