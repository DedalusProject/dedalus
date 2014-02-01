"""
Class for systems of fields.

"""

from collections import OrderedDict
import numpy as np


class System:
    """
    Collection of fields with continguous internal buffer for efficient
    computations.

    Parameters
    ----------
    field_names : list of strings
        Names of fields, to be used as keys in field dictionary
    domain : domain object
        Problem domain

    Attributes
    ----------
    data : ndarray
        Contiguous buffer for field
    fields : ordered dict
        Field objects
    nfields : int
        Number of fields in system

    """

    def __init__(self, field_names, domain):

        # Handle nameless input
        if np.isscalar(field_names):
            nfields = field_names
            field_names = [i for i in range(nfields)]
        else:
            nfields = len(field_names)

        # Build fields
        fields = OrderedDict()
        for fn in field_names:
            fields[fn] = domain.new_field()

        # Allocate data with extended coeff data shape
        coeff_layout = domain.distributor.coeff_layout
        shape = np.copy(coeff_layout.shape)
        shape[-1] *= nfields
        self.data = np.zeros(shape, dtype=coeff_layout.dtype)

        # References
        self.fields = fields
        self.nfields = nfields
        self._coeff_layout = coeff_layout

    def __getitem__(self, name):
        """Return field corresponding to specified name."""

        return self.fields[name]

    def gather(self):
        """Copy fields into system buffer."""

        stride = self.nfields
        for start, field in enumerate(self.fields.values()):
            field.require_coeff_space()
            np.copyto(self.data[..., start::stride], field.data)

    def scatter(self):
        """Extract fields from system buffer."""

        stride = self.nfields
        for start, field in enumerate(self.fields.values()):
            field.layout = self._coeff_layout
            np.copyto(field.data, self.data[..., start::stride])

    def get_pencil(self, pencil):
        """Return pencil from system buffer."""

        return self.data[pencil.indeces]

    def set_pencil(self, pencil, data):
        """Set pencil in system buffer."""

        np.copyto(self.data[pencil.indeces], data)

