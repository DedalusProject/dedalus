"""
Class for systems of fields.

"""

from collections import OrderedDict


class System:
    """
    Collection of fields.

    Parameters
    ----------
    field names : list of strings
        Names of fields, to be used as keys in field dictionary
    domain : domain object
        Problem domain

    Attributes
    ----------
    n_fields : int
        Number of fields in system
    fields : ordered dict
        Field objects

    """

    def __init__(self, field_names, domain):

        # Initial attributes
        self.field_names = field_names
        self.n_fields = len(field_names)
        self.domain = domain

        # Build fields
        self.fields = OrderedDict()
        for fn in field_names:
            self.fields[fn] = domain.new_field()

    def __getitem__(self, name):
        """Return field corresponding to specified name."""

        return self.fields[name]

