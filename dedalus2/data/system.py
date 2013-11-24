

from collections import OrderedDict


class System:
    """Collection of fields."""

    def __init__(self, field_names, domain):

        # Initial attributes
        self.field_names = field_names
        self.n_fields = len(field_names)
        self.domain = domain

        # Build fields
        self.fields = OrderedDict()
        for fn in field_names:
            self.fields[fn] = domain.new_field()

    def __getitem__(self, item):

        return self.fields[item]

    def __setitem__(self, item, data):

        self.fields[item] = data

