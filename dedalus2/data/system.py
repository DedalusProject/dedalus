

from collections import OrderedDict

from .pencil import Pencil


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

        # Handle pencils and field names
        if isinstance(item, Pencil):
            return item.get(self)
        else:
            return self.fields[item]

    def __setitem__(self, item, data):

        # Handle pencils and field names
        if isinstance(item, Pencil):
            item.set(self, data)
        else:
            self.fields[item] = data

