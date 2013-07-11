

from collections import OrderedDict


class System(object):
    """Collection of fields"""

    def __init__(self, field_names, domain):

        # Store inputs
        self.field_names = field_names
        self.domain = domain
        self.N = len(field_names)

        # Build fields
        self.fields = OrderedDict()
        for field_name in field_names:
            self.fields[field_name] = domain.create_field()

    def __getitem__(self, item):

        return self.fields[item]

    # def __iadd__(self, other):

    #     if self.field_names != other.field_names:
    #         raise ValueError("Cannot add systems with incompatible field names.")

    #     for fn in self.field_names:
    #         self.fields[fn] += other.fields[fn]

