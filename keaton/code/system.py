

from collections import OrderedDict

from pencils import Pencil


class System(object):
    """Collection of fields"""

    def __init__(self, field_names, domain):

        # Store inputs
        self.field_names = field_names
        self.domain = domain
        self.N = len(field_names)

        # Build fields
        self.fields = OrderedDict()
        for fn in field_names:
            self.fields[fn] = domain.create_field()

    def __getitem__(self, item):

        if isinstance(item, Pencil):
            return item.get(self)
        else:
            return self.fields[item]

    def __setitem__(self, item, data):

        if isinstance(item, Pencil):
            item.set(self, data)
        else:
            raise NotImplementedError()

    # def __iadd__(self, other):

    #     if self.field_names != other.field_names:
    #         raise ValueError("Cannot add systems with incompatible field names.")

    #     for fn in self.field_names:
    #         self.fields[fn] += other.fields[fn]

