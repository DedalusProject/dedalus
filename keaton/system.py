

from collections import OrderedDict


class System(object):
    """System of fields"""

    def __init__(self, field_names, domain):

        # Store inputs
        self.field_names = field_names
        self.domain = domain

        # Build fields
        self.fields = OrderedDict()
        for field_name in field_names:
            self.fields[field_name] = domain.create_field()

    def __getitem__(self, item):

        return self.fields[item]

