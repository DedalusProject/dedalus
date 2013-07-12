

import numpy as np


class Pencil(object):
    """Pencil object for viewing one k_trans across system"""

    def __init__(self, slice):

        self.slice = slice

    def get(self, system):
        """Retrieve slice of all fields in system"""

        data = []

        for field in system.fields.itervalues():
            data.append(field['k'][self.slice])

        data = np.hstack(data)

        return data

    def set(self, system, data):
        """Set slice of all fields in system"""

        start = 0
        for field in system.fields.itervalues():
            end = start + field.domain.bases[-1].size
            field['k'][self.slice] = data[start:end]
            start = end

