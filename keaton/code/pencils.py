

import numpy as np
from scipy import sparse


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

    def build_matrices(self, problem, basis):

        self.M = (sparse.kron(problem.M0, basis.Eval) +
                  sparse.kron(problem.M1, basis.Deriv) +
                  sparse.kron(problem.ML, basis.Left) +
                  sparse.kron(problem.MR, basis.Right))
        self.L = (sparse.kron(problem.L0, basis.Eval) +
                  sparse.kron(problem.L1, basis.Deriv) +
                  sparse.kron(problem.LL, basis.Left) +
                  sparse.kron(problem.LR, basis.Right))
        self.b = np.kron(problem.b, basis.last)

