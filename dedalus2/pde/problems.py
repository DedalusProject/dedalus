

import numpy as np


class Problem:
    """Class defining PDE through matrices and non-linear constructor"""

    def __init__(self, field_names, order=1):

        # Inputs
        self.field_names = field_names
        self.order = order

        # Parameters
        self.size = len(field_names)
        self.parameters = {}

        # Build matrix lists
        size = self.size
        self.M0 = [lambda d_trans: np.zeros((size, size), dtype=np.complex128) for i in range(order)]
        self.M1 = [lambda d_trans: np.zeros((size, size), dtype=np.complex128) for i in range(order)]
        self.L0 = [lambda d_trans: np.zeros((size, size), dtype=np.complex128) for i in range(order)]
        self.L1 = [lambda d_trans: np.zeros((size, size), dtype=np.complex128) for i in range(order)]
        self.ML = np.zeros((size, size), dtype=np.complex128)
        self.MR = np.zeros((size, size), dtype=np.complex128)
        self.LL = np.zeros((size, size), dtype=np.complex128)
        self.LR = np.zeros((size, size), dtype=np.complex128)
        self.b = np.zeros(size, dtype=np.complex128)
        self.F = [None] * self.size

