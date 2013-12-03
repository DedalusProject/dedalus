

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
        self.F = [None] * self.size
        self.ML = lambda d_trans: np.zeros((size, size), dtype=np.complex128)
        self.MR = lambda d_trans: np.zeros((size, size), dtype=np.complex128)
        self.MI = lambda d_trans: np.zeros((size, size), dtype=np.complex128)
        self.LL = lambda d_trans: np.zeros((size, size), dtype=np.complex128)
        self.LR = lambda d_trans: np.zeros((size, size), dtype=np.complex128)
        self.LI = lambda d_trans: np.zeros((size, size), dtype=np.complex128)
        # self.BL = [None] * self.size
        # self.BR = [None] * self.size
        # self.BI = [None] * self.size
        self.b = lambda d_trans: np.zeros(size, dtype=np.complex128)

