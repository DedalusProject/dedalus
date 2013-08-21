

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
        self.M0 = [np.zeros((size, size), dtype=np.complex128) for i in range(order)]
        self.M1 = [np.zeros((size, size), dtype=np.complex128) for i in range(order)]
        self.L0 = [np.zeros((size, size), dtype=np.complex128) for i in range(order)]
        self.L1 = [np.zeros((size, size), dtype=np.complex128) for i in range(order)]
        self.ML = np.zeros((size, size), dtype=np.complex128)
        self.MR = np.zeros((size, size), dtype=np.complex128)
        self.LL = np.zeros((size, size), dtype=np.complex128)
        self.LR = np.zeros((size, size), dtype=np.complex128)
        self.b = np.zeros(size, dtype=np.complex128)
        self.F = [None] * self.size


# Wave equation:  y_tt = c2 y_xx
#
# y_t - v = 0
# y_x - dy = 0
# v_t - c2 dy_x = 0
#
# y(left) = 0
# y(right) = 0
#
# Index ordering:  Matrix[Order][Equation][Variable]
#
wave_equation_1d = Problem(['y', 'dy', 'v'], 3)

wave_equation_1d.M0[0][0][0] = 1.
wave_equation_1d.L0[0][0][2] = -1.

wave_equation_1d.L0[0][1][1] = -1.
wave_equation_1d.L1[0][1][0] = 1.

wave_equation_1d.M0[0][2][2] = 1.
wave_equation_1d.L1[0][2][1] = -9./8.
wave_equation_1d.L1[1][2][1] = -1.
wave_equation_1d.L1[2][2][1] = -1./8.

wave_equation_1d.LL[1][0] = 1.
wave_equation_1d.LR[2][0] = 1.

# wave_equation_1d.M0[0] = np.array([[1., 0., 0.],
#                                    [0., 0., 0.],
#                                    [0., 0., 1.]])
# wave_equation_1d.L0[0] = np.array([[0., 0., -1.],
#                                    [0., -1., 0.],
#                                    [0., 0., 0.]])
# wave_equation_1d.L1[0] = np.array([[0., 0., 0.],
#                                    [1., 0., 0.],
#                                    [0., -9./8., 0.]])
# wave_equation_1d.L1[1] = np.array([[0., 0., 0.],
#                                    [0., 0., 0.],
#                                    [0., -1., 0.]])
# wave_equation_1d.L1[2] = np.array([[0., 0., 0.],
#                                    [0., 0., 0.],
#                                    [0., -1./8., 0.]])
# wave_equation_1d.LL = np.array([[0., 0., 0.],
#                                 [1., 0., 0.],
#                                 [0., 0., 0.]])
# wave_equation_1d.LR = np.array([[0., 0., 0.],
#                                 [0., 0., 0.],
#                                 [1., 0., 0.]])


# Heat equation: y_t = y_xx
#
# y_x - dy = 0
# y_t - dy_x = 0
#
heat_equation_1d = Problem(['y', 'dy'], 1)
heat_equation_1d.M0[0] = np.array([[0., 0.],
                                   [1., 0.]])
heat_equation_1d.L0[0] = np.array([[0., -1.],
                                   [0., 0.]])
heat_equation_1d.L1[0] = np.array([[1., 0.],
                                   [0., -1.]])
heat_equation_1d.LL = np.array([[1., 0.],
                                [0., 0.]])
heat_equation_1d.LR = np.array([[0., 0.],
                                [1., 0.]])
heat_equation_1d.b = np.array([1., 1.])

