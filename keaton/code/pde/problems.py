

import numpy as np


class Problem(object):
    """Class defining PDE through matrices and non-linear constructor"""

    def __init__(self, field_names):

        self.order = 0

        self.field_names = field_names
        self.size = len(field_names)


# Wave equation:  y_tt = c2 y_xx
#
# y_t - v = 0
# y_x - dy = 0
# v_t - c2 dy_x = 0
#
wave_equation_1d = Problem(['y', 'dy', 'v'])
wave_equation_1d.order = 3
wave_equation_1d.M0 = [np.array([[1., 0., 0.],
                                 [0., 0., 0.],
                                 [0., 0., 1.]])]
wave_equation_1d.M1 = [np.array([[0., 0., 0.],
                                 [0., 0., 0.],
                                 [0., 0., 0.]])]
wave_equation_1d.L0 = [np.array([[0., 0., -1.],
                                 [0., -1., 0.],
                                 [0., 0., 0.]])]
wave_equation_1d.L1 = [np.array([[0., 0., 0.],
                                 [1., 0., 0.],
                                 [0., -9./8., 0.]]),
                       np.array([[0., 0., 0.],
                                 [0., 0., 0.],
                                 [0., -1., 0.]]),
                       np.array([[0., 0., 0.],
                                 [0., 0., 0.],
                                 [0., -1./8., 0.]])]
wave_equation_1d.ML = np.array([[0., 0., 0.],
                                [0., 0., 0.],
                                [0., 0., 0.]])
wave_equation_1d.MR = np.array([[0., 0., 0.],
                                [0., 0., 0.],
                                [0., 0., 0.]])
wave_equation_1d.LL = np.array([[0., 0., 0.],
                                [1., 0., 0.],
                                [0., 0., 0.]])
wave_equation_1d.LR = np.array([[0., 0., 0.],
                                [0., 0., 0.],
                                [1., 0., 0.]])
wave_equation_1d.b = np.array([0., 0., 0.])


# Heat equation: y_t = y_xx
#
# y_x - dy = 0
# y_t - dy_x = 0
#
heat_equation_1d = Problem(['y', 'dy'])
heat_equation_1d.M0 = np.array([[0., 0.],
                                [1., 0.]])
heat_equation_1d.M1 = np.array([[0., 0.],
                                [0., 0.]])
heat_equation_1d.L0 = np.array([[0., -1.],
                                [0., 0.]])
heat_equation_1d.L1 = np.array([[1., 0.],
                                [0., -1.]])
heat_equation_1d.ML = np.array([[0., 0.],
                                [0., 0.]])
heat_equation_1d.MR = np.array([[0., 0.],
                                [0., 0.]])
heat_equation_1d.LL = np.array([[1., 0.],
                                [0., 0.]])
heat_equation_1d.LR = np.array([[0., 0.],
                                [1., 0.]])
heat_equation_1d.b = np.array([1., 1.])

