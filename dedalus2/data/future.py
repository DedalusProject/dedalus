"""
Classes for future evaluation.

"""

from functools import partial


class Future:
    """
    An object whose value is updated as the simulation progresses.

    """

    __array_priority__ = 100.

    # def __getattr__(self, attr):
    #     # Intercept numpy ufunc calls
    #     if attr in operators.UfuncWrapper.supported:
    #         ufunc = operators.UfuncWrapper.supported[attr]
    #         return partial(operators.UfuncWrapper, ufunc, self)
    #     else:
    #         raise AttributeError("%r object has no attribute %r" %(self.__class__.__name__, attr))

    def __abs__(self):
        return operators.Absolute(self)

    def __neg__(self):
        return operators.Negate(self)

    def __add__(self, other):
        return operators.Add(self, other)

    def __radd__(self, other):
        # Addition is commutative
        return operators.Add(self, other)

    def __sub__(self, other):
        # self - other
        return self + (-1)*other
        #return operators.Subtract(self, other)

    def __rsub__(self, other):
        # other - self
        return other + (-1)*self
        #return operators.Subtract(other, self)

    def __mul__(self, other):
        return operators.Multiply(self, other)

    def __rmul__(self, other):
        # Multiplication is commutative
        return operators.Multiply(self, other)

    def __truediv__(self, other):
        # Input: self / other
        return self * other**(-1)
        #return operators.Divide(self, other)

    def __rtruediv__(self, other):
        # Input: other / self
        return other * self**(-1)
        #return operators.Divide(other, self)

    def __pow__(self, other):
        return operators.Power(self, other)


from . import operators

