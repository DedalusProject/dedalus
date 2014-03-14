"""
Classes for future evaluation.

"""


class Future:
    """
    An object whose value is updated as the simulation progresses.

    """

    __array_priority__ = 100.

    def __getattr__(self, attr):
        # Intercept numpy ufunc calls
        if attr in UfuncWrapper.supported:
            ufunc = UfuncWrapper.supported[attr]
            return partial(UfuncWrapper, ufunc, self)
        else:
            raise AttributeError("%r object has no attribute %r" %(self.__class__.__name__, attr))

    def __abs__(self):
        return operators.Absolute(self)

    def __neg__(self):
        return operators.Negate(self)

    def __add__(self, other):
        return operators.Add(self, other)

    def __radd__(self, other):
        return operators.Add(other, self)

    def __sub__(self, other):
        return operators.Subtract(self, other)

    def __rsub__(self, other):
        return operators.Subtract(other, self)

    def __mul__(self, other):
        return operators.Multiply(self, other)

    def __rmul__(self, other):
        return operators.Multiply(other, self)

    def __truediv__(self, other):
        return operators.Divide(self, other)

    def __rtruediv__(self, other):
        return operators.Divide(other, self)

    def __pow__(self, other):
        return operators.Power(self, other)


from . import operators

