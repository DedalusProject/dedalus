

class Space:

    def __init__(self, dim, name=None):
        self.dim = dim
        self.name = name

    def __mul__(self, other):
        return ProductSpace(self, other)


class ProductSpace(Space):

    def __init__(self, *spaces, name=None):
        self.spaces = spaces
        self.dim = sum(space.dim for space in spaces)
        self.name = name




class CoordinateSystem:

    def __init__(self, space, names):
        if space.dim != self.dim:
            raise ValueError("Coordinate and space dimensions must match.")
        if len(names) != self.dim:
            raise ValueError("Number of coordinate names must match dimension.")
        self.space = space
        self.names = names


class Coordinate(CoordinateSystem):

    dim = 1


class CartesianCoordinates(CoordinateSystem):

    def __init__(self, space, names):
        self.dim = len(names)
        super().__init__(space, names)


class S2Coordinates(CoordinateSystem):

    dim = 2


class SphericalCoordinates(CoordinateSystem):

    dim = 3


class PolarCoordinates(CoordinateSystem):

    dim = 2


class CylindricalCoordinates(CoordinateSystem):

    dim = 3



