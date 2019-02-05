

class CoordinateSystem:

    def __init__(self, names, dist, axis):
        if len(names) != self.dim:
            raise ValueError("Number of coordinate names must match dimension.")
        self.names = names
        self.dist = dist
        self.axis = axis


class Coordinate(CoordinateSystem):
    dim = 1


class CartesianCoordinates(CoordinateSystem):

    def __init__(self, names, dist, axis):
        self.dim = len(names)
        super().__init__(names, dist, axis)


class S2Coordinates(CoordinateSystem):
    dim = 2


class SphericalCoordinates(CoordinateSystem):
    dim = 3


class PolarCoordinates(CoordinateSystem):
    dim = 2


