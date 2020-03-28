"""Coordinates and coordinate sytems."""


class CoordinateSystem:
    pass


class Coordinate(CoordinateSystem):
    dim = 1

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    @property
    def axis(self):
        return self.dist.coords.index(self)


class CartesianCoordinates(CoordinateSystem):

    def __init__(self, *names):
        self.names = names
        self.dim = len(names)
        self.coords = tuple(Coordinate(name) for name in names)

    def __str__(self):
        return '{' + ','.join([c.name for c in self.coords]) + '}'


class S2Coordinates(CoordinateSystem):
    dim = 2

    def __init__(self, azimuth, colatitude):
        self.azimuth = Coordinate(azimuth)
        self.colatitude = Coordinate(colatitude)
        self.coords = (self.azimuth, self.colatitude)

class SphericalCoordinates(CoordinateSystem):
    dim = 3

    def __init__(self, azimuth, colatitude, radius):
        self.azimuth = Coordinate(azimuth)
        self.colatitude = Coordinate(colatitude)
        self.radius = Coordinate(radius)
        self.coords = (self.azimuth, self.colatitude, self.radius)

    @property
    def axis(self):
        return self.azimuth.axis


