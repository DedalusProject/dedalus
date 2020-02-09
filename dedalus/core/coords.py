"""Coordinates and coordinate sytems."""


class CoordinateSystem:
    pass


class Coordinate(CoordinateSystem):
    dim = 1

    def __init__(self, name):
        self.name = name

    def __print__(self):
        return self.name


class CartesianCoordinates(CoordinateSystem):

    def __init__(self, *names):
        self.names = names
        self.dim = len(names)
        self.coords = tuple(Coordinate(name) for name in names)


class S2Coordinates(CoordinateSystem):
    dim = 2

    def __init__(self, azimuth, colatitude):
        self.azimuth = Coordinate(azimuth)
        self.colatitude = Coordinate(colatitude)


class SphericalCoordinates(CoordinateSystem):
    dim = 3

    def __init__(self, azimuth, colatitude, radius):
        self.azimuth = Coordinate(azimuth)
        self.colatitude = Coordinate(colatitude)
        self.radius = Coordinate(radius)


