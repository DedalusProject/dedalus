"""Coordinates and coordinate sytems."""


class CoordinateSystem:

    def __eq__(self, other):
        if type(self) is type(other):
            if len(self.coords) == len(other.coords):
                for i, coord in enumerate(self.coords):
                    if coord != other.coords[i]:
                        return False
                    return True
        return False

    def __hash__(self):
        return id(self)

    def set_distributor(self, distributor):
        for coord in self.coords:
            coord.dist = distributor
            

class Coordinate:
    dim = 1

    def __init__(self, name):
        self.name = name
        self.coords = (self,)

    def __str__(self):
        return self.name

    def __eq__(self, other):
        if self.name == other.name: return True
        else: return False

    def __hash__(self):
        return id(self)

    @property
    def axis(self):
        return self.dist.coords.index(self)

    def set_distributor(self, distributor):
        self.dist = distributor


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

    @property
    def axis(self):
        return self.azimuth.axis


class SphericalCoordinates(CoordinateSystem):
    dim = 3

    def __init__(self, azimuth, colatitude, radius):
        self.azimuth = Coordinate(azimuth)
        self.colatitude = Coordinate(colatitude)
        self.radius = Coordinate(radius)
        self.S2cs = S2Coordinates(azimuth, colatitude)
        self.coords = (self.azimuth, self.colatitude, self.radius)

    @property
    def axis(self):
        return self.azimuth.axis

    def set_distributor(self, distributor):
        super().set_distributor(distributor)
        self.S2cs.set_distributor(distributor)

