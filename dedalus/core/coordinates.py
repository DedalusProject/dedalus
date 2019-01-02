"""Classes representing coordinate spaces."""


class Coordinate:
    """Single coordinate."""

    def __init__(self, name):
        self.name = name
        self.names = [name]
        self.dim = 1


class CoordinateSet:
    """Base class for coordinate sets."""

    def __init__(self, *names):
        self.names = names
        self.coords = tuple(Coordinate(n) for n in names)
        self.dim = len(names)
        self._check_dim()

    def _check_dim(self):
        if self.fixed_dim:
            if self.dim != self.fixed_dim:
                raise ValueError("Number of coordinates must match fixed dimension: %i" %self.fixed_dim)


class SphericalCoords2D(Coordinates):
    """2D spherical coordinates (azimuth, colatitude)."""
    fixed_dim = 2


class SphericalCoords3D(Coordinates):
    """3D spherical coordinates (azimuth, colatitute, radius)."""
    fixed_dim = 3


class PolarCoords(Coordinates):
    """2D polar coordinates (azimuth, radius)."""
    fixed_dim = 2










class VectorSpace:
    """Collection of coordinates forming a vector space."""

    def __init__(self, coords):
        self.coords = coords
        self.dim = sum(c.dim for c in coords)






class Domain:
    """Class representing a subdomain of a coordinate space."""

    def _check_coords(self):
        if not isinstance(self.coords, self.coord_dtype):
            raise ValueError("Invalid coordinate type.")


class Interval(Domain):
    pass







class TensorField:

    def __init__(self, dist, tens=None, bases=None):
        if tens is None:
            tens = []
        if bases is None:
            bases = []
        self.dist = dist
        self.tens = tens
        self.bases = bases
        self.order = len(tens)
        self.domains = [b.domain for b in bases]

        self.tens_shape = tuple(t.dim for t in tens)
        self.data_shape = np.concatenate(b.shape for b in bases)
        self.n_components = np.prod(self.tens_shape)