"""Classes representing coordinate spaces."""


class Coordinate:
    """Single coordinate."""
    dim = 1

    def __init__(self, name):
        self.name = name
        self.names = [name]


class CoordinateSet:
    """Base class for coordinate sets."""

    def __init__(self, *names):
        self.names = names
        self.coords = tuple(Coordinate(n) for n in names)
        self._check_dim()

    def _check_dim(self):
        if len(self.coords) != self.dim:
            raise ValueError("Number of coordinates must match fixed dimension: %i" %self.dim)


class PolarCoords(CoordinateSet):
    """2D polar coordinates (azimuth, radius)."""
    dim = 2


class SphericalCoords2D(CoordinateSet):
    """2D spherical coordinates (azimuth, colatitude)."""
    dim = 2


class SphericalCoords3D(CoordinateSet):
    """3D spherical coordinates (azimuth, colatitute, radius)."""
    dim = 3








# class VectorSpace:
#     """Collection of coordinates forming a vector space."""

#     def __init__(self, coords):
#         self.coords = coords
#         self.dim = sum(c.dim for c in coords)



# class TensorField:

#     def __init__(self, dist, tens=None, bases=None):
#         if tens is None:
#             tens = []
#         if bases is None:
#             bases = []
#         self.dist = dist
#         self.tens = tens
#         self.bases = bases
#         self.order = len(tens)
#         self.domains = [b.domain for b in bases]

#         self.tens_shape = tuple(t.dim for t in tens)
#         self.data_shape = np.concatenate(b.shape for b in bases)
#         self.n_components = np.prod(self.tens_shape)