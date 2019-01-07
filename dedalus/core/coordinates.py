


import numpy as np


class VectorSpace:
    """Collection of coordinates forming a vector space."""

    def __init__(self, spaces):
        from .domain import Domain
        self.domain = Domain(spaces[0].dist, spaces)
        self.spaces = self.domain.spaces
        space_dims = [space.dim for space in self.spaces]
        space_indeces = np.cumsum(space_dims) - space_dims[0]
        self.indeces = dict(zip(self.spaces, space_indeces))
        self.dim = sum(space_dims)

    def get_index(self, space):
        return self.indeces[space]

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