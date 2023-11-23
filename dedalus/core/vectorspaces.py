


import numpy as np


"""
Idea: just make VS based on string coordinates?
Eliminates issues with 2sphere basis with 3sphere vectors?

"""


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


class TensorSignature:

    def __init__(self, vector_spaces):
        self.vector_spaces = vector_spaces
        self.tensor_order = len(vector_spaces)
        self.tensor_shape = tuple(vs.dim for vs in vector_spaces)
        self.n_components = prod(self.tensor_shape)


