

import numpy as np

from .distributor import Distributor
from .field import Field
from .pencil import Pencil
from ..tools.general import CachedMethod, reshape_vector


class Domain:
    """Global domain composed of orthogonal bases."""

    def __init__(self, bases, grid_dtype=np.complex128):

        # Initial attributes
        self.bases = bases
        self.dim = len(bases)
        self.grid_dtype = grid_dtype

        # Iteratively set basis data types
        # (Transforms from grid-to-coeff proceed in the listed order)
        for b in self.bases:
            grid_dtype = b.set_dtypes(grid_dtype)

        # Field management
        self._field_list = list()
        self._field_count = 0

        # Create distributor
        self.distributor = Distributor(self)

        # Construct pencil info
        self._construct_pencil_info()

    def _construct_pencil_info(self):

        # Construct pencil slices
        n_pencils = np.prod([b.coeff_size for b in self.bases])
        n_pencils /= self.bases[-1].coeff_size
        n = np.arange(int(n_pencils))
        index_list = []
        dtrans_list = []

        j = 1
        for b in self.bases:
            if b is not self.bases[-1]:
                bi = divmod(n, j)[0] % b.coeff_size
                index_list.append(bi)
                dtrans_list.append(b.trans_diff([bi]))
                j *= b.coeff_size
            else:
                if self.dim == 1:
                    index_list.append([])
                    dtrans_list.append([])
                else:
                    index_list = list(zip(*index_list))
                    dtrans_list = list(zip(*dtrans_list))

        slices = []
        for bl in index_list:
            sli = []
            for i in bl:
                sli.append(slice(i, i+1))
            sli.append(slice(None))
            slices.append(sli)

        self.pencil_slices = slices
        self.pencils_dtrans = dtrans_list

    @CachedMethod
    def grid(self, axis):

        if not self.distributor.grid_layout.local[axis]:
            raise NotImplementedError("Distributed grid not implemented.")
        #sli = self.distributor.grid_layout.slices[axis]
        grid = self.bases[axis].grid#[sli]
        grid = reshape_vector(grid, self.dim, axis)

        return grid

    def _collect_field(self, field):

        # Clean field
        field.layout = self.distributor.grid_layout
        field.data.fill(0)

        # Add to field list
        self._field_list.append(field)

    def new_field(self):

        # Return a free field if available
        if self._field_list:
            field = self._field_list.pop()
        # Otherwise instantiate another field
        else:
            field = Field(self)

        return field

