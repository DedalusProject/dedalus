

import numpy as np

from .field import Field


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
            grid_dtype = b.set_dtype(grid_dtype)

        # Construct domain grids
        grids = [b.grid for b in bases]
        if self.dim > 1:
            self.grids = np.meshgrid(*grids, indexing='ij')

        # Construct pencil slices
        n_pencils = np.prod([b.coeff_size for b in bases])
        n_pencils /= bases[-1].coeff_size
        n = np.arange(int(n_pencils))
        index_list = []
        dtrans_list = []

        j = 1
        for b in bases:
            if b is not bases[-1]:
                bi = divmod(n, j)[0] % b.coeff_size
                index_list.append(bi)
                dtrans_list.append(b.trans_diff([bi]))
                j *= b.coeff_size
            else:
                if len(bases) == 1:
                    index_list.append([])
                    dtrans_list.append([])
                else:
                    index_list = list(zip(*index_list))
                    dtrans_list = list(zip(*dtrans_list))

        self.dtrans_list = dtrans_list
        self.slices = []
        for bl in index_list:
            sli = []
            for i in bl:
                sli.append(slice(i, i+1))
            sli.append(slice(None))
            self.slices.append(sli)

        # Field management
        self._field_list = list()
        self._field_count = 0

    # def grid(self, layout):

    #     slices = layout.slices
    #     grids = self.grids

    #     local_grids = [g[s] for (g, s) in zip(grids, slices)]

    #     return np.meshgrid(*local_grids, indexing='ij')

    def integrate(self, field):
        """Integrate field over domain."""

        # Integrate by coefficients
        data = field['K']
        for b in self.bases:
            data = b.integrate(data, 0)
            data = b.grid_dtype(data)

        return data

    def _collect_field(self, field):

        # Clean field
        # field.layout = distributor.layouts[field.domain][0]
        # field.data *= 0.

        # Add to field list
        self._field_list.append(field)

    def get_field(self):

        # Return a free field if available
        if self._field_list:
            field = self._field_list.pop()
        # Otherwise build a new field
        else:
            field = Field(self)

        return field

