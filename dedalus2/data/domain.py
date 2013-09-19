

import numpy as np


class Domain:
    """Domain composed of orthogonal bases."""

    def __init__(self, bases, dtype=np.complex128):

        # Properties
        self.bases = bases
        self.dtype = dtype

        # Iteratively set basis data types
        # (Transforms from grid-space proceed in the listed order)
        for b in self.bases:
            dtype = b.set_dtype(dtype)

        # Construct pencil slices
        n_pencils = np.prod([b.coeff_size for b in bases])
        n_pencils /= bases[-1].coeff_size
        n = np.arange(int(n_pencils))
        bi_list = []
        d_list = []

        j = 1
        for b in bases:
            if b is not bases[-1]:
                bi = divmod(n, j)[0] % b.coeff_size
                bi_list.append(bi)
                d_list.append(b.diff_factor([bi]))
                j *= b.coeff_size
            else:
                if len(bases) == 1:
                    bi_list.append(bi)
                    d_list.append([])

        self.d_list = np.transpose(d_list)
        self.slices = []
        for bl in np.transpose(bi_list):
            sli = []
            for i in bl:
                sli.append(slice(i, i+1))
            sli.append(slice(None))
            self.slices.append(sli)


        # Build shape
        self.shape = [b.grid_size for b in bases]
        self.dim = len(self.shape)

        # Grid
        self.grids = [b.grid for b in bases]
        if self.dim > 1:
            self.grids = np.meshgrid(*self.grids, indexing='ij')


    #     # Pencil slices
    #     num_pencils = np.prod(self.shape) / self.shape[-1]
    #     self.slices = [slice(None)]

    # def grid(self, layout):

    #     slices = layout.slices
    #     grids = self.grids

    #     local_grids = [g[s] for (g, s) in zip(grids, slices)]

    #     return np.meshgrid(*local_grids, indexing='ij')
