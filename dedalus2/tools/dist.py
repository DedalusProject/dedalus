

import numpy as np
try:
    from mpi4py import MPI
except ImportError:
    MPI = None
    print('Cannot import mpi4py. Parallelism disabled.')


class Distributor:

    def __init__(self, domain):

        # Initial attributes
        self.domain = domain

        # MPI communicator
        if MPI:
            self.communicator = MPI.COMM_WORLD
            self.local_process = self.communicator.Get_rank()
            self.total_processes = self.communicator.Get_size()
        else:
            self.communicator = None
            self.local_process = 0
            self.total_processes = 1

    def build_layouts(self):

        if self.total_processes > 1:
            mesh = [self.total_processes]
        else:
            mesh = []

        # Build layouts
        self.layouts = []
        for i in range(domain.dim+len(mesh)+1):
            self.layouts.append(Layout(domain, mesh, i))

        self.coeff_layout = self.layouts[0]
        self.grid_layout = self.layouts[-1]

        # Compute buffer size
        self.buffer_size = max([l.buffer_size for l in self.layouts])

    # def increment_layout(self, field):

    #     index = field.layout.index
    #     domain = field.domain

    #     self.increment[domain][index](field)

    #     field.layout = self.layouts[domain][index + 1]

    # def decrement_layout(self, field):

    #     index = field.layout.index
    #     domain = field.domain

    #     input = field.data
    #     field.layout = self.layouts[domain][index - 1]
    #     output = field.data
    #     self.decrement[domain][index](input, output)


class Layout:

    def __init__(self, domain, mesh, index):

        # Sizes
        d = domain.dim
        r = len(mesh)

        if r >= d:
            raise ValueError("r must be less than d")

        # Build local and grid space flags
        self.local = [False] * r + [True] * (d-r)
        self.grid_space = [False] * d
        self.dtype = domain.bases[-1].coeff_dtype

        for op in range(index):
            for i in reversed(range(d)):
                if not self.grid_space[i]:
                    if self.local[i]:
                        self.grid_space[i] = True
                        self.dtype = domain.bases[i].grid_dtype
                        break
                    else:
                        self.local[i] = True
                        self.local[i+1] = False
                        break

        # Build global shape
        global_shape = []
        for i in range(d):
            if self.grid_space[i]:
                global_shape.append(domain.bases[i].grid_size)
            else:
                global_shape.append(domain.bases[i].coeff_size)

        # Build local shape
        j = 0
        self.shape = global_shape
        for i in range(d):
            if not self.local[i]:
                self.shape[i] /= mesh[j]
                j += 1

        # Compute necessary buffer size
        n_bytes = np.dtype(self.dtype).itemsize
        self.buffer_size = np.prod(self.shape) * n_bytes

    def view_data(self, buffer):

        data = np.ndarray(shape=self.shape,
                          dtype=self.dtype,
                          buffer=buffer)

        return data

