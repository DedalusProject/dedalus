

import numpy as np
import weakref
import functools
try:
    from mpi4py import MPI
    print('Successfully imported mpi4py.  Parallelism enabled.')
except ImportError:
    MPI = None
    print('Cannot import mpi4py. Parallelism disabled.')


class Distributor:

    def __init__(self, domain):

        # Initial attributes
        self.domain = weakref.ref(domain)

        # MPI communicator
        if MPI:
            self.communicator = MPI.COMM_WORLD
            self.local_process = self.communicator.Get_rank()
            self.total_processes = self.communicator.Get_size()
        else:
            self.communicator = None
            self.local_process = 0
            self.total_processes = 1

        # Build layouts
        self._build_layouts()

    def _build_layouts(self):

        if self.total_processes > 1:
            mesh = [self.total_processes]
        else:
            mesh = []

        # Build layouts
        self.layouts = []
        self.increment = []
        self.decrement = []

        # Sizes
        domain = self.domain()
        d = domain.dim
        r = len(mesh)

        if r >= d:
            raise ValueError("r must be less than d")

        # Build local and grid space flags
        local = [False] * r + [True] * (d-r)
        grid_space = [False] * d
        dtype = domain.bases[-1].coeff_dtype

        for index in range(r+d+1):
            operation = None
            for op in range(index):
                for i in reversed(range(d)):
                    if not grid_space[i]:
                        if local[i]:
                            grid_space[i] = True
                            dtype = domain.bases[i].grid_dtype
                            operation = 'transform'
                            op_index = i
                            break
                        else:
                            local[i] = True
                            local[i+1] = False
                            operation = 'transpose'
                            break

            print(operation)
            self.layouts.append(Layout(domain, local, grid_space, dtype, index))
            if operation == 'transform':
                j = op_index
                self.increment.append(functools.partial(domain.bases[j].backward, axis=j))
                self.decrement.append(functools.partial(domain.bases[j].forward, axis=j))
            elif operation == 'transpose':
                self.increment.append('transpose')
                self.decrement.append('transpose')

        # Directly reference coefficient and grid space layouts
        self.coeff_layout = self.layouts[0]
        self.grid_layout = self.layouts[-1]

        # Allow reference to coefficient and grid space layouts by string
        self.string_references = {'c': self.coeff_layout,
                                  'g': self.grid_layout,
                                  'coeff': self.coeff_layout,
                                  'grid': self.grid_layout}

        # Compute buffer size (in bytes)
        self.buffer_size = max([l.buffer_size for l in self.layouts])

    def increment_layout(self, field):

        index = field.layout.index
        orig_data = field.data
        field.layout = self.layouts[index+1]

        self.increment[index](orig_data, field.data)

    def decrement_layout(self, field):

        index = field.layout.index
        orig_data = field.data
        field.layout = self.layouts[index-1]

        self.decrement[index-1](orig_data, field.data)


class Layout:

    def __init__(self, domain, local, grid_space, dtype, index):

        # Initial attributes
        self.local = tuple(local)
        self.grid_space = tuple(grid_space)
        self.dtype = dtype
        self.index = index

        # Build global shape
        d = domain.dim
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

        # Compute necessary buffer size (in bytes)
        self.buffer_size = np.prod(self.shape) * np.dtype(self.dtype).itemsize

    def view_data(self, buffer):

        data = np.ndarray(shape=self.shape,
                          dtype=self.dtype,
                          buffer=buffer)

        return data

