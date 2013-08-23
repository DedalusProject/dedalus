

import numpy as np
try:
    from mpi4py import MPI
except ImportError:
    MPI = None
    print('Cannot import mpi4py. Parallelism disabled.')


class Distributor:

    def __init__(self, domain, mesh=[]):

        # Inputs
        self.domain = domain
        self.mesh = mesh

        # MPI communicator
        if MPI:
            self.communicator = MPI.COMM_WORLD
            self.local_process = self.communicator.Get_rank()
            self.total_processes = self.communicator.Get_size()
        else:
            self.local_process = 0
            self.total_processes = 1

        # DEBUG: want to enforce something like this after done testing
        # Check total processes
        # if self.total_processes != np.prod(mesh):
        #     raise ValueError("Requires %i processes" %np.prod(mesh))

        # Build layouts
        self.layouts = []
        for i in range(len(domain.bases)+len(mesh)+1):
            self.layouts.append(Layout(domain, mesh, i))


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
        self.dtype = domain.bases[-1]._coeff_dtype

        for op in range(index):
            for i in reversed(range(d)):
                if not self.grid_space[i]:
                    if self.local[i]:
                        self.grid_space[i] = True
                        self.dtype = domain.bases[i]._grid_dtype
                        break
                    else:
                        self.local[i] = True
                        self.local[i+1] = False
                        break

        # Build global shape
        global_shape = []
        for i in range(d):
            if self.grid_space[i]:
                global_shape.append(domain.bases[i]._grid_size)
            else:
                global_shape.append(domain.bases[i]._coeff_size)

        # Build local shape
        j = 0
        self.shape = global_shape
        for i in range(d):
            if not self.local[i]:
                self.shape[i] /= mesh[j]
                j += 1

