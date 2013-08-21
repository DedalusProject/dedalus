

import numpy as np
try:
    from mpi4py import MPI
except ImportError:
    MPI = None
    print('Cannot import mpi4py. Parallelism disabled.')

from graph import Graph


class Distributor:

    def __init__(self, domain, mesh=None):

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

        # Check total processes
        # if self.total_processes != np.prod(mesh):
        #     raise ValueError("Requires %i processes" %np.prod(mesh))

        # Build layouts
        self.layouts = []
        for i in range(len(mesh)+1):
            self.layouts.append(Layout(domain, mesh, i))

        # Build graph
        self.graph = Graph()
        # Add layouts
        for lo in self.layouts:
            self.graph.add_vertex(lo)
        # Add transpose plans
        for i in range(1, len(mesh)+1):
            forward_transpose = '%i_%i' %(0, i)
            backward_transpose = '%i_%i' %(i, 0)
            self.graph.add_edge(self.layouts[0], self.layouts[i], forward_transpose)
            self.graph.add_edge(self.layouts[i], self.layouts[0], backward_transpose)


class Layout:

    def __init__(self, domain, mesh, index):

        # Inputs
        self.domain = domain
        self.mesh = mesh
        self.index = index

        # Sizes
        D = domain
        R = mesh
        d = len(D)
        r = len(R)
        L = index

        if r >= d:
            raise ValueError("r must be less than d")
        if L > r:
            raise ValueError("num must be less than or equal to r")

        shape = []
        local = []
        for i in range(0, L):
            shape.append(D[i] / R[i])
            local.append(False)
        shape.append(D[L])
        local.append(True)
        for i in range(L+1, r+1):
            shape.append(D[i] / R[i-1])
            local.append(False)
        for i in range(r+1, d):
            shape.append(D[i])
            local.append(True)

        print(shape)
        print(local)
        print('-'*10)

# How fields might determine proper transpose
        # distributor.graph.find_shortest_path(self.layout, local(i))
        # path = self.distributor.require_local(i)
