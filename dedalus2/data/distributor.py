

import numpy as np
try:
    from mpi4py import MPI
    print('Successfully imported mpi4py.') #LOG
except ImportError:
    MPI = None
    print('Cannot import mpi4py.') #LOG

from ..tools.general import rev_enumerate
try:
    from ..tools.fftw import fftw_wrappers as fftw
    print("Successfully imported fftw wrappers.") #LOG
    fftw.fftw_mpi_init()
except ImportError:
    print("Cannot import fftw wrappers.") #LOG
    print("Don't forget to build using: 'python3 setup.py build_ext --inplace'")


class Distributor:

    def __init__(self, domain, mesh=None):

        # Initial attributes
        self.domain = domain

        # MPI / global statistics
        if MPI:
            self.comm_world = MPI.COMM_WORLD
            self.rank = self.comm_world.rank
            self.size = self.comm_world.size
        else:
            self.rank = 0
            self.size = 1

        # Default mesh: 1D
        if mesh is None:
            mesh = (np.array([self.size], dtype=int))

        # Squeeze out local (size 1) dimensions
        self.mesh = np.array([i for i in mesh if (i>1)], dtype=int)

        # Reconcile mesh
        if mesh.size >= domain.dim:
            raise ValueError("Mesh must have lower dimension than domain.")
        if np.prod(self.mesh) > self.size:
            raise ValueError("Insufficient processes for specified mesh.")
        elif self.size > np.prod(self.mesh):
            #LOG warn if: More processes than needed for specified mesh.
            pass

        # Create cartesian communicator for parallel runs
        if self.mesh.size:
            self.parallel = True
            self.comm_cart = self.comm_world.Create_cart(self.mesh)
            self.coords = np.array(self.comm_cart.coords, dtype=int)
        else:
            self.parallel = False
            self.coords = np.array([], dtype=int)

        # Log parallelism information
        #LOG: print self.parallel, self.size, self.mesh, self.coords

        # Build layouts
        self._build_layouts()

    def _build_layouts(self):

        # References
        domain = self.domain
        mesh = self.mesh
        coords = self.coords

        # Sizes
        D = domain.dim
        R = mesh.size

        # Initial layout: full coefficient space
        local = [False] * R + [True] * (D-R)
        grid_space = [False] * D
        dtype = domain.bases[-1].coeff_dtype
        layout_0 = Layout(domain, mesh, coords, local, grid_space, dtype)
        layout_0.index = 0

        # Layout and path lists
        self.layouts = [layout_0]
        self.paths = []

        # Subsequent layouts
        for i in range(1, R+D+1):
            # Iterate backwards over bases
            for d, basis in rev_enumerate(domain.bases):
                # Find coeff_space basis
                if not grid_space[d]:
                    # Transform if local
                    if local[d]:
                        grid_space[d] = True
                        dtype = basis.grid_dtype
                        layout_i = Layout(domain, mesh, coords, local, grid_space, dtype)
                        path_i = Transform(self.layouts[-1], layout_i, d, basis)
                        break
                    # Otherwise transpose
                    else:
                        local[d] = True
                        local[d+1] = False
                        layout_i = Layout(domain, mesh, coords, local, grid_space, dtype)
                        path_i = Transpose(self.layouts[-1], layout_i, d, self.comm_cart)
                        break

            layout_i.index = i
            self.layouts.append(layout_i)
            self.paths.append(path_i)

        # Directly reference coefficient and grid space layouts
        self.coeff_layout = self.layouts[0]
        self.grid_layout = self.layouts[-1]

        # Allow string references to coefficient and grid space layouts
        self.string_layouts = {'c': self.coeff_layout,
                               'g': self.grid_layout,
                               'coeff': self.coeff_layout,
                               'grid': self.grid_layout}

        # Compute required buffer size (in doubles)
        self.alloc_doubles = max([p.alloc_doubles for p in self.paths])

    def create_buffer(self):

        # Allocate memory using FFTW for SIMD alignment
        buffer = fftw.create_buffer(self.alloc_doubles)

        return buffer

    def increment_layout(self, field):

        # Call proper path method
        index = field.layout.index
        self.paths[index].increment(field)

    def decrement_layout(self, field):

        # Call proper path method
        index = field.layout.index
        self.paths[index-1].decrement(field)


class Layout:

    def __init__(self, domain, mesh, coords, local, grid_space, dtype):

        # Initial attributes
        self.local = tuple(local)
        self.grid_space = tuple(grid_space)
        self.dtype = dtype

        # Compute global shape and embed
        g_shape = np.zeros(domain.dim, dtype=int)
        g_embed = np.zeros(domain.dim, dtype=int)
        for i, b in enumerate(domain.bases):
            if self.grid_space[i]:
                g_shape[i] = b.grid_size
                g_embed[i] = b.grid_embed
            else:
                g_shape[i] = b.coeff_size
                g_embed[i] = b.coeff_embed

        # Compute distributed global shape and embed
        dg_shape = []
        dg_embed = []
        for i in range(domain.dim):
            if not self.local[i]:
                dg_shape.append(g_shape[i])
                dg_embed.append(g_embed[i])
        dg_shape = np.array(dg_shape, dtype=int)
        dg_embed = np.array(dg_embed, dtype=int)

        # Compute blocks and distributed local start
        blocks = np.ceil(dg_embed / mesh).astype(int)
        dl_start = coords * blocks

        # Compute distributed local shape and embed
        scut = np.floor(dg_shape / blocks).astype(int)
        ecut = np.floor(dg_embed / blocks).astype(int)
        dl_shape = np.zeros(mesh.size, dtype=int)
        dl_embed = np.zeros(mesh.size, dtype=int)
        dl_shape[coords < scut] = blocks[coords < scut]
        dl_embed[coords < ecut] = blocks[coords < ecut]
        dl_shape[coords == scut] = (dg_shape - scut*blocks)[coords == scut]
        dl_embed[coords == ecut] = (dg_embed - ecut*blocks)[coords == ecut]

        # Compute local start, shape, and embed
        l_start = np.zeros(domain.dim, dtype=int)
        l_shape = np.zeros(domain.dim, dtype=int)
        l_embed = np.zeros(domain.dim, dtype=int)
        j = 0
        for i in range(domain.dim):
            if self.local[i]:
                l_shape[i] = g_shape[i]
                l_embed[i] = g_embed[i]
            else:
                l_start[i] = dl_start[j]
                l_shape[i] = dl_shape[j]
                l_embed[i] = dl_shape[j]
                j += 1

        # Compute local strides and required allocation
        nbytes = np.prod(l_embed) * np.dtype(dtype).itemsize
        if nbytes:
            self.strides = nbytes / np.cumprod(l_embed)
        else:
            self.strides = np.zeros(domain.dim, dtype=int)
        self.alloc_doubles = nbytes // 8

        # Create slice
        self.slice = []
        for i in range(domain.dim):
            if self.local[i]:
                self.slice.append(slice(None))
            else:
                start = l_start[i]
                size = l_shape[i]
                self.slice.append(slice(start, start+size))

        # Store local start, shape, embed, and blocks
        self.start = l_start
        self.shape = l_shape
        self.embed = l_embed
        self.blocks = blocks

    def view_data(self, buffer):

        # Create view of buffer
        data = np.ndarray(shape=self.shape,
                          strides=self.strides,
                          dtype=self.dtype,
                          buffer=buffer)

        return data

    def view_embedding(self, buffer):

        # Create view of buffer
        embedding = np.ndarray(shape=self.embed,
                               strides=self.strides,
                               dtype=self.dtype,
                               buffer=buffer)

        return embedding


class Transform:

    def __init__(self, layout0, layout1, axis, basis):

        # Initial attributes
        self.layout0 = layout0
        self.layout1 = layout1
        self.axis = axis
        self.basis = basis

        # Compute required buffer size (in doubles)
        self.alloc_doubles = max(layout0.alloc_doubles, layout1.alloc_doubles)

        if np.prod(layout0.shape):
            # Increasing layout index: moving towards grid space
            self.increment = self._backward
            self.decrement = self._forward
        else:
            # No-op for no local data
            self.increment = self._no_op_backward
            self.decrement = self._no_op_forward

    def _backward(self, field):

        # Get coefficient and grid data views
        cdata = field.data
        field.layout = self.layout1
        gdata = field.data

        # Call basis transform
        self.basis.backward(cdata, gdata, axis=self.axis)

    def _forward(self, field):

        # Get coefficient and grid data views
        gdata = field.data
        field.layout = self.layout0
        cdata = field.data

        # Call basis transform
        self.basis.forward(gdata, cdata, axis=self.axis)

    def _no_op_backward(self, field):

        # Update layout
        field.layout = self.layout1

    def _no_op_forward(self, field):

        # Update layout
        field.layout = self.layout0


class Transpose:

    def __init__(self, layout0, layout1, axis, comm_cart):

        # Initial attributes
        self.layout0 = layout0
        self.layout1 = layout1
        self.axis = axis

        # Create subgrid communicator
        remain_dims = [0] * comm_cart.dim
        remain_dims[axis] = 1
        self.comm_sub = comm_cart.Sub(remain_dims)

        # FFTW transpose parameters
        n0 = layout1.embed[axis]
        n1 = layout0.embed[axis+1]
        howmany = np.prod(layout0.embed[:axis]) * np.prod(layout0.embed[axis+2:])
        block0 = layout0.blocks[axis]
        block1 = layout1.blocks[axis]
        dtype = layout0.dtype

        # Create FFTW transpose plans
        self.fftw_plans = fftw.Transpose(n0, n1, howmany, block0, block1,
                                         dtype, self.comm_sub, flags=['FFTW_MEASURE'])

        # Compute required buffer size (in doubles)
        self.alloc_doubles = self.fftw_plans.alloc_doubles

        # Increasing layout index: gather axis
        if axis == 0:
            self.increment = self._gather_d
            self.decrement = self._scatter_d
        else:
            # Create buffer for intermediate local transposes
            self._temp_buff = fftw.create_buffer(self.alloc_doubles)
            self._rolled_axes = np.roll(np.arange(layout0.embed.size), -axis)
            self.increment = self._gather_ldl
            self.decrement = self._scatter_ldl

    def _gather_d(self, field):

        # Call FFTW plan on embedding view
        view0 = field._embedding
        self.fftw_plans.gather(view0)

        # Update layout
        field.layout = self.layout1

    def _scatter_d(self, field):

        # Call FFTW plan on embedding view
        view1 = field._embedding
        self.fftw_plans.scatter(view1)

        # Update layout
        field.layout = self.layout0

    def _gather_ldl(self, field):

        # Local transpose axis to first index
        view0 = field._embedding
        tr_view0 = np.transpose(view0, self._rolled_axes)
        temp_view = np.ndarray(shape=tr_view0.shape, dtype=tr_view0.dtype, buffer=self._temp_buff)
        np.copyto(temp_view, tr_view0)

        # Global transpose
        self.fftw_plans.gather(temp_view)

        # Update field layout
        field.layout = self.layout1

        # Local transpose first index to axis
        view1 = field._embedding
        tr_view1 = np.transpose(view1, self._rolled_axes)
        temp_view.resize(tr_view1.shape)
        np.copyto(tr_view1, temp_view)

    def _scatter_ldl(self, field):

        # Local transpose axis to first index
        view1 = field._embedding
        tr_view1 = np.transpose(view1, self._rolled_axes)
        temp_view = np.ndarray(shape=tr_view1.shape, dtype=tr_view1.dtype, buffer=self._temp_buff)
        np.copyto(temp_view, tr_view1)

        # Global transpose
        self.fftw_plans.scatter(temp_view)

        # Update field layout
        field.layout = self.layout0

        # Local transpose first index to axis
        view0 = field._embedding
        tr_view0 = np.transpose(view0, self._rolled_axes)
        temp_view.resize(tr_view0.shape)
        np.copyto(tr_view0, temp_view)

