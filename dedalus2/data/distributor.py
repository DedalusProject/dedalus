

import numpy as np
from mpi4py import MPI

from ..tools.logging import logger
from ..tools.general import rev_enumerate
try:
    from ..tools.fftw import fftw_wrappers as fftw
    logger.debug("Successfully imported FFTW wrappers.")
    fftw.fftw_mpi_init()
except ImportError:
    logger.warning("Cannot import FFTW wrappers.")
    logger.warning("Don't forget to build using 'python3 setup.py build_ext --inplace'")


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
        if self.mesh.size >= domain.dim:
            raise ValueError("Mesh must have lower dimension than domain.")
        if np.prod(self.mesh) > self.size:
            raise ValueError("Insufficient processes for specified mesh.")
        elif self.size > np.prod(self.mesh):
            logger.warning("There are more available processes than will be "
                           "utilized by the specified mesh.  Some processes "
                           "may be idle.")

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

        # Compute global shape
        g_shape = np.zeros(domain.dim, dtype=int)
        for i, b in enumerate(domain.bases):
            if self.grid_space[i]:
                g_shape[i] = b.grid_size
            else:
                g_shape[i] = b.coeff_size

        # Compute distributed global shape
        dg_shape = []
        for i in range(domain.dim):
            if not self.local[i]:
                dg_shape.append(g_shape[i])
        dg_shape = np.array(dg_shape, dtype=int)

        # Compute blocks and distributed local start
        blocks = np.ceil(dg_shape / mesh).astype(int)
        dl_start = coords * blocks

        # Compute distributed local shape
        cuts = np.floor(dg_shape / blocks).astype(int)
        dl_shape = np.zeros(mesh.size, dtype=int)
        dl_shape[coords < cuts] = blocks[coords < cuts]
        dl_shape[coords == cuts] = (dg_shape - cuts*blocks)[coords == cuts]

        # Compute local start and shape
        l_start = np.zeros(domain.dim, dtype=int)
        l_shape = np.zeros(domain.dim, dtype=int)
        j = 0
        for i in range(domain.dim):
            if self.local[i]:
                l_shape[i] = g_shape[i]
            else:
                l_start[i] = dl_start[j]
                l_shape[i] = dl_shape[j]
                j += 1

        # Compute local strides and required allocation
        nbytes = np.prod(l_shape) * np.dtype(dtype).itemsize
        self.alloc_doubles = nbytes // 8

        # Store local start, shape, and blocks
        self.start = l_start
        self.shape = l_shape
        self.blocks = blocks

    def view_data(self, buffer):

        # Create view of buffer
        data = np.ndarray(shape=self.shape,
                          dtype=self.dtype,
                          buffer=buffer)

        return data


class Transform:

    def __init__(self, layout0, layout1, axis, basis):

        # Initial attributes
        self.layout0 = layout0
        self.layout1 = layout1
        self.axis = axis
        self.basis = basis

        # Compute embedded shapes
        shape0 = np.copy(layout0.shape)
        shape0[axis] = basis.coeff_embed
        shape1 = np.copy(layout1.shape)
        shape1[axis] = basis.grid_embed

        # Embedded dtypes
        dtype0 = layout0.dtype
        dtype1 = layout1.dtype

        # Size
        doubles0 = np.prod(shape0) * np.dtype(dtype0).itemsize // 8
        doubles1 = np.prod(shape1) * np.dtype(dtype1).itemsize // 8

        # Buffer size
        buffsize = max(doubles0, doubles1)
        self.buffer = fftw.create_buffer(buffsize)

        # Create views
        self.embed0 = np.ndarray(shape=shape0,
                                 dtype=dtype0,
                                 buffer = self.buffer)
        self.embed1 = np.ndarray(shape=shape1,
                                 dtype=dtype1,
                                 buffer = self.buffer)

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
        self.basis.pad(cdata, self.embed0, axis=self.axis)
        self.basis.backward(self.embed0, gdata, axis=self.axis)

    def _forward(self, field):

        # Get coefficient and grid data views
        gdata = field.data
        field.layout = self.layout0
        cdata = field.data

        # Call basis transform
        self.basis.forward(gdata, self.embed0, axis=self.axis)
        self.basis.unpad(self.embed0, cdata, axis=self.axis)

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
        n0 = layout1.shape[axis]
        n1 = layout0.shape[axis+1]
        howmany = np.prod(layout0.shape[:axis]) * np.prod(layout0.shape[axis+2:])
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
            self._rolled_axes = np.roll(np.arange(layout0.shape.size), -axis)
            self.increment = self._gather_ldl
            self.decrement = self._scatter_ldl

    def _gather_d(self, field):

        # Call FFTW plan on data view
        view0 = field.data
        self.fftw_plans.gather(view0)

        # Update layout
        field.layout = self.layout1

    def _scatter_d(self, field):

        # Call FFTW plan on data view
        view1 = field.data
        self.fftw_plans.scatter(view1)

        # Update layout
        field.layout = self.layout0

    def _gather_ldl(self, field):

        # Local transpose axis to first index
        view0 = field.data
        tr_view0 = np.transpose(view0, self._rolled_axes)
        temp_view = np.ndarray(shape=tr_view0.shape, dtype=tr_view0.dtype, buffer=self._temp_buff)
        np.copyto(temp_view, tr_view0)

        # Global transpose
        self.fftw_plans.gather(temp_view)

        # Update field layout
        field.layout = self.layout1

        # Local transpose first index to axis
        view1 = field.data
        tr_view1 = np.transpose(view1, self._rolled_axes)
        temp_view.resize(tr_view1.shape)
        np.copyto(tr_view1, temp_view)

    def _scatter_ldl(self, field):

        # Local transpose axis to first index
        view1 = field.data
        tr_view1 = np.transpose(view1, self._rolled_axes)
        temp_view = np.ndarray(shape=tr_view1.shape, dtype=tr_view1.dtype, buffer=self._temp_buff)
        np.copyto(temp_view, tr_view1)

        # Global transpose
        self.fftw_plans.scatter(temp_view)

        # Update field layout
        field.layout = self.layout0

        # Local transpose first index to axis
        view0 = field.data
        tr_view0 = np.transpose(view0, self._rolled_axes)
        temp_view.resize(tr_view0.shape)
        np.copyto(tr_view0, temp_view)

