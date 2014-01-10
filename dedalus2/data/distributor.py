"""
Classes for available data layouts and the paths between them.

"""

import numpy as np
from mpi4py import MPI

from ..tools.logging import logger
from ..tools.general import rev_enumerate
try:
    from ..tools.fftw import fftw_wrappers as fftw
    fftw.fftw_mpi_init()
except ImportError:
    logger.error("Don't forget to buid using 'python3 setup.py build_ext --inplace'")
    raise


class Distributor:
    """
    Directs parallelized distribution and transformation of fields over a domain.

    Attributes
    ----------
    comm_world : MPI communicator
        Global MPI communicator
    rank : int
        Internal MPI process number
    size : int
        Number of MPI processes
    mesh : tuple of ints, optional
        Process mesh for parallelization (default: 1-D mesh of available processes)
    comm_cart : MPI communicator
        Cartesian MPI communicator over mesh
    coords : array of ints
        Coordinates in cartesian communicator (None if outside mesh)
    layouts : list of layout objects
        Available layouts for domain
    paths : list of path objects
        Transforms and transposes between layouts

    Notes
    -----
    Computations are parallelized by splitting D-dimensional data fields over
    an R-dimensional mesh of MPI processes, where R < D.  In coefficient space,
    we take the first R dimensions of the data to be distributed over the mesh,
    leaving the last (D-R) dimensions local.  To transform such a data cube to
    grid space, we loop backwards over the D dimensions, performing each
    transform if the corresponding dimension is local, and performing an MPI
    transpose with the next dimension otherwise.  This effectively bubbles the
    first local dimension up from the (D-R)-th to the first dimension,
    transforming to grid space along the way.  In grid space, then, the first
    dimensional is local, followed by R dimensions distributed over the mesh,
    and the last (D-R-1) dimensions local.

    The distributor object for a given domain constructs layout objects
    describing each of the (D+R+1) layouts (sets of transform/distribution
    states) and the paths between them (D transforms and R transposes).

    """

    def __init__(self, domain, mesh=None):

        # MPI communicator and statistics
        self.comm_world = MPI.COMM_WORLD
        self.rank = self.comm_world.rank
        self.size = self.comm_world.size

        # Default to 1-D mesh of available processes
        if mesh is None:
            mesh = np.array([self.size], dtype=int)

        # Squeeze out local and bad (size <= 1) dimensions
        self.mesh = np.array([i for i in mesh if (i>1)], dtype=int)

        # Check mesh compatibility
        logger.debug('Mesh: %s' %str(self.mesh))
        if self.mesh.size >= domain.dim:
            raise ValueError("Mesh must have lower dimension than domain.")
        if np.prod(self.mesh) > self.size:
            raise ValueError("Insufficient processes for specified mesh.")
        elif self.size > np.prod(self.mesh):
            logger.warning("There are more available processes than will be "
                           "utilized by the specified mesh.  Some processes "
                           "may be idle.")

        # Create cartesian communicator for parallel runs
        self.comm_cart = self.comm_world.Create_cart(self.mesh)

        # Get cartesian coordinates
        # Non-mesh processes receive null communicators
        if self.comm_cart == MPI.COMM_NULL:
            # UPGRADE: figure out what to do when outside mesh
            self.coords = None
        else:
            self.coords = np.array(self.comm_cart.coords, dtype=int)
            self._build_layouts(domain)

    def _build_layouts(self, domain):
        """Construct layout objects."""

        # References
        mesh = self.mesh
        coords = self.coords
        D = domain.dim
        R = mesh.size

        # First layout: full coefficient space
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
            # Iterate backwards over bases to last coefficient space basis
            for d, basis in rev_enumerate(domain.bases):
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
        logger.debug('Local grid shape: %s' %str(self.grid_layout.shape))
        logger.debug('Local coeff shape: %s' %str(self.coeff_layout.shape))

        # Allow string references to coefficient and grid space layouts
        self.string_layouts = {'c': self.coeff_layout,
                               'g': self.grid_layout,
                               'coeff': self.coeff_layout,
                               'grid': self.grid_layout}

        # Take maximum required buffer size (in doubles)
        self.alloc_doubles = max(i.alloc_doubles for i in (self.layouts+self.paths))

    def create_buffer(self):
        """Allocate memory using FFTW for SIMD alignment."""

        return fftw.create_buffer(self.alloc_doubles)

    def increment_layout(self, field):
        """Transform field to subsequent layout (towards grid space)."""

        index = field.layout.index
        self.paths[index].increment(field)

    def decrement_layout(self, field):
        """Transform field to preceding layout (towards coefficient space)."""

        index = field.layout.index
        self.paths[index-1].decrement(field)


class Layout:
    """
    Specifications for the local part of a given data layout, specified by the
    transform and distribution states

    Attributes
    ----------
    local : array of bools
        Locality flags for each dimension
    grid_space : array of bools
        Grid space flags for each dimension
    dtype : numeric type
        Numeric type of data
    blocks : arrays of ints
        Distributed block sizes
    start : array of ints
        Local data start indices
    shape : array of ints
        Local data shape

    """

    def __init__(self, domain, mesh, coords, local, grid_space, dtype):

        # Freeze local and grid_space lists by creating boolean arrays
        local = np.array(local)
        grid_space = np.array(grid_space)

        # Initial attributes
        self.local = local
        self.grid_space = grid_space
        self.dtype = dtype

        # Compute global shape
        g_shape = np.copy(domain.coeff_shape)
        g_shape[grid_space] = domain.grid_shape[grid_space]

        # Distributed global shape: subset of global shape
        dg_shape = g_shape[~local]

        # Block sizes: FFTW standard
        self.blocks = blocks = np.ceil(dg_shape / mesh).astype(int)

        # Cutoff coordinates: coordinates of first empty/partial blocks
        cuts = np.floor(dg_shape / blocks).astype(int)

        # Distributed local start
        dl_start = coords * blocks

        # Distributed local shape
        dl_shape = np.zeros(mesh.size, dtype=int)
        dl_shape[coords < cuts] = blocks[coords < cuts]
        dl_shape[coords == cuts] = (dg_shape - cuts*blocks)[coords == cuts]

        # Local start
        self.start = np.zeros(domain.dim, dtype=int)
        self.start[~local] = dl_start

        # Local shape
        self.shape = g_shape
        self.shape[~local] = dl_shape

        # Required buffer size (in doubles)
        nbytes = np.prod(self.shape) * np.dtype(dtype).itemsize
        self.alloc_doubles = nbytes // 8

    def view_data(self, buffer):
        """View buffer in this layout."""

        return np.ndarray(shape=self.shape,
                          dtype=self.dtype,
                          buffer=buffer)


class Transform:
    """
    Directs transforms between two layouts.

    Notes
    -----
    A local buffer is used to pad coefficients according to the basis
    parameters, and out-of-place transforms between this buffer and the
    field's buffer are performed.

    """

    def __init__(self, layout0, layout1, axis, basis):

        # Initial attributes
        self.layout0 = layout0
        self.layout1 = layout1
        self.axis = axis
        self.basis = basis

        # Construct buffer for padded coefficients
        pad_shape = np.copy(layout0.shape)
        pad_shape[axis] = basis.coeff_embed
        pad_dtype = layout0.dtype
        pad_doubles = np.prod(pad_shape) * np.dtype(pad_dtype).itemsize // 8
        self.buffer = fftw.create_buffer(pad_doubles)
        self.embed = np.ndarray(shape=pad_shape,
                                dtype=pad_dtype,
                                buffer=self.buffer)

        # By using buffer, transforms/padding don't impact field allocations
        self.alloc_doubles = 0

        # Dispatch based on local distribution
        if np.prod(layout0.shape):
            # Increment layout <==> towards grid space
            self.increment = self._backward
            self.decrement = self._forward
        else:
            # No-op if there's no local data
            self.increment = self._no_op_backward
            self.decrement = self._no_op_forward

    def _backward(self, field):
        """Coefficient-to-grid padding and transform."""

        # Get coefficient and grid space views
        cdata = field.data
        field.layout = self.layout1
        gdata = field.data

        # Call basis padding and transform
        self.basis.pad_coeff(cdata, self.embed, axis=self.axis)
        self.basis.backward(self.embed, gdata, axis=self.axis)

    def _forward(self, field):
        """Grid-to-coefficient transform and unpadding."""

        # Get coefficient and grid space views
        gdata = field.data
        field.layout = self.layout0
        cdata = field.data

        # Call basis transform and unpadding
        self.basis.forward(gdata, self.embed, axis=self.axis)
        self.basis.unpad_coeff(self.embed, cdata, axis=self.axis)

    def _no_op_backward(self, field):
        """Update layout, no data handling."""

        field.layout = self.layout1

    def _no_op_forward(self, field):
        """Update layout, no data handling."""

        field.layout = self.layout0


class Transpose:
    """
    Directs transposes between two layouts.

    Notes
    -----
    If the transpose is not between the first two dimensions, then local
    transposes to an internal buffer are called before and after the global
    MPI transpose to achieve such an ordering.

    """

    def __init__(self, layout0, layout1, axis, comm_cart):

        # Initial attributes
        self.layout0 = layout0
        self.layout1 = layout1
        self.axis = axis

        # Create subgrid communicator along the mesh axis that switches dimension
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

        # Required buffer size (in doubles)
        self.alloc_doubles = self.fftw_plans.alloc_doubles

        # Dispatch based on transpose axis
        # Increment layout <==> gather specified axis
        if axis == 0:
            self.increment = self._gather_d
            self.decrement = self._scatter_d
        else:
            # Create buffer for intermediate local transposes
            self._buffer = fftw.create_buffer(self.alloc_doubles)
            self._rolled_axes = np.roll(np.arange(layout0.shape.size), -axis)
            self.increment = self._gather_ldl
            self.decrement = self._scatter_ldl

    def _gather_d(self, field):
        """FFTW transpose to gather axis == 0."""

        # Call FFTW plan on data view
        self.fftw_plans.gather(field.data)

        # Update layout
        field.layout = self.layout1

    def _scatter_d(self, field):
        """FFTW transpose to scatter axis == 0."""

        # Call FFTW plan on data view
        self.fftw_plans.scatter(field.data)

        # Update layout
        field.layout = self.layout0

    def _gather_ldl(self, field):
        """FFTW and local transposes to gather axis > 0."""

        # Local transpose from field to internal buffer
        view0 = field.data
        tr_view0 = np.transpose(view0, self._rolled_axes)
        temp = np.ndarray(shape=tr_view0.shape, dtype=tr_view0.dtype, buffer=self._buffer)
        np.copyto(temp, tr_view0)

        # Global transpose
        self.fftw_plans.gather(temp)

        # Update field layout
        field.layout = self.layout1

        # Local transpose from internal buffer to field
        view1 = field.data
        tr_view1 = np.transpose(view1, self._rolled_axes)
        temp.resize(tr_view1.shape)
        np.copyto(tr_view1, temp)

    def _scatter_ldl(self, field):
        """FFTW and local transposes to scatter axis > 0."""

        # Local transpose from field to internal buffer
        view1 = field.data
        tr_view1 = np.transpose(view1, self._rolled_axes)
        temp = np.ndarray(shape=tr_view1.shape, dtype=tr_view1.dtype, buffer=self._buffer)
        np.copyto(temp, tr_view1)

        # Global transpose
        self.fftw_plans.scatter(temp)

        # Update field layout
        field.layout = self.layout0

        # Local transpose from internal buffer to field
        view0 = field.data
        tr_view0 = np.transpose(view0, self._rolled_axes)
        temp.resize(tr_view0.shape)
        np.copyto(tr_view0, temp)

