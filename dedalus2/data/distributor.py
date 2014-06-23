"""
Classes for available data layouts and the paths between them.

"""

from functools import partial
import numpy as np
from mpi4py import MPI

from ..tools.config import config
from ..tools.general import rev_enumerate
try:
    from ..libraries.fftw import fftw_wrappers as fftw
    fftw.fftw_mpi_init()
except ImportError:
    logger.error("Don't forget to buid using 'python3 setup.py build_ext --inplace'")
    raise

import logging
logger = logging.getLogger(__name__.split('.')[-1])


# Load config options
use_fftw = config['transforms'].getboolean('use_fftw')
path_barriers = config['parallelism'].getboolean('path_barriers')


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

        # Squeeze out local/bad (size <= 1) dimensions
        self.mesh = np.array([i for i in mesh if (i>1)], dtype=int)

        # Check mesh compatibility
        logger.debug('Mesh: %s' %str(self.mesh))
        if self.mesh.size >= domain.dim:
            raise ValueError("Mesh must have lower dimension than domain.")
        if np.prod(self.mesh) != self.size:
            raise ValueError("Wrong number of processes (%i) for specified mesh (%s)" %(self.size, self.mesh))

        # Create cartesian communicator for parallel runs
        self.comm_cart = self.comm_world.Create_cart(self.mesh)

        # Get cartesian coordinates
        self.coords = np.array(self.comm_cart.coords, dtype=int)
        self._build_layouts(domain)

    def _build_layouts(self, domain, dry_run=False):
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
                        if not dry_run:
                            path_i = Transform(self.layouts[-1], layout_i, d, basis)
                        break
                    # Otherwise transpose
                    else:
                        local[d] = True
                        local[d+1] = False
                        layout_i = Layout(domain, mesh, coords, local, grid_space, dtype)
                        if not dry_run:
                            path_i = Transpose(self.layouts[-1], layout_i, d, self.comm_cart)
                        break

            layout_i.index = i
            self.layouts.append(layout_i)
            if not dry_run:
                self.paths.append(path_i)

        # Directly reference coefficient and grid space layouts
        self.coeff_layout = self.layouts[0]
        self.grid_layout = self.layouts[-1]
        logger.debug('Local grid shape: %s' %str(self.grid_layout.shape))
        logger.debug('Local coeff shape: %s' %str(self.coeff_layout.shape))

        # Allow string references to coefficient and grid space layouts
        self.layout_references = {'c': self.coeff_layout,
                                  'g': self.grid_layout,
                                  'coeff': self.coeff_layout,
                                  'grid': self.grid_layout}

        # Take maximum required buffer size (in doubles)
        self.alloc_doubles = max(i.alloc_doubles for i in (self.layouts+self.paths))

    def get_layout_object(self, input):
        """Dereference layout identifiers."""

        if isinstance(input, Layout):
            return input
        else:
            return self.layout_references[input]

    def create_buffer(self):
        """Allocate memory using FFTW for SIMD alignment."""

        if self.alloc_doubles:
            return fftw.create_buffer(self.alloc_doubles)
        else:
            return np.ndarray(shape=(0,), dtype=np.float64)

    def increment_layout(self, field):
        """Transform field to subsequent layout (towards grid space)."""

        if path_barriers:
            self.comm_cart.Barrier()
        index = field.layout.index
        self.paths[index].increment(field)

    def decrement_layout(self, field):
        """Transform field to preceding layout (towards coefficient space)."""

        if path_barriers:
            self.comm_cart.Barrier()
        index = field.layout.index
        self.paths[index-1].decrement(field)


class Layout:
    """
    Specifications for the local part of a given data layout, i.e. a particular
    transform / distribution state.

    Attributes
    ----------
    local : array of bools
        Locality flags for each dimension
    grid_space : array of bools
        Grid space flags for each dimension
    dtype : numeric type
        Numeric type of data

    All methods require a tuple of the current transform scalings.

    """

    def __init__(self, domain, mesh, coords, local, grid_space, dtype):

        self.domain = domain
        # Freeze local and grid_space lists into boolean arrays
        self.local = np.array(local)
        self.grid_space = np.array(grid_space)
        self.dtype = dtype

        # Extend mesh and coordinates to domain dimension
        self.ext_mesh = np.ones(domain.dim, dtype=int)
        self.ext_mesh[~self.local] = mesh
        self.ext_coords = np.zeros(domain.dim, dtype=int)
        self.ext_coords[~self.local] = coords

    def global_shape(self, scales):
        """Compute global data shape."""

        global_coeff_shape = self.domain.global_coeff_shape
        global_grid_shape = self.domain.global_grid_shape(scales)

        global_shape = global_coeff_shape.copy()
        global_shape[self.grid_space] = global_grid_shape[self.grid_space]
        return global_shape

    def blocks(self, scales):
        """Compute block sizes for data distribution."""

        global_shape = self.global_shape(scales)
        # FFTW standard block sizes
        return np.ceil(global_shape / self.ext_mesh).astype(int)

    def start(self, scales):
        """Compute starting coordinates for local data."""

        blocks = self.blocks(scales)
        return self.ext_coords * blocks

    def local_shape(self, scales):
        """Compute local data shape."""

        global_shape = self.global_shape(scales)
        blocks = self.blocks(scales)
        ext_coords = self.ext_coords

        # Cutoff coordinates: first empty/partial blocks
        cuts = np.floor(global_shape / blocks).astype(int)

        local_shape = blocks.copy()
        local_shape[ext_coords == cuts] = (global_shape - cuts*blocks)[ext_coords == cuts]
        local_shape[ext_coords > cuts] = 0
        return local_shape

    def slices(self, scales):
        """Compute slices for selecting local portion of global data."""

        start = self.start(scales)
        local_shape = self.local_shape(scales)
        return tuple(slice(s, s+l) for (s, l) in zip(start, local_shape))

    def alloc_doubles(self, scales):
        """Compute necessary allocation size."""

        local_shape = self.local_shape(scales)
        nbytes = np.prod(local_shape) * self.dtype.itemsize
        return nbytes // 8

    def view_data(self, buffer, scales):
        """View buffer in this layout."""

        return np.ndarray(shape=self.local_shape(scales),
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
        if np.prod(layout0.shape):
            pad_shape = np.copy(layout0.shape)
            pad_shape[axis] = basis.coeff_embed
            pad_dtype = layout0.dtype
            pad_doubles = np.prod(pad_shape) * np.dtype(pad_dtype).itemsize // 8
            self.buffer = fftw.create_buffer(pad_doubles)
            self.embed = np.ndarray(shape=pad_shape,
                                    dtype=pad_dtype,
                                    buffer=self.buffer)

        # Set transform callables
        if use_fftw and basis.fftw_plan:
            self.fftw_plan = basis.fftw_plan(layout1.shape, axis)
            self._backward_callable = self.fftw_plan.backward
            self._forward_callable = self.fftw_plan.forward
        else:
            self._backward_callable = partial(basis.backward, axis=axis)
            self._forward_callable = partial(basis.forward, axis=axis)

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
        self._backward_callable(self.embed, gdata)

    def _forward(self, field):
        """Grid-to-coefficient transform and unpadding."""

        # Get coefficient and grid space views
        gdata = field.data
        field.layout = self.layout0
        cdata = field.data

        # Call basis transform and unpadding
        self._forward_callable(gdata, self.embed)
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

        # Dispatch based on local distribution
        if howmany:
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
        else:
            # No-op if there's no local data
            self.alloc_doubles = 0
            self.increment = self._no_op_gather
            self.decrement = self._no_op_scatter

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
        temp = np.ndarray(shape=tr_view1.shape, dtype=tr_view1.dtype, buffer=self._buffer)
        # temp.resize(tr_view1.shape)
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
        temp = np.ndarray(shape=tr_view0.shape, dtype=tr_view0.dtype, buffer=self._buffer)
        # temp.resize(tr_view0.shape)
        np.copyto(tr_view0, temp)

    def _no_op_gather(self, field):
        """Update layout, no data handling."""

        field.layout = self.layout1

    def _no_op_scatter(self, field):
        """Update layout, no data handling."""

        field.layout = self.layout0

