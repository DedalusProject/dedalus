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
    Object describing the data distribution for a given transform and
    distribution state.

    Attributes
    ----------
    local : array of bools
        Axis locality flags (True/False for local/distributed)
    grid_space : array of bools
        Axis grid-space flags (True/False for grid/coeff space)
    dtype : numeric type
        Data type

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
    """Directs transforms between two layouts."""

    # To Do: Group transforms for multiple fields (group)

    def __init__(self, layout0, layout1, axis, basis):

        self.layout0 = layout0
        self.layout1 = layout1
        self.axis = axis
        self.basis = basis

    def increment(self, field):
        """Backward transform."""

        # Reference views from both layouts
        cdata = field.data
        field.layout = self.layout1
        gdata = field.data
        # Transform if there's local data
        if np.prod(cdata.shape):
            scale = field.meta[self.axis]['scale']
            self.basis.backward(cdata, axis=self.axis, gdata=gdata, scale=scale)

    def decrement(self, field):
        """Forward transform."""

        # Reference views from both layouts
        gdata = field.data
        field.layout = self.layout0
        cdata = field.data
        # Transform if there's local data
        if np.prod(gdata.shape):
            self.basis.forward(gdata, axis=self.axis, cdata=cdata)


class Transpose:
    """Directs transposes between two layouts."""

    # To Do: Determine how to query plans
    # To Do: Skip transpose for empty arrays (no-op)
    # To Do: Group transposes for multiple fields (group)

    def __init__(self, layout0, layout1, axis, comm_cart):

        self.layout0 = layout0
        self.layout1 = layout1
        self.axis = axis
        # Create subgrid communicator along the moving mesh axis
        remain_dims = [0] * comm_cart.dim
        remain_dims[axis] = 1
        self.comm_sub = comm_cart.Sub(remain_dims)

    @CachedMethod
    def _fftw_setup(self, scales):
        """Build FFTW plans."""

        axis = self.axis
        shape0 = self.layout0.local_shape(scales)
        shape1 = self.layout1.local_shape(scales)
        blocks0 = self.layout0.blocks(scales)
        blocks1 = self.layout1.blocks(scales)
        logger.debug("Building FFTW transpose plan for (nfields, scales, axis) = (%s, %s, %s)" %(nfields, scales, axis))
        # Build FFTW transpose plans
        n0 = shape1[axis]
        n1 = shape0[axis+1]
        howmany = np.prod(shape0[:axis]) * np.prod(shape0[axis+2:])
        block0 = blocks0[axis]
        block1 = blocks1[axis+1]
        dtype = self.layout0.dtype
        flags = ['FFTW_'+FFTW_RIGOR.upper()]
        plan = fftw.Transpose(n0, n1, howmany, block0, block1, dtype, self.comm_sub, flags=flags)
        # Create temporary arrays with transposed data ordering
        tr_shape0 = np.roll(shape0, -axis)
        tr_shape1 = np.roll(shape1, -axis)
        tr_temp0 = np.ndarray(shape=tr_shape0, dtype=dtype, buffer=plan.buffer0)
        tr_temp1 = np.ndarray(shape=tr_shape1, dtype=dtype, buffer=plan.buffer1)
        # Create anti-transposed views of temporary arrays
        # For direct copying to and from field data
        dim = self.layout0.domain.dim
        temp0 = np.transpose(tr_temp0, np.roll(np.arange(dim), axis))
        temp1 = np.transpose(tr_temp1, np.roll(np.arange(dim), axis))

        return plan, temp0, temp1

    def increment(self, field):
        """Gather along specified axis."""

        scales = tuple(axmeta['scale'] for axmeta in field.meta)
        plan, temp0, temp1 = self._fftw_setup(scales)
        # Copy layout0 view of data to plan buffer
        np.copyto(temp0, field.data)
        # Globally transpose between temp buffers
        self.fftw_plans.gather()
        # Update field layout
        # Copy plan buffer to layout1 view of data
        field.layout = self.layout1
        np.copyto(field.data, temp1)

    def decrement(self, fields):
        """Scatter along specified axis."""

        scales = tuple(axmeta['scale'] for axmeta in field.meta)
        plan, temp0, temp1 = self._fftw_setup(scales)
        # Copy layout1 view of data to plan buffer
        np.copyto(temp1, field.data)
        # Globally transpose between temp buffers
        self.fftw_plans.scatter()
        # Update field layout
        # Copy plan buffer to layout0 view of data
        field.layout = self.layout0
        np.copyto(field.data, temp0)

