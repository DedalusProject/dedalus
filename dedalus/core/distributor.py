"""
Classes for available data layouts and the paths between them.

"""

import logging
from mpi4py import MPI
import numpy as np

from ..libraries.fftw import fftw_wrappers as fftw
from ..tools.cache import CachedMethod
from ..tools.config import config
from ..tools.general import rev_enumerate, unify

logger = logging.getLogger(__name__.split('.')[-1])
GROUP_TRANSFORMS = config['transforms'].getboolean('GROUP_TRANSFORMS')
TRANSPOSE_LIBRARY = config['parallelism'].get('TRANSPOSE_LIBRARY').upper()
GROUP_TRANSPOSES = config['parallelism'].getboolean('GROUP_TRANSPOSES')
SYNC_TRANSPOSES = config['parallelism'].getboolean('SYNC_TRANSPOSES')
ALLTOALLV = config['parallelism-mpi'].getboolean('ALLTOALLV')

if TRANSPOSE_LIBRARY == 'FFTW':
    from .transposes import FFTWTranspose as TransposePlanner
elif TRANSPOSE_LIBRARY == 'MPI':
    if ALLTOALLV:
        from .transposes import AlltoallvTranspose as TransposePlanner
    else:
        from .transposes import AlltoallTranspose as TransposePlanner


class Distributor:
    """
    Directs parallelized distribution and transformation of fields over a domain.

    Attributes
    ----------
    comm : MPI communicator
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

    def __init__(self, domain, comm=None, mesh=None):

        # MPI communicator and statistics
        if comm is None:
            comm = MPI.COMM_WORLD
        self.comm = comm
        self.rank = self.comm.rank
        self.size = self.comm.size

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
        self.comm_cart = self.comm.Create_cart(self.mesh)

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

        # Allow string references to coefficient and grid space layouts
        self.layout_references = {'c': self.coeff_layout,
                                  'g': self.grid_layout,
                                  'coeff': self.coeff_layout,
                                  'grid': self.grid_layout}

    def get_layout_object(self, input):
        """Dereference layout identifiers."""

        if isinstance(input, Layout):
            return input
        else:
            return self.layout_references[input]

    @CachedMethod
    def buffer_size(self, scales):
        """Compute necessary buffer size (bytes) for all layouts."""

        return max(layout.buffer_size(scales) for layout in self.layouts)


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

    All methods require a tuple of the current transform scales.

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

    @CachedMethod
    def global_shape(self, scales):
        """Compute global data shape."""

        global_coeff_shape = self.domain.global_coeff_shape
        global_grid_shape = self.domain.global_grid_shape(scales)

        global_shape = global_coeff_shape.copy()
        global_shape[self.grid_space] = global_grid_shape[self.grid_space]
        return global_shape

    @CachedMethod
    def blocks(self, scales):
        """Compute block sizes for data distribution."""

        global_shape = self.global_shape(scales)
        # FFTW standard block sizes
        return np.ceil(global_shape / self.ext_mesh).astype(int)

    @CachedMethod
    def start(self, scales):
        """Compute starting coordinates for local data."""

        blocks = self.blocks(scales)
        return self.ext_coords * blocks

    @CachedMethod
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

    @CachedMethod
    def slices(self, scales):
        """Compute slices for selecting local portion of global data."""

        start = self.start(scales)
        local_shape = self.local_shape(scales)
        return tuple(slice(s, s+l) for (s, l) in zip(start, local_shape))

    @CachedMethod
    def buffer_size(self, scales):
        """Compute necessary buffer size (bytes)."""

        local_shape = self.local_shape(scales)
        return np.prod(local_shape) * self.dtype.itemsize


class Transform:
    """Directs transforms between two layouts."""
    # To Do: group transforms for multiple fields

    def __init__(self, layout0, layout1, axis, basis):

        self.layout0 = layout0
        self.layout1 = layout1
        self.axis = axis
        self.basis = basis

    @CachedMethod
    def group_data(self, nfields, scales):

        local_shape0 = self.layout0.local_shape(scales)
        local_shape1 = self.layout1.local_shape(scales)
        group_shape0 = [nfields] + list(local_shape0)
        group_shape1 = [nfields] + list(local_shape1)
        group_cdata = fftw.create_array(group_shape0, self.layout0.dtype)
        group_gdata = fftw.create_array(group_shape1, self.layout1.dtype)

        return group_cdata, group_gdata

    def increment_group(self, fields):
        fields = list(fields)
        scales = fields[0].meta[:]['scale']
        cdata, gdata = self.group_data(len(fields), scales)
        for i, field in enumerate(fields):
            np.copyto(cdata[i], field.data)
        self.basis.backward(cdata, gdata, self.axis+1, fields[0].meta[self.axis])
        for i, field in enumerate(fields):
            field.layout = self.layout1
            np.copyto(field.data, gdata[i])

    def decrement_group(self, fields):
        fields = list(fields)
        scales = fields[0].meta[:]['scale']
        cdata, gdata = self.group_data(len(fields), scales)
        for i, field in enumerate(fields):
            np.copyto(gdata[i], field.data)
        self.basis.forward(gdata, cdata, self.axis+1, fields[0].meta[self.axis])
        for i, field in enumerate(fields):
            field.layout = self.layout0
            np.copyto(field.data, cdata[i])

    def increment_single(self, field):
        """Backward transform."""
        # Reference views from both layouts
        cdata = field.data
        field.layout = self.layout1
        gdata = field.data
        # Transform if there's local data
        if np.prod(cdata.shape):
            self.basis.backward(cdata, gdata, self.axis, field.meta[self.axis])

    def decrement_single(self, field):
        """Forward transform."""
        # Reference views from both layouts
        gdata = field.data
        field.layout = self.layout0
        cdata = field.data
        # Transform if there's local data
        if np.prod(gdata.shape):
            self.basis.forward(gdata, cdata, self.axis, field.meta[self.axis])

    def increment(self, fields):
        """Backward transform."""
        if len(fields) == 1:
            self.increment_single(*fields)
        elif GROUP_TRANSFORMS:
            self.increment_group(fields)
        else:
            for field in fields:
                self.increment_single(field)

    def decrement(self, fields):
        """Forward transform."""
        if len(fields) == 1:
            self.decrement_single(*fields)
        elif GROUP_TRANSFORMS:
            self.decrement_group(fields)
        else:
            for field in fields:
                self.decrement_single(field)


class Transpose:
    """Directs transposes between two layouts."""

    def __init__(self, layout0, layout1, axis, comm_cart):
        # Create subgrid communicator along the moving mesh axis
        remain_dims = [0] * comm_cart.dim
        remain_dims[axis] = 1
        comm_sub = comm_cart.Sub(remain_dims)
        # Attributes
        self.layout0 = layout0
        self.layout1 = layout1
        self.dtype = layout0.dtype  # same as layout1.dtype
        self.axis = axis
        self.comm_cart = comm_cart
        self.comm_sub = comm_sub

    def _sub_shape(self, scales):
        """Build global shape of data assigned to sub-communicator."""
        local_shape = self.layout0.local_shape(scales)
        global_shape = self.layout0.global_shape(scales)
        # Global shape along transposing axes, local shape along others
        sub_shape = local_shape.copy()
        sub_shape[self.axis] = global_shape[self.axis]
        sub_shape[self.axis+1] = global_shape[self.axis+1]
        return sub_shape

    @CachedMethod
    def _single_plan(self, scales):
        """Build single transpose plan."""
        sub_shape = self._sub_shape(scales)
        if np.prod(sub_shape) == 0:
            return None  # no data
        else:
            return TransposePlanner(sub_shape, self.dtype, self.axis, self.comm_sub)

    @CachedMethod
    def _group_plan(self, nfields, scales):
        """Build group transpose plan."""
        sub_shape = self._sub_shape(scales)
        group_shape = np.hstack([nfields, sub_shape])
        if np.prod(group_shape) == 0:
            return None, None, None  # no data
        else:
            # Create group buffer to hold group data contiguously
            buffer0_shape = np.hstack([nfields, self.layout0.local_shape(scales)])
            buffer1_shape = np.hstack([nfields, self.layout1.local_shape(scales)])
            size = max(np.prod(buffer0_shape), np.prod(buffer1_shape))
            buffer = fftw.create_array(shape=[size], dtype=self.dtype)
            buffer0 = np.ndarray(shape=buffer0_shape, dtype=self.dtype, buffer=buffer)
            buffer1 = np.ndarray(shape=buffer1_shape, dtype=self.dtype, buffer=buffer)
            # Creat plan on subsequent axis of group shape
            plan = TransposePlanner(group_shape, self.dtype, self.axis+1, self.comm_sub)
            return plan, buffer0, buffer1

    def increment(self, fields):
        """Transpose from layout0 to layout1."""
        if SYNC_TRANSPOSES:
            self.comm_sub.Barrier()
        if len(fields) == 1:
            self.increment_single(*fields)
        elif GROUP_TRANSPOSES:
            self.increment_group(*fields)
        else:
            for field in fields:
                self.increment_single(field)

    def decrement(self, fields):
        """Transpose from layout1 to layout0."""
        if SYNC_TRANSPOSES:
            self.comm_sub.Barrier()
        if len(fields) == 1:
            self.decrement_single(*fields)
        elif GROUP_TRANSPOSES:
            self.decrement_group(*fields)
        else:
            for field in fields:
                self.decrement_single(field)

    def increment_single(self, field):
        """Transpose field from layout0 to layout1."""
        scales = field.meta[:]['scale']
        plan = self._single_plan(scales)
        if plan:
            # Setup views of data in each layout
            data0 = field.data
            field.layout = self.layout1
            data1 = field.data
            # Transpose between data views
            plan.localize_columns(data0, data1)
        else:
            # No data: just update field layout
            field.layout = self.layout1

    def decrement_single(self, field):
        """Transpose field from layout1 to layout0."""
        scales = field.meta[:]['scale']
        plan = self._single_plan(scales)
        if plan:
            # Setup views of data in each layout
            data1 = field.data
            field.layout = self.layout0
            data0 = field.data
            # Transpose between data views
            plan.localize_rows(data1, data0)
        else:
            # No data: just update field layout
            field.layout = self.layout0

    def increment_group(self, *fields):
        """Transpose group from layout0 to layout1."""
        scales = unify(field.meta[:]['scale'] for field in fields)
        plan, buffer0, buffer1 = self._group_plan(len(fields), scales)
        if plan:
            # Copy fields to group buffer
            for i, field in enumerate(fields):
                np.copyto(buffer0[i], field.data)
            # Transpose between group buffer views
            plan.localize_columns(buffer0, buffer1)
            # Copy from group buffer to fields in new layout
            for i, field in enumerate(fields):
                field.layout = self.layout1
                np.copyto(field.data, buffer1[i])
        else:
            # No data: just update field layouts
            for field in fields:
                field.layout = self.layout1

    def decrement_group(self, *fields):
        """Transpose group from layout1 to layout0."""
        scales = unify(field.meta[:]['scale'] for field in fields)
        plan, buffer0, buffer1 = self._group_plan(len(fields), scales)
        if plan:
            # Copy fields to group buffer
            for i, field in enumerate(fields):
                np.copyto(buffer1[i], field.data)
            # Transpose between group buffer views
            plan.localize_rows(buffer1, buffer0)
            # Copy from group buffer to fields in new layout
            for i, field in enumerate(fields):
                field.layout = self.layout0
                np.copyto(field.data, buffer0[i])
        else:
            # No data: just update field layouts
            for field in fields:
                field.layout = self.layout0

