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
TRANSPOSE_LIBRARY = config['parallelism'].get('TRANSPOSE_LIBRARY')
GROUP_TRANSPOSES = config['parallelism'].getboolean('GROUP_TRANSPOSES')
SYNC_TRANSPOSES = config['parallelism'].getboolean('SYNC_TRANSPOSES')

if TRANSPOSE_LIBRARY.upper() == 'FFTW':
    from .transposes import FFTWTranspose as TransposePlanner
elif TRANSPOSE_LIBRARY.upper() == 'MPI':
    from .transposes import AlltoallvTranspose as TransposePlanner
from .transposes import RowDistributor, ColDistributor


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
        layout_0 = Layout(domain, mesh, coords, local, grid_space)
        layout_0.index = 0

        # Layout and path lists
        self.layouts = [layout_0]
        self.paths = []

        # Subsequent layouts
        for i in range(1, R+D+1):
            # Iterate backwards over bases to last coefficient space basis
            for d in reversed(range(domain.dim)):
                if not grid_space[d]:
                    # Transform if local
                    if local[d]:
                        grid_space[d] = True
                        layout_i = Layout(domain, mesh, coords, local, grid_space)
                        if not dry_run:
                            path_i = Transform(self.layouts[-1], layout_i, d)
                        break
                    # Otherwise transpose
                    else:
                        local[d] = True
                        local[d+1] = False
                        layout_i = Layout(domain, mesh, coords, local, grid_space)
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
    def buffer_size(self, bases, scales):
        """Compute necessary buffer size (bytes) for all layouts."""
        return max(layout.buffer_size(bases, scales) for layout in self.layouts)


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

    distribution = 'block'

    def __init__(self, domain, mesh, coords, local, grid_space):
        self.domain = domain
        # Freeze local and grid_space lists into boolean arrays
        self.local = np.array(local)
        self.grid_space = np.array(grid_space)
        # Extend mesh and coordinates to domain dimension
        self.ext_mesh = np.ones(domain.dim, dtype=int)
        self.ext_mesh[~self.local] = mesh
        self.ext_coords = np.zeros(domain.dim, dtype=int)
        self.ext_coords[~self.local] = coords

    @CachedMethod
    def global_shape(self, subdomain, scales):
        """Compute global data shape."""
        scales = self.domain.remedy_scales(scales)
        grid_space = self.grid_space
        global_shape = np.zeros(self.domain.dim, dtype=int)
        global_shape[grid_space] = subdomain.global_grid_shape(scales)[grid_space]
        global_shape[~grid_space] = subdomain.global_coeff_shape[~grid_space]
        return global_shape

    def global_groups(self, subdomain, scales):
        """Global group indices by axis."""
        scales = self.domain.remedy_scales(scales)
        groups = []
        for axis, space in enumerate(subdomain.spaces):
            if space is None:
                n_groups = 1
            elif self.grid_space[axis]:
                n_groups = space.grid_size(scales[axis])
            else:
                n_groups = space.coeff_size // space.group_size
            groups.append(np.arange(n_groups))
        return groups

    def local_groups(self, subdomain, scales):
        """Local group indices by axis."""
        global_groups = self.global_groups(subdomain, scales)
        groups = []
        for axis, space in enumerate(subdomain.spaces):
            if self.local[axis] or (space is None):
                groups.append(global_groups[axis])
            else:
                mesh = self.ext_mesh[axis]
                coord = self.ext_coords[axis]
                if self.distribution == 'block':
                    block = len(global_groups[axis]) // mesh
                    start = coord * block
                    groups.append(global_groups[axis][start:start+block])
                elif self.distribution == 'cyclic':
                    groups.append(global_groups[axis][coord::mesh])
        return groups

    def local_elements(self, subdomain, scales):
        """Local element indices by axis."""
        local_groups = self.local_groups(subdomain, scales)
        indices = []
        for axis, space in enumerate(subdomain.spaces):
            if space is None:
                indices.append(np.arange(1))
            else:
                if self.grid_space[axis]:
                    GS = 1
                else:
                    GS = space.group_size
                ind = [GS*G+i for i in range(GS) for G in local_groups[axis]]
                indices.append(np.array(ind))
        return indices
        #return np.ix_(*indices)

    def slices(self, subdomain, scales):
        return np.ix_(*self.local_elements(subdomain, scales))

    def global_array_shape(self, subdomain, scales):
        """Global array shape."""
        scales = self.domain.remedy_scales(scales)
        shape = []
        for axis, space in enumerate(subdomain.spaces):
            if space is None:
                shape.append(1)
            elif self.grid_space[axis]:
                shape.append(space.grid_size(scales[axis]))
            else:
                shape.append(space.coeff_size)
        return tuple(shape)

    def local_array_shape(self, subdomain, scales):
        """Local array shape."""
        local_elements = self.local_elements(subdomain, scales)
        return [LE.size for LE in local_elements]

    def local_group_index(self, group, subdomain, scales):
        """Index of a group within local groups."""
        index = []
        for grp, local_grps in zip(group, self.local_groups(subdomain, scales)):
            if grp is None:
                index.append(None)
            else:
                index.append(local_grps.index(grp))
        return index

    #@CachedMethod
    def buffer_size(self, subdomain, scales):
        """Compute necessary buffer size (bytes)."""
        local_shape = self.local_array_shape(subdomain, scales)
        return np.prod(local_shape) * self.domain.dtype.itemsize



        # if distribution == 'block':
        #     index[~local] = (group - start)[~local]
        # elif distribution == 'cyclic':
        #     index[~local] = np.mod(group, mesh)[~local]

    # def group(self, local_index):
    #     pass
    #     # if distribution == 'block':
    #     #     group = start + index
    #     # elif distribution == 'cyclic':
    #     #     group = mesh *



    # @CachedMethod
    # def groups(self, subdomain, scales):
    #     """Comptue group sizes."""
    #     groups = []
    #     for axis, space in enumerate(subdomain.spaces):
    #         if space is None:
    #             groups.append(1)
    #         elif self.grid_space[axis]:
    #             groups.append(1)
    #         else:
    #             groups.append(space.group_size)
    #     return np.array(groups, dtype=int)

    # @CachedMethod
    # def blocks(self, subdomain, scales):
    #     """Compute block sizes for data distribution."""
    #     global_shape = self.global_shape(subdomain, scales)
    #     groups = self.groups(subdomain, scales)
    #     return groups * np.ceil(global_shape / groups / self.ext_mesh).astype(int)

    # @CachedMethod
    # def start(self, subdomain, scales):
    #     """Compute starting coordinates for local data."""
    #     blocks = self.blocks(subdomain, scales)
    #     start = self.ext_coords * blocks
    #     start[subdomain.constant] = 0
    #     return start

    # @CachedMethod
    # def local_shape(self, subdomain, scales):
    #     """Compute local data shape."""
    #     global_shape = self.global_shape(subdomain, scales)
    #     blocks = self.blocks(subdomain, scales)
    #     start = self.start(subdomain, scales)
    #     local_shape = np.minimum(blocks, global_shape-start)
    #     local_shape = np.maximum(0, local_shape)
    #     return local_shape

    # @CachedMethod
    # def slices(self, subdomain, scales):
    #     """Compute slices for selecting local portion of global data."""
    #     start = self.start(subdomain, scales)
    #     local_shape = self.local_shape(subdomain, scales)
    #     return tuple(slice(s, s+l) for (s, l) in zip(start, local_shape))



class Transform:
    """Directs transforms between two layouts."""

    def __init__(self, layout0, layout1, axis):
        self.domain = layout0.domain
        self.layout0 = layout0
        self.layout1 = layout1
        self.axis = axis

    # @CachedMethod
    # def group_data(self, nfields, scales):

    #     local_shape0 = self.layout0.local_shape(scales)
    #     local_shape1 = self.layout1.local_shape(scales)
    #     group_shape0 = [nfields] + list(local_shape0)
    #     group_shape1 = [nfields] + list(local_shape1)
    #     group_cdata = fftw.create_array(group_shape0, self.layout0.dtype)
    #     group_gdata = fftw.create_array(group_shape1, self.layout1.dtype)

    #     return group_cdata, group_gdata

    # def increment_group(self, fields):
    #     fields = list(fields)
    #     scales = fields[0].meta[:]['scale']
    #     cdata, gdata = self.group_data(len(fields), scales)
    #     for i, field in enumerate(fields):
    #         np.copyto(cdata[i], field.data)
    #     self.basis.backward(cdata, gdata, self.axis+1, fields[0].meta[self.axis])
    #     for i, field in enumerate(fields):
    #         field.layout = self.layout1
    #         np.copyto(field.data, gdata[i])

    # def decrement_group(self, fields):
    #     fields = list(fields)
    #     scales = fields[0].meta[:]['scale']
    #     cdata, gdata = self.group_data(len(fields), scales)
    #     for i, field in enumerate(fields):
    #         np.copyto(gdata[i], field.data)
    #     self.basis.forward(gdata, cdata, self.axis+1, fields[0].meta[self.axis])
    #     for i, field in enumerate(fields):
    #         field.layout = self.layout0
    #         np.copyto(field.data, cdata[i])

    def increment_single(self, field):
        """Backward transform."""
        basis = field.bases[self.axis]
        # Reference views from both layouts
        cdata = field.data
        field.set_layout(self.layout1)
        gdata = field.data
        # Transform if there's local data
        if (basis is not None) and np.prod(cdata.shape):
            plan = basis.transform_plan(cdata.shape, self.axis, field.scales[self.axis])
            plan.backward(cdata, gdata)

    def decrement_single(self, field):
        """Forward transform."""
        basis = field.bases[self.axis]
        # Reference views from both layouts
        gdata = field.data
        field.set_layout(self.layout0)
        cdata = field.data
        # Transform if there's local data
        if (basis is not None) and np.prod(gdata.shape):
            plan = basis.transform_plan(cdata.shape, self.axis, field.scales[self.axis])
            plan.forward(gdata, cdata)

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
        self.dtype = layout0.domain.dtype  # same as layout1.dtype
        self.axis = axis
        self.comm_cart = comm_cart
        self.comm_sub = comm_sub

    def _sub_shape(self, subdomain, scales):
        """Build global shape of data assigned to sub-communicator."""
        local_shape = self.layout0.local_array_shape(subdomain, scales)
        global_shape = self.layout0.global_array_shape(subdomain, scales)
        # Global shape along transposing axes, local shape along others
        sub_shape = np.array(local_shape)
        sub_shape[self.axis] = global_shape[self.axis]
        sub_shape[self.axis+1] = global_shape[self.axis+1]
        return sub_shape

    @CachedMethod
    def _single_plan(self, subdomain, scales):
        """Build single transpose plan."""
        sub_shape = self._sub_shape(subdomain, scales)
        dtype = self.layout0.domain.dtype
        axis = self.axis
        if np.prod(sub_shape) == 0:
            return None  # no data
        elif (subdomain.spaces[axis] is None) and (subdomain.spaces[axis+1] is None):
            return None  # no change
        elif (subdomain.spaces[axis] is None):
            return RowDistributor(sub_shape, dtype, axis, self.comm_sub)
        elif (subdomain.spaces[axis+1] is None):
            return ColDistributor(sub_shape, dtype, axis, self.comm_sub)
        else:
            return TransposePlanner(sub_shape, dtype, axis, self.comm_sub)

    @CachedMethod
    def _group_plan(self, nfields, scales, dtype):
        """Build group transpose plan."""
        sub_shape = self._sub_shape(scales)
        group_shape = np.hstack([nfields, sub_shape])
        if np.prod(group_shape) == 0:
            return None, None, None  # no data
        else:
            # Create group buffer to hold group data contiguously
            buffer0_shape = np.hstack([nfields, self.layout0.local_array_shape(scales)])
            buffer1_shape = np.hstack([nfields, self.layout1.local_array_shape(scales)])
            size = max(np.prod(buffer0_shape), np.prod(buffer1_shape))
            buffer = fftw.create_array(shape=[size], dtype=dtype)
            buffer0 = np.ndarray(shape=buffer0_shape, dtype=dtype, buffer=buffer)
            buffer1 = np.ndarray(shape=buffer1_shape, dtype=dtype, buffer=buffer)
            # Creat plan on subsequent axis of group shape
            plan = TransposePlanner(group_shape, dtype, self.axis+1, self.comm_sub)
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
        scales = field.scales
        plan = self._single_plan(field.subdomain, scales)
        if plan:
            # Setup views of data in each layout
            data0 = field.data
            field.set_layout(self.layout1)
            data1 = field.data
            # Transpose between data views
            plan.localize_columns(data0, data1)
        else:
            # No data: just update field layout
            field.set_layout(self.layout1)

    def decrement_single(self, field):
        """Transpose field from layout1 to layout0."""
        scales = field.scales
        plan = self._single_plan(field.subdomain, scales)
        if plan:
            # Setup views of data in each layout
            data1 = field.data
            field.set_layout(self.layout0)
            data0 = field.data
            # Transpose between data views
            plan.localize_rows(data1, data0)
        else:
            # No data: just update field layout
            field.set_layout(self.layout0)

    def increment_group(self, *fields):
        """Transpose group from layout0 to layout1."""
        scales = unify(field.scales for field in fields)
        plan, buffer0, buffer1 = self._group_plan(len(fields), scales)
        if plan:
            # Copy fields to group buffer
            for i, field in enumerate(fields):
                np.copyto(buffer0[i], field.data)
            # Transpose between group buffer views
            plan.localize_columns(buffer0, buffer1)
            # Copy from group buffer to fields in new layout
            for i, field in enumerate(fields):
                field.set_layout(self.layout1)
                np.copyto(field.data, buffer1[i])
        else:
            # No data: just update field layouts
            for field in fields:
                field.set_layout(self.layout1)

    def decrement_group(self, *fields):
        """Transpose group from layout1 to layout0."""
        scales = unify(field.scales for field in fields)
        plan, buffer0, buffer1 = self._group_plan(len(fields), scales)
        if plan:
            # Copy fields to group buffer
            for i, field in enumerate(fields):
                np.copyto(buffer1[i], field.data)
            # Transpose between group buffer views
            plan.localize_rows(buffer1, buffer0)
            # Copy from group buffer to fields in new layout
            for i, field in enumerate(fields):
                field.set_layout(self.layout0)
                np.copyto(field.data, buffer0[i])
        else:
            # No data: just update field layouts
            for field in fields:
                field.set_layout(self.layout0)

