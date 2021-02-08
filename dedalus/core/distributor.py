"""
Distributor, Layout, Transform, and Transpose class definitions.
"""

import logging
from mpi4py import MPI
import numpy as np
from collections import OrderedDict

from ..tools.cache import CachedMethod, CachedAttribute
from ..tools.config import config

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
    Directs parallelized distribution and transformation of fields.

    Parameters
    ----------
    dim : int
        Dimension
    comm : MPI communicator, optional
        MPI communicator (default: comm world)
    mesh : tuple of ints, optional
        Process mesh for parallelization (default: 1-D mesh of available processes)

    Attributes
    ----------
    comm_cart : MPI communicator
        Cartesian MPI communicator over mesh
    coords : array of ints
        Coordinates in cartesian communicator
    layouts : list of layout objects
        Available layouts

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

    The distributor object for a given dimension constructs layout objects
    describing each of the (D+R+1) layouts (sets of transform/distribution
    states) and the paths between them (D transforms and R transposes).
    """

    def __init__(self, coordsystems, comm=None, mesh=None):
        self.coords = tuple([coord for coordsystem in coordsystems for coord in coordsystem.coords])
        for coordsystem in coordsystems:
            coordsystem.set_distributor(self)
        self.coordsystems = coordsystems
        # Defaults
        if comm is None:
            comm = MPI.COMM_WORLD
        if mesh is None:
            mesh = np.array([comm.size], dtype=int)
        self.dim = dim = len(self.coords)
#        self.dim = dim = sum(coordsystem.dim for coordsystem in coordsystems)
        self.comm = comm
        # Squeeze out local/bad (size <= 1) dimensions
        self.mesh = mesh = np.array([i for i in mesh if (i>1)], dtype=int)
        # Check mesh compatibility
        logger.debug('Mesh: %s' %str(mesh))
        if mesh.size >= dim:
            raise ValueError("Mesh (%s) must have lower dimension than distributor (%i)" %(mesh, dim))
        if np.prod(mesh) != comm.size:
            raise ValueError("Wrong number of processes (%i) for specified mesh (%s)" %(comm.size, mesh))
        # Create cartesian communicator
        self.comm_cart = comm.Create_cart(mesh)
        self.comm_coords = np.array(self.comm_cart.coords, dtype=int)
        # Build layout objects
        self._build_layouts()

    @CachedAttribute
    def cs_by_axis(self):
        cs_dict = {}
        for cs in self.coordsystems:
            for subaxis in range(cs.dim):
                cs_dict[cs.axis+subaxis] = cs
        return cs_dict

    def get_coordsystem(self, axis):
        return self.cs_by_axis[axis]

    def _build_layouts(self, dry_run=False):
        """Construct layout objects."""
        D = self.dim
        R = self.mesh.size
        # First layout: full coefficient space
        local = [False] * R + [True] * (D-R)
        grid_space = [False] * D
        layout_0 = Layout(self, local, grid_space)
        layout_0.index = 0
        # Layout and path lists
        self.layouts = [layout_0]
        self.paths = []
        # Subsequent layouts
        for i in range(1, R+D+1):
            # Iterate backwards over bases to last coefficient space basis
            for d in reversed(range(D)):
                if not grid_space[d]:
                    # Transform if local
                    if local[d]:
                        grid_space[d] = True
                        layout_i = Layout(self, local, grid_space)
                        if not dry_run:
                            path_i = Transform(self.layouts[-1], layout_i, d)
                        break
                    # Otherwise transpose
                    else:
                        local[d] = True
                        local[d+1] = False
                        layout_i = Layout(self, local, grid_space)
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
                                  'g': self.grid_layout}

    def get_layout_object(self, input):
        """Dereference layout identifiers."""
        if isinstance(input, Layout):
            return input
        else:
            return self.layout_references[input]

    def buffer_size(self, domain, scales, dtype):
        """Compute necessary buffer size (bytes) for all layouts."""
        return max(layout.buffer_size(domain, scales, dtype) for layout in self.layouts)

    def remedy_scales(self, scales):
        """Remedy different scale inputs."""
        if scales is None:
            scales = 1
        if np.isscalar(scales):
            scales = [scales] * self.dim
        if 0 in scales:
            raise ValueError("Scales must be nonzero.")
        return tuple(scales)

    def get_transform_object(self, axis):
        for path in self.paths:
            if isinstance(path, Transform):
                if path.axis == axis:
                    return path


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

    """

    def __init__(self, dist, local, grid_space):
        self.dist = dist
        # Freeze local and grid_space lists into boolean arrays
        self.local = np.array(local)
        self.grid_space = np.array(grid_space)
        # Extend mesh and coordinates to distributor dimension
        self.ext_mesh = np.ones(dist.dim, dtype=int)
        self.ext_mesh[~self.local] = dist.mesh
        self.ext_coords = np.zeros(dist.dim, dtype=int)
        self.ext_coords[~self.local] = dist.comm_coords

    def global_shape(self, domain, scales):
        """Global data shape."""
        scales = self.dist.remedy_scales(scales)
        #global_shape = np.array(domain.coeff_shape).copy()
        #global_shape[self.grid_space] = np.array(domain.grid_shape(scales))[self.grid_space]
        global_shape = domain.global_shape(self, scales)
        return tuple(global_shape)

    def chunk_shape(self, domain):
        """Chunk shape."""
        #scales = self.dist.remedy_scales(scales)
        #chunk_shape = np.array(domain.coeff_group_shape).copy()
        #chunk_shape[self.grid_space] = 1
        chunk_shape = domain.chunk_shape(self)
        return tuple(chunk_shape)

    def local_chunks(self, domain, scales):
        """Local chunk indices by axis."""
        global_shape = self.global_shape(domain, scales)
        chunk_shape = self.chunk_shape(domain)
        chunk_nums = -(-np.array(global_shape) // np.array(chunk_shape))  # ceil
        local_chunks = []
        for axis, basis in enumerate(domain.full_bases):
            if self.local[axis]:
                # All chunks for local dimensions
                local_chunks.append(np.arange(chunk_nums[axis]))
            # elif basis is None:
            #     # Copy across constant dimensions
            #     local_chunks.append(np.arange(chunk_nums[axis]))
            else:
                # Block distribution otherwise
                mesh = self.ext_mesh[axis]
                coord = self.ext_coords[axis]
                block = -(-chunk_nums[axis] // mesh)
                start = min(chunk_nums[axis], block*coord)
                end = min(chunk_nums[axis], block*(coord+1))
                local_chunks.append(np.arange(start, end))
        return tuple(local_chunks)

    def local_elements(self, domain, scales):
        """Local element indices by axis."""
        chunk_shape = self.chunk_shape(domain)
        local_chunks = self.local_chunks(domain, scales)
        indices = []
        for GS, LG in zip(chunk_shape, local_chunks):
            indices.append(np.array([GS*G+i for G in LG for i in range(GS)], dtype=int))
        return indices

    def slices(self, domain, scales):
        """Local element slices by axis."""
        return np.ix_(*self.local_elements(domain, scales))

    @CachedMethod
    def local_shape(self, domain, scales):
        """Local data shape."""
        local_elements = self.local_elements(domain, scales)
        return tuple(LE.size for LE in local_elements)

    def buffer_size(self, bases, scales, dtype):
        """Local buffer size (bytes)."""
        local_shape = self.local_shape(bases, scales)
        return np.prod(local_shape) * np.dtype(dtype).itemsize

    # def local_group_index(self, group, domain, scales):
    #     """Index of a group within local groups."""
    #     index = []
    #     for grp, local_grps in zip(group, self.local_groups(domain, scales)):
    #         if grp is None:
    #             index.append(None)
    #         else:
    #             index.append(local_grps.index(grp))
    #     return index

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
    # def groups(self, domain, scales):
    #     """Comptue group sizes."""
    #     groups = []
    #     for axis, space in enumerate(domain.spaces):
    #         if space is None:
    #             groups.append(1)
    #         elif self.grid_space[axis]:
    #             groups.append(1)
    #         else:
    #             groups.append(space.group_size)
    #     return np.array(groups, dtype=int)

    # @CachedMethod
    # def blocks(self, domain, scales):
    #     """Compute block sizes for data distribution."""
    #     global_shape = self.global_shape(domain, scales)
    #     groups = self.groups(domain, scales)
    #     return groups * np.ceil(global_shape / groups / self.ext_mesh).astype(int)

    # @CachedMethod
    # def start(self, domain, scales):
    #     """Compute starting coordinates for local data."""
    #     blocks = self.blocks(domain, scales)
    #     start = self.ext_coords * blocks
    #     start[domain.constant] = 0
    #     return start

    # @CachedMethod
    # def local_shape(self, domain, scales):
    #     """Compute local data shape."""
    #     global_shape = self.global_shape(domain, scales)
    #     blocks = self.blocks(domain, scales)
    #     start = self.start(domain, scales)
    #     local_shape = np.minimum(blocks, global_shape-start)
    #     local_shape = np.maximum(0, local_shape)
    #     return local_shape

    # @CachedMethod
    # def slices(self, domain, scales):
    #     """Compute slices for selecting local portion of global data."""
    #     start = self.start(domain, scales)
    #     local_shape = self.local_shape(domain, scales)
    #     return tuple(slice(s, s+l) for (s, l) in zip(start, local_shape))



class Transform:
    """
    Directs spectral transforms between two layouts.

    TODO:
        - Implement grouped transforms
    """

    def __init__(self, layout0, layout1, axis):
        self.layout0 = layout0
        self.layout1 = layout1
        self.axis = axis

    def increment(self, fields):
        """Backward transform a list of fields."""
        if len(fields) == 1:
            self.increment_single(*fields)
        elif GROUP_TRANSFORMS:
            self.increment_group(fields)
        else:
            for field in fields:
                self.increment_single(field)

    def decrement(self, fields):
        """Forward transform a list of fields."""
        if len(fields) == 1:
            self.decrement_single(*fields)
        elif GROUP_TRANSFORMS:
            self.decrement_group(fields)
        else:
            for field in fields:
                self.decrement_single(field)

    def increment_single(self, field):
        """Backward transform a field."""
        axis = self.axis
        basis = field.domain.full_bases[axis]
        # Reference views from both layouts
        cdata = field.data
        field.set_layout(self.layout1)
        gdata = field.data
        # Transform non-constant bases with local data
        if (basis is not None) and np.prod(cdata.shape):
            basis.backward_transform(field, axis, cdata, gdata)
            #basis.backward_transform(cdata, gdata, axis, field.scales[axis], field.tensorsig)
            #plan = basis.transform_plan(cdata.shape, self.axis, field.scales[self.axis], field.dtype)
            #plan.backward(cdata, gdata)

    def decrement_single(self, field):
        """Forward transform a field."""
        axis = self.axis
        basis = field.domain.full_bases[axis]
        # Reference views from both layouts
        gdata = field.data
        field.set_layout(self.layout0)
        cdata = field.data
        # Transform non-constant bases with local data
        if (basis is not None) and np.prod(gdata.shape):
            basis.forward_transform(field, axis, gdata, cdata)
            #basis.forward_transform(gdata, cdata, axis, field.scales[axis], field.tensorsig)
            #plan = basis.transform_plan(cdata.shape, self.axis, field.scales[self.axis], field.dtype)
            #plan.forward(gdata, cdata)

    def increment_group(self, fields):
        """Backward transform multiple fields simultaneously."""
        #logger.warning("Group transforms not implemented.")
        for field in fields:
            self.increment_single(field)

    def decrement_group(self, fields):
        """Forward transform multiple fields simultaneously."""
        #logger.warning("Group transforms not implemented.")
        for field in fields:
            self.decrement_single(field)

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


class Transpose:
    """
    Directs distributed transposes between two layouts.

    TODO:
        - Implement grouped transposes
        - Transpose all components simultaneously
    """

    def __init__(self, layout0, layout1, axis, comm_cart):
        self.layout0 = layout0
        self.layout1 = layout1
        self.axis = axis
        self.comm_cart = comm_cart
        # Create subgrid communicator along the moving mesh axis
        remain_dims = [0] * comm_cart.dim
        remain_dims[axis] = 1
        self.comm_sub = comm_cart.Sub(remain_dims)

    def _sub_shape(self, domain, scales):
        """Build global shape of data assigned to sub-communicator."""
        local_shape = self.layout0.local_shape(domain, scales)
        global_shape = self.layout0.global_shape(domain, scales)
        # Global shape along transposing axes, local shape along others
        sub_shape = np.array(local_shape)
        sub_shape[self.axis] = global_shape[self.axis]
        sub_shape[self.axis+1] = global_shape[self.axis+1]
        return tuple(sub_shape)

    @CachedMethod
    def _plan(self, ncomp, sub_shape, chunk_shape, dtype):
        axis = self.axis
        if np.prod(sub_shape) == 0:
            return None  # no data
        elif (sub_shape[axis] == chunk_shape[axis]) and (sub_shape[axis+1] == chunk_shape[axis+1]):
            return None  # no change
        else:
            # Add axis for components
            full_sub_shape = (ncomp,) + sub_shape
            full_chunk_shape = (ncomp,) + chunk_shape
            return TransposePlanner(full_sub_shape, full_chunk_shape, dtype, axis+1, self.comm_sub)

    def _single_plan(self, field):
        """Build single transpose plan."""
        ncomp = np.prod([cs.dim for cs in field.tensorsig], dtype=int)
        sub_shape = self._sub_shape(field.domain, field.scales)
        chunk_shape = field.domain.chunk_shape(self.layout0)
        return self._plan(ncomp, sub_shape, chunk_shape, field.dtype)

    def _group_plans(self, fields):
        """Build group transpose plan."""
        # Segment fields by sub_shapes and chunk_shapes
        field_groups = OrderedDict()
        for field in fields:
            sub_shape = self._sub_shape(field.domain, field.scales)
            chunk_shape = field.domain.chunk_shape(self.layout0)
            if (sub_shape, chunk_shape) in field_groups:
                field_groups[(sub_shape, chunk_shape)].append(field)
            else:
                field_groups[(sub_shape, chunk_shape)] = [field]
        # Plan for each field group
        plans = []
        for (sub_shape, chunk_shape), fields in field_groups.items():
            ncomp = 0
            for field in fields:
                ncomp += np.prod([cs.dim for cs in field.tensorsig], dtype=int)
            plan = self._plan(ncomp, sub_shape, chunk_shape, field.dtype) # Assumes last field's dtype is good for everybody
            plans.append((fields, plan))
        return plans

    def increment(self, fields):
        """Backward transpose a list of fields."""
        if SYNC_TRANSPOSES:
            self.comm_sub.Barrier()
        if len(fields) == 1:
            self.increment_single(*fields)
        elif GROUP_TRANSPOSES:
            self.increment_group(fields)
        else:
            for field in fields:
                self.increment_single(field)

    def decrement(self, fields):
        """Forward transpose a list of fields."""
        if SYNC_TRANSPOSES:
            self.comm_sub.Barrier()
        if len(fields) == 1:
            self.decrement_single(*fields)
        elif GROUP_TRANSPOSES:
            self.decrement_group(fields)
        else:
            for field in fields:
                self.decrement_single(field)

    def increment_single(self, field):
        """Backward transpose a field."""
        plan = self._single_plan(field)
        if plan:
            # Reference views from both layouts
            data0 = field.data
            field.set_layout(self.layout1)
            data1 = field.data
            # Transpose between data views
            plan.localize_columns(data0, data1)
        else:
            # No communication: just update field layout
            field.set_layout(self.layout1)

    def decrement_single(self, field):
        """Forward transpose a field."""
        plan = self._single_plan(field)
        if plan:
            # Reference views from both layouts
            data1 = field.data
            field.set_layout(self.layout0)
            data0 = field.data
            # Transpose between data views
            plan.localize_rows(data1, data0)
        else:
            # No communication: just update field layout
            field.set_layout(self.layout0)

    def increment_group(self, fields):
        """Backward transpose multiple fields simultaneously."""
        plans = self._group_plans(fields)
        for fields, plan in plans:
            if plan:
                if len(fields) == 1:
                    field = fields[0]
                    # Reference views from both layouts
                    data0 = field.data
                    field.set_layout(self.layout1)
                    data1 = field.data
                    # Transpose between data views
                    plan.localize_columns(data0, data1)
                else:
                    # Gather data across fields
                    data0 = []
                    data1 = []
                    for field in fields:
                        rank = len(field.tensorsig)
                        # Reference views from both layouts
                        flat_comp_shape = (-1,) + field.data.shape[rank:]
                        data0.append(field.data.reshape(flat_comp_shape))
                        field.set_layout(self.layout1)
                        flat_comp_shape = (-1,) + field.data.shape[rank:]
                        data1.append(field.data.reshape(flat_comp_shape))
                    data0 = np.concatenate(data0)
                    data1 = np.concatenate(data1)
                    # Transpose between data views
                    plan.localize_columns(data0, data1)
                    # Split up transposed data
                    i = 0
                    for field in fields:
                        ncomp = np.prod([cs.dim for cs in field.tensorsig], dtype=int)
                        data = data1[i:i+ncomp]
                        field.data[:] = data.reshape(field.data.shape)
                        i += ncomp
            else:
                # No communication: just update field layouts
                for field in fields:
                    field.set_layout(self.layout1)

    def decrement_group(self, fields):
        """Forward transpose multiple fields simultaneously."""
        plans = self._group_plans(fields)
        for fields, plan in plans:
            if plan:
                if len(fields) == 1:
                    field = fields[0]
                    # Reference views from both layouts
                    data1 = field.data
                    field.set_layout(self.layout0)
                    data0 = field.data
                    # Transpose between data views
                    plan.localize_rows(data1, data0)
                else:
                    # Gather data across fields
                    data0 = []
                    data1 = []
                    for field in fields:
                        rank = len(field.tensorsig)
                        # Reference views from both layouts
                        flat_comp_shape = (-1,) + field.data.shape[rank:]
                        data1.append(field.data.reshape(flat_comp_shape))
                        field.set_layout(self.layout0)
                        flat_comp_shape = (-1,) + field.data.shape[rank:]
                        data0.append(field.data.reshape(flat_comp_shape))
                    data0 = np.concatenate(data0)
                    data1 = np.concatenate(data1)
                    # Transpose between data views
                    plan.localize_rows(data1, data0)
                    # Split up transposed data
                    i = 0
                    for field in fields:
                        ncomp = np.prod([cs.dim for cs in field.tensorsig], dtype=int)
                        data = data0[i:i+ncomp]
                        field.data[:] = data.reshape(field.data.shape)
                        i += ncomp
            else:
                # No communication: just update field layouts
                for field in fields:
                    field.set_layout(self.layout1)

    # def increment_group(self, *fields):
    #     """Transpose group from layout0 to layout1."""
    #     scales = unify(field.scales for field in fields)
    #     plan, buffer0, buffer1 = self._group_plan(len(fields), scales)
    #     if plan:
    #         # Copy fields to group buffer
    #         for i, field in enumerate(fields):
    #             np.copyto(buffer0[i], field.data)
    #         # Transpose between group buffer views
    #         plan.localize_columns(buffer0, buffer1)
    #         # Copy from group buffer to fields in new layout
    #         for i, field in enumerate(fields):
    #             field.set_layout(self.layout1)
    #             np.copyto(field.data, buffer1[i])
    #     else:
    #         # No data: just update field layouts
    #         for field in fields:
    #             field.set_layout(self.layout1)

    # def decrement_group(self, *fields):
    #     """Transpose group from layout1 to layout0."""
    #     scales = unify(field.scales for field in fields)
    #     plan, buffer0, buffer1 = self._group_plan(len(fields), scales)
    #     if plan:
    #         # Copy fields to group buffer
    #         for i, field in enumerate(fields):
    #             np.copyto(buffer1[i], field.data)
    #         # Transpose between group buffer views
    #         plan.localize_rows(buffer1, buffer0)
    #         # Copy from group buffer to fields in new layout
    #         for i, field in enumerate(fields):
    #             field.set_layout(self.layout0)
    #             np.copyto(field.data, buffer0[i])
    #     else:
    #         # No data: just update field layouts
    #         for field in fields:
    #             field.set_layout(self.layout0)

