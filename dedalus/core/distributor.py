"""
Distributor, Layout, Transform, and Transpose class definitions.
"""

import logging
from mpi4py import MPI
import numpy as np
import itertools
from collections import OrderedDict
from math import prod
import numbers
from weakref import WeakSet

from .coords import CoordinateSystem, DirectProduct
from ..tools.array import reshape_vector
from ..tools.cache import CachedMethod, CachedAttribute
from ..tools.config import config
from ..tools.general import OrderedSet

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

# Public interface
__all__ = ['Distributor']


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

    def __init__(self, coordsystems, comm=None, mesh=None, dtype=None):
        # Accept single coordsys in place of tuple/list
        if not isinstance(coordsystems, (tuple, list)):
            coordsystems = (coordsystems,)
        # Note if only a single coordsys for simplicity
        if len(coordsystems) == 1:
            self.single_coordsys = coordsystems[0]
        else:
            self.single_coordsys = False
        # Get coords
        self.coords = sum((cs.coords for cs in coordsystems), ())
        self.coordsystems = coordsystems
        # Defaults
        if comm is None:
            comm = MPI.COMM_WORLD
        if mesh is None:
            mesh = np.array([comm.size], dtype=int)
        else:
            if isinstance(mesh, list) or isinstance(mesh, tuple):
                mesh = np.array(mesh, dtype=int)
        # Trim trailing ones
        mesh = 1 + np.trim_zeros(mesh - 1, trim='b')
        self.dim = dim = len(self.coords)
#        self.dim = dim = sum(coordsys.dim for coordsys in coordsystems)
        self.comm = comm
        self.mesh = mesh = np.array(mesh)
        # Check mesh compatibility
        logger.debug('Mesh: %s' %str(mesh))
        if mesh.size >= dim:
            raise ValueError("Mesh (%s) must have lower dimension than distributor (%i)" %(mesh, dim))
        if prod(mesh) != comm.size:
            raise ValueError("Wrong number of processes (%i) for specified mesh (%s)" %(comm.size, mesh))
        self.dtype = dtype
        # Create cartesian communicator, ignoring axes with m=1
        reduced_mesh = [m for m in mesh if m > 1]
        self.comm_cart = comm.Create_cart(reduced_mesh)
        self.comm_coords = np.array(self.comm_cart.coords, dtype=int)
        # Build layout objects
        self._build_layouts()
        # Keep set of weak field references
        self.fields = WeakSet()

    @CachedAttribute
    def cs_by_axis(self):
        cs_dict = {}
        for cs in self.coordsystems:
            for subaxis in range(cs.dim):
                axis = self.get_axis(cs)
                cs_dict[axis+subaxis] = cs
        return cs_dict

    def get_coordsystem(self, axis):
        return self.cs_by_axis[axis]

    def _build_layouts(self, dry_run=False):
        """Construct layout objects."""
        D = self.dim
        R = np.sum(self.mesh > 1)
        # First layout: full coefficient space
        local = np.array([True] * D)
        local[:self.mesh.size][self.mesh > 1] = False
        grid_space = [False] * D
        layout_0 = Layout(self, local, grid_space)
        layout_0.index = 0
        # Layout and path lists
        self.layouts = [layout_0]
        self.paths = []
        self.transforms = []
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
                            self.transforms.insert(0, path_i)
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
        if isinstance(scales, numbers.Number):
            scales = (scales,) * self.dim
        scales = tuple(scales)
        if 0 in scales:
            raise ValueError("Scales must be nonzero.")
        return scales

    def get_transform_object(self, axis):
        return self.transforms[axis]

    def get_axis(self, coord):
        if isinstance(coord, CoordinateSystem):
            coord = coord.coords[0]
        return self.coords.index(coord)

    def get_basis_axis(self, basis):
        return self.get_axis(basis.coordsys.coords[0])

    def first_axis(self, basis):
        return self.get_basis_axis(basis)

    def last_axis(self, basis):
        return self.first_axis(basis) + basis.dim - 1

    def Field(self, *args, **kw):
        """Alternate constructor for fields."""
        from .field import Field
        return Field(self, *args, **kw)

    def ScalarField(self, *args, **kw):
        """Alternate constructor for scalar fields."""
        from .field import ScalarField
        return ScalarField(self, *args, **kw)

    def VectorField(self, *args, **kw):
        """Alternate constructor for vector fields."""
        from .field import VectorField
        return VectorField(self, *args, **kw)

    def TensorField(self, *args, **kw):
        """Alternate constructor for tensor fields."""
        from .field import TensorField
        return TensorField(self, *args, **kw)

    def IdentityTensor(self, coordsys_in, coordsys_out=None, bases=None, dtype=None):
        """Identity tensor field."""
        if coordsys_out is None:
            coordsys_out = coordsys_in
        from .field import TensorField
        I = TensorField(self, (coordsys_out, coordsys_in), bases=bases, dtype=dtype)
        if coordsys_in is coordsys_out:
            for i in range(coordsys_in.dim):
                I['g'][i, i] = 1
        elif isinstance(coordsys_in, DirectProduct) and (coordsys_out in coordsys_in.coordsystems):
            i0 = coordsys_in.subaxis_by_cs[coordsys_out]
            for i in range(coordsys_out.dim):
                I['g'][i, i0+i] = 1
        elif isinstance(coordsys_out, DirectProduct) and (coordsys_in in coordsys_out.coordsystems):
            i0 = coordsys_out.subaxis_by_cs[coordsys_in]
            for i in range(coordsys_in.dim):
                I['g'][i0+i, i] = 1
        else:
            raise ValueError("Unsupported coordinate systems.")
        return I

    def local_grid(self, basis, scale=None):
        # TODO: remove from bases and do it all here?
        if scale is None:
            scale = 1
        if basis.dim == 1:
            return basis.local_grid(self, scale=scale)
        else:
            raise ValueError("Use `local_grids` for multidimensional bases.")

    # def global_grids(self, *bases, scales=None):
    #     """Global grids."""
    #     grids = []
    #     scales = self.remedy_scales(scales)
    #     for basis in bases:
    #         basis_axis = self.get_basis_axis(basis)
    #         basis_scales = scales[basis_axis:basis_axis+basis.dim]
    #         global_grids = basis.global_grids(scales=basis_scales)
    #         for subaxis in range(basis.dim):
    #             axis = basis_axis + subaxis
    #             grids.append(reshape_vector(global_grids[subaxis], dim=self.dim, axis=axis))
    #     return tuple(grids)

    # def local_grids(self, *bases, scales=None):
    #     """Local grid."""
    #     grids = []
    #     scales = self.remedy_scales(scales)
    #     for basis in bases:
    #         basis_axis = self.get_basis_axis(basis)
    #         basis_scales = scales[basis_axis:basis_axis+basis.dim]
    #         local_elements = self.grid_layout.local_elements(basis.domain(self), scales=scales)
    #         global_grids = basis.global_grids(scales=basis_scales)
    #         for subaxis in range(basis.dim):
    #             axis = basis_axis + subaxis
    #             local_grid = global_grids[subaxis][local_elements[axis]]
    #             grids.append(reshape_vector(local_grid, dim=self.dim, axis=axis))
    #     return tuple(grids)

    def local_grids(self, *bases, scales=None):
        scales = self.remedy_scales(scales)
        grids = []
        for basis in bases:
            basis_scales = scales[self.first_axis(basis):self.last_axis(basis)+1]
            grids.extend(basis.local_grids(self, scales=basis_scales))
        return grids

    def local_modes(self, basis):
        # TODO: remove from bases and do it all here?
        return basis.local_modes(self)

    @CachedAttribute
    def default_nonconst_groups(self):
        return sum((cs.default_nonconst_groups for cs in self.coordsystems), ())


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
        reduced_mesh = [m for m in dist.mesh if m > 1]
        self.ext_mesh[~self.local] = reduced_mesh
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

    def group_shape(self, domain):
        """Chunk shape."""
        return tuple(domain.group_shape(self))

    def local_chunks(self, domain, scales, rank=None, broadcast=False):
        """Local chunk indices by axis."""
        global_shape = self.global_shape(domain, scales)
        chunk_shape = self.chunk_shape(domain)
        chunk_nums = -(-np.array(global_shape) // np.array(chunk_shape))  # ceil
        local_chunks = []
        # Get coordinates
        if rank is None:
            ext_coords = self.ext_coords
        else:
            ext_coords = np.zeros(self.dist.dim, dtype=int)
            ext_coords[~self.local] = self.dist.comm_cart.Get_coords(rank)
        # Get chunks axis by axis
        for axis, basis in enumerate(domain.full_bases):
            if self.local[axis]:
                # All chunks for local dimensions
                local_chunks.append(np.arange(chunk_nums[axis]))
            else:
                # Block distribution otherwise
                mesh = self.ext_mesh[axis]
                if broadcast and (basis is None):
                    coord = 0
                else:
                    coord = ext_coords[axis]
                block = -(-chunk_nums[axis] // mesh)
                start = min(chunk_nums[axis], block*coord)
                end = min(chunk_nums[axis], block*(coord+1))
                local_chunks.append(np.arange(start, end))
        return tuple(local_chunks)

    def global_elements(self, domain, scales):
        """Global element indices by axis."""
        global_shape = self.global_shape(domain, scales)
        indices = [np.arange(n) for n in global_shape]
        return tuple(indices)

    def local_elements(self, domain, scales, rank=None, broadcast=False):
        """Local element indices by axis."""
        chunk_shape = self.chunk_shape(domain)
        local_chunks = self.local_chunks(domain, scales, rank=rank, broadcast=broadcast)
        indices = []
        for chunk_size, chunks in zip(chunk_shape, local_chunks):
            ax_indices = chunk_size*np.repeat(chunks, chunk_size) + np.tile(np.arange(chunk_size), len(chunks))
            indices.append(ax_indices)
        return tuple(indices)

    @CachedMethod
    def valid_elements(self, tensorsig, domain, scales, rank=None, broadcast=False):
        """Make dense array of mode inclusion."""
        # Make dense array of local elements
        elements = self.local_elements(domain, scales, rank=rank, broadcast=broadcast)
        elements = np.array(np.meshgrid(*elements, indexing='ij'))
        # Check validity basis-by-basis
        grid_space = self.grid_space
        vshape = tuple(cs.dim for cs in tensorsig) + elements[0].shape
        valid = np.ones(shape=vshape, dtype=bool)
        for basis in domain.bases:
            basis_axes = slice(self.dist.first_axis(basis), self.dist.last_axis(basis)+1)
            valid &= basis.valid_elements(tensorsig, grid_space[basis_axes], elements[basis_axes])
        return valid

    def _group_arrays(self, elements, domain):
        # Convert to groups basis-by-basis
        grid_space = self.grid_space
        groups = np.zeros_like(elements)
        groups = np.ma.masked_array(groups)
        for basis in domain.bases:
            basis_axes = slice(self.dist.first_axis(basis), self.dist.last_axis(basis)+1)
            groups[basis_axes] = basis.elements_to_groups(grid_space[basis_axes], elements[basis_axes])
        return groups

    @CachedMethod
    def local_group_arrays(self, domain, scales, rank=None, broadcast=False):
        """Dense array of local groups (first axis)."""
        # Make dense array of local elements
        elements = self.local_elements(domain, scales, rank=rank, broadcast=broadcast)
        elements = np.array(np.meshgrid(*elements, indexing='ij'))
        return self._group_arrays(elements, domain)

    @CachedMethod
    def global_group_arrays(self, domain, scales):
        """Dense array of local groups (first axis)."""
        # Make dense array of local elements
        elements = self.global_elements(domain, scales)
        elements = np.array(np.meshgrid(*elements, indexing='ij'))
        return self._group_arrays(elements, domain)

    @CachedMethod
    def local_groupsets(self, group_coupling, domain, scales, rank=None, broadcast=False):
        local_groupsets = self.local_group_arrays(domain, scales, rank=rank, broadcast=broadcast).astype(object)
        # Replace non-enumerated axes with None
        for axis in range(local_groupsets.shape[0]):
            if group_coupling[axis]:
                local_groupsets[axis] = None
        # Flatten local groupsets
        local_groupsets = local_groupsets.reshape((local_groupsets.shape[0], -1))
        # Drop masked groups
        local_groupsets = np.ma.compress_cols(local_groupsets)
        # Return unique groupsets
        local_groupsets = tuple(map(tuple, local_groupsets.T))
        return OrderedSet(local_groupsets)

    @CachedMethod
    def local_groupset_slices(self, groupset, domain, scales, rank=None, broadcast=False):
        groups = self.local_group_arrays(domain, scales, rank=rank, broadcast=broadcast)
        dim = groups.shape[0]
        group_shape = self.group_shape(domain)
        # find all elements which match group
        selections = np.ones(groups[0].shape, dtype=int)
        for i, subgroup in enumerate(groupset):
            if subgroup is not None:
                selections *= (subgroup == groups[i])
                # Note: seems to exclude masked elements for ==, unlike other comparisons
        # determine which axes to loop over, which to find bounds for
        slices = []
        for i, subgroup in enumerate(groupset):
            if subgroup is None:
                subslices = [slice(None)]
            else:
                # loop over axis i but taking into account group_shape
                subslices = [slice(j, j+group_shape[i]) for j in range(0, groups.shape[i+1], group_shape[i])]
            slices.append(subslices)
        group_slices = []
        for s in itertools.product(*slices):
            sliced_selections = selections[tuple(s)]
            if np.any(sliced_selections): # some elements match group
                # assume selected groups are cartesian product, find left and right bounds
                lefts = list(map(np.min, np.where(sliced_selections)))
                rights = list(map(np.max, np.where(sliced_selections)))
                # build multidimensional group slice
                group_slice = []
                for i in range(dim):
                    if s[i] != slice(None):
                        group_slice.append(s[i])
                    else:
                        group_slice.append(slice(lefts[i], rights[i]+1))
                group_slices.append(tuple(group_slice))
        return group_slices

    def slices(self, domain, scales):
        """Local element slices by axis."""
        local_elements = self.local_elements(domain, scales)
        slices = []
        for LE in local_elements:
            if LE.size:
                slices.append(slice(LE.min(), LE.max()+1))
            else:
                slices.append(slice(0, 0))
        return tuple(slices)

    @CachedMethod
    def local_shape(self, domain, scales, rank = None):
        """Local data shape."""
        local_elements = self.local_elements(domain, scales, rank = rank)
        shape = tuple(LE.size for LE in local_elements)
        return shape

    def buffer_size(self, bases, scales, dtype):
        """Local buffer size (bytes)."""
        local_shape = self.local_shape(bases, scales)
        return prod(local_shape) * np.dtype(dtype).itemsize

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
        field.preset_layout(self.layout1)
        gdata = field.data
        # Transform non-constant bases with local data
        if (basis is not None) and prod(cdata.shape):
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
        field.preset_layout(self.layout0)
        cdata = field.data
        # Transform non-constant bases with local data
        if (basis is not None) and prod(gdata.shape):
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
        # No comm cart across axes where mesh = 1
        mesh = layout0.dist.mesh
        comm_cart_axis = axis - np.sum(mesh[:axis]==1)
        remain_dims[comm_cart_axis] = 1
        self.comm_sub = comm_cart.Sub(remain_dims)

    @CachedMethod
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
        if prod(sub_shape) == 0:
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
        ncomp = prod([cs.dim for cs in field.tensorsig])
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
                ncomp += prod([cs.dim for cs in field.tensorsig])
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

    def plan_localize_columns(self, plan, data0, data1):
        plan.localize_columns(data0, data1)

    def increment_single(self, field):
        """Backward transpose a field."""
        plan = self._single_plan(field)
        if plan:
            # Reference views from both layouts
            data0 = field.data
            field.preset_layout(self.layout1)
            data1 = field.data
            # Transpose between data views
            self.plan_localize_columns(plan, data0, data1)
        else:
            # No communication: just update field layout
            field.preset_layout(self.layout1)

    def plan_localize_rows(self, plan, data0, data1):
        plan.localize_rows(data1, data0)

    def decrement_single(self, field):
        """Forward transpose a field."""
        plan = self._single_plan(field)
        if plan:
            # Reference views from both layouts
            data1 = field.data
            field.preset_layout(self.layout0)
            data0 = field.data
            # Transpose between data views
            self.plan_localize_rows(plan, data0, data1)
        else:
            # No communication: just update field layout
            field.preset_layout(self.layout0)

    def increment_group(self, fields):
        """Backward transpose multiple fields simultaneously."""
        plans = self._group_plans(fields)
        for fields, plan in plans:
            if plan:
                if len(fields) == 1:
                    field = fields[0]
                    # Reference views from both layouts
                    data0 = field.data
                    field.preset_layout(self.layout1)
                    data1 = field.data
                    # Transpose between data views
                    self.plan_localize_columns(plan, data0, data1)
                else:
                    # Gather data across fields
                    data0 = []
                    data1 = []
                    for field in fields:
                        rank = len(field.tensorsig)
                        # Reference views from both layouts
                        flat_comp_shape = (-1,) + field.data.shape[rank:]
                        if field.data.size:
                            data0.append(field.data.reshape(flat_comp_shape))
                        field.preset_layout(self.layout1)
                        flat_comp_shape = (-1,) + field.data.shape[rank:]
                        if field.data.size:
                            data1.append(field.data.reshape(flat_comp_shape))
                    if data0:
                        data0 = np.concatenate(data0)
                    else:
                        data0 = np.zeros(0, dtype=fields[0].dtype) # Assumes same dtypes
                    if data1:
                        data1 = np.concatenate(data1)
                    else:
                        data1 = np.zeros(0, dtype=fields[0].dtype) # Assumes same dtypes
                    # Transpose between data views
                    self.plan_localize_columns(plan, data0, data1)
                    # Split up transposed data
                    i = 0
                    for field in fields:
                        ncomp = prod([cs.dim for cs in field.tensorsig])
                        data = data1[i:i+ncomp]
                        field.data[:] = data.reshape(field.data.shape)
                        i += ncomp
            else:
                # No communication: just update field layouts
                for field in fields:
                    field.preset_layout(self.layout1)

    def decrement_group(self, fields):
        """Forward transpose multiple fields simultaneously."""
        plans = self._group_plans(fields)
        for fields, plan in plans:
            if plan:
                if len(fields) == 1:
                    field = fields[0]
                    # Reference views from both layouts
                    data1 = field.data
                    field.preset_layout(self.layout0)
                    data0 = field.data
                    # Transpose between data views
                    self.plan_localize_rows(plan, data0, data1)
                else:
                    # Gather data across fields
                    data0 = []
                    data1 = []
                    for field in fields:
                        rank = len(field.tensorsig)
                        # Reference views from both layouts
                        flat_comp_shape = (-1,) + field.data.shape[rank:]
                        if field.data.size:
                            data1.append(field.data.reshape(flat_comp_shape))
                        field.preset_layout(self.layout0)
                        flat_comp_shape = (-1,) + field.data.shape[rank:]
                        if field.data.size:
                            data0.append(field.data.reshape(flat_comp_shape))
                    if data0:
                        data0 = np.concatenate(data0)
                    else:
                        data0 = np.zeros(0, dtype=fields[0].dtype) # Assumes same dtypes
                    if data1:
                        data1 = np.concatenate(data1)
                    else:
                        data1 = np.zeros(0, dtype=fields[0].dtype) # Assumes same dtypes
                    # Transpose between data views
                    self.plan_localize_rows(plan, data0, data1)
                    # Split up transposed data
                    i = 0
                    for field in fields:
                        ncomp = prod([cs.dim for cs in field.tensorsig])
                        data = data0[i:i+ncomp]
                        field.data[:] = data.reshape(field.data.shape)
                        i += ncomp
            else:
                # No communication: just update field layouts
                for field in fields:
                    field.preset_layout(self.layout1)

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
    #             field.preset_layout(self.layout1)
    #             np.copyto(field.data, buffer1[i])
    #     else:
    #         # No data: just update field layouts
    #         for field in fields:
    #             field.preset_layout(self.layout1)

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
    #             field.preset_layout(self.layout0)
    #             np.copyto(field.data, buffer0[i])
    #     else:
    #         # No data: just update field layouts
    #         for field in fields:
    #             field.preset_layout(self.layout0)
