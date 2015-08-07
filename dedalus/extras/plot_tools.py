

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib import transforms

from ..core.field import Field
from ..tools.array import reshape_vector


class FieldWrapper:
    """Class to mimic h5py dataset interface for Dedalus fields."""

    def __init__(self, field):
        self.field = field
        self.attrs = {'name': field.name}
        self.dims = [DimWrapper(field, axis) for axis in range(field.domain.dim)]

    def __getitem__(self, item):
        return self.field.data[item]

    @property
    def shape(self):
        return self.field.data.shape


class DimWrapper:
    """Wrapper class to mimic h5py dimension scales."""

    def __init__(self, field, axis):
        self.field = field
        self.axis = axis
        self.basis = field.domain.bases[axis]

    @property
    def label(self):
        if self.field.layout.grid_space[self.axis]:
            return self.basis.name
        else:
            return self.basis.element_name

    def __getitem__(self, item):
        if self.field.layout.grid_space[self.axis]:
            scale = self.field.meta[self.axis]['scale']
            return self.basis.grid(scale)
        else:
            return self.basis.elements


def plot_bot(dset, image_axes, data_slices, image_scales=(0,0), clim=None, even_scale=False, cmap='RdBu_r', axes=None, figkw={}, title=None, func=None):
    """
    Plot a 2d slice of the grid data of a dset/field.

    Parameters
    ----------
    dset : h5py dset or Dedalus Field object
        Dataset to plot
    image_axes: tuple of ints (xi, yi)
        Data axes to use for image x and y axes
    data_slices: tuple of slices, ints
        Slices selecting image data from global data
    image_scales: tuple of ints or strs (xs, ys)
        Axis scales (default: (0,0))
    clim : tuple of floats, optional
        Colorbar limits (default: (data min, data max))
    even_scale : bool, optional
        Expand colorbar limits to be symmetric around 0 (default: False)
    cmap : str, optional
        Colormap name (default: 'RdBu_r')
    axes : matplotlib.Axes object, optional
        Axes to overplot.  If None (default), a new figure and axes will be created.
    figkw : dict, optional
        Keyword arguments to pass to plt.figure (default: {})
    title : str, optional
        Title for plot (default: dataset name)
    func : function(xmesh, ymesh, data), optional
        Function to apply to selected meshes and data before plotting (default: None)

    """

    # Wrap fields
    if isinstance(dset, Field):
        dset = FieldWrapper(dset)

    # Unpack image axes
    xaxis, yaxis = image_axes
    xscale, yscale = image_scales

    # Get meshes and data
    xmesh, ymesh, data = get_plane(dset, xaxis, yaxis, data_slices, xscale, yscale)
    if func is not None:
        xmesh, ymesh, data = func(xmesh, ymesh, data)

    # Setup figure
    if axes is None:
        fig = plt.figure(**figkw)
        axes = fig.add_subplot(1, 1, 1)

    # Setup axes
    # Bounds (left, bottom, width, height) relative-to-axes
    pbbox = transforms.Bbox.from_bounds(0.03, 0, 0.94, 0.94)
    cbbox = transforms.Bbox.from_bounds(0.03, 0.95, 0.94, 0.05)
    # Convert to relative-to-figure
    to_axes_bbox = transforms.BboxTransformTo(axes.get_position())
    pbbox = pbbox.transformed(to_axes_bbox)
    cbbox = cbbox.transformed(to_axes_bbox)
    # Create new axes and suppress base axes
    paxes = axes.figure.add_axes(pbbox)
    caxes = axes.figure.add_axes(cbbox)
    axes.axis('off')

    # Colormap options
    cmap = matplotlib.cm.get_cmap(cmap)
    cmap.set_bad('0.7')

    # Plot
    plot = paxes.pcolormesh(xmesh, ymesh, data, cmap=cmap, zorder=1)
    paxes.axis(pad_limits(xmesh, ymesh))
    paxes.tick_params(length=0, width=0)
    if clim is None:
        if even_scale:
            lim = max(abs(data.min()), abs(data.max()))
            clim = (-lim, lim)
        else:
            clim = (data.min(), data.max())
    plot.set_clim(*clim)

    # Colorbar
    cbar = plt.colorbar(plot, cax=caxes, orientation='horizontal',
        ticks=ticker.MaxNLocator(nbins=5))
    cbar.outline.set_visible(False)
    caxes.xaxis.set_ticks_position('top')

    # Labels
    if title is None:
        try:
            title = dset.attrs['name']
        except KeyError:
            title = dset.name
    caxes.set_xlabel(title)
    caxes.xaxis.set_label_position('top')
    if isinstance(xscale, str):
        paxes.set_xlabel(xscale)
    else:
        paxes.set_xlabel(dset.dims[xaxis].label)
    if isinstance(yscale, str):
        paxes.set_ylabel(yscale)
    else:
        paxes.set_ylabel(dset.dims[yaxis].label)

    return paxes, caxes


def plot_bot_2d(dset, transpose=False, **kw):
    """
    Plot the grid data of a 2d field.

    Parameters
    ----------
    field : field object
        Field to plot
    transpose : bool, optional
        Flag for transposing plot (default: False)

    Other keyword arguments are passed on to plot_bot.

    """

    # Wrap fields
    if isinstance(dset, Field):
        dset = FieldWrapper(dset)

    # Check dimension
    if len(dset.shape) != 2:
        raise ValueError("This function is for plotting 2d datasets only.")

    # Call general plotting function
    image_axes = (0, 1)
    if transpose:
        image_axes = image_axes[::-1]
    data_slices = (slice(None), slice(None))

    return plot_bot(dset, image_axes, data_slices, **kw)


def plot_bot_3d(dset, normal_axis, normal_index, transpose=False, **kw):
    """
    Plot a 2d slice of the grid data of a 3d field.

    Parameters
    ----------
    field : field object
        Field to plot
    normal_axis: int or str
        Index or name of normal axis
    normal_index: int
        Index along normal direction to plot
    transpose : bool, optional
        Flag for transposing plot (default: False)

    Other keyword arguments are passed on to plot_bot.

    """

    # Wrap fields
    if isinstance(dset, Field):
        dset = FieldWrapper(dset)

    # Check dimension
    if len(dset.shape) != 3:
        raise ValueError("This function is for plotting 3d datasets only.")

    # Resolve axis name to axis index
    if isinstance(normal_axis, str):
        for axis, dim in enumerate(dset.dims):
            if normal_axis == dim.label:
                normal_axis = axis
                break
        else:
            raise ValueError("Axis name not found.")

    # Call general plotting function
    axes = (0, 1, 2)
    image_axes = axes[:normal_axis] + axes[normal_axis+1:]
    if transpose:
        image_axes = image_axes[::-1]
    data_slices = [slice(None), slice(None), slice(None)]
    data_slices[normal_axis] = normal_index

    return plot_bot(dset, image_axes, tuple(data_slices), **kw)


class MultiFigure:
    """
    An array of generic images within a matplotlib figure.

    Parameters
    ----------
    nrows, ncols : int
        Number of image rows/columns.
    image : Box instance
        Box describing the image shape.
    pad : Frame instance
        Frame describing the padding around each image.
    margin : Frame instance
        Frame describing the margin around the array of images.
    scale : float, optional
        Scaling factor to convert from provided box/frame units to figsize.
        Margin will be automatically expanded so that fig dimensions are integers.

    Other keywords passed to plt.figure.

    """

    def __init__(self, nrows, ncols, image, pad, margin, scale=1., **kw):

        # Build composite boxes
        subfig = pad + image
        fig = margin + nrows*subfig.ybox + ncols*subfig.xbox

        # Rectify scaling so fig dimensions are integers
        intscale = np.ceil(scale*fig.y) / fig.y
        extra_w = np.ceil(intscale*fig.x) - intscale*fig.x

        # Apply scale
        image *= intscale
        pad *= intscale
        margin *= intscale
        margin.left += extra_w / 2
        margin.right += extra_w / 2

        # Rebuild composite boxes
        subfig  = pad + image
        fig = margin + nrows*subfig.ybox + ncols*subfig.xbox

        # Build figure
        figx = int(np.rint(fig.x))
        figy = int(np.rint(fig.y))
        self.figure = plt.figure(figsize=(figx, figy), **kw)

        # Attributes
        self.nrows = nrows
        self.ncols = ncols
        self.image = image
        self.pad = pad
        self.margin = margin
        self.fig = fig

    def add_axes(self, i, j, rect, **kw):
        """
        Add axes to a subfigure.

        Parameters
        ----------
        i, j  : int
            Image row/column
        rect : tuple of floats
            (left, bottom, width, height) in fractions of image width and height

        Other keywords passed to Figure.add_axes.

        """

        # Get image offset in figure coordinates
        irev = self.nrows - 1 - i
        subfig = self.pad + self.image
        offset = self.margin.bottom_left + irev*subfig.ybox + j*subfig.xbox + self.pad.bottom_left

        # Convert image rect to figure rect
        imstart = Box(rect[0], rect[1])
        imshape = Box(rect[2], rect[3])
        figstart = (offset + imstart * self.image) / self.fig
        figshape = imshape * self.image / self.fig
        figrect = [figstart.x, figstart.y, figshape.x, figshape.y]

        return self.figure.add_axes(figrect, **kw)


class Box:
    """
    2d-vector-like object for representing image sizes and offsets.

    Parameters
    ----------
    x, y : float
        Box width/height.

    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    @property
    def xbox(self):
        return Box(self.x, 0)

    @property
    def ybox(self):
        return Box(0, self.y)

    def __add__(self, other):
        if isinstance(other, Box):
            return Box(self.x+other.x, self.y+other.y)
        return NotImplemented

    def __radd__(self, other):
        return self.__radd__(other)

    def __mul__(self, other):
        if np.isscalar(other):
            return Box(self.x*other, self.y*other)
        elif isinstance(other, Box):
            return Box(self.x*other.x, self.y*other.y)
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if np.isscalar(other):
            return Box(self.x/other, self.y/other)
        elif isinstance(other, Box):
            return Box(self.x/other.x, self.y/other.y)
        return NotImplemented


class Frame:
    """
    Object for representing a non-uniform frame around an image.

    Parameters
    ----------
    top, bottom, left, right : float
        Frame widths.

    """

    def __init__(self, top, bottom, left, right):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right

    @property
    def bottom_left(self):
        return Box(self.left, self.bottom)

    @property
    def top_right(self):
        return Box(self.right, self.top)

    def __add__(self, other):
        if isinstance(other, Box):
            x = self.left + other.x + self.right
            y = self.bottom + other.y + self.top
            return Box(x, y)
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if np.isscalar(other):
            return Frame(other*self.top,
                         other*self.bottom,
                         other*self.left,
                         other*self.right)
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)


def quad_mesh(x, y, cut_x_edges=False, cut_y_edges=False):
    """
    Construct quadrilateral mesh arrays from two grids.
    Intended for use with e.g. plt.pcolor.

    Parameters
    ----------
    x : 1d array
        Grid for last axis of the mesh.
    y : 1d array
        Grid for first axis of the mesh.
    cut_x_edges, cut_y_edges : bool, optional
        True to truncate edge quadrilaterals at x/y grid edges.
        False (default) to center edge quadrilaterals at x/y grid edges.

    """

    # Get 1d vertex vectors
    xvert = get_1d_vertices(x, cut_edges=cut_x_edges)
    yvert = get_1d_vertices(y, cut_edges=cut_y_edges)
    # Reshape as multidimensional vectors
    xvert = reshape_vector(xvert, dim=2, axis=1)
    yvert = reshape_vector(yvert, dim=2, axis=0)
    # Broadcast up to arrays
    xmesh = xvert * np.ones_like(yvert)
    ymesh = yvert * np.ones_like(xvert)

    return xmesh, ymesh


def get_1d_vertices(grid, cut_edges=False):
    """
    Get vertices dividing a 1d grid.

    Parameters
    ----------
    grid : 1d array
        Grid.
    cut_edges : bool, optional
        True to set edge vertices at grid edges.
        False (default) to center edge segments at grid edges.

    """

    if len(grid.shape) > 1:
        raise ValueError("grid must be 1d array.")
    diff = np.diff(grid)
    vert = np.zeros(grid.size+1)
    # Interior vertices: halfway between points
    vert[1:-1] = grid[0:-1] + diff/2
    # Edge vertices: tight or reflect
    if cut_edges:
        vert[0] = grid[0]
        vert[-1] = grid[-1]
    else:
        vert[0] = grid[0] - diff[0]/2
        vert[-1] = grid[-1] + diff[-1]/2

    return vert


def pad_limits(xgrid, ygrid, xpad=0., ypad=0., square=None):
    """
    Compute padded image limits for x and y grids.

    Parameters
    ----------
    xgrid : array
        Grid for x axis of image.
    ygrid : array
        Grid for y axis of image.
    xpad : float, optional
        Padding fraction for x axis (default: 0.).
    ypad : float, optional
        Padding fraction for y axis (default: 0.).
    square : axis object, optional
        Extend limits to have a square aspect ratio within an axis.

    """

    xmin, xmax = xgrid.min(), xgrid.max()
    ymin, ymax = ygrid.min(), ygrid.max()
    dx = xmax - xmin
    dy = ymax - ymin
    x0 = xmin - xpad*dx
    x1 = xmax + xpad*dx
    y0 = ymin - ypad*dy
    y1 = ymax + ypad*dy

    if square:
        axis = square
        ax_position = axis.get_position()
        ax_height = ax_position.height * axis.figure.get_figheight()
        ax_width  = ax_position.width  * axis.figure.get_figwidth()
        ax_aspect = ax_height / ax_width

        im_height = y1 - y0
        im_width  = x1 - x0
        im_aspect = im_height / im_width

        if (im_height/im_width) > (ax_height/ax_width):
            # Image too tall
            extra_w = im_height/ax_aspect - im_width
            x0 -= extra_w / 2
            x1 += extra_w / 2
        else:
            # Image too wide
            extra_h = im_width*ax_aspect - im_height
            y0 -= extra_h / 2
            y1 += extra_h / 2

    return [x0, x1, y0, y1]


def get_plane(dset, xaxis, yaxis, slices, xscale=0, yscale=0, **kw):
    """
    Select plane from dataset.
    Intended for use with e.g. plt.pcolor.

    Parameters
    ----------
    dset : h5py dataset
        Dataset
    xaxis, yaxis : int
        Axes for plotting
    xscale, yscale : int or str
        Corresponding axis scales
    slices : tuple of ints, slice objects
        Selection object for dataset

    Other keywords passed to quad_mesh

    """

    # Make sure slices are in tuple
    slices = tuple(slices)

    # Build quad meshes from sorted grids
    xgrid = dset.dims[xaxis][xscale][slices[xaxis]]
    ygrid = dset.dims[yaxis][yscale][slices[yaxis]]
    xorder = np.argsort(xgrid)
    yorder = np.argsort(ygrid)
    xmesh, ymesh = quad_mesh(xgrid[xorder], ygrid[yorder], **kw)

    # Select and arrange data
    data = dset[slices]
    if xaxis < yaxis:
        data = data.T
    data = data[yorder]
    data = data[:, xorder]

    return xmesh, ymesh, data





# def visit(filename, main):
#     file = h5py.File(filename, mode='r')
#     try:
#         for i in range(len(file['scales']['sim_time'])):
#             main(file, i)
#     finally:
#         file.close()

# if __name__ == "__main__":

#     if MPI.COMM_WORLD.rank == 0:
#         clean_path(saveroot)
#     MPI.COMM_WORLD.barrier()

#     filename = sys.argv[1]
#     visit(filename, main)


