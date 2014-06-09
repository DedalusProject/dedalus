

import numpy as np
import matplotlib.pyplot as plt

from dedalus2.tools.array import reshape_vector


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


def get_plane(dset, xaxis, yaxis, slices, **kw):
    """
    Select plane from dataset.
    Intended for use with e.g. plt.pcolor.

    Parameters
    ----------
    dset : h5py dataset
        Dataset
    xaxis, yaxis : int
        Axes for plotting
    slices : tuple of ints, slice objects
        Selection object for dataset

    Other keywords passed to quad_mesh

    """

    # Build quad meshes from sorted grids
    xgrid = dset.dims[xaxis][0][indices[xaxis]]
    ygrid = dset.dims[yaxis][0][indices[yaxis]]
    xorder = np.argsort(xgrid)
    yorder = np.argsort(ygrid)
    xmesh, ymesh = quad_mesh(xgrid[xorder], ygrid[yorder], **kw)

    # Select and arrange data
    data = dset[slices]
    if xi < yi:
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


