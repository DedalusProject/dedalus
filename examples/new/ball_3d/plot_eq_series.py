

import h5py
import sys
import os
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpi4py import MPI
import pathlib
import multiprocessing

from dedalus.tools import post
from dedalus.tools.array import reshape_vector
from dedalus.tools import general

# Some settings
nrows = 1
ncols = 1

cmap_name = 'RdBu_r'
dpi = 300

start = 0

dir = sys.argv[1]

start_num = 0

saveroot = 'frames/'

name = 'T_eq'

savename = lambda index: '%s_%06i.png' %(name, index+start_num)

def main(file, index, first, plot_list, timestring, mfig):

    fnames = ['T eq']

    for k, task in enumerate(fnames):
        dset = file['tasks'][task]
        write_num = file['scales/write_number'][index]
        pcolormesh(mfig, k, task, dset, index, first, plot_list, write_num)

    del dset

    # Title
    title = 't = %g' %file['scales/sim_time'][index]
    if first:
      title_height = 1 - 0.5 * mfig.margin.top / mfig.fig.y
      timestring = mfig.figure.suptitle(title, y=title_height)
    else:
      timestring.set_text(title)

    # Save
    path = os.path.join(saveroot, savename(file['scales/write_number'][index]))
    mfig.figure.savefig(path, dpi=dpi)
#    plt.close(mfig.figure)

    return timestring

def pcolormesh(mfig, k, taskname, dset, index, first, plot_list, write_num):

    # Pick data axes for x and y plot axes
    # Note: we use the full time-space data array, so remember that axis 0 is time
    xi, yi = (1, 3)

    # Slices for data.
    # First is time axis, here sliced by the "index" argument.
    # The "xi" and "yi" entries should be "slice(None)",
    # Others (for >2 spatial dimensions) should be an integer.
    datslices = (index, slice(None), 0, slice(None))

    if first:
      # Create axes
      i, j = divmod(k, ncols)
      paxes = mfig.add_axes(i, j, [0., 0., 1., 0.93])
      caxes = mfig.add_axes(i, j, [0., 0.95, 1., 0.05])

    # Get vertices
    y = dset.dims[yi][0][datslices[yi]]
    x = dset.dims[xi][0][datslices[xi]]
    xorder = np.argsort(x)
    yorder = np.argsort(y)
    th = x[xorder]
    r = y[yorder]
    r = np.concatenate(([0], r))
    th = np.concatenate((th, [2*np.pi]))
    xv = r[:,None]*np.cos(th[None,:])
    yv = r[:,None]*np.sin(th[None,:])
#    xv, yv = get_vertex_arrays(x, y)

    # Colormap
    cmap = matplotlib.cm.get_cmap(cmap_name)
    cmap.set_bad('0.7')

    # Slice data
    data = dset[datslices]

    # Transpose if necessary
    if xi < yi:
        data = data.T

    # Sort data
    datsort = data[yorder]
    datsort = datsort[:, xorder]

    if first:
      # Plot
      plot_list.append(paxes.pcolormesh(xv, yv, datsort, cmap=cmap, zorder=1))
      paxes.axis(build_limits(paxes, xv, yv, ypad=0.0, square=False))
      paxes.tick_params(length=0, width=0)

      # Colorbar
      cbar = mfig.figure.colorbar(plot_list[-1], cax=caxes, orientation='horizontal',
          ticks=ticker.MaxNLocator(nbins=5))
      cbar.outline.set_visible(False)
      caxes.xaxis.set_ticks_position('top')

      # Labels
      caxes.set_xlabel(taskname)
      caxes.xaxis.set_label_position('top')
      paxes.set_ylabel(dset.dims[yi].label)
      paxes.set_xlabel(dset.dims[xi].label)

    else:
      #plot_list[k].set_array(np.ravel(datsort[:-1,:-1]))
      plot_list[k].set_array(np.ravel(datsort))

    plot_list[k].set_clim(datsort.min(), datsort.max())

def build_limits(axis, x, y, xpad=0., ypad=0., square=False):

    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    dx = xmax - xmin
    dy = ymax - ymin
    x0 = xmin - xpad*dx
    x1 = xmax + xpad*dx
    y0 = ymin - ypad*dy
    y1 = ymax + ypad*dy
    if square:
        ax_position = axis.get_position()
        ax_aspect = (ax_position.height*axis.figure.get_figheight() /
                     (ax_position.width *axis.figure.get_figwidth()))
        if (y1 - y0) / (x1 - x0) > ax_aspect:
            extra_w = (y1 - y0) / ax_aspect - (x1 - x0)
            x0 -= extra_w / 2
            x1 += extra_w / 2
        else:
            extra_h = ax_aspect * (x1 - x0) - (y1 - y0)
            y0 -= extra_h / 2
            y1 += extra_h / 2
    return [x0, x1, y0, y1]

def clean_path(path):

    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)


def get_vertex_arrays(x, y, trim_edges=False):

    x_vvec = get_vertex_vector(x, 1, trim_edges=trim_edges)
    y_vvec = get_vertex_vector(y, 0, trim_edges=trim_edges)
    x_varr = x_vvec * np.ones_like(y_vvec)
    y_varr = np.ones_like(x_vvec) * y_vvec

    return x_varr, y_varr


def get_vertex_vector(r, axis, trim_edges=False):

    #axis = vector_axis(r)
    vflat = get_vertex_flat(r.ravel(), trim_edges=trim_edges)

    return reshape_vector(vflat, dim=2, axis=axis)


def vector_axis(r):

    extended = [i for (i,s) in enumerate(r.shape) if (s>1)]
    if len(extended) == 0:
        raise ValueError("No extended direction")
    elif len(extended) > 1:
        raise ValueError("Multiple extended directions")

    return extended[0]


def get_vertex_flat(grid, trim_edges=False):

    diff = np.diff(grid)
    vflat = np.zeros(grid.size+1)
    vflat[1:-1] = grid[0:-1] + diff/2
    if trim_edges:
        vflat[0] = grid[0]
        vflat[-1] = grid[-1]
    else:
        vflat[0] = grid[0] - diff[0]/2
        vflat[-1] = grid[-1] + diff[-1]/2

    return vflat


def visit(filename):
    file = h5py.File(filename, mode='r')

    # Layout
    image = Vector(1., 1.)
    pad = Frame(0.2, 0.1, 0.1, 0.1)
    margin = Frame(0.2, 0.1, 0.1, 0.1)
    scale = 3.


    # Create multifigure
    mfig = MultiFigure(nrows, ncols, image, pad, margin, scale)

    first = True
    plot_list = []
    timestring = ''
    try:
        for i in range(start,len(file['scales']['sim_time'])):
            timestring = main(file, i, first, plot_list, timestring, mfig)
            first = False
    finally:
        file.close()

    plt.close(mfig.figure)


class Vector:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    @property
    def vx(self):
        return Vector(self.x, 0)

    @property
    def vy(self):
        return Vector(0, self.y)

    def __add__(self, other):
        if np.isscalar(other):
            return Vector(self.x + other, self.y + other)
        elif isinstance(other, Vector):
            return Vector(self.x + other.x, self.y + other.y)
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if np.isscalar(other):
            return Vector(self.x * other, self.y * other)
        elif isinstance(other, Vector):
            return Vector(self.x * other.x, self.y * other.y)
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if np.isscalar(other):
            return Vector(self.x / other, self.y / other)
        elif isinstance(other, Vector):
            return Vector(self.x / other.x, self.y / other.y)
        return NotImplemented


class Frame:

    def __init__(self, top, bottom, left, right):
        self.bottom_left = Vector(left, bottom)
        self.top_right = Vector(right, top)

    @staticmethod
    def from_vectors(bottom_left, top_right):
        return Frame(top_right.y, bottom_left.y, bottom_left.x, top_right.x)

    @property
    def bottom(self):
        return self.bottom_left.y

    @property
    def left(self):
        return self.bottom_left.x

    @property
    def top(self):
        return self.top_right.y

    @property
    def right(self):
        return self.top_right.x

    def __add__(self, other):
        bottom_left = self.bottom_left + other
        top_right = self.top_right + other
        return Frame.from_vectors(bottom_left, top_right)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        bottom_left = self.bottom_left * other
        top_right = self.top_right * other
        return Frame.from_vectors(bottom_left, top_right)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        bottom_left = self.bottom_left / other
        top_right = self.top_right / other
        return Frame.from_vectors(bottom_left, top_right)


class MultiFigure:

    def __init__(self, nrows, ncols, image, pad, margin, scale=1., **kw):

        # Build composite vectors
        sub = pad.bottom_left + image + pad.top_right
        subs = nrows*sub.vy + ncols*sub.vx
        fig = margin.bottom_left + subs + margin.top_right

        # Rectify scaling so fig dimensions are integers
        newscale = np.ceil(scale*fig.y) / fig.y
        x_extra = np.ceil(newscale*fig.x) - newscale*fig.x
        extra = Vector(x_extra, 0)

        # Apply scale
        image *= newscale
        pad *= newscale
        margin *= newscale
        margin += extra / 2

        # Build composite vectors
        sub = pad.bottom_left + image + pad.top_right
        subs = nrows*sub.vy + ncols*sub.vx
        fig = margin.bottom_left + subs + margin.top_right

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
        self.sub = sub
        self.subs = subs
        self.fig = fig

    def add_axes(self, i, j, rect, **kw):

        # References
        margin = self.margin
        nrows = self.nrows
        sub = self.sub
        pad = self.pad
        image = self.image
        fig = self.fig

        start = margin.bottom_left + (nrows-1-i)*sub.vy + j*sub.vx + pad.bottom_left
        left, bottom, width, height = rect
        axstart = (start + Vector(left, bottom)*image) / fig
        axshape = (Vector(width, height)*image) / fig
        normrect = [axstart.x, axstart.y, axshape.x, axshape.y]
        ax = self.figure.add_axes(normrect, **kw)

        return ax

if __name__ == "__main__":

    if MPI.COMM_WORLD.rank == 0:
        if not os.path.exists('%s/' %saveroot):
            os.mkdir('%s/' %saveroot)
    MPI.COMM_WORLD.barrier()

    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size

    dir = sys.argv[1]

    if len(sys.argv)>2:
      num_start = sys.argv[2]
    else:
      num_start = 1

    if len(sys.argv)>3:
      num_end = sys.argv[3]
    else:
      # find number of files
      base_path = pathlib.Path(sys.argv[1]).absolute()
      folder_paths = base_path.glob('%s_f*' %base_path.stem)
      folder_paths = general.natural_sort(folder_paths)
      num_end = len(folder_paths) + (num_start-1)


    for i in range(int(num_start)+rank,int(num_end)+1,size):
      filename = dir+"/"+dir+"_s"+str(i)+".h5"
      print(filename)
      visit(filename)
#      proc = multiprocessing.Process(target=visit, args=(filename,main))
#      proc.daemon=True
#      proc.start()
#      proc.join()


