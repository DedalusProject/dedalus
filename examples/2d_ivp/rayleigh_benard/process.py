"""
Plot planes from joint analysis files.

Usage:
    plot.py join <base_path>
    plot.py plot <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./frames]

"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from mpi4py import MPI
MPI_RANK = MPI.COMM_WORLD.rank

from dedalus2.extras import plot_tools


even_scale = True

def main(filename, start, count, output):

    # Layout
    nrows, ncols = 4, 1
    image = plot_tools.Box(4, 1.1)
    pad = plot_tools.Frame(0.2, 0.1, 0.1, 0.1)
    margin = plot_tools.Frame(0.2, 0.1, 0.1, 0.1)
    scale = 3.
    # Plot settings
    dpi = 100
    # Create multifigure
    mfig = plot_tools.MultiFigure(nrows, ncols, image, pad, margin, scale)

    with h5py.File(filename, mode='r') as file:
        for index in range(start, start+count):
            print(MPI_RANK, filename, start, index, start+count)
            # Plot datasets
            #taskkey = lambda taskname: write[taskname].attrs['task_number']
            #for k, taskname in enumerate(sorted(write, key=taskkey)):
            for k, task in enumerate(file['tasks']):
                dset = file['tasks'][task]
                pcolormesh(mfig, k, task, dset, index)
            # Title
            title = 't = %g' %file['scales/sim_time'][index]
            title_height = 1 - 0.5 * mfig.margin.top / mfig.fig.y
            mfig.figure.suptitle(title, y=title_height)
            # Save
            write = file['scales']['write_number'][index]
            savename = lambda index: 'write_%06i.png' %write
            fig_path = output.joinpath(savename(write))
            mfig.figure.savefig(str(fig_path), dpi=dpi)
            mfig.figure.clear()
    plt.close(mfig.figure)


def pcolormesh(mfig, k, taskname, dset, index):

    # Pick data axes for x and y plot axes
    # Note: we use the full time-space data array, so remember that axis 0 is time
    xi, yi = (1, 2)

    # Slices for data.
    # First is time axis, here sliced by the "index" argument.
    # The "xi" and "yi" entries should be "slice(None)",
    # Others (for >2 spatial dimensions) should be an integer.
    datslices = (index, slice(None), slice(None))

    # Create axes
    i, j = divmod(k, mfig.ncols)
    paxes = mfig.add_axes(i, j, [0., 0., 1., 0.91])
    caxes = mfig.add_axes(i, j, [0., 0.93, 1., 0.05])

    # Get vertices
    xmesh, ymesh, data = plot_tools.get_plane(dset, xi, yi, datslices)

    # Colormap
    cmap = matplotlib.cm.get_cmap('RdBu_r')
    cmap.set_bad('0.7')

    # Plot
    plot = paxes.pcolormesh(xmesh, ymesh, data, cmap=cmap, zorder=1)
    paxes.axis(plot_tools.pad_limits(xmesh, ymesh, ypad=0.0, square=False))
    paxes.tick_params(length=0, width=0)
    if even_scale:
        lim = max(abs(data.min()), abs(data.max()))
        plot.set_clim(-lim, lim)

    # Colorbar
    cbar = mfig.figure.colorbar(plot, cax=caxes, orientation='horizontal',
        ticks=ticker.MaxNLocator(nbins=5))
    cbar.outline.set_visible(False)
    caxes.xaxis.set_ticks_position('top')

    # Labels
    caxes.set_xlabel(taskname)
    caxes.xaxis.set_label_position('top')
    paxes.set_ylabel(dset.dims[yi].label)
    paxes.set_xlabel(dset.dims[xi].label)


if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus2_burns.tools import logging
    from dedalus2_burns.tools import post
    from dedalus2_burns.tools.parallel import Sync

    args = docopt(__doc__)

    if args['join']:
        post.merge_analysis(args['<base_path>'])
    elif args['plot']:
        output_path = pathlib.Path(args['--output']).absolute()
        # Create output directory if needed
        with Sync() as sync:
            if sync.comm.rank == 0:
                if not output_path.exists():
                    output_path.mkdir()
        post.visit(args['<files>'], main, output=output_path)

