"""
Plot T on equatorial plane from joint analysis files.

Usage:
    plot_eq_series.py <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./frames]

"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import ticker
import matplotlib.pyplot as plt

from dedalus.extras import plot_tools

def build_d2_coord_vertices(phi, r):
    phi = phi.ravel()
    phi_vert = np.concatenate([phi, [2*np.pi]])
    phi_vert -= phi_vert[1] / 2
    r = r.ravel()
    r_mid = (r[:-1] + r[1:]) / 2
    r_vert = np.concatenate([[0], r_mid, [1]])
    return np.meshgrid(phi_vert, r_vert, indexing='ij')

def main(filename, start, count, output):
    """Save plot of specified tasks for given range of analysis writes."""
    # Plot settings
    task = 'T eq'
    cmap = plt.cm.RdBu_r
    savename_func = lambda write: 'T_eq_{:06}.png'.format(write)
    title_func = lambda sim_time: 't = {:.3f}'.format(sim_time)
    names = [r'$T(\theta=\pi)$']

    # Create figure
    nrows, ncols = 1, 1
    image = plot_tools.Box(1, 1)
    pad = plot_tools.Frame(0.1, 0.1, 0.1, 0.1)
    margin = plot_tools.Frame(0.3, 0.2, 0.1, 0.1)
    scale = 2.5
    dpi = 200

    mfig = plot_tools.MultiFigure(nrows, ncols, image, pad, margin, scale)
    fig = mfig.figure

    paxes = []
    cbaxes = []
    for n in range(ncols*nrows):
        i, j = divmod(n, ncols)
        paxes.append(mfig.add_axes(i, j, [0.03, 0, 0.94, 0.94]))
        cbaxes.append(mfig.add_axes(i, j, [0.03, 0.95, 0.94, 0.05]))

    # Plot writes
    with h5py.File(filename, mode='r') as file:

        title_text = title_func(file['scales/sim_time'][start])
        title_height = 1 - 0.5 * mfig.margin.top / mfig.fig.y
        title = fig.suptitle(title_text, x=0.48, y=title_height, ha='left')

        dset = file['tasks'][task]
        phi = dset.dims[1][0][:].ravel()
        r = dset.dims[3][0][:].ravel()
        phi_vert, r_vert = build_d2_coord_vertices(phi, r)
        x = np.cos(phi_vert) * r_vert
        y = np.sin(phi_vert) * r_vert
        pcms = []
        cbs = []
        for index in range(start, start+count):
            data_slices = (index, slice(None), 0, slice(None))
            data = dset[data_slices]
            for i in range(1):
                if index == start:
                    pcms.append( paxes[i].pcolormesh(x, y, data, cmap=cmap) )
                    cbs.append( plt.colorbar(pcms[i], cax=cbaxes[i], orientation='horizontal',
                                             ticks=ticker.MaxNLocator(nbins=5)) )
                    cbs[i].outline.set_visible(False)
                    cbaxes[i].xaxis.set_ticks_position('top')
                    cbaxes[i].xaxis.set_label_position('top')
                    cbaxes[i].set_xlabel(names[i])
                    paxes[i].set_ylabel(r'$y$')
                    paxes[i].set_xlabel(r'$x$')
                else:
                    pcms[i].set_array(np.ravel(data))
                    pcms[i].set_clim(data.min(), data.max())
                    cbaxes[i].xaxis.set_ticks_position('top')
                    cbaxes[i].xaxis.set_label_position('top')

            title_text = title_func(file['scales/sim_time'][index])
            title.set_text(title_text)

            # Save figure
            savename = savename_func(file['scales/write_number'][index])
            savepath = output.joinpath(savename)
            fig.savefig(str(savepath), dpi=dpi)
    plt.close(fig)

if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync

    args = docopt(__doc__)

    output_path = pathlib.Path(args['--output']).absolute()
    # Create output directory if needed
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()
    post.visit_writes(args['<files>'], main, output=output_path)

