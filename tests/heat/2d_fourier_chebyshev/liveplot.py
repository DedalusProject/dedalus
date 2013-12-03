

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import shelve


# Options
fnames = ['y', 'dy']
xstr = 'z'
ystr = 'x'
cmapname = 'Spectral_r'
even_scale = False
units = True
static_scale = False

# Read storage
data = shelve.open('data_0.db', flag='r')
t = data['t']
x = data['z'][0,:]
y = data['x'][:,0]
fields = [data[fn].real for fn in fnames]

# Storage
images = []
image_axes = []
cbar_axes = []

# Determine grid size
nrows = 1
ncols = len(fields)

# Setup spacing [top, bottom, left, right] and [height, width]
t_mar, b_mar, l_mar, r_mar = (0.2, 0.2, 0.2, 0.2)
t_pad, b_pad, l_pad, r_pad = (0.15, 0.03, 0.03, 0.03)
h_cbar, w_cbar = (0.05, 1.)
h_data, w_data = (1., 1.)

h_im = t_pad + h_cbar + h_data + b_pad
w_im = l_pad + w_data + r_pad
h_total = t_mar + nrows * h_im + b_mar
w_total = l_mar + ncols * w_im + r_mar
scale = 4.0

# Create figure and axes
fig = plt.figure(1, figsize=(scale * w_total,
                             scale * h_total))
for j, (fname, field) in enumerate(zip(fnames, fields)):

    cindex = j
    row = 0

    left = (l_mar + w_im * cindex + l_pad) / w_total
    bottom = 1 - (t_mar + h_im * (row + 1) - b_pad) / h_total
    width = w_data / w_total
    height = h_data / h_total
    image_axes.append(fig.add_axes([left, bottom, width, height]))
    image_axes[j].lastrow = (row == nrows - 1)
    image_axes[j].firstcol = (cindex == 0)

    left = (l_mar + w_im * cindex + l_pad) / w_total
    bottom = 1 - (t_mar + h_im * row + t_pad + h_cbar) / h_total
    width = w_cbar / w_total
    height = h_cbar / h_total
    cbar_axes.append(fig.add_axes([left, bottom, width, height]))

# Title
height = 1 - (0.6 * t_mar) / h_total
timestring = fig.suptitle(r'', y=height, size=16)


def create_limits_mesh(x, y):
    xd = np.diff(x)
    yd = np.diff(y)
    shape = x.shape
    xm = np.zeros((y.size+1, x.size+1))
    ym = np.zeros((y.size+1, x.size+1))
    xm[:, 0] = x[0] - xd[0] / 2.
    xm[:, 1:-1] = x[:-1] + xd / 2.
    xm[:, -1] = x[-1] + xd[-1] / 2.
    ym[0, :] = y[0] - yd[0] / 2.
    ym[1:-1, :] = (y[:-1] + yd / 2.)[:, None]
    ym[-1, :] = y[-1] + yd[-1] / 2.

    return xm, ym


def add_image(fig, imax, cbax, x, y, data):

    cmap = matplotlib.cm.get_cmap(cmapname)
    cmap.set_bad('0.7')

    if units:
        xm, ym = create_limits_mesh(x, y)
        im = imax.pcolormesh(xm, ym, data, cmap=cmap, zorder=1)
        plot_extent = [xm.min(), xm.max(), ym.min(), ym.max()]
        imax.axis(plot_extent)
    else:
        im = imax.imshow(data, cmap=cmap, zorder=1, aspect='auto',
            interpolation='nearest', origin='lower')
        shape = data.shape
        plot_extent = [-0.5, shape[1] - 0.5, -0.5, shape[0] - 0.5]
        imax.axis(plot_extent)

    fig.colorbar(im, cax=cbax, orientation='horizontal',
        ticks=MaxNLocator(nbins=5, prune='both'))

    return im


def update_image(im, data):

    if units:
        im.set_array(np.ravel(data))
    else:
        im.set_data(data)

    if not static_scale:
        if even_scale:
            lim = na.max(na.abs([data.min(), data.max()]))
            im.set_clim(-lim, lim)
        else:
            im.set_clim(data.min(), data.max())


def add_labels(imax, cbax, fname):

    # Title
    title = imax.set_title('%s' %fname, size=14)
    title.set_y(1.1)

    # Colorbar
    cbax.xaxis.set_ticks_position('top')
    plt.setp(cbax.get_xticklabels(), size=10)

    if imax.lastrow:
        imax.set_xlabel(xstr, size=12)
        plt.setp(imax.get_xticklabels(), size=10)
    else:
        plt.setp(imax.get_xticklabels(), visible=False)

    if imax.firstcol:
        imax.set_ylabel(ystr, size=12)
        plt.setp(imax.get_yticklabels(), size=10)
    else:
        plt.setp(imax.get_yticklabels(), visible=False)


# Plot images
for j, (fname, field) in enumerate(zip(fnames, fields)):
    imax = image_axes[j]
    cbax = cbar_axes[j]
    images.append(add_image(fig, imax, cbax, x, y, field[0]))
    add_labels(imax, cbax, fname)
for i in range(1, t.size):
    for j, field in enumerate(fields):
        update_image(images[j], field[i])

    # Update time title
    tstr = 't = %6.3f' % t[i]
    timestring.set_text(tstr)

    # Draw
    fig.canvas.draw()

