

save = True


import os
import shutil
import numpy as np
import matplotlib
if save:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import shelve
import time


start_time = time.time()

proc = 1

# Options
fnames = ['p', 'h', 'u', 'w']
xstr = 'x'
ystr = 'z'
cmapname = 'Spectral_r'
even_scale = True
units = True
static_scale = False
dpi = 100
figi = 15

# Read storage
data = shelve.open('data_%i.db' %proc, flag='r')
t = data['time']
y = data['z'][0]
x = data['x'][:,0]
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
wh_max = max(h_total, w_total)

# Create figure and axes
fig = plt.figure(1, figsize=(figi * w_total / wh_max,
                             figi * h_total / wh_max))
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
            lim = np.max(np.abs([data.min(), data.max()]))
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
    images.append(add_image(fig, imax, cbax, x, y, field[0].T))
    add_labels(imax, cbax, fname)

path = 'frames_%i/' %proc

if save:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)
    fig.savefig(path+'frame_%06i.png' %0, dpi=dpi)
else:
    fig.savefig('start.png')

for i in range(1, t.size):
    for j, field in enumerate(fields):
        update_image(images[j], field[i].T)

    # Update time title
    tstr = 't = %6.3f' % t[i]
    timestring.set_text(tstr)

    if save:
        fig.savefig(path+'frame_%06i.png' %i, dpi=dpi)
    else:
        fig.canvas.draw()

if not save:
    fig.savefig('end.png')

end_time = time.time()
print('Elapsed time = %f' %(end_time - start_time))
