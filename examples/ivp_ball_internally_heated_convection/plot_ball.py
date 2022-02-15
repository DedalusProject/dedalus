"""
Plot cutaway ball outputs.

Usage:
    plot_ball.py <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./frames]

"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def build_s2_vertices(phi, theta):
    phi = phi.ravel()
    phi_vert = np.concatenate([phi, [2*np.pi]])
    phi_vert -= phi_vert[1] / 2
    theta = theta.ravel()
    theta_mid = (theta[:-1] + theta[1:]) / 2
    theta_vert = np.concatenate([[np.pi], theta_mid, [0]])
    return phi_vert, theta_vert


def build_spherical_vertices(phi, theta, r, Ri, Ro):
    phi_vert, theta_vert = build_s2_vertices(phi, theta)
    r = r.ravel()
    r_mid = (r[:-1] + r[1:]) / 2
    r_vert = np.concatenate([[Ri], r_mid, [Ro]])
    return phi_vert, theta_vert, r_vert


def spherical_to_cartesian(phi, theta, r):
    phi, theta, r = np.meshgrid(phi, theta, r, indexing='ij')
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])



def main(filename, start, count, output):
    """Save plot of specified tasks for given range of analysis writes."""
    # Plot settings
    # r: outer radial face
    # s: starting meridional face
    # e: ending meridional face
    task_r = 'T(r=1)'
    task_s = 'T(phi=0)'
    task_e = 'T(phi=3/2*pi)'
    cmap_r = plt.cm.magma
    cmap_phi = plt.cm.magma
    clim_r = (-3, 3) # std
    clim_phi = (-3, 3) # std
    R = 1
    phis = 0
    phie = 3*np.pi/2
    dpi = 100
    figsize = (8, 8)
    savename_func = lambda write: 'write_{:06}.png'.format(write)
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    # Plot writes
    with h5py.File(filename, mode='r') as file:
        dset_o = file['tasks'][task_r]
        dset_s = file['tasks'][task_s]
        dset_e = file['tasks'][task_e]
        phi = dset_o.dims[1][0][:].ravel()
        theta = dset_o.dims[2][0][:].ravel()
        r = dset_s.dims[3][0][:].ravel()
        phi_vert, theta_vert, r_vert = build_spherical_vertices(phi, theta, r, 0, R)
        phi_vert_pick = (phis < phi_vert) * (phi_vert < phie)
        phi_vert = np.concatenate([[phis], phi_vert[phi_vert_pick], [phie]], axis=0)
        xo, yo, zo = spherical_to_cartesian(phi_vert, theta_vert, [R])[:, :, :, 0]
        xs, ys, zs = spherical_to_cartesian([phis], theta_vert, r_vert)[:, 0, :, :]
        xe, ye, ze = spherical_to_cartesian([phie], theta_vert, r_vert)[:, 0, :, :]
        norm_r = matplotlib.colors.Normalize(*clim_r)
        norm_phi = matplotlib.colors.Normalize(*clim_phi)
        for index in range(start, start+count):
            dphi = phi[1] - phi[0]
            phi_pick = (phis-dphi/2 < phi) * (phi < phie+dphi/2)
            data_o = dset_o[index, :, :, 0][phi_pick]
            data_s = dset_s[index, 0, :, :]
            data_e = dset_e[index, 0, :, :]
            mean = np.mean(np.concatenate([data_s, data_e], axis=0))
            std = np.std(np.concatenate([data_s, data_e], axis=0))
            fc_o = cmap_r(norm_r((data_o - mean) / std))
            fc_s = cmap_phi(norm_phi((data_s - mean) / std))
            fc_e = cmap_phi(norm_phi((data_e - mean) / std))
            fc_p = np.concatenate([fc_e.transpose((1,0,2))[::-1], fc_s.transpose((1,0,2))], axis=0)
            if index == start:
                xp = np.concatenate([xe.T[::-1], xs.T[1:]], axis=0)
                yp = np.concatenate([ye.T[::-1], ys.T[1:]], axis=0)
                zp = np.concatenate([ze.T[::-1], zs.T[1:]], axis=0)
                surf_r = ax.plot_surface(xo, yo, zo, facecolors=fc_o, cstride=1, rstride=1, linewidth=0, antialiased=False, shade=False)
                surf_p = ax.plot_surface(xp, yp, zp, facecolors=fc_p, cstride=1, rstride=1, linewidth=0, antialiased=False, shade=False)
                ax.set_box_aspect((1,1,1))
                ax.set_xlim(-0.7*R, 0.7*R)
                ax.set_ylim(-0.7*R, 0.7*R)
                ax.set_zlim(-0.7*R, 0.7*R)
                ax.axis('off')
                ax.view_init(azim=-75)
            else:
                surf_r.set_facecolors(fc_o.reshape(fc_o.size//4, 4))
                surf_p.set_facecolors(fc_p.reshape(fc_p.size//4, 4))
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

