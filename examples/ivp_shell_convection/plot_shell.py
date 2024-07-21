"""
Plot cutaway spherical shell outputs.

Usage:
    plot_shell.py <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./frames]

"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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
    task = 'flux'
    Ri, Ro = 14, 15
    phis = 0
    phie = 3*np.pi/2
    cmap = plt.cm.magma
    clim = (-0.05, 0.05)
    inner_fade = 0.75
    dpi = 100
    figsize = (8, 8)
    savename_func = lambda write: 'write_{:06}.png'.format(write)
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    # Plot writes
    with h5py.File(filename, mode='r') as file:
        dset_i = file['tasks'][task+'_r_inner']
        dset_o = file['tasks'][task+'_r_outer']
        dset_s = file['tasks'][task+'_phi_start']
        dset_e = file['tasks'][task+'_phi_end']
        phi = dset_i.dims[1][0][:].ravel()
        theta = dset_i.dims[2][0][:].ravel()
        r = dset_s.dims[3][0][:].ravel()
        phi_vert, theta_vert, r_vert = build_spherical_vertices(phi, theta, r, Ri, Ro)
        phi_vert_pick = (phis < phi_vert) * (phi_vert < phie)
        phi_vert = np.concatenate([[phis], phi_vert[phi_vert_pick], [phie]], axis=0)
        xi, yi, zi = spherical_to_cartesian(phi_vert, theta_vert, [Ri])[:, :, :, 0]
        xo, yo, zo = spherical_to_cartesian(phi_vert, theta_vert, [Ro])[:, :, :, 0]
        xs, ys, zs = spherical_to_cartesian([phis], theta_vert, r_vert)[:, 0, :, :]
        xe, ye, ze = spherical_to_cartesian([phie], theta_vert, r_vert)[:, 0, :, :]
        norm = matplotlib.colors.Normalize(*clim)
        for index in range(start, start+count):
            dphi = phi[1] - phi[0]
            phi_pick = (phis-dphi/2 < phi) * (phi < phie+dphi/2)
            fc_i = cmap(norm(dset_i[index, :, :, 0][phi_pick]))
            fc_o = cmap(norm(dset_o[index, :, :, 0][phi_pick]))
            fc_s = cmap(norm(dset_s[index, 0, :, :]))
            fc_e = cmap(norm(dset_e[index, 0, :, :]))
            fc_i[:, :, :3] *= inner_fade
            fc = np.concatenate([fc_o, fc_e.transpose((1,0,2))[::-1], fc_i[::-1], fc_s.transpose((1,0,2))], axis=0)
            if index == start:
                X = np.concatenate([xo, xe.T[1:-1][::-1], xi[::-1], xs.T[1:]], axis=0)
                Y = np.concatenate([yo, ye.T[1:-1][::-1], yi[::-1], ys.T[1:]], axis=0)
                Z = np.concatenate([zo, ze.T[1:-1][::-1], zi[::-1], zs.T[1:]], axis=0)
                surf = ax.plot_surface(X, Y, Z, facecolors=fc, cstride=1, rstride=1, linewidth=0, antialiased=False, shade=False, zorder=3)
                # surf_i = ax.plot_surface(Ri*x, Ri*y, Ri*z, facecolors=fc_i, cstride=1, rstride=1, linewidth=0, antialiased=False, shade=False, zorder=3)
                # surf_o = ax.plot_surface(Ro*x, Ro*y, Ro*z, facecolors=fc_o, cstride=1, rstride=1, linewidth=0, antialiased=False, shade=False, zorder=3)
                ax.set_box_aspect((1,1,1))
                ax.set_xlim(-0.7*Ro, 0.7*Ro)
                ax.set_ylim(-0.7*Ro, 0.7*Ro)
                ax.set_zlim(-0.7*Ro, 0.7*Ro)
                ax.axis('off')
                ax.view_init(azim=-75)
            else:
                surf.set_facecolors(fc.reshape(fc.size//4, 4))
                # surf_i.set_facecolors(fc_i.reshape(fc_i.size//4, 4))
                # surf_o.set_facecolors(fc_o.reshape(fc_o.size//4, 4))
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

