

import numpy as np
import matplotlib.pyplot as plt
import shelve


# Read storage
data = shelve.open('data.db', flag='r')
n = -1
t = data['t'][:n]
x = data['x']
u = data['u'][:n].real
v = data['v'][:n].real

amp = u*u + v*v
phase = np.arctan2(v, u)

# Amp/phase plot
fig = plt.figure(1, figsize=(14, 8))
fig.clear()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

px = np.zeros_like(amp)
py = np.zeros_like(amp)
px[:] = x
py.T[:] = t
pc1 = amp
pc2 = phase
clim = np.max(np.abs(pc1))
cmap = 'Spectral_r'
size = 5
sc1 = ax1.scatter(px, py, c=pc1, lw=0, cmap=cmap, s=size)
sc2 = ax2.scatter(px, py, c=pc2, lw=0, cmap=cmap, s=size)
sc1.set_clim(-clim, clim)
sc2.set_clim(-np.pi, np.pi)
fig.colorbar(sc1, fraction=0.1, pad=0.02, aspect=40, ax=ax1)
fig.colorbar(sc2, fraction=0.1, pad=0.02, aspect=40, ax=ax2)
ax1.set_xlim(x.min(), x.max())
ax1.set_ylim(t.min(), t.max())
ax2.set_xlim(x.min(), x.max())
ax2.set_ylim(t.min(), t.max())
ax1.set_title(r'$|A|^2$')
ax2.set_title(r'$\mathrm{arg}(A)$')
ax1.set_xlabel(r'$z$')
ax2.set_xlabel(r'$z$')
ax1.set_ylabel(r'$t$')
ax2.set_ylabel(r'$t$')
fig.savefig('amp_phase.png', dpi=200)

# Plot
fig = plt.figure(1, figsize=(14, 8))
fig.clear()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

px = np.zeros_like(u)
py = np.zeros_like(u)
px[:] = x
py.T[:] = t
pc1 = u
pc2 = v
clim1 = np.max(np.abs(pc1))
clim2 = np.max(np.abs(pc2))
cmap = 'Spectral_r'
size = 5
sc1 = ax1.scatter(px, py, c=pc1, lw=0, cmap=cmap, s=size)
sc2 = ax2.scatter(px, py, c=pc2, lw=0, cmap=cmap, s=size)
sc1.set_clim(-clim1, clim1)
sc2.set_clim(-clim2, clim2)
fig.colorbar(sc1, fraction=0.1, pad=0.02, aspect=40, ax=ax1)
fig.colorbar(sc2, fraction=0.1, pad=0.02, aspect=40, ax=ax2)
ax1.set_xlim(x.min(), x.max())
ax1.set_ylim(t.min(), t.max())
ax2.set_xlim(x.min(), x.max())
ax2.set_ylim(t.min(), t.max())
ax1.set_title(r'$\mathrm{Re}(A)$')
ax2.set_title(r'$\mathrm{Im}(A)$')
ax1.set_xlabel(r'$z$')
ax2.set_xlabel(r'$z$')
ax1.set_ylabel(r'$t$')
ax2.set_ylabel(r'$t$')
fig.savefig('real_imag.png', dpi=200)

