

import numpy as np
import matplotlib.pyplot as plt
import shelve


# Read storage
data = shelve.open('data.db', flag='r')
t = data['t']
x = data['x']
y = data['y'].real

# Plot
fig = plt.figure(1, figsize=(6, 10))
fig.clear()
ax = fig.add_subplot(1, 1, 1)

px = np.zeros_like(y)
py = np.zeros_like(y)
px[:] = x
py.T[:] = t
pc = y
clim = np.max(np.abs(pc))

sc = ax.scatter(px, py, c=pc, lw=0, cmap='Spectral_r')
sc.set_clim(-clim, clim)
fig.colorbar(sc, fraction=0.1, pad=0.02, aspect=40)
ax.set_xlim(x.min(), x.max())
ax.set_ylim(t.min(), t.max())
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$t$')
fig.savefig('time_space.png', dpi=200)

