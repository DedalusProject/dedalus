"""
Plot KE trace.

Usage:
    plot_KE.py <file>

"""


import numpy as np
import matplotlib.pyplot as plt
import h5py
import pathlib
from docopt import docopt

args = docopt(__doc__)

with h5py.File(args['<file>'], mode='r') as file:
    t = np.array(file['scales/sim_time'])
    KE = np.array(file['tasks/KE'][:,0,0])

plt.plot(t, KE)
plt.xlabel(r'$t$')
plt.ylabel(r'$KE$')
plt.yscale('log')

plt.savefig('KE.png', dpi=300)

