

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
data = shelve.open('data.db', flag='r')
t = data['t']
E = data['E']


plt.figure(1)
plt.semilogy(t, E, '.b')

fit = np.polyfit(t, np.log(E), 1)
print('-k_sq = ', fit[0]/2., ' = ', '-pi^2 * %f' %(-fit[0]/2./np.pi**2))

