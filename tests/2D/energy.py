

import numpy as np
import matplotlib.pyplot as plt
import shelve


# Read storage
data = shelve.open('data.db', flag='r')
t = data['t'][1:]
E = data['E'][1:]


plt.figure(1)
plt.semilogy(t, E, '.b')

fit = np.polyfit(t, np.log(E), 1)
print('E growth rate = %f' %fit[0])

plt.semilogy(t, np.exp(fit[0]*t + fit[1]), '--k')

plt.savefig('energy.png')
