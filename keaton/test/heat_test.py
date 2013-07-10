

import numpy as np
import matplotlib.pyplot as plt
import time
import shelve
from fk.public import *


# Set domain
basis = primary.Chebyshev(64, range=[0.,1.])
domain = OneDimensionalDomain(basis)

# Choose PDE and integrator
pde = problems.heat_equation_1d
ts = timesteppers.CNAB3

# Build solver
int = Integrator(pde, domain, ts)

# Initial conditions
x = domain.grid
a = 0.1
y = int.state['y']
dy = int.state['dy']
y['x'] = np.sin(np.pi * 4 * x)
dy['k'] = y.differentiate(0)

# Integration parameters
int.dt = 1e-4
int.sim_stop_time = 0.02
int.wall_stop_time = np.inf
int.stop_iteration = np.inf

# Create storage lists
t_list = [int.time]
y_list = [np.copy(y['x'])]
copy_cadence = 1

# Main loop
start_time = time.time()
while int.ok:

    # Advance
    int.advance()

    # Update storage lists
    if int.iteration % copy_cadence == 0:
        t_list.append(int.time)
        y_list.append(np.copy(y['x']))

    # Print progress
    if int.iteration % copy_cadence == 0:
        print 'Iteration: %i, Time: %e' %(int.iteration, int.time)

# Store final state
if int.iteration % copy_cadence != 0:
    t_list.append(int.time)
    y_list.append(np.copy(y['x']))

end_time = time.time()

# Print statistics
print '-' * 20
print 'Total time:', end_time - start_time
print 'Iterations:', int.iteration
print 'Average timestep:', int.time / int.iteration
print '-' * 20

# Write storage lists
shelf = shelve.open('data.db', flag='n')
shelf['t'] = np.array(t_list)
shelf['x'] = x
shelf['y'] = np.array(y_list)
shelf.close()

# Plot error
computed = y['x'].real
expected = np.sin(np.pi * 4 * x) * np.exp(-int.time * (4 * np.pi)**2)

fig = plt.figure(1)
ax1 = fig.add_subplot(211)
ax1.plot(x, expected, '-k')
ax1.plot(x, computed, 'ob')
ax2 = fig.add_subplot(212)
ax2.plot(x, computed - expected, 'o-')
plt.savefig('error.png', dpi=200)
