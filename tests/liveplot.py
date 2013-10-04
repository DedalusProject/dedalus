

import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import shelve
import time


# Read storage
data = shelve.open('data.db', flag='r')
t = data['t']
x = data['x']
y = data['y'].real

# Plot
fig = plt.figure(1, figsize=(18, 4))
fig.clear()
ax = fig.add_subplot(1, 1, 1)

line = ax.plot(x, y[0], 'ob')
title = ax.set_title('t = %e' %t[0])

ax.set_xlim(x.min(), x.max())
ax.set_ylim(y.min(), y.max())

ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')

for i in range(1, t.size):
    line[0].set_ydata(y[i])
    title.set_text('t = %e' %t[i])
    #ax.set_ylim(y[i].min(), y[i].max())
    fig.canvas.draw()
    time.sleep(0.01)

