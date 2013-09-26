

import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import shelve
import time


# Read storage
data = shelve.open('data.db', flag='r')
t = data['t']
x = data['x']
u = data['u'].real
v = data['v'].real

amp = u*u + v*v
phase = np.arctan2(v, u)
amp=u
phase=v

# Plot
fig = plt.figure(1, figsize=(16, 8))
fig.clear()
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

line1 = ax1.plot(x, amp[0], 'ob')
line2 = ax2.plot(x, phase[0], 'ob')
title = ax1.set_title('t = %e' %t[0])

ax1.set_xlim(x.min(), x.max())
ax2.set_xlim(x.min(), x.max())
#ax2.set_ylim(-np.pi, np.pi)

for i in range(1, t.size):
    line1[0].set_ydata(amp[i])
    line2[0].set_ydata(phase[i])

    ax1.set_ylim(amp[i].min(), amp[i].max())
    ax2.set_ylim(0., phase[i].max())

    title.set_text('t = %e' %t[i])
    fig.canvas.draw()
    time.sleep(0.01)

