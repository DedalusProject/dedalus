import numpy as np
import shelve
import pylab as P

# Read storage
data = shelve.open('data.db', flag='r')
t = data['t']
energy = data['energy']

log_en = np.log(energy)
K, A_log = np.polyfit(t, log_en, 1)

print("decay coefficent = %10.5e" % K)
k2 = 8*np.pi**2
nu = -0.5*K/k2
print("nu should be = %10.5e" % nu)
P.semilogy(t,energy,'ko',label='data')
P.semilogy(t,np.exp(A_log)*np.exp(K*t), label='model')
P.legend(loc='upper left').draw_frame(False)
P.xlabel("$t$")
P.ylabel(r"$|u|^2$")
P.savefig('energy_vs_t.png')

