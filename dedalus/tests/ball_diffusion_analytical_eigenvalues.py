import numpy as np
from scipy.special import spherical_jn as j

def dispersion_zeros(ell,n,a=0,r0=1,guess=None,imax=20,nk=10,eps=0.1):

    def F(k,deriv=False):
        return j(ell,k*r0,derivative=deriv) - a*j(ell+2,k*r0,derivative=deriv)

    if guess == None:
        kmax = np.pi*(n+ell/2 + eps)/r0
        k = np.linspace(0,kmax,int(kmax*nk))
        S = np.sign(F(k))
        i = np.where(np.abs(np.roll(S,-1)-S)==2)[0]
        k = 0.5*(k[i]+k[i+1])
    else:
        k = guess

    for i in range(imax):
        dk =  F(k)/F(k,deriv=True)
        k -= dk

    print('dk =',np.max(np.abs(dk)))

    return k


def wavenumbers(ell,n,BC, **kwargs):

    k = {'toroidal':0,'poloidal':0}

    if BC=='Bessel':
        k = dispersion_zeros(ell,n,**kwargs)
    elif BC=="no-slip":
        k['toroidal'] = dispersion_zeros(ell,n,**kwargs)
        k['poloidal'] = dispersion_zeros(ell+1,n,**kwargs)
    elif BC=="stress-free":
        if ell == 1:
            k['toroidal'] = dispersion_zeros(2,n,**kwargs)
        else:
            k['toroidal'] = dispersion_zeros(ell-1,n,a=(ell+2)/(ell-1),**kwargs)
        k['poloidal'] = dispersion_zeros(ell,n,a=2/(2*ell+1),**kwargs)
    elif BC=="potential":
        k['toroidal'] = dispersion_zeros(ell-1,n,**kwargs)
        k['poloidal'] = dispersion_zeros(ell,n,**kwargs)
    elif BC=="conducting":
        k['toroidal'] = dispersion_zeros(ell,n,**kwargs)
        k['poloidal'] = dispersion_zeros(ell-1,n,a=ell/(ell+1),**kwargs)
    elif BC=="pseudo":
        k['toroidal'] = dispersion_zeros(ell-1,n,a=ell/(ell+1),**kwargs)
        k['poloidal'] = dispersion_zeros(ell,n,**kwargs)
    else:
        raise ValueError("BC='{}' is not a valid choice".format(BC))

    return k

def eigenvalues(ell,n,BC, **kwargs):
    k = wavenumbers(ell,n,BC, **kwargs)
    k = np.sort(np.concatenate((k['toroidal'],k['poloidal'])))
    return k**2
