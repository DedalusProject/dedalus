import numpy             as np
from . import jacobi

# The defalut configurations for the base Jacobi parameters.
alpha = (-0.5,-0.5)

def quadrature(Nmax,alpha=alpha,**kw):
    return jacobi.quadrature(Nmax,alpha[0],alpha[1],**kw)

def trial_functions(Nmax,z,alpha=alpha):

    init = 1/np.sqrt(jacobi.mass(alpha[0],alpha[1])) + 0*z
    return jacobi.recursion(Nmax,alpha[0],alpha[1],z,init)

def operator(dimension,op,Nmax,k,ell,radii,pad=0,alpha=alpha):
    # Pad the matrices by a safe amount before outputting the correct size.
    
    if radii[1] <= radii[0]: raise ValueError('Inner radius must be greater than outer radius.')
    
    gapwidth    =  radii[1] - radii[0]
    aspectratio = (radii[1] + radii[0])/gapwidth
    
    a, b, N = k+alpha[0], k+alpha[1], Nmax + pad
    
    # zeros
    if (op == '0'):  return jacobi.operator('0',N,a,b)
    
    # identity
    if (op == 'I'):  return jacobi.operator('I',N,a,b)
    
    Z =  aspectratio*jacobi.operator('I',N+2,a,b)
    Z += jacobi.operator('J',N+2,a,b)
    
    # r multiplication
    if op == 'R': return (gapwidth/2)*Z[:N+1,:N+1]
    
    E = jacobi.operator('A+',N+2,a,b+1) @ jacobi.operator('B+',N+2,a,b)
    
    # conversion
    if op == 'E': return 0.5*(E @ Z)[:N+1,:N+1]
    
    D = jacobi.operator('D+',N+2,a,b) @ Z
    
    # derivatives
    if op == 'D+': return (D - (ell+k+1          )*E)[:N+1,:N+1]/gapwidth
    if op == 'D-': return (D + (ell-k+dimension-3)*E)[:N+1,:N+1]/gapwidth
    
    # restriction
    if op == 'r=Ri': return jacobi.operator('z=-1',N,a,b)
    if op == 'r=Ro': return jacobi.operator('z=+1',N,a,b)

