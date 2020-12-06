import numpy as np
from . import jacobi  as Jacobi
from .operators    import Operator, Codomain, infinite_csr


# The defalut configuration for the base Jacobi parameter.
alpha = (-1/2,-1/2)

def operator(dimension,radii,name,alpha=alpha):
    """
    Shell.operator function.

    Parameters
    ----------

    """

    width  =  radii[1] - radii[0]

    if name == 'Z':
        def Z(n, k):
            return Jacobi.operator('Z')(n, k+alpha[0], k+alpha[1])
        return Operator(Z, ShellCodomain(0, 0))

    Z = (radii[1] + radii[0])/width + Jacobi.operator('Z')

    if name == 'Id':
        def I(n,k):
            return Jacobi.operator('Id')(n,k+alpha[0],k+alpha[1])
        return Operator(I,ShellCodomain(0,0))

    if name == 'R':
        def R(n,k):
            return (0.5*width)*Z(n,k+alpha[0],k+alpha[1])
        return Operator(R,ShellCodomain(1,0))

    AB = Jacobi.operator('A')(+1) @ Jacobi.operator('B')(+1)

    if name == 'AB':
        def AB_(n, k):
            return AB(n, k+alpha[0], k+alpha[1])
        return Operator(AB_, ShellCodomain(0, 1))

    if name == 'E':
        def E(n,k):
            return 0.5 * (AB @ Z)(n,k+alpha[0],k+alpha[1])
        return Operator(E,ShellCodomain(1,1))

    if name == 'D':

        def D(dl,l):

            def D(n,k):

                D = Jacobi.operator('D')(+1) @ Z

                K = Jacobi.operator('A')(0) - alpha[0]

                K += dl*l + (dl == -1)*(2-dimension)

                D = ( D - K @ AB )/width

                return D(n,k+alpha[0],k+alpha[1])

            return Operator(D,ShellCodomain(0,1))

        return D


class ShellCodomain(Codomain):

    def __init__(self,dn=0,dk=0):
        Codomain.__init__(self,dn,dk,Output=ShellCodomain)

    def __str__(self):
        s = f'(n->n+{self[0]},k->k+{self[1]})'
        return s.replace('+0','').replace('+-','-')

    def __eq__(self,other):
        return self[1] == other[1]

    def __or__(self,other):
        if self != other:
            raise TypeError('operators have incompatible codomains.')
        if self[0] >= other[0]:
            return self
        return other
