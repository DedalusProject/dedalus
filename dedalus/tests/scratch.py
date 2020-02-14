
import numpy as np
from mpi4py import MPI
from memory_profiler import profile

from dedalus_burns.core import distributor
from dedalus_burns.core import spaces
from dedalus_burns.core import domain


@profile
def main():

    dist = distributor.Distributor(dim=2)

    s0 = spaces.FiniteInterval(a=1, b=1, name='x', size=32, bounds=(0,1), dist=dist, axis=0, dealias=2)
    s1 = spaces.FiniteInterval(a=1, b=1, name='y', size=16, bounds=(0,1), dist=dist, axis=1, dealias=3)

    d0 = domain.Domain([s0])
    d1 = domain.Domain([s1])
    d01 = domain.Domain([s0, s1])

    print(d0.spaces)
    print(d1.spaces)
    print(d01.spaces)
    print()
    print(d0.group_shape)
    print(d1.group_shape)
    print(d01.group_shape)


if __name__ == '__main__':
    main()
