
import numpy as np
import dedalus.public as d3
import dedalus
from dedalus.tools.parallel import RotateProcesses


shape = (8, 8)
dtype = np.float64
scales = 1
enum = (True, False)

coords = d3.S2Coordinates('phi', 'theta')
dist = d3.Distributor(coords, dtype=dtype)
basis = d3.SphereBasis(coords, shape, dtype=dtype)
domain = dedalus.core.domain.Domain(dist, (basis,))

for layout in dist.layouts:
    with RotateProcesses():
        print("Rank:", dist.comm.rank)
        print("Grid space:", layout.grid_space, " Local:", layout.local)
        print("Global shape:", layout.global_shape(domain, scales), " Chunk shape:", layout.chunk_shape(domain))
        print(layout.local_group_arrays(domain, scales))
        print(layout.local_groupsets(enum, domain, scales))
        print(basis.m_maps)
        print(basis.ell_maps)
        print()