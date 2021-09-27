
import numpy as np
import dedalus.public as d3
import dedalus
from dedalus.tools.parallel import RotateProcesses


shape = (16, 8)
dtype = np.float64
scales = 1

coords = d3.S2Coordinates('phi', 'theta')
dist = d3.Distributor(coords, dtype=dtype)
sb = d3.SphereBasis(coords, shape, dtype=dtype)
domain = dedalus.core.domain.Domain(dist, (sb,))

for layout in dist.layouts:
    with RotateProcesses():
        print("Rank:", dist.comm.rank)
        print("Grid space:", layout.grid_space, " Local:", layout.local)
        print("Global shape:", layout.global_shape(domain, scales), " Chunk shape:", layout.chunk_shape(domain))
        print(layout.local_groups(domain, scales))
        print()