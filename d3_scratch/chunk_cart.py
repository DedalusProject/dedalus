import dedalus.public as d3
import dedalus
from dedalus.tools.parallel import RotateProcesses

scales = 1

coords = d3.CartesianCoordinates('x', 'y')
dist = d3.Distributor(coords, dtype=float)
xb = d3.RealFourier(coords['x'], 32, (0, 1))
yb = d3.Chebyshev(coords['y'], 64, (0, 1))
domain = dedalus.core.domain.Domain(dist, (xb, yb))

for layout in dist.layouts:
    with RotateProcesses():
        print("Rank:", dist.comm.rank)
        print("Grid space:", layout.grid_space, " Local:", layout.local)
        print("Global shape:", layout.global_shape(domain, scales), " Chunk shape:", layout.chunk_shape(domain))
        print(layout.local_groups(domain, scales))
        print()
