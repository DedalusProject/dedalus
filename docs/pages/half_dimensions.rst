Half Dimensions
***************

A *half dimensional* problem is one in which the solution is independent of one of the spatial dimensions, for instance a 2.5D problem is one with a 3D coordinate system (and vectors with 3 components) but only 2D bases.
Dedalus can efficiently solve such problems, but care needs to be taken when parallelizing them to ensure the problem is well distributed.

Cartesian problems
------------------

In Cartesian coordinate systems, the constant dimensions (with no bases) should be placed as the last dimensions in the coordinate system.
Note this may require marking the coordinate system as being left-handed so that the proper signs are used in curls and cross products.
The reason is that Dedalus distributes problems over the leading axes, and if one of these is constant, the problem will end up entirely on a single core, bottleneckling the scaling.

For instance, consider a 2.5D problem in a 3D Cartesian domain with coordinates :math:`(x, y, z)`, where the solution is independent of :math:`y`, and the coordinate system is defined normally with a 1D process mesh of size :math:`p=8`:

 .. code-block:: python

    coords = d3.CartesianCoordinates('x', 'y', 'z')
    dist = d3.Distributor(coords, mesh=(8,), dtype=dtype)
    xbasis = d3.RealFourier(coords['x'], 64, bounds=(0, 1))
    zbasis = d3.RealFourier(coords['z'], 64, bounds=(0, 1))
    f = dist.Field(bases=(xbasis, zbasis))

In coefficient space, the first dimension is distributed and the local data shape on each process will be :math:`(\lceil N_x/p \rceil, 1, N_z) = (8, 1, 64)`.
However, in grid space the second dimension is distributed.
So the root process will contain the entire problem, with a local data shape of :math:`(N_x, \lceil 1/p \rceil) = (64, 1, 64)`.
The other processes will be empty, since there are no higher modes in the :math:`y`-direction.
The computation will therefore be bottlenecked in grid space, and will not scale well beyond a single process.

To fix this, we need to make the :math:`y`-dimension the last dimension in the coordinate system, and mark the coordinate system as left-handed:

.. code-block:: python

    coords = d3.CartesianCoordinates('x', 'z', 'y', right_handed=False)
    dist = d3.Distributor(coords, mesh=(8,), dtype=dtype)
    xbasis = d3.RealFourier(coords['x'], 64, bounds=(0, 1))
    zbasis = d3.RealFourier(coords['z'], 64, bounds=(0, 1))
    f = dist.Field(bases=(xbasis, zbasis))

Now, in coefficient space, the local data shape on each process will be :math:`(\lceil N_x/p \rceil, N_z, 1) = (8, 64, 1)`.
In grid space, the local data shape will be :math:`(N_x, \lceil N_z/p \rceil, 1) = (64, 8, 1)`.
Now the global problem is well distributed in both spaces, and the computation will scale in the same manner as a 2D problem over just :math:`(x, z)`.

Curvilinear problems
--------------------

In curvilinear coordinates, axisymmetric problems can be handled by setting the azimuthal resolution to 1.
However, unlike Cartesian coordinate, the coordinate ordering is fixed.
Since the azimuthal coordinate is the first in the curvilinear coordinate systems, distributing over this dimension can be avoided by prepending a 1 to the process mesh shape.
So for instance, an axisymmetric problem in the ball can be parallelized over :math:`p=8` processes as follows:

.. code-block:: python

    coords = d3.SphericalCoordinates('φ', 'θ', 'r')
    dist = d3.Distributor(coords, mesh=(1, 8), dtype=dtype)
    ball = d3.BallBasis(coords, (1, 64, 64), dtype=dtype)
    f = dist.Field(bases=ball)

Because of the prepended 1 in the process mesh, the data will be distributed over the second (:math:`\theta`) dimension in coefficient space, and over the third (:math:`r`) dimension in grid space.
Again, this will result in no empty processes, and the computation will scale correctly like a 2D problem over a 1D process mesh.
