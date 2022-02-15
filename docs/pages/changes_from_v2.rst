Changes from Dedalus v2
***********************

This document outlines some of the major conceptual and API differences between v2 and v3 of Dedalus.
If you are entirely new to Dedalus, the tutorial notebooks may be a better introduction to the new API.

Coordinate and Distributor objects
----------------------------------

In Dedalus v3, fields no longer have to be defined over just a single domain or set of bases.
You must therefore construct ``Coordinate``/``CoordinateSystem`` and ``Distributor`` objects so that fields with different dimensions and bases can be consistently distrubuted in parallel.
Coordinate or CoordinateSystem objects must be created for all the coordinates in your problem.
These are combined into a Distributor object, mapping the coordinates to specific array axes and constructing the Layout objects.
Coodinates are also used when creating ``Basis`` objects to indicate the coordinates of the basis.
The datatype of the fields in a problem can also be set when the distributor is constructed.

For instance, to setup the bases for a real-valued problem in 2D Cartesian coordiantes:

.. code-block:: python

    coords = d3.CartesianCoordinates('x', 'y')
    dist = d3.Distributor(coords, dtype=np.float64)
    xbasis = d3.RealFourier(coords['x'], size=128, bounds=(0, 1))
    ybasis = d3.Chebyshev(coords['y'], size=256, bounds=(0, 1))

For problems in curvilinear coordinates, the entire coordinate system object is passed when creating a basis, since the spectral bases for these coordinates are not generally of direct-product form:

.. code-block:: python

    coords_sph = d3.SphericalCoordinates('phi', 'theta', 'r')
    dist_sph = d3.Distributor(coords_sph, dtype=np.float64)
    ball_basis = d3.BallBasis(coords_sph, shape=(512, 256, 256), radius=1)

The distributor and bases are both used when creating fields, ensuring that all fields are distributed consistently in parallel.

No global domain / bases
------------------------

In Dedalus v2, each problem had a single domain object with a fixed set a bases.
Field metadata was used to indicate if certain fields were e.g. constant in any coordinates.

In Dedalus v3, there is no global domain object, and fields can be constructed with different combinations of bases as needed.
This simplifies the creation and specification of fields of different dimensions, and allows operators to directly map between different bases as required for maximum sparsity.
For brevity, fields can be constructed using methods on the distributor object.

For instance, 1D and 2D fields :math:`f(x)` and :math:`g(x,y)` in a 2D simulation can be built like:

.. code-block:: python

    f = dist.Field(bases=xbasis) # Constant in y
    g = dist.Field(bases=(xbasis, ybasis))

Spatially constant fields can even be created by not specifying any bases.
This is often useful for problems with gauge conditions, etc.:

.. code-block:: python

    c = dist.Field() # Constant-valued field

Vector and tensor-valued fields
-------------------------------

In Dedalus v2, vector and tensor-valued fields and equations were handled component-wise.
In Dedalus v3, field objects can be scalar, vector, or arbitrary-rank-tensor-valued.
Component-wise representations are still allowed for Cartesian problems, but not in curvilinear coordinates since the components of vectors and tensors in these coordinates generally do not have the same analytic behavior as smooth scalar-valued fields near the coordinate singularities.

To construct a vector-valued field, the coordinates corresponding to the components you want the vector to contain must be passed at instantiation.
Technically, this is specifying the `tangent bundle <https://en.wikipedia.org/wiki/Tangent_bundle>`_ of the vector field.
This is necessary to distinguish between the components we want the vector field to contain, and the coordinates on which the vector field depends.

The components of the vector field are stored in the first index of the field data.
In grid space, these components correspond to the coordinates in the specified tangent bundle.
In coefficient space, the components are the same for Cartesian problems, but different for curvilinear problems where spin and/or regularity recombinations of the components taken.
It's generally recommended to avoid directly interacting with coefficient data for this reason, unless you're experienced with these representations.

For instance, we can construct vector fields in our 2D Cartesian coordinate system that depend on just :math:`x` or both :math:`x` and :math:`y` as:

.. code-block:: python

    u = dist.VectorField(coords, bases=xbasis) # Constant in y, but with x and y components
    v = dist.VectorField(coords, bases=(xbasis, ybasis))

Arbitrary-rank tensor fields can also be created by combining coordinate objects to represent the `tensor bundle <https://en.wikipedia.org/wiki/Tensor_field#Tensor_bundles>`_ of the field.
The components of the field are stored in the first rank-many indices of the field data.
For instance, to create spatially constant rank-2 identity tensor:

.. code-block:: python

    I = dist.TensorField((coords, coords))
    I['g'][0,0] = 1
    I['g'][1,1] = 1

Vector calculus operators
-------------------------

Along with vector and tensor-valued fields, vectorial differential operators (``Gradient``, ``Divergence``, ``Curl``, and ``Laplacian``) are now available.
This dramatically simplifies the symbolic specification of vector and tensor-valued equations, particularly in curvilinear coordinates.
Individual partial derivative operators are now usually just used in 1D equations.
Vector algebra operations (dot products, cross products, and outer products) are also available through the ``Dot``, ``Cross``, and regular multiplication operators.

For instance, an operator for computing the strain rate tensor from a velocity field can be created like:

.. code-block:: python

    E = (d3.grad(u) + d3.transpose(d3.grad(u))) / 2

Problem namespaces
------------------

In Dedalus v2, problems were instantiated with a domain and a list of variable names, and the corresponding field objects were internally created to form the solver's state vector.

In Dedalus v3, you should create the field objects for your problem variables, and pass a list of these variables when instantiating a problem object.
You can then specify equations by adding ``(LHS, RHS)`` tuples of operators.
This more object-oriented (as opposed to string-based) form of equation entry can make it much easier to programatically construct complex equations and substitutions.

It remains possible to enter equations in string form, to be parsed in the internal problem namespace.
This can reduce the boilerplate associated with retrieving the built in operators from the Dedalus module, etc., and allows the equations to be specified in the nice ``"LHS = RHS"`` form.
A best-of-both-worlds approach is to pass your scripts entire local namespace to the problem object, to make it available when parsing string-based equations.
This is achieved by passing the keyword ``namespace=locals()`` when instantiating problem objects.
See the built in examples for illustrations of this approach to equation construction.

Gauge conditions
----------------

In Dedalus v2, gauge conditions (like the pressure gauge in incompressible hydrodynamics) were usually set by changing the equations for certain modes with the ``condition`` keyword when entering equations.
In Dedalus v3, it's recommended to instead add spatially-constant gauge variables to the equations to introduce degrees of freedom that allow the gauge conditions to be directly imposed alongside the other equations.
In most cases, the ``condition`` keyword can still be used if desired, but for technical/performance reasons it is no longer available in fully-Fourier problems.
In any event, we find that the new approach (with gauge variables instead of equation conditions) makes for more readable equations.
See the :doc:`gauge_conditions` page and the examples for more details.

Tau terms
---------

In Dedalus v2, problems were required to be first-order in Chebyshev derivatives and rows were automatically dropped from the differential equations to be replaced with the specified boundary conditions.
In Dedalus v3, first-order reductions are no longer required, allowing for smaller and faster higher-order problem formulations.
However, this makes it more complicated to determine how boundary conditions should be imposed, particularly in curvilinear coordinates.

Currently in Dedalus v3, boundary condition enforcement is not fully automated.
Instead, you must explicitly add "tau terms" to your differential equations, which introduce degrees of freedom than allow for your specified boundary conditions to be imposed.
See the :doc:`tau_method` page and the examples for more details.

Virtual HDF5 datasets
---------------------

In Dedalus v2, each process wrote to its own HDF5 file, and these process files had to be manually merged after a simulation was completed.
In Dedalus v3, each process still writes to its own HDF5 file, but a "virtual file" is also created that allows you to access the global data as if it has already been merged.
This file uses the `Virtual Dataset <https://docs.h5py.org/en/stable/vds.html>`_ feature of HDF5/h5py, and eliminates the need to merge outputs after each simulation.
Note, however, that these virtual files contains no data themselves -- if you want to relocate the data, you must copy the underlying process along with each virtual file.

