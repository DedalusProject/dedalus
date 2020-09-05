..  _performance_tips:

Performance Tips
****************

Stack Configuration
===================

Disable multithreading
----------------------

Dedalus does not fully implement hybrid parallelism, so the best performance is typically seen when there is one MPI process for each available core.
Some underlying stack components (e.g. Numpy or Scipy or the libraries they wrap) may still attempt to use multiple threads behind the scenes, though, and this can substantially degrade performance.
We therefore recommend explicitly disabling threading by setting environment variables such as ``OMP_NUM_THREADS`` to ``1`` before running Dedalus.

Domain Specification
====================

Resolutions for faster transforms
---------------------------------

The transforms for the ``Fourier``, ``SinCos``, and ``Chebyshev`` bases are computed using fast Fourier transforms (FFTs).
The underlying FFT algorithms are most efficient when the transform sizes are products of small primes.
We recommend choosing basis resolutions that are powers of two, or powers of two multiplied by other small factors.

Process meshes for better load balancing
----------------------------------------

Dedalus uses multidimensional block distributions when decomposing domains in parallel.
By default, problems are parallelized over a 1D process mesh of all available MPI processes.
Multidimensional parallelization is easily enabled by specifing a mesh shape using the ``mesh`` keyword when instantiating a ``Domain`` object.
The specified mesh shape should be a tuple with a length one less than the problem dimension.

Ideal load balancing occurs when the size of a distributed dimension is evenly divisible by the corresponding mesh size.
We recommend choosing mesh sizes that are powers of two, or powers of two multiplied by other small factors.
An "isotropic" mesh with the same number of processes in each mesh dimension, e.g. ``(8, 8)``, will theoretically be the most efficient for a given number of processes.

Avoid empty cores
-----------------

Note that it is possible to end up with empty cores in certain layouts if the mesh shape is chosen improperly or if a mesh is not specified for a 3D problem ran over many cores.
For example, consider a problem with global shape ``N = (64, 64, 64)`` distributed on ``P = 256`` cores.
Keeping the default 1D mesh or choosing a mesh of shape ``(128, 2)`` or ``(2, 128)`` will result in many empty cores -- a better choice would be a mesh of shape ``(16, 16)``.

Problem Formulation
===================

Minimize the number of problem variables
----------------------------------------

The number of variables used in a problem can have a large impact on the overall simulation performance.
We recommend formulating problems to use as few variables as possible, within the constraints posed by Dedalus itself, which require that PDEs are written as first-order systems in terms of temporal and non-Fourier spatial derivatives.
Often the ``problem.substitutions`` interface can be used to simplify the entry of complex equations without introducing new variables into a problem.

Formulate boundary conditions as Dirichlet conditions
-----------------------------------------------------

Maintaining a high degree of bandedness in the LHS matrices can greatly improve the performance of Dedalus.
This requires using Dirichlet rather than integral, Neumann, or other mixed boundary conditions.
Note that the first-order formulation required by Dedalus often means that such boundary conditions can be posed as Dirichlet conditions on the first-order variables, e.g. the Neumann condition ``dz(u) = 0`` for an elliptic or parabolic problem can be written as the Dirichlet condition ``uz = 0``, assuming the first-order reduction definition ``"uz - dz(u) = 0"`` is part of the PDE system.

Avoid non-smooth or rational-function NCCs
------------------------------------------

High bandedness also requires that non-constant coefficients (NCCs) appearing on the LHS are spectrally smooth.
An amplitude cutoff and a limit to the number of terms used when expanding the LHS NCCs can be specified with the ``ncc_cutoff`` and ``max_ncc_terms`` keywords when instantiating a problem.
These settings can have a large performance impact for problems with NCCs that are not low-degree polynomials.
Problems with NCCs that are rational functions (such as ``1/r**2`` terms that appear in problems in curvilinear coordinates) should usually be multiplied through to clear the polynomial denominators, resulting in purely polynomial and hence well-banded NCC operators.

Timestepping
============

Avoid changing the simulation timestep unless necessary
-------------------------------------------------------

Changing the simulation timestep requires refactorizing the LHS matrices, so the timestep should be changed as infrequently as possible for maximum performance.
If you are using the built-in CFL tools to calculate the simulation timestep, be sure to specify a ``threshold`` parameter greater than zero to prevent the timestep from changing due to small variations in the flow.

