GPU Support
***********

.. note::
    GPU support is currently experimental and may not be fully functional for all use cases.
    Cartesian problems should generally work, but curvilinear problems are not yet supported.
    Variable timestepping is still slow, so constant timestepping is strongly recommended.

Dedalus supports GPU acceleration on NVIDIA and AMD GPUs through the use of the CuPy array library.
Specifically, Dedalus utilizes the ``array-api-compat`` library to provide a unified interface for Numpy and CuPy array types and operations.

Installation
------------

In add  ition to the regular Dedalus dependencies, you will need to install CuPy to enable GPU support. CuPy itself can be installed using pip, but also requires a compatible CUDA or ROCm environment to be installed on your system.

Selecting GPUs
--------------

To utilize GPU acceleration, your Dedalus script needs to import CuPy and pass it as the ``array_namespace`` keyword to the Distributor object.
Fields built with that Distributor will then use CuPy arrays for their data and computations.
No other explicit changes are required to most scripts, but some care may need to be taken for setting field initial conditions, etc., and converting other Numpy arrays to CuPy field data.

.. code-block:: python

    import dedalus.public as d3
    import cupy as cp
    ...
    dist = d3.Distributor(coords, dtype=np.float64, array_namespace=cp)


Guidelines
----------

GPU support is preliminary and many standard Dedalus features have not yet been fully implemented or optimized for GPUs.
Please keep in mind the following guidelines for best performance under the current capabilities:

- Single and double precision, real and complex, are supported.
- Curvilinear problems are not yet supported.
- Distributed GPUs (combining MPI with multiple GPUs) are not yet supported.
- By default, GPU problems are treated collectively as a single subproblem to speed up linear algebra on the GPU (looping explicitly over subproblems is slow). The trade off is that this can make matrix factorizations quite slow, **so we strongly recommend using constant timesteps when possible to avoid refactorizations**. We hope to address this issue shortly.

