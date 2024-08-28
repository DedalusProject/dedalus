General Functions
*****************

The ``GeneralFunction`` and ``UnaryGridFunction`` classes enables users to simply define new explicit operators for the right-hand side and analysis tasks of their simulations.
Such operators can be used to apply arbitrary user-defined functions to the grid values or coefficients of some set of input fields, or even do things like introduce random data or read data from an external source.

A ``GeneralFunction`` object is instantiated with a Dedalus distributor, domain, tensor signature, dtype, layout object or descriptor (e.g. ``'g'`` or ``'c'`` for grid or coefficient space), function, list of arguments, and dictionary of keywords.
The resulting object is a Dedalus operator that can be evaluated and composed like other Dedalus operators.
It operates by first ensuring that any arguments that are Dedalus field objects are in the specified layout, then calling the function with the specified arguments and keywords, and finally setting the result as the output data in the specified layout.

A simpler option that should work for many use cases is the ``UnaryGridFunction`` class, which specifically applies a function to the grid data of a single field.
The output field's distributor, domain/bases, tensor signature, and dtype are all taken to be idential to those of the input field.
Only the function and input field need to be specified.
The function must be vectorized, take a single Numpy array as input, and include an ``out`` argument that specifies the output array.
Applying most Numpy or Scipy universal functions to a Dedalus field will automatically produce the corresponding ``UnaryGridFunction`` operator.

Here's an example of using the ``UnaryGridFunction`` class to apply a custom function to the grid data of a single Dedalus field.
First, we define the underlying function we want to apply to the field data:

.. code-block:: python

    # Custom function acting on grid data
    def custom_grid_function(x, out):
        out[:] = (x + np.abs(x)) / 2
        return out

Second, we make a wrapper that returns a ``UnaryGridFunction`` instance that applies ``custom_grid_function`` to a specified field.
This wrapper produces a Dedalus operator, so it's what we want to use on the RHS or in analysis tasks:

.. code-block:: python

    # Operator wrapper for custom function
    custom_grid_operator = lambda field: d3.UnaryGridFunction(custom_grid_function, field)

    # Analysis task applying custom operator to a field
    snapshots.add_task(custom_grid_operator(u), name="custom(u)")
