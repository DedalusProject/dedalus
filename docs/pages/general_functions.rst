General Functions
*****************

The ``GeneralFunction`` class enables users to simply define new explicit operators for the right-hand side and analysis tasks of their simulations.
Such operators can be used to apply arbitrary user-defined functions to the grid values or coefficients of some set of input fields, or even do things like introduce random data or read data from an external source.

A ``GeneralFunction`` object is instantiated with a Dedalus domain, a layout object or descriptor (e.g. ``'g'`` or ``'c'`` for grid or coefficient space), a function, a list of arguments, and a dictionary of keywords.
The resulting object is a Dedalus operator that can be evaluated and composed like other Dedalus operators.
It operates by first ensuring that any arguments that are Dedalus field objects are in the specified layout, then calling the function with the specified arguments and keywords, and finally setting the result as the output data in the specified layout.

Here's an example how you can use this class to apply a nonlinear function to the grid data of a single Dedalus field.
First, we define the underlying function we want to apply to the field data -- say the error function from scipy:

.. code-block:: python

    from scipy import special

    def erf_func(field):
        # Call scipy erf function on the field's data
        return special.erf(field.data)

Second, we make a wrapper that returns a ``GeneralFunction`` instance that applies ``erf_func`` to a provided field in grid space.
This function produces a Dedalus operator, so it's what we want to use on the RHS or in analysis tasks:

.. code-block:: python

    import dedalus.public as de

    def erf_operator(field):
        # Return GeneralFunction instance that applies erf_func in grid space
        return de.operators.GeneralFunction(
            field.domain,
            layout = 'g',
            func = erf_func,
            args = (field,)
        )

Finally, we add this wrapper to the parsing namespace to make it available in string-specified equations and analysis tasks:

.. code-block:: python

    de.operators.parseables['erf'] = erf_operator
