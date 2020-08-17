Troubleshooting
***************

Singular matrix errors
======================

If you come across an error in the linear solver stating that a matrix/factor is singular, that means that the linear LHS portion of the PDE system is not solvable.
This error indicates that some degrees of freedom of the solution are unconstrained and some of the specified equations are redundant (these are equivalent since the LHS matrices must be square).

These errors are often due to imposing boundary conditions that are redundant for some set of modes and/or failing to constrain a gauge freedom in the solution.
For instance, when simulating incompressibly hydrodynamics in a channel, specifying the mean wall-normal velocity on both walls is redundant with the incompressiblity constraint (the mean wall-normal velocity at one wall must be the same as at the other wall for the flow to be divergence-free).
Instead, the wall-normal velocity boundary condition at one wall should be removed and replaced with a gauge condition setting the constant portion of the pressure.
This is why you'll see these types of boundary conditions for the wall-normal velocity (e.g. ``w``) in Dedalus scripts for incompressible flow:

.. code-block:: python

    problem.add_bc("left(w) = 0")
    problem.add_bc("right(w) = 0", condition="(nx != 0)")
    problem.add_bc("right(p) = 0", condition="(nx == 0)")

Maintaining Hermitian symmetry with real variables
==================================================

In certain problems with real variables, numerical instabilities may arise due to the loss of Hermitian symmetry in the Fourier components of the solution.
When the grid datatype is set to ``float`` or ``np.float64`` in Dedalus, that means that the first Fourier transform (say in the x direction) will be computed with a real-to-complex FFT where the negative half of the spectrum (kx < 0) is discarded.
Subsequent transforms are then complex-to-complex.

This strategy maintains the proper Hermitian symmetry for the majority of the modes, since only the non-negative kx half of the spectrum is evolved and the negative kx modes are computed from the conjugates of the positive kx modes whenever they are needed (i.e. for interpolation along x).
However, modes with kx = 0 should have coefficients that are real after the x-transform, and should have Hermitian symmetry in e.g. ky if there's a subsequent Fourier transform in y.
This is not enforced by Dedalus, so if the PDE has a linear instability for modes with kx = 0, then the imaginary part of the solution can grow tihout bound (the nonlinear terms are computed in grid space, so they only affect the real portion of the solution and can’t saturate a growing imaginary part).

A simple solution is to periodically transform all the state variables to grid space, which will project away any imaginary component that may have been building up during timestepping.
This can be done with a simple statement in the main loop like:

.. code-block:: python

    if solver.iteration % 100 == 0:
        for field in solver.state.fields:
            field.require_grid_space()

Out of memory errors
====================

Spectral simulations with implicit timestepping can require a large amount of memory to store the LHS matrices and their factorizations.
The best way to minimize the required memory is to minimize the LHS matrix size by using as few variables as possible and to minimize the LHS matrix bandwidth (see the :ref:`performance_tips` page).
Beyond this, several of the Dedalus configuration options can be changed the minimize the simulation's memory footprint, potentially at the cost of reduced performance (see the :ref:`configuration` page).

Reducing memory consumption in Dedalus is an ongoing effort.
Any assistance with memory profiling and contributions reducing the code's memory footprint would be greatly appreciated!

