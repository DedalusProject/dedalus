Troubleshooting
***************

Singular matrix errors
======================

If you come across an error in the linear solver stating that a matrix/factor is singular, that means that the linear LHS portion of the PDE system is not uniquely solvable.
This error indicates that some degrees of freedom of the solution are unconstrained and some of the specified equations are redundant (these are equivalent since the LHS matrices must be square).
These errors are often due to imposing boundary conditions that are redundant for some set of modes and/or failing to constrain a gauge freedom in the solution.
See the :doc:`gauge_constraints` and :doc:`tau_method` pages for more information on fixing these issues.

Out of memory errors
====================

Spectral simulations with implicit timestepping can require a large amount of memory to store the LHS matrices and their factorizations.
The best way to minimize the required memory is to minimize the LHS matrix size by using as few variables as possible and to minimize the LHS matrix bandwidth (see the :doc:`performance_tips` page).
Beyond this, several of the Dedalus configuration options can be changed the minimize the simulation's memory footprint, potentially at the cost of reduced performance (see the :doc:`configuration` page).

Reducing memory consumption in Dedalus is an ongoing effort.
Any assistance with memory profiling and contributions reducing the code's memory footprint would be greatly appreciated!

Maintaining Hermitian symmetry with real variables
==================================================

In certain problems with real variables, numerical instabilities may arise due to the loss of Hermitian symmetry in the Fourier components of the solution.
A simple solution is to periodically transform all the state variables to grid space, which will project away any imaginary component that may have been building up during timestepping.
This is done in Dedalus every 100 timesteps by default.
This cadence can be modified via the ``enforce_real_cadence`` keyword when instantiating an IVP solver, and may need to be decreased in simulations with strong linear instabilities.

