Gauge Conditions
****************

When you enter a system of PDEs in Dedalus, the left-hand side (LHS) of the equations is parsed into a sparse linear system.
For the solver to succeed, this LHS matrix must be square and nonsingular.
This means it must have a unique solution for *any* possible values on the right-hand side (RHS) of the equations, not just the particular RHS that you have entered.
This also means that it must constrain *all* degrees of freedom of the variables, including gauge freedoms.

For example, let's consider solving incompressible hydrodynamics in a fully periodic domain discretized with Fourier bases in each dimension.
At first glance, we might think to build the problem variables and specify the equations simply as:

.. code-block:: python

    # Fields
    p = dist.Field(name='p', bases=bases)
    u = dist.VectorField(coords, name='u', bases=bases)

    # Problem
    problem = d3.IVP([p, u], namespace=locals())
    problem.add_equation("div(u) = 0")
    problem.add_equation("dt(u) + grad(p) - nu*lap(u) = - dot(u,grad(u))")

This formulation produces square matrices (same number of modes in the equations as the variables), but they are not all nonsingular.
The problem is that the pressure gauge is undetermined -- any constant can be added to the pressure without affecting any of the equations.

However, since the system is square, if one variable is underdetermined, then there must also be a degenerate constraint.
In this case, the mean of the divergence equation is degenerate with the periodic boundary conditions imposed by the Fourier discretization.
That is, for the mean :math:`\vec{k} = 0` Fourier mode, the divergence equation simply becomes ``"0 = 0"``.
This seems consistent -- but again the system must be solvable for *any* RHS, not just the one we entered, and clearly we would have a problem if we entered ``"div(u) = 1"`` since for the mean mode this would become ``"0 = 1"``.

To fix these problems, we need to add another equation that fixes the pressure gauge, and we need to remove the degenerate constraint.
Algorithmically, we could simply replace the divergence constraint for the mean Fourier mode with a pressure constraint (as was done in Dedalus v2 using equation conditions).
Another option is to expand the system by adding a spatially-constant variable (let's call it :math:`\tau_p`) to the divergence equation to absorb this degeneracy, and then impose the pressure gauge as a separate equation:

.. code-block:: python

    # Fields
    p = dist.Field(name='p', bases=bases)
    u = dist.VectorField(coords, name='u', bases=bases)
    tau_p = dist.Field(name='tau_p')

    # Problem
    problem = d3.IVP([p, u, tau_p], namespace=locals())
    problem.add_equation("div(u) + tau_p = 0")
    problem.add_equation("dt(u) + grad(p) - nu*lap(u) = - dot(u,grad(u))")
    problem.add_equation("integ(p) = 0")

We've added a single additional degree of freedom to the variables and a single additional constraint, so this system is still square.
It is now also nonsingular, since the mean pressure is fixed.
The degeneracy of the mean divergence equation is also lifted as the tau variable can simply absorb/acquire the mean value of any possible RHS.
The mean divergence of the velocity will always be zero, as required by the periodic discretization.

Similar modifications work for other types of gauges and geometries.
For example, for incompressible hydrodynamics in a bounded domain, we still need the above type of modification so that the integral of the divergence equation is compatible with the specified inflow boundary conditions.
If the prescribed net inflow is nonzero, then the tau variable will acquire a corresponding nonzero value.
From the modified equation, we can see that the velocity will then have a spatially uniform convergence equal to this tau value.
Of course, for properly specified boundary conditions with no net inflow, the tau variable will be zero and the velocity will be divergence free.

See the included :doc:`example scripts <tutorials>` for more examples of gauge conditions in various domains.
