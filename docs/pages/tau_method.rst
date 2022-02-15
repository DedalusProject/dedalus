Tau Method
**********

The *generalized tau method* is a system for imposing boundary boundary conditions (BCs) when solving partial differential equations (PDEs) using polynomial spectral methods.
It consists of explicitly adding *tau terms* to the PDE which introduce degrees of freedom that allow the problem to be solved exactly over polynomials.
Determining the optimal tau terms to add to a given PDE is an open problem, and while we hope to eventually automate this process in Dedalus, currently these terms must be manually added when specifying equations in Dedalus v3.

The underyling mathematical issue is that most PDEs that we wish to solve do not have exact polynomial solutions.
Instead, we seek polynomial solutions that approximate the true solution.
All spectral methods find such a solution by modifying the underlying equations in some way, and then solve for the exact polynomial solution to the modified equations.
The tau method makes these modifications explicit in the problem specification rather than hiding them within the solution algorithm.

A first example
---------------

As an example, let's consider the equation:

.. math::
    \begin{gathered}
        \partial_x u(x) - u(x) = 0, \quad x \in [0, 1] \\
        u(0) = 1
    \end{gathered}

The exact solution is :math:`u(x) = e^x`, but we seek an approximate polynomial solution.
The generalized tau method modifies the PDE to be:

.. math::
    \partial_x u(x) - u(x) + \tau P(x) = 0

where :math:`\tau` is an undetermined constant and :math:`P(x)` is a specified polynomial.
If :math:`P(x)` is a polynomial of degree :math:`N`, then the modified equation also has an exact polynomial solution :math:`u_N(x)` that is also of degree :math:`N`.
For instance, taking :math:`P(x) = x^2`, the modified PDE has the solution :math:`u_2(x) = (x^2 + 2 x + 2) / 2`` with :math:`\tau = 1 / 2`.
The *classical tau method* picks the tau polynomials to be the Chebyshev polynomials, :math:`P(x) = T_N(x)`, but the generalized method allows more freedom in picking :math:`P(x)`, as we'll see below.

Systems in first-order form
---------------------------

When solving a system of nonsingular PDEs, the number of tau terms and the number of boundary conditions will generally match the total number of derivatives in the system.
This is most easily counted by converting the system to first-order form, which is how equations were required to be entered in Dedalus v2.
For instance, let's consider linearized 2D incompressible hydrodynamics with velocity :math:`\vec{u} = (u, v)`, pressure :math:`p`, and a generic forcing :math:`\vec{f} = (f, g)`.
Let's consider a domain that is periodic in :math:`x` and bounded in :math:`y \in [-1, 1]` with no-slip conditions at the boundaries and a zero-mean gauge condition for the pressure.
The component-wise equations are:

.. math::
    \begin{gathered}
        \partial_x u + \partial_y v = 0 \\
        \partial_t u - \nu (\partial_x^2 u + \partial_y^2 u) + \partial_x p = f \\
        \partial_t v - \nu (\partial_x^2 v + \partial_y^2 v) + \partial_y p = g \\
    \end{gathered}

Introducing :math:`u_y` and :math:`v_y` variables, the equations can be written in first-order form in :math:`y`-derivatives as:

.. math::
    \begin{gathered}
        u_y - \partial_y u = 0 \\
        v_y - \partial_y v = 0 \\
        \partial_x u + v_y = 0 \\
        \partial_t u - \nu (\partial_x^2 u + \partial_y u_y) + \partial_x p = f \\
        \partial_t v - \nu (\partial_x^2 v + \partial_y v_y) + \partial_y p = g
    \end{gathered}

We see that in first-order form, four of the equations have :math:`y`-derivatives, and we also have four boundary conditions to impose.
This can be achieved by adding four tau terms, one to each :math:`y`-differential equation:

.. math::
    \begin{gathered}
        u_y - \partial_y u + \tau_1(x) P(y) = 0 \\
        v_y - \partial_y v + \tau_2(x) P(y) = 0 \\
        \partial_x u + v_y = 0 \\
        \partial_t u - \nu (\partial_x^2 u + \partial_y u_y) + \partial_x p + \tau_3(x) P(y) = f \\
        \partial_t v - \nu (\partial_x^2 v + \partial_y v_y) + \partial_y p + \tau_4(x) P(y) = g
    \end{gathered}

Note that the tau variables themselves are now functions of the tangential coordinates, in this case :math:`x`.
If the RHS terms are truncated at degree :math:`N` in :math:`y` and the tau polynomial :math:`P(y)` is of degree :math:`N` in :math:`y`, then the system will have exact polynomial solutions for :math:`u`, :math:`v`, :math:`p`, :math:`u_y`, and :math:`v_y` also of degree :math:`N` in :math:`y`.

In Dedalus v2, equations were required to be entered in first-order form as above.
Tau terms were then automatically added to the differential equations with :math:`P(y) = U_N(y)`, using the second-kind Chebyshev polynomials :math:`U_n(x)`.
This system consituted a generalized tau method using a first-order form of the Cheybshev ultraspherical method.
Algorithmically, it is equivalent to dropping the last rows from the differential equations after they have been discretized with the ultraspherical method (using sparse Chebyshev T-to-U operators).
Enforcing boundary conditions in this fashion is easily automatable, but results in larger linear systems due to the first-order reduction.

Higher-order systems
--------------------

To more efficiently handle high-order systems, and to more easily accomodate the singular equations arising in curvilinear domains, Dedalus v3 supports equations with arbitrary differential order.
For Chebyshev dimensions as well as annuli and spherical shells, we recommend adding tau terms to the equations in a manner consistent with the first-order formulations discussed above.
For example, after adding the tau terms above, we can eliminate the first-order-reduction variables to recover the original second-order equations, but containing the same tau corrections:

.. math::
    \begin{gathered}
        \partial_x u + \partial_y v - \tau_2(x) P(y) = 0 \\
        \partial_t u - \nu (\partial_x^2 u + \partial_y^2 u) + \partial_x p + \tau_3(x) P(y) - \nu \tau_1(x) \partial_y P(y) = f \\
        \partial_t v - \nu (\partial_x^2 v + \partial_y^2 v) + \partial_y p + \tau_4(x) P(y) - \nu \tau_2(x) \partial_y P(y) = g
    \end{gathered}

This system has the same solution as the first-order system, but is more efficient to solve.
This type of system is specified in Dedalus v3 by creating problem variables that correspond to the tau fields, using substitutions in place of the first-order reductions, and entering the higher-order equations using these substitutions.
The equations are entered in vectorial rather than component form, so the tau variables and terms similarly need to be promoted to vectors as :math:`\vec{\tau}_1 = (\tau_1, \tau_2)` and :math:`\vec{\tau}_2 = (\tau_3, \tau_4)`.
Defining :math:`G = \nabla \vec{u} - \vec{e}_y \vec{\tau}_1(x) P(y)`, the above equations can be written in vectorial form as:

.. math::
    \begin{gathered}
        \mathrm{tr}(G) = 0 \\
        \partial_t \vec{u} - \nu \nabla \cdot G + \nabla p + \vec{\tau}_2(x) P(y) = \vec{f}
    \end{gathered}

since

.. math::
    \mathrm{tr}(G) = \nabla \cdot \vec{u} - \vec{e}_y \cdot \vec{\tau}_1(x) P(y)
.. math::
    \nabla \cdot G = \nabla^2 \vec{u} - \vec{\tau}_1(x) \partial_y P(y)

Let's walk through setting up such a problem in Dedalus v3, assuming we're discretizing :math:`x` and :math:`y` with Fourier and Chebyshev bases, respectively.
First, we need to create the necessary problem variable fields, including fields for the tau variables and a constant scalar tau for imposing the pressure gauge (see the :doc:`gauge_conditions` page):

.. code-block:: python

    # Fields
    p = dist.Field(name='p', bases=(xbasis,ybasis))
    u = dist.VectorField(coords, name='u', bases=(xbasis,ybasis))
    tau_u1 = dist.VectorField(coords, name='tau_u1', bases=xbasis)
    tau_u2 = dist.VectorField(coords, name='tau_u2', bases=xbasis)
    tau_p = dist.Field(name='tau_p')

We then create substitutions for :math:`G` and :math:`P(y)`.
Specification of and multiplication by :math:`P(y)` are handled through the ``Lift`` operator, which here simply multiplies its argument by the specified mode/element of a selected basis.
Here we'll take :math:`P(y)` to be the highest mode in the Chebyshev-U basis, in accordance with the first-order ultraspherical method described above:

.. code-block:: python

    # Substitutions
    ex, ey = coords.unit_vector_fields(dist)
    lift_basis = ybasis.clone_with(a=1/2, b=1/2) # Chebyshev U basis
    lift = lambda A, n: d3.Lift(A, lift_basis, -1) # Shortcut for multiplying by U_{N-1}(y)
    grad_u = d3.grad(u) - ey*lift(tau_u1) # Operator representing G

We can then create a problem and enter the tau-modified PDEs, boundary condtions, and pressure gauge in vectorial form using these substitutions.
Note that we will need to add the contant tau variable to the divergence equation as described in the :doc:`gauge_conditions` page.
This allows us to impose the pressure gauge and removes the redundancy between the integral of the divergence equation and the integral of the inflow boundary conditions.

.. code-block:: python

    # Problem
    problem = d3.IVP([p, u, tau_u1, tau_u2, tau_p], namespace=locals())
    problem.add_equation("trace(grad_u) + tau_p = 0")
    problem.add_equation("dt(u) - nu*div(grad_u) + grad(p) + lift(tau_u2) = f")
    problem.add_equation("u(y=-1) = 0")
    problem.add_equation("u(y=+1) = 0")
    problem.add_equation("integ(p) = 0")

The same approach can be used to add scalar taus for tracer fields/equations, as illustrated in the :doc:`example scripts <tutorials>`.
Overall, we have so far found that this method of "first-order taus" in higher-order equations works well for problems in Cartesian domains, annuli, and spherical shells.

Disks and balls
---------------

In the disk and ball, the radial dimension only has a single (outer) boundary.
This means that second-order elliptic and parabolic equations generally only need one boundary condition (since there is only one boundary) rather than two.
Therefore only one tau term needs to be introduced to the evolutionary equation, and no first-order reduction is necessary.
For instance, to enter the above equation set with homogeneous Dirichlet boundary conditions in the disk, we only need the following problem fields:

.. code-block:: python

    # Fields
    p = dist.Field(name='p', bases=disk_basis)
    u = dist.VectorField(coords, name='u', bases=disk_basis)
    tau_u = dist.VectorField(coords, name='tau_u', bases=phi_basis)
    tau_p = dist.Field(name='tau_p')

The disk and ball bases are not direct-product bases, so the tau terms can't actually be written just as the tau variable times a radial polynomial.
Instead, for each horizontal mode (azimuthal mode :math:`m` in the disk and spherical harmonic :math:`\ell` in the ball), that mode of the tau variable is multiplied by the highest degree radial polynomial in the basis for that particular mode.
The ``Lift`` operator does this under the hood, and is why we use it rather than explicitly writing out the tau polynomials.
We've found that using tau polynomials from the original bases seems to give good results in the disk and ball:

.. code-block:: python

    # Substitutions
    lift = lambda A, n: d3.Lift(A, disk_basis, -1)

Now we can enter the PDE with just the single tau term in the momentum equation:

.. code-block:: python

    # Problem
    problem = d3.IVP([p, u, tau_u, tau_p], namespace=locals())
    problem.add_equation("div(u) + tau_p = 0")
    problem.add_equation("dt(u) - nu*lap(u) + grad(p) + lift(tau_u) = f")
    problem.add_equation("u(r=1) = 0")
    problem.add_equation("integ(p) = 0")

Again, the same appraoch can be used to add scalar taus for tracer fields/equations, as illustrated in the :doc:`example scripts <tutorials>`.

Summary
-------

To summarize, the main points regarding tau formulations are:

1. To impose PDEs in Dedalus v3, you need to add tau fields (which are supported on the boundary) to your problem formulation.
2. You need the same number and type of tau fields as boundary conditions (e.g. 2 vector tau fields if you have two velocity-vector boundary conditions).
3. For problems in Cartesian geometries, annuli, and spherical shells, we recommend a first-order-style implementation of the tau terms. Note that this only requires defining first-order substitutions that include tau terms, rather than increasing the problem size with first-order variables, as in Dedalus v2.
4. For problems in the disk and ball, only a single tau term is needed for second-order elliptic/parabolic problems, and no first-order substitutions are necessary.

See the included :doc:`example scripts <tutorials>` for more examples of tau modifications in various domains.


.. .. math::
..     \begin{gathered}
..         u + r \partial_r u + \partial_\phi v = 0 \\
..         r^2 \partial_t u - \nu (r \partial_r (r \partial_r u) + \partial_\phi^2 u - u - 2 \partial_\phi v) + r^2 \partial_r p = r^2 f \\
..         r^2 \partial_t v - \nu (r \partial_r (r \partial_r v) + \partial_\phi^2 v - v + 2 \partial_\phi u) + r \partial_\phi p = r^2 g \\
..     \end{gathered}
