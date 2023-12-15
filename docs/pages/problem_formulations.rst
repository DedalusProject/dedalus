Problem Formulations
********************

Dedalus parses all equations, including constraints and boundary conditions, into common forms based on the problem type.

Linear Boundary-Value Problems (LBVPs)
--------------------------------------

Equations in linear boundary-value problems must all take the form:

.. math::
    L \cdot X = F,

where :math:`X` is the state-vector of problem variables, :math:`L` are linear operators, and :math:`F` are inhomogeneous terms that are independent of :math:`X`.
LBVPs are solved by explicitly evaluating the RHS and solving the sparse-matrix representation of the LHS to find :math:`X`.

Initial-Value Problems (IVPs)
-----------------------------

Equations in initial value problems must all take the form:

.. math::
    M \cdot \partial_t X + L \cdot X = F(X,t),

where :math:`X(t)` is the state-vector of problem variables, :math:`M` and :math:`L` are time-independent linear operators, and :math:`F` are inhomogeneous terms or nonlinear terms.
Initial conditions :math:`X(t=0)` are set for the state, and the state is then evolved forward in time using mixed implicit-explicit timesteppers.
During this process, the RHS is explicitly evaluated using :math:`X(t)` and the LHS is implicitly solved using the sparse-matrix representations of :math:`M` and :math:`L` to produce :math:`X(t+ \Delta t)`.

Eigenvalue Problems (EVPs)
--------------------------

Equations in eigenvalue problems must all take the generalized form:

.. math::
    \lambda M \cdot X + L \cdot X = 0,

where :math:`\lambda` is the eigenvalue, :math:`X` is the state-vector of problem variables, and :math:`M` and :math:`L` are linear operators. The standard *right eigenmodes* :math:`(\lambda_i, X_i)` are solved using the sparse-matrix representations of :math:`M` and :math:`L`, and satisfy:

.. math::
    \lambda_i M \cdot X_i + L \cdot X_i = 0.

The *left eigenmodes* :math:`(\lambda_i, Y_i)` are solved (if requested) using the sparse-matrix representations of :math:`M` and :math:`L`, and satisfy:

.. math::
    \lambda_i Y_i^* \cdot M + Y_i^* \cdot L = 0.

The left and right eigenmodes satisfy the generalized :math:`M`-orthogonality condition:

.. math::
    Y_i^* \cdot M \cdot X_j = 0 \quad \mathrm{if} \quad \lambda_i \neq \lambda_j.

For convenience, we also provide *modified left eigenmodes* :math:`Z_i = M^* \cdot Y_i`.
When the eigenvalues are nondegenerate, the left and modified left eigenvectors are rescaled by :math:`(X_i^* \cdot M^* \cdot Y_i)^{-1}` so that

.. math::
    Z_i^* \cdot X_j = \delta_{ij}.

Nonlinear Boundary-Value Problems (NLBVPs)
--------------------------------------

Equations in nonlinear boundary-value problems must all take the form:

.. math::
    G(X) = H(X),

where :math:`X` is the state-vector of problem variables and :math:`G` and :math:`H` are generic operators.
All equations are immediately reformulated into the root-finding problem

.. math::
    F(X) = G(X) - H(X) = 0.

NLBVPs are solved iteratively via Newton's method.
The problem is reduced to a LBVP for an update :math:`\delta X` to the state using the symbolically-computed Frechet differential as:

.. math::
    F(X_n + \delta X) \approx 0 \quad \implies \quad \partial F(X_n) \cdot \delta X = - F(X_n)

Each iteration entails reforming the LHS matrix, explicitly evaluating the RHS, solving the LHS to find :math:`\delta X`, and updating the new solution as :math:`X_{n+1} = X_n + \delta X`.
The iterations proceed until convergence criteria are satisfied.

