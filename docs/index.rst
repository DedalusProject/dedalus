.. Dedalus Project documentation master file, created by
   sphinx-quickstart on Tue Mar  4 10:25:54 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Dedalus
===================================

Dedalus is a framework for solving partial differential equations
(PDEs), including eigenvalue problems, boundary value problems, and
initial value problems (i.e. simulations) for nearly arbitrary sets of
equations. Simply enter the equations in an easy-to-learn text format,
choose a timestepper and the type of basis functions (currently Fourier and Chebyshev
bases, with more on the way), and run! 

The code is `community developed`_ using the mercurial_ (hg)
distributed version control system (DVCS). The `development team`_ are
astrophysicists and applied mathematicians, working on a wide variety
of astrophysical and geophysical fluid dynamics problems. 

You can solve equations like 

.. math::
  :nowrap:

  \begin{equation}
  \partial_t \mathbf{u + u \cdot \nabla u} = -\nabla p + \nu \nabla^2 \mathbf{u}\\
  \nabla \cdot \mathbf{u} = 0
  \end{equation}


by typing::

  problem.add_equation("dt(u) - + dx(p) + nu*(dx(ux) + dy(dy(u))) = -u*ux")
  problem.add_equation("dt(v) - + dy(p) + nu*(dx(vx) + dy(dy(v))) = -u*ux")
  problem.add_equation("ux + dy(u) = 0")
  problem.add_equation("ux - dx(u) = 0")
  problem.add_equation("vx - dx(v) = 0")


.. _`community developed`: https://bitbucket.org/dedalus-project
.. _mercurial: http://mercurial.selenic.com/
.. _`development team`: http://dedalus-project.org/community.html#developers

Table of Contents
==========


.. toctree::
   :maxdepth: 1

   installation
   getting_started 
   examples

API Documentation
======

.. toctree::
   :maxdepth: 1

   dedalus2

