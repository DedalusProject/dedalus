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
choose the type of basis functions (currently Fourier and Chebyshev
bases are available) and a timestepper, and run! 

The code is `community developed`_ using the mercurial_ (hg)
distributed version control system (DVCS). The `development team`_ are
astrophysicists and applied mathematicians, working on a wide variety
of astrophysical and geophysical fluid dynamics problems. 

.. _`community developed`: https://bitbucket.org/dedalus-project
.. _mercurial: http://mercurial.selenic.com/
.. _`development team`: http://dedalus-project.org/community.html#developers

Installing Dedalus 
==========

There are several ways to install Dedalus

.. toctree::
   :maxdepth: 1

   installation

Getting Started
==========

We have a series of ipython notebooks giving an overview of the code.

.. toctree::
   :maxdepth: 1

   getting_started 

Examples 
=========

We provide a number of examples_ in a separate mercurial
repository. These are sample scripts that run typical physical
problems, including Rayleigh-Benard convection in 2D and the KdV
equation in 1D.

.. _examples: https://bitbucket.org/dedalus-project/dedalus-examples

API Documentation
======

.. toctree::
   :maxdepth: 1

   dedalus2

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
