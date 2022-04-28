Tutorials \& Examples
*********************

**Note: please check you're referencing the intended version of the documentation.
This documentation is in reference to v2 of the code.
The latest documentation for v3 can be accessed through the sidebar.**

Tutorial Notebooks
==================

This tutorial on using Dedalus consists of four short IPython notebooks, which can be downloaded and ran interactively, or viewed on-line through the links below.

The notebooks cover the basics of setting up and interacting with the primary facets of the code, culminating in the setup and simulation of the 1D KdV-Burgers equation.

.. toctree::
   :maxdepth: 1

   Tutorial 1: Bases and Domains </notebooks/dedalus_tutorial_bases_domains.ipynb>
   Tutorial 2: Fields and Operators </notebooks/dedalus_tutorial_fields_operators.ipynb>
   Tutorial 3: Problems and Solvers </notebooks/dedalus_tutorial_problems_solvers.ipynb>
   Tutorial 4: Analysis and Post-processing </notebooks/dedalus_tutorial_analysis_postprocessing.ipynb>

Example Notebooks
=================

Below are several notebooks that walk through the setup and execution of more complicated multidimensional example problems.

.. toctree::
   :maxdepth: 1

   Example 1: Kelvin-Helmholtz Instability </notebooks/KelvinHelmholtzInstability.ipynb>
   Example 2: Taylor-Couette Flow </notebooks/TaylorCouetteFlow.ipynb>

Example Scripts
===============

A wider range of examples are available under the ``examples`` subdirectory of the main code repository, which you can browse `here <https://github.com/DedalusProject/dedalus/tree/master/examples>`_.
These example scripts can be copied to any working directory with the command ``python3 -m dedalus get_examples``.
These examples cover a wider range of use cases, including larger multidimensional problems designed for parallel execution.
Basic post-processing and plotting scripts are also provided with many problems.
These simulation and processing scripts may be useful as a starting point for implementing different problems and equation sets.
