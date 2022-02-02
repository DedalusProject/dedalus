Tutorials \& Examples
*********************

Tutorial Notebooks
==================

This tutorial on using Dedalus consists of four short IPython notebooks, which can be downloaded and ran interactively, or viewed on-line through the links below.

These notebooks cover the basics of setting up and interacting with the primary facets of the code, culminating in the setup and simulation of the 1D KdV-Burgers equation.

.. toctree::
    :maxdepth: 1

    /notebooks/dedalus_tutorial_1.ipynb
    /notebooks/dedalus_tutorial_2.ipynb
    /notebooks/dedalus_tutorial_3.ipynb
    /notebooks/dedalus_tutorial_4.ipynb

Example Scripts
===============

A range of examples are available under the ``examples`` subdirectory of the main code repository, which you can browse :repo:`here <examples>` and preview below.
These example scripts can be copied to any working directory with the command ``python3 -m dedalus get_examples``.
These examples cover a wide range of use cases, including multidimensional problems designed for parallel execution.
Basic post-processing and plotting scripts are also provided with many problems.
These simulation and processing scripts may be useful as a starting point for implementing different problems and equation sets.

Cartesian examples
------------------

.. nbgallery::
    :name: cartesian-example-gallery
    :glob:

    /pages/examples/*1d*
    /pages/examples/*2d*
    /pages/examples/*3d*

Curvilinear examples
--------------------

.. nbgallery::
    :name: curvilinear-example-gallery
    :glob:

    /pages/examples/*disk*
    /pages/examples/*sphere*
    /pages/examples/*shell*
    /pages/examples/*ball*
