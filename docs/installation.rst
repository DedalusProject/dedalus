Installing Dedalus
******************

Installing the Dedalus Package
==============================

Dedalus is a Python 3 package that includes custom C-extensions (compiled with Cython) and that relies on MPI, FFTW (linked to MPI), HDF5, and a basic scientific-Python stack (numpy, scipy, mpi4py, and h5py).

If you have the necessary C dependencies (MPI, FFTW+MPI, and HDF5), as well as Python 3, you should be able to install Dedalus from PyPI or build it from source.
Otherwise, see one of the alternate sections below for instructions for building the dependency stack.

We currently only provide Dedalus on PyPI as a source distribution so that the cython extensions are properly linked to your MPI and FFTW libraries at build-time.
To install Dedalus from PyPI, set the ``FFTW_PATH`` and ``MPI_PATH`` environment variables to the prefix paths for those libraries, and install Dedalus with pip::

    export FFTW_PATH=/path/to/your/fftw_prefix
    export MPI_PATH=/path/to/your/mpi_prefix
    python3 -m pip install --pre --extra-index-url https://testpypi.python.org/pypi dedalus

Alternately, to build the lastest version of Dedalus from source, clone the repository from `<https://bitbucket.org/dedalus-project/dedalus>`_, set the path variables, install the build requirements, and run setup.py::

    export FFTW_PATH=/path/to/your/fftw_prefix
    export MPI_PATH=/path/to/your/mpi_prefix
    python3 -m pip install -r requirements.txt
    python3 setup.py install

Dedalus can then be uninstalled simply using::

    python3 -m pip uninstall dedalus

Conda Installation
==================

We preliminarily support installation through conda if you do not require linking against custom FFTW/MPI/HDF5 libraries, and are happy with builds of those packages that are available through conda.

First, install conda/miniconda for your system if you don't already have it, following the `instructions from conda <https://conda.io/docs/user-guide/install/index.html>`_.
Then download the Dedalus conda-environment file from `this link <https://raw.githubusercontent.com/DedalusProject/conda_dedalus/master/env-dedalus.yaml>`_ or using::

    wget https://raw.githubusercontent.com/DedalusProject/conda_dedalus/master/env-dedalus.yaml

Create a new conda environment from this file using::

    conda env create -n dedalus -f env-dedalus.yaml

Once you activate the environment, you should have all the necessary requirements to build Dedalus from source.
Full conda recipes for Dedalus are still in-progress, but for now you should be able to install Dedalus from PyPI using pip inside your environment::

    conda activate dedalus
    python3 -m pip install --pre --extra-index-url https://testpypi.python.org/pypi dedalus

Installation Script
===================

Dedalus provides an all-in-one installation script that will build an isolated stack containing a Python installation and the other dependencies needed to run Dedalus.
In most cases, the script can be modified to link with system installations of FFTW, MPI, and linear algebra libraries.

You can get the installation script from `this link <https://bitbucket.org/dedalus-project/dedalus/raw/tip/docs/install.sh>`_, or download it using::

    wget https://bitbucket.org/dedalus-project/dedalus/raw/tip/docs/install.sh

and execute it using::

    bash install.sh

The installation script has been tested on a number of Linux distributions and OS X.
If you run into trouble using the script, please get in touch on the `user list <https://groups.google.com/forum/#!forum/dedalus-users>`_.

Manual Installation
===================

Dependencies
------------

Dedalus primarily relies on the basic components of a scientific Python stack using Python 3.
Below are instructions for building the dependency stack on a variety of machines and operating systems:

.. toctree::
    :maxdepth: 1

    machines/mac_os/mac_os
    machines/stampede/stampede
    machines/nasa_pleiades/pleiades
    machines/nasa_discover/discover
    machines/bridges/bridges
    machines/trestles/trestles
    machines/janus/janus
    machines/savio/savio
    machines/engaging/engaging

Dedalus Package
---------------

Dedalus is distributed using the mercurial version control system, and hosted on Bitbucket.
To install Dedalus itself, first install `mercurial <http://mercurial.selenic.com>`_, and then clone the main repository using::

    hg clone https://bitbucket.org/dedalus-project/dedalus

Move into the newly cloned repository, and use pip to install any remaining Python dependencies with the command::

    pip3 install -r requirements.txt

To help Dedalus find the proper libraries, it may be necessary to set the ``FFTW_PATH`` and ``MPI_PATH`` environment variables (see system-specific documentation).
Dedalus's C-extensions can then be built in-place using::

    python3 setup.py build_ext --inplace

Finally, add the repository directory to your ``PYTHONPATH`` environment variable to ensure that the ``dedalus`` package within can be found by the Python interpreter.

Updating Dedalus
================

To update your installation of Dedalus, move into the repository directory (located at ``src/dedalus`` within the installation script's build, or where you manually cloned it) and issue the mercurial update commands::

    hg pull
    hg update

Then rerun the pip requirements installation and python build, in case the dependencies or C-extensions have changes::

    pip3 install -r requirements.txt
    python3 setup.py build_ext --inplace

Dedalus should be updated.
