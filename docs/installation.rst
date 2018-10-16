Installing Dedalus
******************

Installing the Dedalus Package
==============================

Dedalus is a Python 3 package that includes custom C-extensions (compiled with Cython) and that relies on MPI, FFTW (linked to MPI), HDF5, and a basic scientific-Python stack: numpy, scipy, mpi4py (linked to the same MPI), and h5py.

If you have the necessary C dependencies (MPI, FFTW+MPI, and HDF5), as well as Python 3, you should be able to install Dedalus from PyPI or build it from source.
Otherwise, see one of the alternate sections below for instructions for building the dependency stack.

Installing from PyPI
--------------------

We currently only provide Dedalus on PyPI as a source distribution so that the Cython extensions are properly linked to your FFTW library at build-time.
To install Dedalus from PyPI, first set the ``FFTW_PATH`` environment variable to the prefix paths for FFTW and then install using pip::

    export FFTW_PATH=/path/to/your/fftw_prefix
    pip3 install dedalus

Building from source
--------------------

Alternately, to build the lastest version of Dedalus from source: clone the repository, set the ``FFTW_PATH`` variable, install the build requirements, and install using pip::

    hg clone https://bitbucket.org/dedalus-project/dedalus
    cd dedalus
    export FFTW_PATH=/path/to/your/fftw_prefix
    pip3 install -r requirements.txt
    pip3 install .

Updating Dedalus
----------------

If Dedalus was installed from PyPI, it can be updated using::

    export FFTW_PATH=/path/to/your/fftw_prefix
    pip3 install --upgrade Dedalus

If Dedalus was built from source, it can be updated by first pulling new changes from the source repository, and then reinstalling with pip::

    cd /path/to/dedalus/repo
    hg pull
    hg update
    export FFTW_PATH=/path/to/your/fftw_prefix
    pip3 install -r requirements.txt
    pip3 install --upgrade --force-reinstall .

Uninstalling Dedalus
--------------------

If Dedalus was installed using pip, it can be uninstalled using::

    pip3 uninstall dedalus

Conda Installation
==================

We preliminarily support installation through conda via a custom script that allows you to link against custom MPI/FFTW/HDF5 libraries, or opt for the builds of those packages that are available through conda.

First, install conda/miniconda for your system if you don't already have it, following the `instructions from conda <https://conda.io/docs/user-guide/install/index.html>`_.
Then download the Dedalus conda installation script from `this link <https://raw.githubusercontent.com/DedalusProject/conda_dedalus/master/install_conda.sh>`_ or using::

    wget https://raw.githubusercontent.com/DedalusProject/conda_dedalus/master/install_conda.sh

Modify the options at the top of the script to link against custom MPI/FFTW/HDF5 libraries, choose between OpenBLAS and MKL-based numpy and scipy, and set the name of the resulting conda environment.
Then activate the base conda environment and run the script to build a new conda environment with Dedalus and its dependencies, as requested::

    conda activate base
    bash install_conda.sh

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

Once the dependency stack has been installed, Dedalus can be installed `as described above <#installing-the-dedalus-package>`_.
