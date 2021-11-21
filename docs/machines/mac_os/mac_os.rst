Install notes for Mac OS X (10.9)
*******************************************

These instructions assume you're starting with a clean Mac OS X system,
which will need ``python3`` and all scientific packages installed.

Mac OS X cookbook
-----------------

::

    #!bash

    # Homebrew
    ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
    brew update
    brew doctor
    # ** Fix any errors raised by brew doctor before proceeding **

    # Prep system
    brew install gcc
    brew install swig

    # Python 3
    brew install python3

    # Scientific packages for Python 3
    brew tap homebrew/science
    brew install suite-sparse
    pip3 install nose
    pip3 install numpy
    pip3 install scipy
    brew install libpng
    brew install freetype
    pip3 install matplotlib

    # MPI
    brew install openmpi
    pip3 install mpi4py

    # FFTW
    brew install fftw --with-mpi

    # HDF5
    brew install hdf5
    pip3 install h5py

    # Dedalus
    # ** Change to the directory where you want to keep the Dedalus repository **
    brew install hg
    hg clone https://bitbucket.org/dedalus-project/dedalus
    cd dedalus
    pip3 install -r requirements.txt
    python3 setup.py build_ext --inplace

Detailed install notes for Mac OS X (10.9)
==========================================

Preparing a Mac system
----------------------

First, install Xcode from the App Store and seperately install the Xcode Command Line
Tools. To install the command line tools, open Xcode, go to
``Preferences``, select the ``Downloads`` tab and ``Components``. These
command line tools install ``make`` and other requisite tools that are
no longer automatically included in Mac OS X (as of 10.8).

Next, you should install the `Homebrew`_ package manager for OS X. Run the
following from the Terminal:

.. _Homebrew: http://brew.sh/

::

    #!bash
    ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
    brew update
    brew doctor

Cleanup any problems identified by ``brew doctor`` before proceeding.

To complete the ``scipy`` install process, we'll need ``gfortran`` from ``gcc``
and ``swig``, which you can install from Homebrew:

::

    #!bash
    brew install gcc
    brew install swig

Install Python 3
----------------

Now, install python3 from Homebrew:

::

    #!bash
    brew install python3


Scientific packages for Python3
-------------------------------

Next install the numpy and scipy scientific packages. To adequately warn
you before proceeding, properly installing numpy and scipy on a Mac can
be a frustrating experience.

Start by proactively installing UMFPACK from suite-sparse, located in
homebrew-science on https://github.com/Homebrew/homebrew-science.
Failing to do this may lead to a series of perplexing UMFPACK errors
during the scipy install.

::

    #!bash
    brew tap homebrew/science
    brew install suite-sparse

Now use pip, the (the standard Python package management system, installed with
Python via Homebrew) to install ``nose``, ``numpy``, and ``scipy``
in order:

::

    #!bash
    pip3 install nose
    pip3 install numpy
    pip3 install scipy

The ``scipy`` install can fail in a number of surprising ways. Be
especially wary of custom settings to ``LDFLAGS``, ``CPPFLAGS``, etc.
within your shell; these may cause the ``gfortran`` compile step to fail
spectacularly.

Also install ``matplotlib``, the main Python plotting library, along with its
dependencies, using Homebrew and pip:

::

    #!bash
    brew install libpng
    brew install freetype
    pip3 install matplotlib

Other libraries
---------------

Dedalus is parallelized using MPI, and we recommend using the Open MPI library
on OS X.  The Open MPI library and Python wrappers can be installed using
Homebrew and pip:

::

    #!bash
    brew install openmpi
    pip3 install mpi4py

Dedalus uses the FFTW library for transforms and parallelized transposes, and
can be installed using Homebrew:

::

    #!bash
    brew install fftw --with-mpi

Dedalus uses HDF5 for data storage.  The HDF5 library and Python wrappers can be
installed using Homebrew and pip:

::

    #!bash
    brew install hdf5
    pip3 install h5py

Installing the Dedalus package
------------------------------

Dedalus is managed using the Mercurial distributed version control system, and
hosted online though Bitbucket.  Mercurial (referred to as ``hg``) can be
installed using homebrew, and can then be used to download the latest copy of
Dedalus (note: you should change to the directory where you want the put the
Dedalus repository):

::

    #!bash
    brew install hg
    hg clone https://bitbucket.org/dedalus-project/dedalus
    cd dedalus

A few other Python packages needed by Dedalus are listed in the
``requirements.txt`` file in the Dedalus repository, and can be installed using
pip:

::

    #!bash
    pip3 install -r requirements.txt

You then need to build Dedalus's Cython extensions from within the repository
using the ``setup.py`` script.  This step should be perfomed whenever updates
are pulled from the main repository (but it is only strictly necessary when the
Cython extensions are modified).

::

    #!bash
    python3 setup.py build_ext --inplace

Finally, you need to add the Dedalus repository to the Python search path so
that the ``dedalus`` package can be imported.  To do this, add the following
to your ``~/.bash_profile``, substituting in the path to the Dedalus repository
you cloned using Mercurial:

::

    # Add Dedalus repository to Python search path
    export PYTHONPATH=<PATH/TO/DEDALUS/REPOSITORY>:$PYTHONPATH

Other resources
---------------

http://www.lowindata.com/2013/installing-scientific-python-on-mac-os-x/

http://stackoverflow.com/questions/12574604/scipy-install-on-mountain-lion-failing

https://github.com/jonathansick/dotfiles/wiki/Notes-for-Mac-OS-X
