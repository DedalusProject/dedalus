Installing Dedalus
******************

**Note: please check you're referencing the intended version of the documentation.
This documentation is in reference to v2 of the code.
The latest documentation for v3 can be accessed through the sidebar.**

Dedalus is a Python 3 package that includes custom C-extensions (compiled with Cython) and that relies on MPI, FFTW (linked to MPI), HDF5, and a basic scientific-Python stack: numpy, scipy, mpi4py (linked to the same MPI), and h5py.

We recommend using conda to build a Python environment with all the necessary prerequisites, as described in the conda instructions below.
This procedure can be easily customized to link to existing MPI/FFTW/HDF5 libraries, which may be preferable when installing Dedalus on a cluster.

Once you have the necessary C dependencies (MPI, FFTW+MPI, and HDF5), as well as Python 3, you should be able to install Dedalus from PyPI or build it from source.


Conda installation (recommended)
================================

We recommend installing Dedalus via a conda script that will create a new conda environment with a complete Dedalus installation.
The script allows you to link against custom MPI/FFTW/HDF5 libraries or opt for builds of those packages that are available through conda.

First, install conda/miniconda for your system if you don't already have it, following the `instructions from conda <https://conda.io/en/latest/miniconda.html>`_.
Then download the Dedalus conda installation script from `this link <https://raw.githubusercontent.com/DedalusProject/dedalus_conda/master/install_conda.sh>`_ or using::

    curl https://raw.githubusercontent.com/DedalusProject/dedalus_conda/master/install_conda.sh --output install_conda.sh

Modify the options at the top of the script to change the name of the resulting conda environment, link against custom MPI/FFTW/HDF5 libraries, choose between OpenBLAS and MKL-based numpy/scipy, and more.
Then activate the base conda environment and run the script to build a new conda environment with Dedalus and its dependencies, as requested::

    conda activate base
    bash install_conda.sh

To use Dedalus, you simply need to activate the new environment. You can test the installation works using the command-line interface::

    conda activate dedalus
    python3 -m dedalus test

The Dedalus package within the environment can be updated using pip as described below.


Installing the Dedalus package
==============================

Once the necessary C dependencies and Python 3 are present, Dedalus can be installed from PyPI or built from source using pip.

**Note**: the instructions in this section assume the ``pip3`` command is hitting the right Python 3 installation.
You can check this by making sure that ``which pip3`` and ``which python3`` reside in the same location.
If not, you may need to use ``pip`` or ``python3 -m pip`` instead of ``pip3`` in the following commands.

Installing from PyPI
--------------------

We currently only provide Dedalus on PyPI as a source distribution so that the Cython extensions are properly linked to your FFTW/MPI libraries at build-time.
To install Dedalus from PyPI, first set the ``FFTW_PATH`` and ``MPI_PATH`` environment variables to the prefix paths for FFTW/MPI and then install using pip::

    export FFTW_PATH=/path/to/your/fftw_prefix
    export MPI_PATH=/path/to/your/mpi_prefix
    pip3 install dedalus

Building from source
--------------------

Alternately, to build the lastest version of Dedalus from source: clone the repository, set FFTW/MPI paths, and install using pip::

    git clone https://github.com/DedalusProject/dedalus
    cd dedalus
    export FFTW_PATH=/path/to/your/fftw_prefix
    export MPI_PATH=/path/to/your/mpi_prefix

    pip3 install .

Updating Dedalus
----------------

If Dedalus was installed using the conda script or manually from PyPI, it can be updated pip::

    pip3 install --upgrade dedalus

**Note**: any custom FFTW/MPI paths set in the conda script or during the original installation will also need to be exported for the update command to work.

If Dedalus was built from source, it can be updated by first pulling new changes from the source repository, and then reinstalling with pip::

    cd /path/to/dedalus/repo
    git pull
    export FFTW_PATH=/path/to/your/fftw_prefix
    export MPI_PATH=/path/to/your/mpi_prefix
    pip3 install --upgrade --force-reinstall .

Uninstalling Dedalus
--------------------

If Dedalus was installed using pip, it can be uninstalled using::

    pip3 uninstall dedalus


Alternative installation procedures
===================================

**Note**: We strongly recommend installing Dedalus using conda, as described above.
These alternative procedures may be out-of-date and are provided for historical reference and expert use.

.. toctree::
    :maxdepth: 1

    /machines/shell_script/shell_script
    /machines/bridges/bridges
    /machines/computecanada/computecanada
    /machines/engaging/engaging
    /machines/janus/janus
    /machines/mac_os/mac_os
    /machines/nasa_discover/discover
    /machines/nasa_pleiades/pleiades
    /machines/savio/savio
    /machines/stampede/stampede
    /machines/trestles/trestles

