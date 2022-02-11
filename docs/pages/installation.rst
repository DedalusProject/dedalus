Installing Dedalus
******************

**Note: please check you're referencing the intended version of the documentation.
This documentation is in reference to v3 of the code, which is currently under beta-release.
Documentation for v2 (the latest on PyPI) can be accessed through the sidebar.**

Dedalus is a Python 3 package that includes custom C-extensions (compiled with Cython) and that relies on MPI, FFTW, HDF5, and a basic scientific-Python stack (numpy, scipy, mpi4py, and h5py).

We recommend using conda to build a Python environment with all the necessary prerequisites, as described in the conda instructions below.
This procedure can be easily customized to link to existing MPI/FFTW/HDF5 libraries, which may be preferable when installing Dedalus on a cluster.

Once you have the necessary C dependencies (MPI, FFTW, and HDF5) and Python 3 environment (with properly linked mpi4py and h5py, in particular), you should be able to install Dedalus using pip.


Conda installation (recommended)
================================

We recommend installing Dedalus via a conda script that will create a new conda environment with a complete Dedalus installation.
The script allows you to link against custom MPI/FFTW/HDF5 libraries or to install builds of those packages from conda.

First, install conda/miniconda for your system if you don't already have it, following the `instructions from conda <https://conda.io/en/latest/miniconda.html>`_.
Then download the Dedalus v3 conda installation script from `this link <https://raw.githubusercontent.com/DedalusProject/dedalus_conda/master/install_dedalus3_conda.sh>`_ or using::

    curl https://raw.githubusercontent.com/DedalusProject/dedalus_conda/master/install_dedalus3_conda.sh --output install_dedalus3_conda.sh

Modify the options at the top of the script to change the name of the resulting conda environment, link against custom MPI/FFTW/HDF5 libraries, choose between OpenBLAS and MKL-based numpy/scipy, and more.
Then activate the base conda environment and run the script to build a new conda environment with Dedalus and its dependencies, as requested::

    conda activate base
    bash install_dedalus3_conda.sh

To use Dedalus, you simply need to activate the new environment. You can test the installation works using the command-line interface::

    conda activate dedalus3
    python3 -m dedalus test

The Dedalus package within the environment can be updated using pip as described below.


Installing the Dedalus package
==============================

Once the necessary C dependencies and Python 3 are present, Dedalus can be installed from PyPI or built from source using pip.

**Note**: the instructions in this section assume the ``pip3`` command is hitting the right Python 3 installation.
You can check this by making sure that ``which pip3`` and ``which python3`` reside in the same location.
If not, use ``python3 -m pip`` instead of ``pip3`` in the following commands.

**Note**: it is strongly recommended that you disable threading, as described on the :doc:`performance_tips` page, when running Dedalus.
This is done automatically when Dedalus is installed using the conda procedure above, but must be done manually otherwise.

Installing from PyPI
--------------------

**Note: Dedalus v3 is currently under beta-release, and not yet available on PyPI.
Please build directly from source as described below.**

.. We currently only provide Dedalus on PyPI as a source distribution so that the Cython extensions are properly linked to your FFTW/MPI libraries at build-time.
.. To install Dedalus from PyPI, first set the ``FFTW_PATH`` and ``MPI_PATH`` environment variables to the prefix paths for FFTW/MPI and then install using pip::

..     export FFTW_PATH=/path/to/your/fftw_prefix
..     export MPI_PATH=/path/to/your/mpi_prefix
..     pip3 install dedalus

Building from source
--------------------

To build and install the most recent version of Dedalus, first set the ``FFTW_PATH`` and ``MPI_PATH`` environment variables to the prefix paths for FFTW and MPI::

    export FFTW_PATH=/path/to/your/fftw_prefix
    export MPI_PATH=/path/to/your/mpi_prefix

You can then install Dedalus directly from GitHub using pip::

    pip3 install --no-cache http://github.com/dedalusproject/dedalus/zipball/d3/

Alternatively, you can clone the d3 branch from the source repository and install::

    git clone -b d3 https://github.com/DedalusProject/dedalus
    cd dedalus
    pip3 install .

Updating Dedalus
----------------

If Dedalus was installed using the conda script or from GitHub with pip, it can also be updated using pip::

    pip3 install --upgrade --no-cache http://github.com/dedalusproject/dedalus/zipball/d3/

If Dedalus was built from a clone of the source repository, first pull new changes and then reinstall with pip::

    cd /path/to/dedalus/repo
    git pull
    pip3 install --upgrade --force-reinstall .

**Note**: any custom FFTW/MPI paths set in the conda script or during the original installation will also need to be exported for the upgrade commands to work.

Uninstalling Dedalus
--------------------

Dedalus can be uninstalled using::

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

