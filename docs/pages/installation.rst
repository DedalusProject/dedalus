Installing Dedalus
******************

**Note: please check you're referencing the intended version of the documentation.
This documentation is in reference to v2 of the code.
The latest documentation for v3 can be accessed through the sidebar.**

Dedalus is a Python 3 package that includes custom C-extensions (compiled with Cython) and that relies on MPI, FFTW, HDF5, and a basic scientific-Python stack (numpy, scipy, mpi4py, and h5py).

The easiest way to install Dedalus and its dependencies is using conda.
The instructions below include options for building the full stack from conda or custom stacks linking to existing MPI/FFTW/HDF5 libraries, which may be preferable when installing on a cluster.
Once you have the necessary C and Python dependencies, you can install and upgrade Dedalus using pip.


Full-stack conda installation (recommended)
===========================================

A full-stack installation of Dedalus v2 is available via conda-forge for Linux and macOS.
This installation route is recommended for laptops, workstations, and some cluster environments.
If linking to the existing MPI libraries on your cluster is recommended, see the alternative "Custom conda installation" instructions below.

#. Install conda on your system if you don't already have it (we recommend the `miniforge variant <https://github.com/conda-forge/miniforge/#download>`_).

#. Create a new environment for your Dedalus installation (here called ``dedalus2``, but you can pick any name) and activate it::

       # Create and activate environment
       conda create -n dedalus2
       conda activate dedalus2

#. *(Apple Silicon only)* There are currently upstream issues in scipy that prevent Dedalus from natively running on arm64 at this time.
   If you are using a Mac with an Apple Silicon processor, you will need to instruct the environment to use x86 packages::

       # Macs with Apple Silicon only!
       conda config --env --set subdir osx-64

#. We strongly recommend disabling threading when running Dedalus for maximum performance on all platforms.
   The conda environment can be configured to do this automatically::

       # Disable threading in the environment (strongly recommended for performance)
       conda env config vars set OMP_NUM_THREADS=1
       conda env config vars set NUMEXPR_MAX_THREADS=1

#. Install Dedalus v2 and all its requirements from the conda-forge channel::

       # Install Dedalus v2 from conda-forge
       conda install -c conda-forge dedalus

To use Dedalus, you simply need to activate the new environment.
You can test the installation using the command-line interface::

    conda activate dedalus2
    python3 -m dedalus test

The Dedalus package within the environment can be updated using pip as described below.


Custom conda installation
=========================

Alternatively, you can use a build script that will create a custom conda environment with all dependencies and then install Dedalus via pip.
This script allows you to optionally link against custom MPI/FFTW/HDF5 libraries, which may provide better performance on some clusters.

#. Install conda on your system if you don't already have it (we recommend the `miniforge variant <https://github.com/conda-forge/miniforge/#download>`_).

#. Download the Dedalus v2 conda installation script from `this link <https://raw.githubusercontent.com/DedalusProject/dedalus_conda/master/conda_install_dedalus2.sh>`_ or using::

       curl https://raw.githubusercontent.com/DedalusProject/dedalus_conda/master/conda_install_dedalus2.sh --output conda_install_dedalus2.sh

#. Modify the options at the top of the script to change the name of the resulting conda environment, link against custom MPI/FFTW/HDF5 libraries, choose between OpenBLAS and MKL-based numpy/scipy, and more.

#. Activate the base conda environment and run the script to build a new conda environment with Dedalus and its dependencies, as requested::

       conda activate base
       bash conda_install_dedalus2.sh

To use Dedalus, you simply need to activate the new environment.
You can test the installation using the command-line interface::

    conda activate dedalus2
    python3 -m dedalus test

The Dedalus package within the environment can be updated using pip as described below.


Installing the Dedalus package
==============================

Once the necessary C dependencies and Python 3 are present, Dedalus can be installed from PyPI or built from source using pip.

**Note**: the instructions in this section assume the ``pip3`` command is hitting the right Python 3 installation.
You can check this by making sure that ``which pip3`` and ``which python3`` reside in the same location.
If not, use ``python3 -m pip`` instead of ``pip3`` in the following commands.

**Note**: it is strongly recommended that you disable threading, as described on the :doc:`performance_tips` page, when running Dedalus.
This is done automatically when Dedalus is installed using the conda build script described above, but must be done manually otherwise.

Installing from PyPI
--------------------

We currently only provide Dedalus on PyPI as a source distribution so that the Cython extensions are properly linked to your FFTW/MPI libraries at build-time.
To install Dedalus from PyPI, first set the ``FFTW_PATH`` and ``MPI_PATH`` environment variables to the prefix paths for FFTW/MPI and then install using pip, ensuring that the C extensions are properly linked to MPI by using ``mpicc``::

    export FFTW_PATH=/path/to/your/fftw_prefix
    export MPI_PATH=/path/to/your/mpi_prefix
    CC=mpicc pip3 install dedalus

Building from source
--------------------

To build and install the most recent version of Dedalus v2, first set the ``MPI_PATH`` and ``FFTW_PATH`` environment variables to your prefix paths for MPI and FFTW::

    export MPI_PATH=/path/to/your/mpi_prefix
    export FFTW_PATH=/path/to/your/fftw_prefix

You can then install Dedalus directly from GitHub using pip, ensuring that the C extensions are properly linked to MPI by using ``mpicc``::

    CC=mpicc pip3 install --no-cache http://github.com/dedalusproject/dedalus/zipball/v2_master/

Alternatively, you can clone the v2_master branch from the source repository and install locally using pip::

    git clone -b v2_master https://github.com/DedalusProject/dedalus
    cd dedalus
    CC=mpicc pip3 install --no-cache .

Updating Dedalus
----------------

If Dedalus was installed using the conda script or from GitHub with pip, it can also be updated using pip::

    CC=mpicc pip3 install --upgrade --force-reinstall --no-deps --no-cache http://github.com/dedalusproject/dedalus/zipball/v2_master/

If Dedalus was built from a clone of the source repository, first pull new changes and then reinstall with pip::

    cd /path/to/dedalus/repo
    git pull
    CC=mpicc pip3 install --upgrade --force-reinstall --no-deps --no-cache .

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
    /machines/nasa_discover/discover
    /machines/nasa_pleiades/pleiades
    /machines/savio/savio
    /machines/stampede/stampede
    /machines/trestles/trestles

