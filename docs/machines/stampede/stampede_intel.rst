Building with intel/impi stack
***************************************************************************
Install notes for building our python3 stack on TACC/Stampede, using the intel compiler suite.  
Many thanks to Yaakoub El Khamra at TACC for help in sorting out the python3 build and numpy linking against a fast MKL BLAS.

Modules
==========================================

Here is my current build environment (from running ``module list``)

  1) TACC-paths   
  2) Linux   
  3) cluster-paths   
  4) TACC   
  5) cluster   
  6) intel/13.0.2.146   
  7) impi/4.1.0.030   
  8) fftw3/3.3.2

.. note ::
    To get here from a gcc default do the following:

    module unload mkl
    module swap gcc intel
    module swap mvapich2 impi

In the ``intel`` compiler stack, we can use either ``mvapich2`` or ``impi``, and ``fftw3`` 
is built on both of these.
See the  `Stampede user guide <https://www.tacc.utexas.edu/user-services/user-guides/stampede-user-guide#compenv-modules-login>`_
for more details.  If you would like to always auto-load the same
modules at startup, build your desired module configuration and then
run::

     module save

Python stack
=========================

Building Python3
--------------------------

Create ``~\build_intel`` and then proceed with downloading and installing Python-3.3::

    cd ~/build_intel
    wget http://www.python.org/ftp/python/3.3.3/Python-3.3.3.tgz
    tar -xzf Python-3.3.3.tgz
    cd Python-3.3.3

    ./configure --prefix=$HOME/build_ifort CC=icc CFLAGS="-mkl -O3 -xHost -fPIC -ipo" CPPFLAGS="-mkl -O3 -xHost -fPIC -ipo" CXX=icpc --enable-shared --with-cxx-main=icpc LDFLAGS="-lpthread" --with-system-ffi

    make
    make install

On ``make``, we're getting one important error::

    icc -fPIC -Wno-unused-result -DNDEBUG -g -O3 -Wall -Wstrict-prototypes -mkl -O3 -xHost -fPIC -ipo -Ibuild/temp.linux-x86_64-3.3/libffi/include -Ibuild/temp.linux-x86_64-3.3/libffi -I/home1/00364/tg456434/build_ifort/Python-3.3.3/Modules/_ctypes/libffi/src -I./Include -I/home1/00364/tg456434/build_ifort/include -I. -IInclude -I/usr/local/include -I/home1/00364/tg456434/build_ifort/Python-3.3.3/Include -I/home1/00364/tg456434/build_ifort/Python-3.3.3 -c /home1/00364/tg456434/build_ifort/Python-3.3.3/Modules/_ctypes/libffi/src/x86/ffi64.c -o build/temp.linux-x86_64-3.3/home1/00364/tg456434/build_ifort/Python-3.3.3/Modules/_ctypes/libffi/src/x86/ffi64.o -Wall -fexceptions
    icc: command line warning #10006: ignoring unknown option '-Wno-unused-result'
    /home1/00364/tg456434/build_ifort/Python-3.3.3/Modules/_ctypes/libffi/src/x86/ffi64.c(56): error: identifier "__m128" is undefined
        UINT128 i128;
        ^

    compilation aborted for /home1/00364/tg456434/build_ifort/Python-3.3.3/Modules/_ctypes/libffi/src/x86/ffi64.c (code 2)

    Failed to build these modules:
    _ctypes                                               

    running build_scripts

Here we are building everything in ``~/build_intel``; you can do it
whereever, but adjust things appropriately in the above instructions.
The build proceeeds quickly (few minutes).

Updating shell settings
------------------------------

At this point, ``python3`` is installed in ``~/build_intel/bin/``.  Add this
to your path and confirm (currently there is no ``python3`` in the
default path, so doing a ``which python3`` will fail if you haven't
added ``~/build_intel/bin``).  

On Stampede, login shells (interactive connections via ssh) source
only ``~/.bash_profile``, ``~/.bash_login`` or ``~/.profile``, in that
order, and do not source ``~/.bashrc``.  Meanwhile non-login shells
only launch ``~/.bashrc`` 
(see Stampede `user guide <https://www.tacc.utexas.edu/user-services/user-guides/stampede-user-guide#compenv-startup-technical>`_).

In the bash shell, add the following to
``.bashrc``::

     export PATH=~/build_intel/bin:$PATH
     export LD_LIBRARY_PATH=~/build_intel/lib:$LD_LIBRARY_PATH

and the following to ``.profile``::

     if [ -f ~/.bashrc ]; then . ~/.bashrc; fi

(from `bash reference manual <https://www.gnu.org/software/bash/manual/html_node/Bash-Startup-Files.html>`_) 
to obtain the same behaviour in both shell types.

Installing pip
-------------------------

We'll use ``pip`` to install our python library depdencies.
Instructions on doing this are `available here <http://www.pip-installer.org/en/latest/installing.html>`_ 
and summarized below.  First
download and install setup tools::

    cd ~/build
    wget https://bitbucket.org/pypa/setuptools/raw/bootstrap/ez_setup.py
    python3 ez_setup.py

Then install ``pip``::

    wget https://raw.github.com/pypa/pip/master/contrib/get-pip.py
    python3 get-pip.py

You will now have ``pip3`` and ``pip`` installed in ``~/build/bin``.
You might try doing ``pip -V`` to confirm that ``pip`` is built
against python 3.3.  We will use ``pip3`` throughout this
documentation to remain compatible with systems (e.g., Mac OS) where
multiple versions of python coexist.

Installing nose
-------------------------

Nose is useful for unit testing, especially in checking our numpy build::

    pip3 install nose



Installing virtualenv
-------------------------

In order to test multiple numpys and scipys (and really, their
underlying BLAS libraries), we will use ``virtualenv``::

     pip3 install virtualenv

Next, construct a virtualenv to hold all of your python modules. We
suggest doing this in your home directory::

     mkdir ~/venv




Numpy and BLAS libraries
======================================

Numpy will be built against a specific BLAS library.  Detailed
instructions appear below for both MKL and OpenBlas.  
Follow these and then return to this document.

MKL
--------------------------

.. toctree::
    :maxdepth: 1
    
    stampede_intel_mkl




Python library stack
=====================

After ``numpy`` has been built (see links above) 
we will proceed with the rest of our python stack.
Right now, all of these need to be installed in each existing
virtualenv instance (e.g., ``openblas``, ``mkl``, etc.).  

Installing Scipy
-------------------------

Scipy is easier, because it just gets its config from numpy.  Download
an install in your appropriate ``~/venv/INSTANCE`` directory::

     wget http://sourceforge.net/projects/scipy/files/scipy/0.13.2/scipy-0.13.2.tar.gz
     tar -xvf scipy-0.13.2.tar.gz
     cd scipy-0.13.2

Then run ::

    python3 setup.py config --compiler=intelem --fcompiler=intelem build_clib \
                                            --compiler=intelem --fcompiler=intelem build_ext \
                                            --compiler=intelem --fcompiler=intelem install


Installing mpi4py
-------------------------

This should just be pip installed::

      pip3 install -v http://mpi4py.googlecode.com/files/mpi4py-1.3.1.tar.gz

.. note::
    
      If we use use ::

           pip3 install mpi4py
           
      then stampede tries to pull version 0.6.0 of mpi4py.  Hence the
      explicit version pull above.

Installing cython
-------------------------

This should just be pip installed::

     pip3 install cython


Installing matplotlib
-------------------------

This should just be pip installed::


     pip3 install -v https://downloads.sourceforge.net/project/matplotlib/matplotlib/matplotlib-1.3.1/matplotlib-1.3.1.tar.gz

.. note::

      If we use use ::

           pip3 install matplotlib
           
      then stampede tries to pull version 1.1.1 of matplotlib.  Hence the
      explicit version pull above.



Dedalus2
========================================

With the modules set as above, set::

     export FFTW_PATH=$TACC_FFTW3_DIR
     export MPI_PATH=$MPICH_HOME

Then change into your root dedalus directory and run::

     python setup.py build_ext --inplace

Running Dedalus on Stampede
========================================

Source the appropriate virtualenv::

     source ~/venv/openblas/bin/activate

or::

     source ~/venv/mkl/bin/activate


grab an interactive dev node with ``idev``.  Play.





Skipped libraries
==============================

Installing freetype2
--------------------------

Freetype is necessary for matplotlib ::

     cd ~/build
     wget http://sourceforge.net/projects/freetype/files/freetype2/2.5.2/freetype-2.5.2.tar.gz
     tar -xvf freetype-2.5.2.tar.gz 
     cd freetype-2.5.2
     ./configure --prefix=$HOME/build
     make
     make install

.. note::
     Skipping for now

Installing libpng
--------------------------

May need this for matplotlib?::

     cd ~/build
     wget http://prdownloads.sourceforge.net/libpng/libpng-1.6.8.tar.gz
     ./configure --prefix=$HOME/build
     make
     make install

.. note::
     Skipping for now

UMFPACK
-------

We may wish to deploy UMFPACK for sparse matrix solves.  Keaton is
starting to look at this now.  If we do, both numpy and scipy will
require UMFPACK, so we should build it before proceeding with those builds.

UMFPACK requires AMD (another package by the same group, not processor) and SuiteSparse_config, too.

If we need UMFPACK, we
can try installing it from ``suite-sparse`` as in the Mac install.
Here are links to `UMFPACK docs <http://www.cise.ufl.edu/research/sparse/umfpack/>`_ 
and `Suite-sparse <http://www.cise.ufl.edu/research/sparse/>`_

.. note::
     We'll check and update this later. (1/9/14)
