Building with intel compiler stack
***************************************************************************
Install notes for building our python3 stack on TACC/Stampede, using the intel compiler suite.  
Many thanks to Yaakoub El Khamra at TACC for help in sorting out the python3 build and numpy linking against a fast MKL BLAS.

.. note::
   
     Just got a direct call from Yaakoub.  Very, very helpful.  Here's
     the quick rundown.

     He got _ctypes to work by editing the following file:

          vim /work/00364/tg456434/yye00/src/Python-3.3.3/Modules/_ctypes/libffi/src/x86/ffi64.c

     Do build with intel 14
     use mvapich2/2.0b
     Will need to do our own build of fftw3

     set mpicc as c compiler rather than icc, same for CXX, FC and
     others, when configuring python.  should help with mpi4py.

     in mpi4py, can edit mpi.cfg (non-pip install).

     Keep Yaakoub updated with direct e-mail on progress.

     Also, Yaakoub is spear-heading TACCs efforts in doing 
     auto-offload to Xenon Phi.
    

     Beware of disk quotas if you're trying many builds; I hit 5GB
     pretty fast and blew my matplotlib install due to quota limits :)

     

Modules
==========================================

Here is my current build environment (from running ``module list``)

    1) TACC-paths   
    2) Linux   
    3) cluster-paths   
    4) TACC   
    5) cluster
    6) intel/14.0.1.106   
    7) mvapich2/2.0b

.. note ::
    To get here from a gcc default do the following:

    module unload mkl
    module swap gcc intel/14.0.1.106

In the ``intel`` compiler stack, we need to use ``mvapich2/2.0b``,
which then implies ``intel/14.0.1.106``.  Right now, TACC has not built
``fftw3`` for this stack, so we'll be doing our own FFTW build.

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

    cp -p  /work/00364/tg456434/yye00/src/Python-3.3.3/Modules/_ctypes/libffi/src/x86/ffi64.c Modules/_ctypes/libffi/src/x86/ffi64.c 

    ./configure --prefix=$HOME/build_intel \
                         CC=icc CFLAGS="-mkl -O3 -xHost -fPIC -ipo" \
                         CXX=icpc CPPFLAGS="-mkl -O3 -xHost -fPIC -ipo" \
                         F90=ifort F90FLAGS="-mkl -O3 -xHost -fPIC -ipo" \
                         --enable-shared LDFLAGS="-lpthread" \
                         --with-cxx-main=icpc --with-system-ffi

    make
    make install

.. note::

     With help from Yaakoub, we now build ``_ctypes`` successfully.
     Key is the ffi64.c replacement above.  I'll wrap this into a
     crude, deployable tar file patch.

     Also, the mpicc build is much, much slower than icc.  Interesting.
     And we crashed out.  Here's what we tried with mpicc::

        ./configure --prefix=$HOME/build_intel \
                         CC=mpicc CFLAGS="-mkl -O3 -xHost -fPIC -ipo" \
                         CXX=mpicxx CPPFLAGS="-mkl -O3 -xHost -fPIC -ipo" \
                         F90=mpif90 F90FLAGS="-mkl -O3 -xHost -fPIC -ipo" \
                         --enable-shared LDFLAGS="-lpthread" \
                         --with-cxx-main=mpicxx --with-system-ffi


Here we are building everything in ``~/build_intel``; you can do it
whereever, but adjust things appropriately in the above instructions.
The build proceeeds quickly (few minutes).

Installing FFTW3
------------------------------

We need to build our own FFTW3, under intel 14 and mvapich2/2.0b::

    wget http://www.fftw.org/fftw-3.3.3.tar.gz
    tar -xzf fftw-3.3.3.tar.gz
    cd fftw-3.3.3

    ./configure --prefix=$HOME/build_intel \
                         CC=icc CFLAGS="-mkl -O3 -xHost -fPIC" \
                         CXX=icpc CPPFLAGS="-mkl -O3 -xHost -fPIC" \
                         F77=ifort FFLAGS="-mkl -O3 -xHost -fPIC" \
                         MPICC=mpicc MPICXX=mpicxx \
                         --enable-shared LDFLAGS="-lpthread" \
                         --enable-mpi

    make
    make install

Last one works!

Also:
`Intel docs <http://software.intel.com/en-us/articles/performance-tools-for-software-developers-building-fftw-with-the-intel-compilers>`_

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

For now, skip the venv process.

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

     export BUILD_HOME=$HOME/build_intel
     export FFTW_PATH=$BUILD_HOME
     export MPI_PATH=$MPICH_HOME

Then change into your root dedalus directory and run::

     python setup.py build_ext --inplace

Our new stack (``intel/14``, ``mvapich2/2.0b``) builds to completion
without the ``mpi.h`` import/conflict error that I was seeing under
``mvapich2/1.9``.  However, it looks like I've got a potential problem
in my fftw build (see below).

Current error::

    login2$ python3 bessel_disk_test.py 
    2014-01-24 16:58:25,198 Dedalus2 0/1 ERROR   : Don't forget to buid using 'python3 setup.py build_ext --inplace'
    Traceback (most recent call last):
      File "bessel_disk_test.py", line 4, in <module>
        import special_functions
      File "/home1/00364/tg456434/code/examples2-bpbrown/dev/bessel_disk/special_functions.py", line 1, in <module>
        import dedalus2.public as d2
      File "/home1/00364/tg456434/build_intel/dedalus2/dedalus2/public.py", line 7, in <module>
        from .data.domain import Domain
      File "/home1/00364/tg456434/build_intel/dedalus2/dedalus2/data/domain.py", line 8, in <module>
        from .distributor import Distributor
      File "/home1/00364/tg456434/build_intel/dedalus2/dedalus2/data/distributor.py", line 12, in <module>
        from ..libraries.fftw import fftw_wrappers as fftw
    ImportError: /home1/00364/tg456434/build_intel/lib/libfftw3_mpi.so.3: undefined symbol: MPI_Bcast
    login2$ 


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



All I want for christmas is suitesparse
----------------------------------------

Well, maybe :)  Let's give it a try, and lets grab the whole library::

     wget http://www.cise.ufl.edu/research/sparse/SuiteSparse/current/SuiteSparse.tar.gz
     tar xvf SuiteSparse.tar.gz

     <edit SuiteSparse_config/SuiteSparse_config.mk>
     
