Install notes for NASA/Pleiades
***************************************************************************

An initial Pleiades environment is pretty bare-bones.  There are no
modules, and your shell is likely a csh varient.  To switch to bash::

    chsh -s /bin/bash

Then add the following to your ``.profile``::

  # Add your commands here to extend your PATH, etc.

  module load comp-intel/2013.5.192
  module load mpi-sgi/mpt.2.08r7

  PATH=$PATH:$HOME/bin:$HOME/build/bin	# Add private commands to PATH

  export BUILD_HOME=$HOME/build
  export LD_LIBRARY_PATH=$BUILD_HOME/lib:$LD_LIBRARY_PATH



Python stack
=========================

Building Python3
--------------------------

Create ``$BUILD_HOME`` and then proceed with downloading and installing Python-3.3::

    cd $BUILD_HOME
    wget http://www.python.org/ftp/python/3.3.3/Python-3.3.3.tgz
    tar -xzf Python-3.3.3.tgz
    cd Python-3.3.3

    # make sure you have the python patch, put it in Python-3.3.3
    tar xvf python_intel_patch.tar 

    ./configure --prefix=$BUILD_HOME \
                         CC=icc CFLAGS="-mkl -O3 -xHost -fPIC -ipo" \
                         CXX=icpc CPPFLAGS="-mkl -O3 -xHost -fPIC -ipo" \
                         F90=ifort F90FLAGS="-mkl -O3 -xHost -fPIC -ipo" \
                         --enable-shared LDFLAGS="-lpthread" \
                         --with-cxx-main=icpc --with-system-ffi

    make
    make install

To successfully build ``python3``, 
the key is replacing the file ``ffi64.c``, which is done
automatically by downloading and unpacking this crude patch
:download:`python_intel_patch.tar<python_intel_patch.tar>` in
your ``Python-3.3.3`` directory.   Unpack it in ``Python-3.3.3``
(``tar xvf python_intel_patch.tar`` line above) 
and it will overwrite ``ffi64.c``.  If you forget to do this, you'll
see a warning/error that ``_ctypes`` couldn't be built.  This is
important.  This patch is identical to the patch on stampede.

.. note::
     We're getting a problem on ``_curses_panel``; ignoring for now.

Installing FFTW3
------------------------------

We need to build our own FFTW3, under intel 14 and mvapich2/2.0b::

    wget http://www.fftw.org/fftw-3.3.3.tar.gz
    tar -xzf fftw-3.3.3.tar.gz
    cd fftw-3.3.3

   ./configure --prefix=$BUILD_HOME \
                         CC=mpicc \
                         CXX=mpicxx \
                         F77=mpif90 \
                         MPICC=mpicc MPICXX=mpicxx \
                         --enable-shared \
                         --enable-mpi --enable-openmp --enable-threads
    make
    make install

It's critical that you use ``mpicc`` as the C-compiler, etc.
Otherwise the libmpich libraries are not being correctly linked into
``libfftw3_mpi.so`` and dedalus failes on fftw import.




Updating shell settings
------------------------------

At this point, ``python3`` is installed in ``$BUILD_HOME/bin/``.  Add this
to your path and confirm (currently there is no ``python3`` in the
default path, so doing a ``which python3`` will fail if you haven't
added ``$BUILD_HOME/bin``).  

On Stampede, login shells (interactive connections via ssh) source
only ``~/.bash_profile``, ``~/.bash_login`` or ``~/.profile``, in that
order, and do not source ``~/.bashrc``.  Meanwhile non-login shells
only launch ``~/.bashrc`` 
(see Stampede `user guide <https://www.tacc.utexas.edu/user-services/user-guides/stampede-user-guide#compenv-startup-technical>`_).

In the bash shell, add the following to
``.bashrc``::

     export PATH=$BUILD_HOME/bin:$PATH
     export LD_LIBRARY_PATH=$BUILD_HOME/lib:$LD_LIBRARY_PATH

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

    cd $BUILD_HOME
    wget https://bitbucket.org/pypa/setuptools/raw/bootstrap/ez_setup.py
    python3 ez_setup.py

Then install ``pip``::

    wget https://raw.github.com/pypa/pip/master/contrib/get-pip.py
    python3 get-pip.py --cert /etc/ssl/certs/ca-bundle.crt

Now edit ``.pip/pip.conf``::

     [global]
     cert = /etc/ssl/certs/ca-bundle.crt

You will now have ``pip3`` and ``pip`` installed in ``$BUILD_HOME/bin``.
You might try doing ``pip -V`` to confirm that ``pip`` is built
against python 3.3.  We will use ``pip3`` throughout this
documentation to remain compatible with systems (e.g., Mac OS) where
multiple versions of python coexist.

Installing nose
-------------------------

Nose is useful for unit testing, especially in checking our numpy build::

    pip3 install nose



Numpy and BLAS libraries
======================================

Numpy will be built against a specific BLAS library.  On Pleiades we
will build against the Intel MKL libraries.  For now we'll do the
build directly in ``$HOME_BUILD`` rather than using virtualenvs.


Building numpy against MKL
----------------------------------

Now, acquire ``numpy`` (1.8.0)::

     cd ~/venv/mkl
     wget http://sourceforge.net/projects/numpy/files/NumPy/1.8.0/numpy-1.8.0.tar.gz
     tar -xvf numpy-1.8.0.tar.gz
     cd numpy-1.8.0

We'll now need to make sure that ``numpy`` is building against the MKL
libraries.  

Create ``site.cfg`` with information for the MKL
library directory so that it correctly point to NASA's
``$MKLROOT/lib/intel64/``.  
With the modules loaded above, this looks like::

     [mkl]
     library_dirs = /nasa/intel/Compiler/2013.5.192/composer_xe_2013.5.192/mkl/lib/intel64/
     include_dirs = /nasa/intel/Compiler/2013.5.192/composer_xe_2013.5.192/mkl/include
     mkl_libs = mkl_rt
     lapack_libs =

.. note:: 
     we should roll a ``$MKLROOT`` into these and distribute this as
     part of the patch.
 
These are based on intels instructions for 
`compiling numpy with ifort <http://software.intel.com/en-us/articles/numpyscipy-with-intel-mkl>`_
and they seem to work so far.

Further following those instructions, you'll need to hand edit two
files in ``numpy/distutils``; these are ``intelccompiler.py`` and
``fcompiler/intel.py``.  I've built a crude patch,
:download:`numpy_intel_patch.tar<numpy_intel_patch.tar>` 
which can be auto-deployed by within the ``numpy-1.8.0`` directory by
doing::
    
      tar -xvf numpy_intel_patch.tar

This will unpack and overwrite::

      numpy/distutils/intelccompiler.py
      numpy/distutils/fcompiler/intel.py

Then proceed with::

    python3 setup.py config --compiler=intelem build_clib --compiler=intelem build_ext --compiler=intelem install

This will config, build and install numpy.


Test numpy install
------------------------------

Test that things worked with this executable script
:download:`numpy_test_full<numpy_test_full>`, 
or do so manually by launching ``python3`` 
and then doing::

     import numpy as np
     np.__config__.show()

If you've installed ``nose`` (with ``pip3 install nose``), 
we can further test our numpy build with::

     np.test()
     np.test('full')

We fail ``np.test()`` with two failures, while ``np.test('full')`` has
3 failures and 19 errors.  But we do successfully link against the
fast BLAS libraries (look for ``FAST BLAS`` output, and fast dot
product time).

.. note::
     We should check what impact these failed tests have on our results.




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


Installing cython
-------------------------

This should just be pip installed::

     pip3 install -v https://pypi.python.org/packages/source/C/Cython/Cython-0.20.tar.gz

The Feb 11, 2014 update to cython (0.20.1) seems to have broken (at
least with intel compilers).::

     pip3 install cython


Installing matplotlib
-------------------------

This should just be pip installed::

     pip3 install matplotlib

Installing mpi4py
-------------------------

This should be pip installed::

    pip3 install mpi4py

.. note::

    This is failing to find mpi.h

Installing HDF5 with parallel support
--------------------------------------------------

The new analysis package brings HDF5 file writing capbaility.  This
needs to be compiled with support for parallel (mpi) I/O::

     wget http://www.hdfgroup.org/ftp/HDF5/current/src/hdf5-1.8.12.tar
     tar xvf hdf5-1.8.12.tar
     cd hdf5-1.8.12
     ./configure --prefix=$BUILD_HOME \
                         CC=mpicc \
                         CXX=mpicxx \
                         F77=mpif90 \
                         MPICC=mpicc MPICXX=mpicxx \
                         --enable-shared --enable-parallel
     make
     make install

Next, install h5py.  If we just want HDF5 file access (in serial),
then we can pip install, though we'll need to set env variables.  Here
we build against the parallel HDF5:

     export CC=mpicc
     export HDF5_DIR=$BUILD_HOME
     pip3 install h5py

Alternatively, we may wish for full HDF5 parallel goodness, so we can
do parallel file access during analysis as well.  This will require
building directly from source (see 
`Parallel HDF5 in h5py<http://docs.h5py.org/en/latest/mpi.html#parallel>`_
for further details).  Here we go::

     git clone https://github.com/h5py/h5py.git
     cd h5py
     export CC=mpicc
     export HDF5_DIR=$BUILD_HOME
     python3 setup.py build --mpi   
     python3 setup.py install --mpi

I'm having difficulty getting this h5py build to actually install to
``site-packages``.  More later.


Dedalus2
========================================

Preliminaries
----------------------------------------
On NASA Pleiades, the first thing we need to install is mercurial
itself::

     wget http://mercurial.selenic.com/release/mercurial-2.9.tar.gz


With the modules set as above, set::

     export BUILD_HOME=$BUILD_HOME
     export FFTW_PATH=$BUILD_HOME
     export MPI_PATH=$MPICH_HOME

Then change into your root dedalus directory and run::

     python setup.py build_ext --inplace

Our new stack (``intel/14``, ``mvapich2/2.0b``) builds to completion
and runs test problems successfully.  We have good scaling in limited
early tests.


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

     cd $BUILD_HOME
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

     cd $BUILD_HOME
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
     



.. note::
     
     Notes from the original successful build process:
   
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

     
