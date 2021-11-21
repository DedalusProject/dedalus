Install notes for TACC/Stampede
***************************************************************************
Install notes for building our python3 stack on TACC/Stampede, using the intel compiler suite.
Many thanks to Yaakoub El Khamra at TACC for help in sorting out the
python3 build and numpy linking against a fast MKL BLAS.

On Stampede, we can in principle either install with a ``gcc/mpvapich2/fftw3``
stack with ``OpenBLAS``, or with an ``intel/mvapich2/fftw3 stack`` with
``MKL``.  Mpvaich2 is causing problems for  us, and this
appears to be a known issue with ``mvapich2/1.9``, so for now we must
use the ``intel/mvapich2/fftw3`` stack, which has ``mvapich2/2.0b``.
The intel stack should also, in principle,
allow us to explore auto-offloading with the Xenon MIC hardware
accelerators.   Current ``gcc`` instructions can be found under NASA Pleiades.


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


For ease in structuring the build, for now we'll define::

     export BUILD_HOME=$HOME/build_intel


Python stack
=========================

Building Python3
--------------------------

Create ``~\build_intel`` and then proceed with downloading and installing Python-3.3::

    cd ~/build_intel
    wget http://www.python.org/ftp/python/3.3.3/Python-3.3.3.tgz
    tar -xzf Python-3.3.3.tgz
    cd Python-3.3.3

    # make sure you have the python patch, put it in Python-3.3.3
    wget http://dedalus-project.readthedocs.org/en/latest/_downloads/python_intel_patch.tar
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
see a warning/error that ``_ctypes`` couldn't be built.  This is important.


Here we are building everything in ``~/build_intel``; you can do it
whereever, but adjust things appropriately in the above instructions.
The build proceeeds quickly (few minutes).

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

    wget --no-check-certificate https://raw.github.com/pypa/pip/master/contrib/get-pip.py
    python3 get-pip.py --cert /etc/ssl/certs/ca-bundle.crt

Now edit ``~/.pip/pip.conf``::

     [global]
     cert = /etc/ssl/certs/ca-bundle.crt


You will now have ``pip3`` and ``pip`` installed in ``~/build/bin``.
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

Building numpy against MKL
----------------------------------

Now, acquire ``numpy`` (1.8.0)::

     cd ~/build_intel
     wget http://sourceforge.net/projects/numpy/files/NumPy/1.8.0/numpy-1.8.0.tar.gz
     tar -xvf numpy-1.8.0.tar.gz
     cd numpy-1.8.0
     wget http://lcd-www.colorado.edu/bpbrown/dedalus_documentation/_downloads/numpy_intel_patch.tar
     tar xvf numpy_inte_patch.tar

This last step saves you from needing to hand edit two
files in ``numpy/distutils``; these are ``intelccompiler.py`` and
``fcompiler/intel.py``.  I've built a crude patch,
:download:`numpy_intel_patch.tar<numpy_intel_patch.tar>`
which can be auto-deployed by within the ``numpy-1.8.0`` directory by
the instructions above.  This will unpack and overwrite::

      numpy/distutils/intelccompiler.py
      numpy/distutils/fcompiler/intel.py

We'll now need to make sure that ``numpy`` is building against the MKL
libraries.  Start by making a ``site.cfg`` file::

     cp site.cfg.example site.cfg
     emacs -nw site.cfg

Edit ``site.cfg`` in the ``[mkl]`` section; modify the
library directory so that it correctly point to TACC's
``$MKLROOT/lib/intel64/``.
With the modules loaded above, this looks like::

     [mkl]
     library_dirs = /opt/apps/intel/13/composer_xe_2013_sp1.1.106/mkl/lib/intel64
     include_dirs = /opt/apps/intel/13/composer_xe_2013_sp1.1.106/mkl/include
     mkl_libs = mkl_rt
     lapack_libs =

These are based on intels instructions for
`compiling numpy with ifort <http://software.intel.com/en-us/articles/numpyscipy-with-intel-mkl>`_
and they seem to work so far.


Then proceed with::

    python3 setup.py config --compiler=intelem build_clib --compiler=intelem build_ext --compiler=intelem install

This will config, build and install numpy.


Test numpy install
------------------------------

Test that things worked with this executable script
:download:`numpy_test_full<numpy_test_full>`.  You can do this
full-auto by doing::

     wget http://lcd-www.colorado.edu/bpbrown/dedalus_documentation/_downloads/numpy_test_full
     chmod +x numpy_test_full
     ./numpy_test_full

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


Installing mpi4py
-------------------------

This should just be pip installed::

      pip3 install mpi4py==2.0.0

.. note::

      If we use use ::

           pip3 install mpi4py

      then stampede tries to pull version 0.6.0 of mpi4py.  Hence the
      explicit version pull above.

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


     pip3 install -v https://downloads.sourceforge.net/project/matplotlib/matplotlib/matplotlib-1.3.1/matplotlib-1.3.1.tar.gz

.. note::

      If we use use ::

           pip3 install matplotlib

      then stampede tries to pull version 1.1.1 of matplotlib.  Hence the
      explicit version pull above.

Installing sympy
-------------------------

Do this with a regular pip install::

      pip3 install sympy


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

Installing h5py
----------------------------------------------------

Next, install h5py. We wish for full HDF5 parallel goodness, so we can
do parallel file access during both simulations and post analysis as
well.   This will require building directly from source (see
`Parallel HDF5 in h5py <http://docs.h5py.org/en/latest/mpi.html#parallel>`_
for further details).  Here we go::

     git clone https://github.com/h5py/h5py.git
     cd h5py
     export CC=mpicc
     export HDF5_DIR=$BUILD_HOME
     python3 setup.py configure --mpi
     python3 setup.py build
     python3 setup.py install

After this install, ``h5py`` shows up as an ``.egg`` in
``site-packages``, but it looks like we pass the ``suggested demo2.py``
test from `Parallel HDF5 in h5py <http://docs.h5py.org/en/latest/mpi.html#parallel>`_.


Installing h5py with collectives
----------------------------------------------------
We've been exploring the use of collectives for faster parallel file
writing.  To build that version of the h5py library::

     git clone https://github.com/andrewcollette/h5py.git
     cd h5py
     git checkout mpi_collective
     export CC=mpicc
     export HDF5_DIR=$BUILD_HOME
     python3 setup.py configure --mpi
     python3 setup.py build
     python3 setup.py install

To enable collective outputs within dedalus, edit ``dedalus2/data/evaluator.py`` and
replace::

            # Assemble nonconstant subspace
            subshape, subslices, subdata = self.get_subspace(out)
            dset = task_group.create_dataset(name=name, shape=subshape, dtype=dtype)
            dset[subslices] = subdata

with ::

            # Assemble nonconstant subspace
            subshape, subslices, subdata = self.get_subspace(out)
            dset = task_group.create_dataset(name=name, shape=subshape, dtype=dtype)
            with dset.collective:
                dset[subslices] = subdata

Alternatively, you can see this same edit in some of the forks
(Lecoanet, Brown).

.. note::

     There are some serious problems with this right now; in
     particular, there seems to be an issue with empty arrays causing h5py
     to hang.  Troubleshooting is ongoing.

Dedalus2
========================================

With the modules set as above, set::

     export BUILD_HOME=$HOME/build_intel
     export FFTW_PATH=$BUILD_HOME
     export MPI_PATH=$MPICH_HOME
     export HDF5_DIR=$BUILD_HOME
     export CC=mpicc

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




Installing virtualenv (skipped)
----------------------------------

In order to test multiple numpys and scipys (and really, their
underlying BLAS libraries), we will use ``virtualenv``::

     pip3 install virtualenv

Next, construct a virtualenv to hold all of your python modules. We
suggest doing this in your home directory::

     mkdir ~/venv




Python3
---------------------------------

.. note::

     With help from Yaakoub, we now build ``_ctypes`` successfully.


     Also, the mpicc build is much, much slower than icc.  Interesting.
     And we crashed out.  Here's what we tried with mpicc::

        ./configure --prefix=$BUILD_HOME \
                         CC=mpicc CFLAGS="-mkl -O3 -xHost -fPIC -ipo" \
                         CXX=mpicxx CPPFLAGS="-mkl -O3 -xHost -fPIC -ipo" \
                         F90=mpif90 F90FLAGS="-mkl -O3 -xHost -fPIC -ipo" \
                         --enable-shared LDFLAGS="-lpthread" \
                         --with-cxx-main=mpicxx --with-system-ffi
