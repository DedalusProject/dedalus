Install notes for PSC/Bridges: Intel stack
***************************************************************************

Here we build using the recommended Intel compilers.  Bridges comes
with python 3.4 at present, but for now we'll maintain a boutique
build to keep access to python >=3.5 and to tune numpy performance by
hand (though the value proposition of this should be tested).

Then add the following to your ``.bash_profile``::

  # Add your commands here to extend your PATH, etc.

  export BUILD_HOME=$HOME/build

  export PATH=$BUILD_HOME/bin:$BUILD_HOME:/$PATH  # Add private commands to PATH
  export LD_LIBRARY_PATH=$BUILD_HOME/lib:$LD_LIBRARY_PATH

  export CC=mpiicc

  export I_MPI_CC=icc

  #pathing for Dedalus
  export LOCAL_PYTHON_VERSION=3.5.1
  export LOCAL_NUMPY_VERSION=1.11.0
  export LOCAL_SCIPY_VERSION=0.17.0
  export LOCAL_HDF5_VERSION=1.8.16
  export LOCAL_MERCURIAL_VERSION=3.7.3

  export PYTHONPATH=$BUILD_HOME/dedalus:$PYTHONPATH
  export MPI_PATH=$MPI_ROOT
  export FFTW_PATH=$BUILD_HOME
  export HDF5_DIR=$BUILD_HOME


Python stack
=========================
Here we use the recommended Intel mpi compilers, rather than our own
openmpi.

Building Python3
--------------------------

Create ``$BUILD_HOME`` and then proceed with downloading and installing Python-3::

    cd $BUILD_HOME
    wget https://www.python.org/ftp/python/$LOCAL_PYTHON_VERSION/Python-$LOCAL_PYTHON_VERSION.tgz --no-check-certificate
    tar xzf Python-$LOCAL_PYTHON_VERSION.tgz
    cd Python-$LOCAL_PYTHON_VERSION
    ./configure --prefix=$BUILD_HOME \
                         OPT="-w -vec-report0 -opt-report0" \
                         FOPT="-w -vec-report0 -opt-report0" \
                         CFLAGS="-mkl -O3 -ipo -xCORE-AVX2 -fPIC" \
                         CPPFLAGS="-mkl -O3 -ipo -xCORE-AVX2 -fPIC" \
                         F90FLAGS="-mkl -O3 -ipo -xCORE-AVX2 -fPIC" \
                         CC=mpiicc  CXX=mpiicpc F90=mpiifort  \
                         LDFLAGS="-lpthread"
    make -j
    make install

The previous intel patch is no longer required.


Installing pip
-------------------------

Python 3.4+ now automatically includes pip.

You will now have ``pip3`` installed in ``$BUILD_HOME/bin``.
You might try doing ``pip3 -V`` to confirm that ``pip3`` is built
against python 3.4.  We will use ``pip3`` throughout this
documentation to remain compatible with systems (e.g., Mac OS) where
multiple versions of python coexist.

We suggest doing the following immediately to suppress version warning
messages::

     pip3 install --upgrade pip

Installing mpi4py
--------------------------

This should be pip installed::

   pip3 install mpi4py

This required setting the ``I_MPI_CC=icc`` envirnoment variable above;
otherwise we keep hitting gcc.

Installing FFTW3
------------------------------

We build our own FFTW3::

    wget http://www.fftw.org/fftw-3.3.4.tar.gz
    tar -xzf fftw-3.3.4.tar.gz
    cd fftw-3.3.4

   ./configure --prefix=$BUILD_HOME \
                         CC=mpiicc        CFLAGS="-O3 -xCORE-AVX2" \
                         CXX=mpiicpc CPPFLAGS="-O3 -xCORE-AVX2" \
                         F77=mpiifort  F90FLAGS="-O3 -xCORE-AVX2" \
                         MPICC=mpiicc MPICXX=mpiicpc \
                         LDFLAGS="-lmpi" \
                         --enable-shared \
                         --enable-mpi --enable-openmp --enable-threads
    make -j
    make install

It's critical that you use ``mpicc`` as the C-compiler, etc.
Otherwise the libmpich libraries are not being correctly linked into
``libfftw3_mpi.so`` and dedalus failes on fftw import.


Installing nose
-------------------------

Nose is useful for unit testing, especially in checking our numpy build::

    pip3 install nose


Installing cython
-------------------------

This should just be pip installed::

     pip3 install cython



Numpy and BLAS libraries
======================================

Numpy will be built against a specific BLAS library.

Building numpy against MKL
----------------------------------

Now, acquire ``numpy``.  The login nodes for Bridges are 14-core
Haswell chips, just like the compute nodes, so let's try doing it with
the normal numpy settings (no patching to adjust the compiler commands
in distutils for cross-compiling).  Ah shoots.  Nope.  The numpy
distutils only employs xSSE4.2 and none of the AVX2 arch flags, nor a
basic xhost.  Well.  On we go.  Change ``-xSSE4.2`` to ``-xCORE-AVX2``
in ``numpy/distutils/intelccompiler.py`` and
``numpy/distutils/fcompiler/intel.py``.  We should really put in a PR
and an ability to pass flags via ``site.cfg`` or other approach.

Here's an automated way to do this, using :download:`numpy_intel.patch<numpy_intel.patch>`.::

     cd $BUILD_HOME
     wget http://sourceforge.net/projects/numpy/files/NumPy/$LOCAL_NUMPY_VERSION/numpy-$LOCAL_NUMPY_VERSION.tar.gz
     tar -xvf numpy-$LOCAL_NUMPY_VERSION.tar.gz
     cd numpy-$LOCAL_NUMPY_VERSION
     wget http://dedalus-project.readthedocs.org/en/latest/_downloads/numpy_intel.patch
     patch -p1 < numpy_intel.patch

We'll now need to make sure that ``numpy`` is building against the MKL
libraries.  Start by making a ``site.cfg`` file::

     cat >> site.cfg << EOF
     [mkl]
     library_dirs = /opt/packages/intel/compilers_and_libraries/linux/mkl/lib/intel64
     include_dirs = /opt/packages/intel/compilers_and_libraries/linux/mkl/include
     mkl_libs = mkl_rt
     lapack_libs =
     EOF

Then proceed with::

    python3 setup.py config --compiler=intelem build_clib --compiler=intelem build_ext --compiler=intelem install

This will config, build and install numpy.







Test numpy install
------------------------------

Test that things worked with this executable script
:download:`numpy_test_full<numpy_test_full>`.  You can do this
full-auto by doing::

     wget http://dedalus-project.readthedocs.org/en/latest/_downloads/numpy_test_full
     chmod +x numpy_test_full
     ./numpy_test_full

Numpy has changed the location of _dotblas, so our old test doesn't
work.  From the dot product speed, it looks like we have succesfully
linked against fast BLAS and the test results look relatively normal,
but this needs to be looked in to.



Python library stack
=====================

After ``numpy`` has been built
we will proceed with the rest of our python stack.

Installing Scipy
-------------------------

Scipy is easier, because it just gets its config from numpy.  Scipy
now is no longer hosted at sourceforge for anything past v0.16, so
lets try git::

    git clone git://github.com/scipy/scipy.git scipy
    cd scipy
    # fall back to stable version
    git checkout tags/v$LOCAL_SCIPY_VERSION
    python3 setup.py config --compiler=intelem --fcompiler=intelem build_clib \
                                            --compiler=intelem --fcompiler=intelem build_ext \
                                            --compiler=intelem --fcompiler=intelem install

.. note::

   We do not have umfpack; we should address this moving forward, but
   for now I will defer that to a later day.  Again.  Still.


Installing matplotlib
-------------------------

This should just be pip installed.  In versions of matplotlib>1.3.1,
Qhull has a compile error if the C compiler is used rather than C++,
so we force the C complier to be icpc ::

     export CC=icpc
     pip3 install matplotlib


Installing HDF5 with parallel support
--------------------------------------------------

The new analysis package brings HDF5 file writing capbaility.  This
needs to be compiled with support for parallel (mpi) I/O.  Intel
compilers are failing on this when done with mpi-sgi, and on NASA's
recommendation we're falling back to gcc for this library::

     wget http://www.hdfgroup.org/ftp/HDF5/releases/hdf5-$LOCAL_HDF5_VERSION/src/hdf5-$LOCAL_HDF5_VERSION.tar.gz
     tar xzvf hdf5-$LOCAL_HDF5_VERSION.tar.gz
     cd hdf5-$LOCAL_HDF5_VERSION
     ./configure --prefix=$BUILD_HOME CC=mpiicc CXX=mpiicpc F77=mpiifort \
                         --enable-shared --enable-parallel
     make
     make install


H5PY via pip
-----------------------

This can now just be pip installed (>=2.6.0)::

     pip3 install h5py

For now we drop our former instructions on attempting to install
parallel h5py with collectives. See the NASA/Pleiades repo history for those notes.

Installing Mercurial
----------------------------------------------------
Here we install mercurial itself.  Following NASA/Pleiades approaches,
we will use gcc.  I haven't checked whether the default bridges
install has mercurial::

     cd $BUILD_HOME
     wget http://mercurial.selenic.com/release/mercurial-$LOCAL_MERCURIAL_VERSION.tar.gz
     tar xvf mercurial-$LOCAL_MERCURIAL_VERSION.tar.gz
     cd mercurial-$LOCAL_MERCURIAL_VERSION
     module load gcc
     export CC=gcc
     make install PREFIX=$BUILD_HOME

I suggest you add the following to your ``~/.hgrc``::
  cat >> ~/.hgrc << EOF
  [ui]
  username = <your bitbucket username/e-mail address here>
  editor = emacs

  [extensions]
  graphlog =
  color =
  convert =
  mq =
  EOF

Dedalus
========================================

Preliminaries
----------------------------------------

Then do the following::

     cd $BUILD_HOME
     hg clone https://bitbucket.org/dedalus-project/dedalus
     cd dedalus
     # this has some issues with mpi4py versioning --v
     pip3 install -r requirements.txt
     python3 setup.py build_ext --inplace



Running Dedalus on Bridges
========================================

Our scratch disk system on Bridges is ``/pylon1/group-name/user-name``.  On
this and other systems, I suggest soft-linking your scratch directory
to a local working directory in home; I uniformly call mine ``workdir``::

      ln -s /pylon1/group-name/user-name workdir

Long-term spinning storage is on ``/pylon2`` and is provided by
allocation request.



