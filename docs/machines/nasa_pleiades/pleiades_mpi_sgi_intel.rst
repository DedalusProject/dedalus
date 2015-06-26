Install notes for NASA/Pleiades: Intel stack with MPI-SGI
***************************************************************************

Here we build using the recommended MPI-SGI environment, with Intel compilers.
An initial Pleiades environment is pretty bare-bones.  There are no
modules, and your shell is likely a csh varient.  To switch shells,
send an e-mail to support@nas.nasa.gov; I'll be using ``bash``.

Then add the following to your ``.profile``::

  # Add your commands here to extend your PATH, etc.

  module load mpi-sgi/mpt
  module load comp-intel
  module load git
  module load openssl
  module load emacs

  export BUILD_HOME=$HOME/build

  export PATH=$BUILD_HOME/bin:$BUILD_HOME:/$PATH  # Add private commands to PATH                                                                                         

  export LD_LIBRARY_PATH=$BUILD_HOME/lib:$LD_LIBRARY_PATH
  export LD_LIBRARY_PATH=/nasa/openssl/1.0.1h/lib/:$LD_LIBRARY_PATH

  # proper wrappers for using Intel rather than GNU compilers,
  # Thanks to Daniel Kokron at NASA.
  export MPICC_CC=icc
  export MPICXX_CXX=icpc

  export CC=mpicc

  #pathing for Dedalus
  export LOCAL_PYTHON_VERSION=3.4.3
  export LOCAL_NUMPY_VERSION=1.9.2
  export LOCAL_SCIPY_VERSION=0.15.1
  export LOCAL_HDF5_VERSION=1.8.14

  export PYTHONPATH=$BUILD_HOME/dedalus:$PYTHONPATH
  export MPI_PATH=$MPI_ROOT
  export FFTW_PATH=$BUILD_HOME
  export HDF5_DIR=$BUILD_HOME

  # Pleaides workaround for QP errors 8/25/14 from NAS (only for MPI-SGI)                                                                                         
  export MPI_USE_UD=true


Python stack
=========================
Here we use the recommended MPI-SGI compilers, rather than our own
openmpi.

.. note::
      Right now, mpi4py is building, but when we do a "from mpi4py
      import MPI", we core dump.  This happens on all tested
      architectures at present.  This stack is stuck until this
      problem is fixed.

Building Python3
--------------------------

Create ``$BUILD_HOME`` and then proceed with downloading and installing Python-3.4::

    cd $BUILD_HOME
    wget https://www.python.org/ftp/python/$LOCAL_PYTHON_VERSION/Python-$LOCAL_PYTHON_VERSION.tgz --no-check-certificate
    tar xzf Python-$LOCAL_PYTHON_VERSION.tgz
    cd Python-$LOCAL_PYTHON_VERSION
    ./configure --prefix=$BUILD_HOME \
                         OPT="-w -vec-report0 -opt-report0" \
                         FOPT="-w -vec-report0 -opt-report0" \
                         CFLAGS="-mkl -O3 -ipo -axCORE-AVX2 -xSSE4.2 -fPIC" \
                         CPPFLAGS="-mkl -O3 -ipo -axCORE-AVX2 -xSSE4.2 -fPIC" \
                         F90FLAGS="-mkl -O3 -ipo -axCORE-AVX2 -xSSE4.2 -fPIC" \
                         CC=mpicc  CXX=mpicxx F90=mpif90  \
                         --with-cxx-main=mpicxx --with-gcc=mpicc \
                         LDFLAGS="-lpthread" \
                        --enable-shared --with-system-ffi

    make
    make install

The previous intel patch is no longer required.


Installing pip
-------------------------

Python 3.4 now automatically includes pip.

On Pleiades, you'll need to edit ``.pip/pip.conf``::

     [global]
     cert = /etc/ssl/certs/ca-bundle.trust.crt

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


Installing FFTW3
------------------------------

We build our own FFTW3::

    wget http://www.fftw.org/fftw-3.3.4.tar.gz
    tar -xzf fftw-3.3.4.tar.gz
    cd fftw-3.3.4

   ./configure --prefix=$BUILD_HOME \
                         CC=icc        CFLAGS="-O3 -axCORE-AVX2 -xSSE4.2" \
                         CXX=icpc CPPFLAGS="-O3 -axCORE-AVX2 -xSSE4.2" \
                         F77=ifort  F90FLAGS="-O3 -axCORE-AVX2 -xSSE4.2" \
                         MPICC=icc MPICXX=icpc \
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

Numpy will be built against a specific BLAS library.  On Pleiades we
will build against the OpenBLAS libraries.  

All of the intel patches, etc. are unnecessary in the gcc stack.

Building numpy against MKL
----------------------------------

Now, acquire ``numpy`` (1.8.2)::

     cd $BUILD_HOME
     wget http://sourceforge.net/projects/numpy/files/NumPy/$LOCAL_NUMPY_VERSION/numpy-$LOCAL_NUMPY_VERSION.tar.gz
     tar -xvf numpy-$LOCAL_NUMPY_VERSION.tar.gz
     cd numpy-$LOCAL_NUMPY_VERSION
     wget http://dedalus-project.readthedocs.org/en/latest/_downloads/numpy_pleiades_intel_patch.tar
     tar xvf numpy_pleiades_intel_patch.tar

This last step saves you from needing to hand edit two
files in ``numpy/distutils``; these are ``intelccompiler.py`` and
``fcompiler/intel.py``.  I've built a crude patch, :download:`numpy_pleiades_intel_patch.tar<numpy_pleiades_intel_patch.tar>` 
which is auto-deployed within the ``numpy-$LOCAL_NUMPY_VERSION`` directory by
the instructions above.  This will unpack and overwrite::

      numpy/distutils/intelccompiler.py
      numpy/distutils/fcompiler/intel.py

This differs from prior versions in that "-xhost" is replaced with
 "-axAVX -xSSE4.1".   NOTE: this needs to be updated for Haswell.

We'll now need to make sure that ``numpy`` is building against the MKL
libraries.  Start by making a ``site.cfg`` file::

     cp site.cfg.example site.cfg
     emacs -nw site.cfg

Edit ``site.cfg`` in the ``[mkl]`` section; modify the
library directory so that it correctly point to TACC's
``$MKLROOT/lib/intel64/``.  
With the modules loaded above, this looks like::

     [mkl]
     library_dirs = /nasa/intel/Compiler/2015.0.090/composer_xe_2015.0.090/mkl/lib/intel64
     include_dirs = /nasa/intel/Compiler/2015.0.090/composer_xe_2015.0.090/mkl/include
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

     wget http://dedalus-project.readthedocs.org/en/latest/_downloads/numpy_test_full
     chmod +x numpy_test_full
     ./numpy_test_full

We succesfully link against fast BLAS and the test results look normal.



Python library stack
=====================

After ``numpy`` has been built
we will proceed with the rest of our python stack.

Installing Scipy
-------------------------

Scipy is easier, because it just gets its config from numpy.  Dong a
pip install fails, so we'll keep doing it the old fashioned way::

    wget http://sourceforge.net/projects/scipy/files/scipy/$LOCAL_SCIPY_VERSION/scipy-$LOCAL_SCIPY_VERSION.tar.gz
    tar -xvf scipy-$LOCAL_SCIPY_VERSION.tar.gz
    cd scipy-$LOCAL_SCIPY_VERSION
    python3 setup.py config --compiler=intelem --fcompiler=intelem build_clib \
                                            --compiler=intelem --fcompiler=intelem build_ext \
                                            --compiler=intelem --fcompiler=intelem install

.. note::

   We do not have umfpack; we should address this moving forward, but
   for now I will defer that to a later day.


Installing matplotlib
-------------------------

This should just be pip installed::

     pip3 install matplotlib

Hmmm... version 1.4.0 of matplotlib has just dropped, but seems to
have a higher freetype versioning requirement (2.4).  Here's a
build script for freetype 2.5.3::

    wget http://sourceforge.net/projects/freetype/files/freetype2/2.5.3/freetype-2.5.3.tar.gz/download
    tar xvf freetype-2.5.3.tar.gz
    cd freetype-2.5.3
    ./configure --prefix=$BUILD_HOME
    make
    make install

Well... that works, but then we fail on a qhull compile during 
``pip3 install matplotlib`` later on.
Let's fall back to 1.3.1::

     pip3 install matplotlib==1.3.1



Installing HDF5 with parallel support
--------------------------------------------------

The new analysis package brings HDF5 file writing capbaility.  This
needs to be compiled with support for parallel (mpi) I/O::


     wget http://www.hdfgroup.org/ftp/HDF5/releases/hdf5-$LOCAL_HDF5_VERSION/src/hdf5-$LOCAL_HDF5_VERSION.tar.gz
     tar xzvf hdf5-$LOCAL_HDF5_VERSION.tar.gz
     cd hdf5-$LOCAL_HDF5_VERSION
     ./configure --prefix=$BUILD_HOME \
                         CC=icc         CFLAGS="-O3 -axCORE-AVX2 -xSSE4.2" \
                         CXX=icpc CPPFLAGS="-O3 -axCORE-AVX2 -xSSE4.2" \
                         F77=ifort  F90FLAGS="-O3 -axCORE-AVX2 -xSSE4.2" \
                         MPICC=icc MPICXX=icpc \
                         LDFLAGS="-lmpi" \
                         --enable-shared --enable-parallel
     make
     make install


H5PY via pip
-----------------------

This works (Dec 21, 2014)::

     pip3 install h5py==2.4.0b1

Installing h5py (working)
----------------------------------------------------

Next, install h5py.  For reasons that are currently unclear to me, 
this cannot be done via pip install (fails)::

     git clone https://github.com/h5py/h5py.git
     cd h5py
     python3 setup.py configure --mpi
     python3 setup.py build
     python3 setup.py install 

This will install ``h5py==2.4.0a0``, and it appears to work (!).


Installing h5py with collectives (not currently working)
------------------------------------------------------------------------
We've been exploring the use of collectives for faster parallel file
writing.  

git is having some problems, especially with it's SSL version.  
I suggest adding the following to ``~/.gitconfig``::

    [http]
    sslCAinfo = /etc/ssl/certs/ca-bundle.crt


This is still not working, owing (most likely) to git being built on
an outdated SSL version.  Here's a short-term hack::

    export GIT_SSL_NO_VERIFY=true

To build that version of the h5py library::

     git clone git://github.com/andrewcollette/h5py
     cd h5py
     git checkout mpi_collective
     export CC=mpicc
     export HDF5_DIR=$BUILD_HOME
     python3 setup.py configure --mpi
     python3 setup.py build
     python3 setup.py install 


Here's the original h5py repository::

     git clone git://github.com/h5py/h5py
     cd h5py
     export CC=mpicc
     export HDF5_DIR=$BUILD_HOME
     python3 setup.py configure --mpi
     python3 setup.py build
     python3 setup.py install 

.. note::
     This is ugly.  We're getting a "-R" error at link, triggered by
     distutils not recognizing that mpicc is gcc or something like
     that.   Looks like we're failing ``if self._is_gcc(compiler)``
     For now, I've hand-edited unixccompiler.py in 
     ``lib/python3.3/distutils`` and changed this line:

           def _is_gcc(self, compiler_name):
                return "gcc" in compiler_name or "g++" in compiler_name

        to:

           def _is_gcc(self, compiler_name):
       	        return "gcc" in compiler_name or "g++" in compiler_name or "mpicc" in compiler_name

     This is a hack, but it get's us running and alive!

.. note::
     Ahh... I understand what's happening here.  We built with
     ``mpicc``, and the test ``_is_gcc`` looks for whether gcc appears
     anywhere in the compiler name.  It doesn't in ``mpicc``, so the
     ``gcc`` checks get missed.  This is only ever used in the
     ``runtime_library_dir_option()`` call.  So we'd need to either
     rename the mpicc wrapper something like ``mpicc-gcc`` or do a
     test on ``compiler --version`` or something.  Oh boy.  Serious
     upstream problem for mpicc wrapped builds that cythonize and go
     to link.  Hmm...

Installing Mercurial
----------------------------------------------------
On NASA Pleiades, we need to install mercurial itself.  I can't get
mercurial to build properly on intel compilers, so for now use gcc::

     wget http://mercurial.selenic.com/release/mercurial-3.1.tar.gz
     tar xvf mercurial-3.1.tar.gz 
     cd mercurial-3.1
     module load gcc
     export CC=gcc
     make install PREFIX=$BUILD_HOME

I suggest you add the following to your ``~/.hgrc``::

  [ui]
  username = <your bitbucket username/e-mail address here>
  editor = emacs

  [web]
  cacerts = /etc/ssl/certs/ca-bundle.crt

  [extensions]
  graphlog =
  color =
  convert =
  mq =


Dedalus2
========================================

Preliminaries
----------------------------------------

With the modules set as above, set::

     export BUILD_HOME=$BUILD_HOME
     export FFTW_PATH=$BUILD_HOME
     export MPI_PATH=$BUILD_HOME/$LOCAL_MPI_VERSION

Then change into your root dedalus directory and run::

     pip3 install -r requirements.txt 
     python setup.py build_ext --inplace


Running Dedalus on Pleiades
========================================

Our scratch disk system on Pleiades is ``/nobackup/user-name``.  On
this and other systems, I suggest soft-linking your scratch directory
to a local working directory in home; I uniformly call mine ``workdir``::

      ln -s /nobackup/bpbrown workdir

Long-term mass storage is on LOU.



