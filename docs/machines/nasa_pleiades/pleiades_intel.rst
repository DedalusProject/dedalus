Install notes for NASA/Pleiades: Intel stack
***************************************************************************

An initial Pleiades environment is pretty bare-bones.  There are no
modules, and your shell is likely a csh varient.  To switch shells,
send an e-mail to support@nas.nasa.gov; I'll be using ``bash``.

Then add the following to your ``.profile``::

  # Add your commands here to extend your PATH, etc.

  module load comp-intel
  module load git
  module load openssl
  module load emacs

  export BUILD_HOME=$HOME/build

  export PATH=$BUILD_HOME/bin:$BUILD_HOME:/$PATH  # Add private commands to PATH

  export LD_LIBRARY_PATH=$BUILD_HOME/lib:$LD_LIBRARY_PATH
  export LD_LIBRARY_PATH=/nasa/openssl/1.0.1h/lib/:$LD_LIBRARY_PATH

  export CC=mpicc

  #pathing for Dedalus
  export LOCAL_MPI_VERSION=openmpi-1.10.1
  export LOCAL_MPI_SHORT=v1.10

  # falling back to 1.8 until we resolve tcp wireup errors
  # (probably at runtime with MCA parameters)
  export LOCAL_MPI_VERSION=openmpi-1.8.6
  export LOCAL_MPI_SHORT=v1.8

  export LOCAL_PYTHON_VERSION=3.5.0
  export LOCAL_NUMPY_VERSION=1.10.1
  export LOCAL_SCIPY_VERSION=0.16.1
  export LOCAL_HDF5_VERSION=1.8.15-patch1
  export LOCAL_MERCURIAL_VERSION=3.6

  export MPI_ROOT=$BUILD_HOME/$LOCAL_MPI_VERSION
  export PYTHONPATH=$BUILD_HOME/dedalus:$PYTHONPATH
  export MPI_PATH=$MPI_ROOT
  export FFTW_PATH=$BUILD_HOME
  export HDF5_DIR=$BUILD_HOME

  # Openmpi forks:
  export OMPI_MCA_mpi_warn_on_fork=0

  # don't mess up Pleiades for everyone else
  export OMPI_MCA_btl_openib_if_include=mlx4_0:1



Doing the entire build took about 2 hours.  This was with several (4)
open ssh connections to Pleaides to do poor-mans-parallel building
(of python, hdf5, fftw, etc.), and one was on a dev node for the
openmpi compile.  The openmpi compile is time intensive and mus be
done first.  The fftw and hdf5 libraries take a while to build.
Building scipy remains the most significant time cost.


Python stack
=========================

Interesting update.  Pleiades now appears to have a python3 module.
Fascinating.  It comes with matplotlib (1.3.1), scipy (0.12), numpy
(1.8.0) and cython (0.20.1) and a few others.  Very interesting.  For
now we'll proceed with our usual build-it-from-scratch approach, but
this should be kept in mind for the future.  No clear mpi4py, and the
``mpi4py`` install was a hangup below for some time.

Building Openmpi
--------------------------

The suggested ``mpi-sgi/mpt`` MPI stack breaks with mpi4py; existing
versions of openmpi on Pleiades are outdated and suffer from a
previously identified bug (v1.6.5), so we'll roll our own.  This needs
to be built on a compute node so that the right memory space is identified.::

    # do this on a main node (where you can access the outside internet):
    cd $BUILD_HOME
    wget http://www.open-mpi.org/software/ompi/$LOCAL_MPI_SHORT/downloads/$LOCAL_MPI_VERSION.tar.gz
    tar xvf $LOCAL_MPI_VERSION.tar.gz

    # get ivy-bridge compute node
    qsub -I -q devel -l select=1:ncpus=24:mpiprocs=24:model=has -l walltime=02:00:00

    # once node exists
    cd $BUILD_HOME
    cd $LOCAL_MPI_VERSION
    ./configure \
	--prefix=$BUILD_HOME \
	--enable-mpi-interface-warning \
	--without-slurm \
	--with-tm=/PBS \
	--without-loadleveler \
	--without-portals \
	--enable-mpirun-prefix-by-default \
        CC=icc CXX=icc FC=ifort

    make -j
    make install

These compilation options are based on ``/nasa/openmpi/1.6.5/NAS_config.sh``,
and are thanks to advice from Daniel Kokron at NAS.  Compiling takes
about 10-15 minutes with make -j.


Building Python3
--------------------------

Create ``$BUILD_HOME`` and then proceed with downloading and installing Python-3.4::

    cd $BUILD_HOME
    wget https://www.python.org/ftp/python/$LOCAL_PYTHON_VERSION/Python-$LOCAL_PYTHON_VERSION.tgz --no-check-certificate
    tar xzf Python-$LOCAL_PYTHON_VERSION.tgz
    cd Python-$LOCAL_PYTHON_VERSION

    ./configure --prefix=$BUILD_HOME \
                         OPT="-mkl -O3 -axCORE-AVX2 -xSSE4.2 -fPIC -ipo -w -vec-report0 -opt-report0" \
                         FOPT="-mkl -O3 -axCORE-AVX2 -xSSE4.2 -fPIC -ipo -w -vec-report0 -opt-report0" \
                         CC=mpicc CXX=mpicxx F90=mpif90 \
                         LDFLAGS="-lpthread" \
                         --enable-shared --with-system-ffi \
                         --with-cxx-main=mpicxx --with-gcc=mpicc

    make
    make install

The previous intel patch is no longer required.


Installing pip
-------------------------

Python 3.4 now automatically includes pip.  We suggest you do the
following immediately to suppress version warning messages::

     pip3 install --upgrade pip


On Pleiades, you'll need to edit ``.pip/pip.conf``::

     [global]
     cert = /etc/ssl/certs/ca-bundle.trust.crt

You will now have ``pip3`` installed in ``$BUILD_HOME/bin``.
You might try doing ``pip3 -V`` to confirm that ``pip3`` is built
against python 3.4.  We will use ``pip3`` throughout this
documentation to remain compatible with systems (e.g., Mac OS) where
multiple versions of python coexist.

Installing mpi4py
--------------------------

This should be pip installed::

    pip3 install mpi4py


Installing FFTW3
------------------------------

We need to build our own FFTW3, under intel 14 and mvapich2/2.0b, or
under openmpi::

    wget http://www.fftw.org/fftw-3.3.4.tar.gz
    tar -xzf fftw-3.3.4.tar.gz
    cd fftw-3.3.4

   ./configure --prefix=$BUILD_HOME \
                         CC=mpicc        CFLAGS="-O3 -axCORE-AVX2 -xSSE4.2" \
                         CXX=mpicxx CPPFLAGS="-O3 -axCORE-AVX2 -xSSE4.2" \
                         F77=mpif90  F90FLAGS="-O3 -axCORE-AVX2 -xSSE4.2" \
                         MPICC=mpicc MPICXX=mpicxx \
                         --enable-shared \
                         --enable-mpi --enable-openmp --enable-threads
    make
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
will build against the Intel MKL BLAS.

Building numpy against MKL
----------------------------------

Now, acquire ``numpy`` (1.9.2)::

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
 "-axCORE-AVX2 -xSSE4.2".   NOTE: this is now updated for Haswell.

We'll now need to make sure that ``numpy`` is building against the MKL
libraries.  Start by making a ``site.cfg`` file::

     cp site.cfg.example site.cfg
     emacs -nw site.cfg

.. note::
    If you're doing many different builds, it may be helpful to have
    the numpy site.cfg shared between builds.  If so, you can edit
    ~/.numpy-site.cfg instead of site.cfg.  This is per site.cfg.example.


Edit ``site.cfg`` in the ``[mkl]`` section; modify the
library directory so that it correctly point to TACC's
``$MKLROOT/lib/intel64/``.
With the modules loaded above, this looks like::

     [mkl]
     library_dirs = /nasa/intel/Compiler/2015.3.187/composer_xe_2015.3.187/mkl/lib/intel64/
     include_dirs = /nasa/intel/Compiler/2015.3.187/composer_xe_2015.3.187/mkl/include
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

This should just be pip installed.  However, we're hitting errors with
qhull compilation in every part of the 1.4.x branch, so we fall back
to 1.3.1::

     pip3 install matplotlib==1.3.1


Installing HDF5 with parallel support
--------------------------------------------------

The new analysis package brings HDF5 file writing capbaility.  This
needs to be compiled with support for parallel (mpi) I/O::


     wget http://www.hdfgroup.org/ftp/HDF5/releases/hdf5-$LOCAL_HDF5_VERSION/src/hdf5-$LOCAL_HDF5_VERSION.tar.gz
     tar xzvf hdf5-$LOCAL_HDF5_VERSION.tar.gz
     cd hdf5-$LOCAL_HDF5_VERSION
     ./configure --prefix=$BUILD_HOME \
                         CC=mpicc         CFLAGS="-O3 -axCORE-AVX2 -xSSE4.2" \
                         CXX=mpicxx CPPFLAGS="-O3 -axCORE-AVX2 -xSSE4.2" \
                         F77=mpif90  F90FLAGS="-O3 -axCORE-AVX2 -xSSE4.2" \
                         MPICC=mpicc MPICXX=mpicxx \
                         --enable-shared --enable-parallel
     make
     make install


H5PY via pip
-----------------------

This can now just be pip installed (>=2.6.0)::

     pip3 install h5py

For now we drop our former instructions on attempting to install
parallel h5py with collectives.  See the repo history for those notes.

Installing Mercurial
----------------------------------------------------
On NASA Pleiades, we need to install mercurial itself.  I can't get
mercurial to build properly on intel compilers, so for now use gcc::

     cd $BUILD_HOME
     wget http://mercurial.selenic.com/release/mercurial-$LOCAL_MERCURIAL_VERSION.tar.gz
     tar xvf mercurial-$LOCAL_MERCURIAL_VERSION.tar.gz
     cd mercurial-$LOCAL_MERCURIAL_VERSION
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


Dedalus
========================================

Preliminaries
----------------------------------------

Then do the following::

     cd $BUILD_HOME
     hg clone https://bitbucket.org/dedalus-project/dedalus
     cd dedalus
     pip3 install -r requirements.txt
     python3 setup.py build_ext --inplace


Running Dedalus on Pleiades
========================================

Our scratch disk system on Pleiades is ``/nobackup/user-name``.  On
this and other systems, I suggest soft-linking your scratch directory
to a local working directory in home; I uniformly call mine ``workdir``::

      ln -s /nobackup/bpbrown workdir

Long-term mass storage is on LOU.



