Install notes for CU/Janus
***************************************************************************

As with NASA/Pleiades, an initial Janus environment is pretty
bare-bones.  There are no
modules, and your shell is likely a bash varient.   Here we'll do a
full build of our stack, using only the prebuilt openmpi compilers.
Later we'll try a module heavy stack to see if we can avoid this.

Add the following to your ``.my.bash_profile``::

  # Add your commands here to extend your PATH, etc.

  module load intel

  export BUILD_HOME=$HOME/build

  export PATH=$BUILD_HOME/bin:$BUILD_HOME:/$PATH  # Add private commands to PATH                                                                                         

  export LD_LIBRARY_PATH=$BUILD_HOME/lib:$LD_LIBRARY_PATH

  export CC=mpicc

  #pathing for Dedalus
  export LOCAL_MPI_VERSION=openmpi-1.8.5
  export LOCAL_MPI_SHORT=v1.8
  export LOCAL_PYTHON_VERSION=3.4.3
  export LOCAL_NUMPY_VERSION=1.9.2
  export LOCAL_SCIPY_VERSION=0.15.1
  export LOCAL_HDF5_VERSION=1.8.15


  export MPI_ROOT=$BUILD_HOME/$LOCAL_MPI_VERSION
  export PYTHONPATH=$BUILD_HOME/dedalus:$PYTHONPATH
  export MPI_PATH=$MPI_ROOT
  export FFTW_PATH=$BUILD_HOME
  export HDF5_DIR=$BUILD_HOME

Do your builds on the janus compile nodes (see MOTD).  As a positive
note, Janus compile nodes have access to the internet (e.g., wget), so
you can download and compile on-node.  For now we're using stock
Pleiades compile flags and patch files.  With intel 15.0.1 the cython
install is now working well, as does h5py.


Building Openmpi
--------------------------
Tim Dunn has pointed out that we may (may) be able to get some speed
improvements by building our own openmpi.  Why not give it a try!
Compiling on the janus-compile nodes seems to do a fine job, and
unlike Pleiades we can grab software from the internet on the compile
nodes too.   This streamlines the process.::

    cd $BUILD_HOME
    wget http://www.open-mpi.org/software/ompi/$LOCAL_MPI_SHORT/downloads/$LOCAL_MPI_VERSION.tar.gz
    tar xvf $LOCAL_MPI_VERSION.tar.gz
    cd $LOCAL_MPI_VERSION
    ./configure \
        --prefix=$BUILD_HOME \
        --with-slurm \
        --with-threads=posix \
        --enable-mpi-thread-multiple \
        CC=icc CXX=icpc FC=ifort 

    make -j
    make install

Config flags thanks to Tim Dunn; the CFLAGS etc are from Pleiades and
should be general.


Building Python3
--------------------------

Create ``$BUILD_HOME`` and then proceed with downloading and installing Python-3.4::

    cd $BUILD_HOME
    wget https://www.python.org/ftp/python/$LOCAL_PYTHON_VERSION/Python-$LOCAL_PYTHON_VERSION.tgz
    tar xzf Python-$LOCAL_PYTHON_VERSION.tgz
    cd Python-$LOCAL_PYTHON_VERSION

    ./configure --prefix=$BUILD_HOME \
                         CC=mpicc         CFLAGS="-mkl -O3 -axAVX -xSSE4.1 -fPIC -ipo" \
                         CXX=mpicxx CPPFLAGS="-mkl -O3 -axAVX -xSSE4.1 -fPIC -ipo" \
                         F90=mpif90  F90FLAGS="-mkl -O3 -axAVX -xSSE4.1 -fPIC -ipo" \
                         --enable-shared LDFLAGS="-lpthread" \
                         --with-cxx-main=mpicxx --with-system-ffi

    make -j
    make install

The former patch for Intel compilers to handle ctypes is no longer necessary.

Installing pip
-------------------------

Python 3.4 now automatically includes pip.

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

    cd $BUILD_HOME
    wget http://www.fftw.org/fftw-3.3.4.tar.gz
    tar xvzf fftw-3.3.4.tar.gz
    cd fftw-3.3.4

   ./configure --prefix=$BUILD_HOME \
                         CC=mpicc        CFLAGS="-O3 -axAVX -xSSE4.1" \
                         CXX=mpicxx CPPFLAGS="-O3 -axAVX -xSSE4.1" \
                         F77=mpif90  F90FLAGS="-O3 -axAVX -xSSE4.1" \
                         MPICC=mpicc MPICXX=mpicxx \
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

Cython is now working (intel 15.0/openmpi-1.8.5).

Numpy and BLAS libraries
======================================

Numpy will be built against a specific BLAS library.  On Pleiades we
will build against the OpenBLAS libraries.  

All of the intel patches, etc. are unnecessary in the gcc stack.

Building numpy against MKL
----------------------------------

Now, acquire ``numpy`` (1.9.0)::

     cd $BUILD_HOME
     wget http://sourceforge.net/projects/numpy/files/NumPy/$LOCAL_NUMPY_VERSION/numpy-$LOCAL_NUMPY_VERSION.tar.gz
     tar -xvf numpy-$LOCAL_NUMPY_VERSION.tar.gz
     cd numpy-$LOCAL_NUMPY_VERSION
     wget http://dedalus-project.readthedocs.org/en/latest/_downloads/numpy_janus_intel_patch.tar
     tar xvf numpy_janus_intel_patch.tar

This last step saves you from needing to hand edit two
files in ``numpy/distutils``; these are ``intelccompiler.py`` and
``fcompiler/intel.py``.  I've built a crude patch, :download:`numpy_janus_intel_patch.tar<numpy_janus_intel_patch.tar>` 
which is auto-deployed within the ``numpy-$LOCAL_NUMPY_VERSION`` directory by
the instructions above.  This will unpack and overwrite::

      numpy/distutils/intelccompiler.py
      numpy/distutils/fcompiler/intel.py

This differs from prior versions in that "-xhost" is replaced with
 "-axAVX -xSSE4.1". 

We'll now need to make sure that ``numpy`` is building against the MKL
libraries.  Start by making a ``site.cfg`` file::

     cp site.cfg.example site.cfg
     emacs -nw site.cfg

Edit ``site.cfg`` in the ``[mkl]`` section; modify the
library directory so that it correctly point to TACC's
``$MKLROOT/lib/intel64/``.  
With the modules loaded above, this looks like::

     [mkl]
     library_dirs = /curc/tools/x_86_64/rh6/intel/15.0.1/composer_xe_2015.1.133/mkl/lib/intel64
     include_dirs = /curc/tools/x_86_64/rh6/intel/15.0.1/composer_xe_2015.1.133/mkl/include
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

This should just be pip installed.  In versions of matplotlib>1.3.1,
Qhull has a compile error if the C compiler is used rather than C++,
so we force the C complier to be icpc ::

     export CC=icpc
     pip3 install matplotlib




Installing HDF5 with parallel support
--------------------------------------------------

The new analysis package brings HDF5 file writing capbaility.  This
needs to be compiled with support for parallel (mpi) I/O::

     wget http://www.hdfgroup.org/ftp/HDF5/releases/hdf5-$LOCAL_HDF5_VERSION/src/hdf5-$LOCAL_HDF5_VERSION.tar.gz

     tar xvzf hdf5-$LOCAL_HDF5_VERSION.tar.gz
     cd hdf5-$LOCAL_HDF5_VERSION
     ./configure --prefix=$BUILD_HOME \
                         CC=mpicc         CFLAGS="-O3 -axAVX -xSSE4.1" \
                         CXX=mpicxx CPPFLAGS="-O3 -axAVX -xSSE4.1" \
                         F77=mpif90  F90FLAGS="-O3 -axAVX -xSSE4.1" \
                         MPICC=mpicc MPICXX=mpicxx \
                         --enable-shared --enable-parallel

     make -j
     make install



Installing h5py
----------------------------------------------------

This now can be pip installed::

     pip3 install hypy


Installing Mercurial
----------------------------------------------------
On Janus, we need to install mercurial itself.  I can't get
mercurial to build properly on intel compilers, so for now use gcc.
Ah, and we also need python2 for the mercurial build (only)::
  
     module unload openmpi intel
     module load gcc/gcc-4.9.1
     module load python/anaconda-2.0.0
     wget http://mercurial.selenic.com/release/mercurial-3.4.tar.gz
     tar xvf mercurial-3.4.tar.gz 
     cd mercurial-3.4
     export CC=gcc
     make install PREFIX=$BUILD_HOME

I suggest you add the following to your ``~/.hgrc``::

  [ui]
  username = <your bitbucket username/e-mail address here>
  editor = emacs

  [extensions]
  graphlog =
  color =
  convert =
  mq =


Dedalus
========================================

Preliminaries
----------------------------------------

With the modules set as above, set::

     export BUILD_HOME=$BUILD_HOME
     export FFTW_PATH=$BUILD_HOME
     export MPI_PATH=$BUILD_HOME/$LOCAL_MPI_VERSION

Pull the dedalus repository:::

     hg clone https://bitbucket.org/dedalus-project/dedalus

Then change into your root dedalus directory and run::

     pip3 install -r requirements.txt 
     python3 setup.py build_ext --inplace


Running Dedalus on Janus
========================================

Our scratch disk system on Pleiades is ``/lustre/janus_scratch/user-name``.  On
this and other systems, I suggest soft-linking your scratch directory
to a local working directory in home; I uniformly call mine ``workdir``::

      ln -s /lustre/janus_scratch/bpbrown workdir

I also suggest you move your stack to the ``projects`` directory,
``/projects/user-name``.  There, I bring back a symbolic link:

     ln -s /projects/bpbrown projects
     ln -s projects/build build


