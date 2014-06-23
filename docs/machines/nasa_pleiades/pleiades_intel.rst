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

  export CC=mpicc

  #pathing for Dedalus2        
  export MPI_ROOT=$BUILD_HOME/openmpi-1.7.3
  export PYTHONPATH=$BUILD_HOME/dedalus2:$PYTHONPATH
  export MPI_PATH=$MPI_ROOT
  export FFTW_PATH=$BUILD_HOME

.. note::
   We are moving here to a python 3.4 build.  Also, it looks like
   scipy-0.14 and numpy 1.9 are going to have happier sparse matrix performance.

Doing the entire build took about 1 hour.  This was with several (4) 
open ssh connections to Pleaides to do poor-mans-parallel building 
(of openBLAS, hdf5, fftw, etc.), and one was on a dev node for the
openmpi and openblas compile.


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
    wget http://www.open-mpi.org/software/ompi/v1.7/downloads/openmpi-1.7.3.tar.gz
    tar xvf openmpi-1.7.3.tar.gz

    # get ivy-bridge compute node
    qsub -I -q devel -l select=1:ncpus=20:mpiprocs=20:model=ivy -l walltime=02:00:00

    # once node exists
    cd $BUILD_HOME
    cd openmpi-1.7.3
    ./configure \
	--prefix=$BUILD_HOME \
	--enable-mpi-interface-warning \
	--without-slurm \
	--with-tm=/PBS \
	--without-loadleveler \
	--without-portals \
	--enable-mpirun-prefix-by-default \
        CC=icc CXX=icc FC=ifort F77=ifort

    make
    make install

These compilation options are based on ``/nasa/openmpi/1.6.5/NAS_config.sh``, 
and are thanks to advice from Daniel Kokron at NAS.

We're using openmpi 1.7.3 here because something substantial changes
in 1.7.4 and from that point onwards instances of mpirun hang on
Pleiades, when used on more than 1 node worth of cores.  I've tested
this extensively with a simple hello world program
(http://www.dartmouth.edu/~rc/classes/intro_mpi/hello_world_ex.html)
and for now suggest we move forward until this is resolved.


Building Python3
--------------------------

Create ``$BUILD_HOME`` and then proceed with downloading and installing Python-3.4::

    cd $BUILD_HOME
    wget https://www.python.org/ftp/python/3.4.0/Python-3.4.0.tgz --no-check-certificate
    tar xzf Python-3.4.0.tgz
    cd Python-3.4.0
    wget http://dedalus-project.readthedocs.org/en/latest/_downloads/python_intel_patch.tar
    tar xvf python_intel_patch.tar 

    ./configure --prefix=$BUILD_HOME \
                         CC=mpicc         CFLAGS="-mkl -O3 -axAVX -xSSE4.1 -fPIC -ipo" \
                         CXX=mpicxx CPPFLAGS="-mkl -O3 -axAVX -xSSE4.1 -fPIC -ipo" \
                         F90=mpif90  F90FLAGS="-mkl -O3 -axAVX -xSSE4.1 -fPIC -ipo" \
                         --enable-shared LDFLAGS="-lpthread" \
                         --with-cxx-main=mpicxx --with-system-ffi

    make
    make install

All of the intel patches, etc. are unnecessary in the gcc stack.

.. note::
     We're getting a problem on ``_curses_panel`` and on ``_sqlite3``; ignoring for now.


Installing pip
-------------------------

Python 3.4 now automatically includes pip.

On Pleiades, you'll need to edit ``.pip/pip.conf``::

     [global]
     cert = /etc/ssl/certs/ca-bundle.crt

You will now have ``pip3`` installed in ``$BUILD_HOME/bin``.
You might try doing ``pip3 -V`` to confirm that ``pip3`` is built
against python 3.4.  We will use ``pip3`` throughout this
documentation to remain compatible with systems (e.g., Mac OS) where
multiple versions of python coexist.

Installing mpi4py
--------------------------

This should be pip installed::

    pip3 install mpi4py

.. note::

   Test that this works by doing a:

   from mpi4py import MPI

   This will segfault on sgi-mpi, but appears to work fine on
   openmpi-1.8, 1.7.3, etc.



Installing FFTW3
------------------------------

We need to build our own FFTW3, under intel 14 and mvapich2/2.0b::

    wget http://www.fftw.org/fftw-3.3.4.tar.gz
    tar -xzf fftw-3.3.4.tar.gz
    cd fftw-3.3.4

   ./configure --prefix=$BUILD_HOME \
                         CC=mpicc        CFLAGS="-O3 -axAVX -xSSE4.1" \
                         CXX=mpicxx CPPFLAGS="-O3 -axAVX -xSSE4.1" \
                         F77=mpif90  F90FLAGS="-O3 -axAVX -xSSE4.1" \
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

The Feb 11, 2014 update to cython (0.20.1) seems to work with gcc.




Numpy and BLAS libraries
======================================

Numpy will be built against a specific BLAS library.  On Pleiades we
will build against the OpenBLAS libraries.  

All of the intel patches, etc. are unnecessary in the gcc stack.

Building numpy against MKL
----------------------------------

Now, acquire ``numpy`` (1.8.1)::

     cd $BUILD_HOME
     wget http://sourceforge.net/projects/numpy/files/NumPy/1.8.1/numpy-1.8.1.tar.gz
     tar -xvf numpy-1.8.1.tar.gz
     cd numpy-1.8.1
     wget http://lcd-www.colorado.edu/bpbrown/dedalus_documentation/_downloads/numpy_intel_patch.tar
     tar xvf numpy_intel_patch.tar

This last step saves you from needing to hand edit two
files in ``numpy/distutils``; these are ``intelccompiler.py`` and
``fcompiler/intel.py``.  I've built a crude patch, 
:download:`numpy_intel_patch.tar<numpy_intel_patch.tar>` 
which can be auto-deployed by within the ``numpy-1.8.1`` directory by
the instructions above.  This will unpack and overwrite::

      numpy/distutils/intelccompiler.py
      numpy/distutils/fcompiler/intel.py

Crap.  For now I'm hand editing these to replace "-xhost" with
 "-axAVX -xSSE4.1".  Crap crap crap.

We'll now need to make sure that ``numpy`` is building against the MKL
libraries.  Start by making a ``site.cfg`` file::

     cp site.cfg.example site.cfg
     emacs -nw site.cfg

Edit ``site.cfg`` in the ``[mkl]`` section; modify the
library directory so that it correctly point to TACC's
``$MKLROOT/lib/intel64/``.  
With the modules loaded above, this looks like::

     [mkl]
     library_dirs = /nasa/intel/Compiler/2013.5.192/composer_xe_2013.5.192/mkl/lib/intel64
     include_dirs = /nasa/intel/Compiler/2013.5.192/composer_xe_2013.5.192/mkl/include
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

    wget http://sourceforge.net/projects/scipy/files/scipy/0.14.0/scipy-0.14.0.tar.gz
    tar -xvf scipy-0.14.0.tar.gz
    cd scipy-0.14.0
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


Installing sympy
-------------------------

This should just be pip installed::

     pip3 install sympy


Installing HDF5 with parallel support
--------------------------------------------------

The new analysis package brings HDF5 file writing capbaility.  This
needs to be compiled with support for parallel (mpi) I/O::

     wget http://www.hdfgroup.org/ftp/HDF5/current/src/hdf5-1.8.12.tar
     tar xvf hdf5-1.8.12.tar
     cd hdf5-1.8.12
     ./configure --prefix=$BUILD_HOME \
                         CC=mpicc         CFLAGS="-O3 -axAVX -xSSE4.1" \
                         CXX=mpicxx CPPFLAGS="-O3 -axAVX -xSSE4.1" \
                         F77=mpif90  F90FLAGS="-O3 -axAVX -xSSE4.1" \
                         MPICC=mpicc MPICXX=mpicxx \
                         --enable-shared --enable-parallel
     make
     make install

Next, install h5py.  For reasons that are currently unclear to me, 
this cannot be done via pip install.




Installing h5py with collectives
----------------------------------------------------
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
     python3 setup.py build --mpi   
     python3 setup.py install --mpi


Here's the original h5py repository::

     git clone git://github.com/h5py/h5py
     cd h5py
     export CC=mpicc
     export HDF5_DIR=$BUILD_HOME
     python3 setup.py build --mpi
     python3 setup.py install --mpi

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

     wget http://mercurial.selenic.com/release/mercurial-2.9.tar.gz
     tar xvf mercurial-2.9.tar.gz 
     cd mercurial-2.9
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
     export MPI_PATH=$BUILD_HOME/openmpi-1.8

Then change into your root dedalus directory and run::

     python setup.py build_ext --inplace

further packages needed for Keaton's branch::

     pip3 install tqdm
     pip3 install pathlib


Running Dedalus on Pleiades
========================================

Our scratch disk system on Pleiades is ``/nobackup/user-name``.  On
this and other systems, I suggest soft-linking your scratch directory
to a local working directory in home; I uniformly call mine ``workdir``::

      ln -s /nobackup/bpbrown workdir

Long-term mass storage is on LOU.



