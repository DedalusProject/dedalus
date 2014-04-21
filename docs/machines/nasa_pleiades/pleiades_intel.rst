Install notes for NASA/Pleiades
***************************************************************************

An initial Pleiades environment is pretty bare-bones.  There are no
modules, and your shell is likely a csh varient.  To switch shells,
send an e-mail to support@nas.nasa.gov; I'll be using ``bash``.


Then add the following to your ``.profile``::

  # Add your commands here to extend your PATH, etc.

  module load comp-intel
  module load mpi-sgi
  module load hdf5
  module load git


  module load gcc
  module load mpi-sgi
  module load hdf5
  module load git


  export BUILD_HOME=$HOME/build

  export PATH=$BUILD_HOME/bin:$BUILD_HOME:/$PATH  # Add private commands to PATH                                                                                         

  export LD_LIBRARY_PATH=$BUILD_HOME/lib:$LD_LIBRARY_PATH

  export CC=mpicc

  #pathing for Dedalus2        
  export MPI_ROOT=$BUILD_HOME/openmpi-1.8                                                                                                                                          
  export PYTHONPATH=$BUILD_HOME/dedalus2:$PYTHONPATH
  export MPI_PATH=$MPI_ROOT
  export FFTW_PATH=$BUILD_HOME

.. note::
   We should roll to a python 3.4 build.  Also, it looks like
   scipy-0.14 and numpy 1.9 are going to have happier sparse matrix performance.


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
    wget http://www.open-mpi.org/software/ompi/v1.8/downloads/openmpi-1.8.tar.gz
    tar xvf openmpi-1.8.tar.gz

    # get ivy-bridge compute node
    qsub -I -q devel -l select=1:ncpus=20:mpiprocs=20:model=ivy -l walltime=02:00:00

    # once node exists
    cd $BUILD_HOME
    cd openmpi-1.8
    ./configure --prefix=$BUILD_HOME CC=gcc
    make
    make install

Building Python3
--------------------------

Create ``$BUILD_HOME`` and then proceed with downloading and installing Python-3.3::

    cd $BUILD_HOME
    wget http://www.python.org/ftp/python/3.3.3/Python-3.3.3.tgz --no-check-certificate
    tar xzf Python-3.3.3.tgz
    cd Python-3.3.3

    ./configure --prefix=$BUILD_HOME \
                         CC=mpicc \
                         CXX=mpicxx \
                         F90=mpif90 \
                         --enable-shared LDFLAGS="-lpthread" \
                         --with-cxx-main=mpicxx --with-system-ffi

    make
    make install

All of the intel patches, etc. are unnecessary in the gcc stack.

.. note::
     We're getting a problem on ``_curses_panel``; ignoring for now.

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

    wget https://raw.github.com/pypa/pip/master/contrib/get-pip.py  --no-check-certificate
    python3 get-pip.py --cert /etc/ssl/certs/ca-bundle.crt

Now edit ``.pip/pip.conf``::

     [global]
     cert = /etc/ssl/certs/ca-bundle.crt

You will now have ``pip3`` and ``pip`` installed in ``$BUILD_HOME/bin``.
You might try doing ``pip -V`` to confirm that ``pip`` is built
against python 3.3.  We will use ``pip3`` throughout this
documentation to remain compatible with systems (e.g., Mac OS) where
multiple versions of python coexist.

Installing mpi4py
--------------------------

This should be pip installed::

    pip3 install mpi4py

.. note::

   Test that this works by doing a:

   from mpi4py import MPI

   This will segfault on sgi-mpi, but appears to work fine on openmpi-1.8.



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


Building OpenBLAS
----------------------------------

From Stampede instructions::

      # this needs to be done on a frontend
      cd $BUILD_HOME
      git clone git://github.com/xianyi/OpenBLAS

      # suggest doing this build on a compute node, so we get the
      # right number of openmp threads and architecture
      cd $BUILD_HOME
      cd OpenBLAS
      make
      make PREFIX=$BUILD_HOME install

Here's the build report before the ``make install``::

  OpenBLAS build complete. (BLAS CBLAS LAPACK LAPACKE)

  OS               ... Linux             
  Architecture     ... x86_64               
  BINARY           ... 64bit                 
  C compiler       ... GCC  (command line : mpicc)
  Fortran compiler ... GFORTRAN  (command line : gfortran)
  Library Name     ... libopenblas_sandybridgep-r0.2.9.rc2.a (Multi threaded; Max num-threads is 40)



Building numpy against OpenBLAS
----------------------------------------

Now, acquire ``numpy`` (1.8.1)::

     wget http://sourceforge.net/projects/numpy/files/NumPy/1.8.1/numpy-1.8.1.tar.gz
     tar xvf numpy-1.8.1.tar.gz
     cd numpy-1.8.1


Create ``site.cfg`` with information for the OpenBLAS
library directory

Next, make a site specific config file::

      cp site.cfg.example site.cfg
      emacs -nw site.cfg

Edit ``site.cfg`` to uncomment the ``[openblas]`` section; modify the
library and include directories so that they correctly point to your
``~/build/lib`` and ``~/build/include`` (note, you may need to do fully expanded
paths).  With my account settings, this looks like::

     [openblas]
     libraries = openblas
     library_dirs = /u/bpbrown/build/lib
     include_dirs = /u/bpbrown/build/include

where ``$BUILD_HOME=/u/bpbrown/build``.  Now build::
 
     python3 setup.py config build_clib build_ext install

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

Scipy is easier, because it just gets its config from numpy.  Download
an install::

     wget http://sourceforge.net/projects/scipy/files/scipy/0.13.2/scipy-0.13.2.tar.gz
     tar -xvf scipy-0.13.2.tar.gz
     cd scipy-0.13.2

Then run ::

    python3 setup.py config build_clib build_ext install

.. note::

   We do not have umfpack; we should address this moving forward, but
   will defer that to a later day.


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
                         CC=mpicc \
                         CXX=mpicxx \
                         F77=mpif90 \
                         MPICC=mpicc MPICXX=mpicxx \
                         --enable-shared --enable-parallel
     make
     make install

Next, install h5py.  




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
     that.  For now, I've hand-edited unixccompiler.py in 
     ``lib/python3.3/distutils`` and changed this line:

     else:
                # No idea how --enable-new-dtags would be passed on to
                # ld if this system was using GNU ld.  Don't know if a
                # system like this even exists.
                return "-R" + dir

     to:

     else:
                # No idea how --enable-new-dtags would be passed on to
                # ld if this system was using GNU ld.  Don't know if a
                # system like this even exists.
                return "-Wl,-R" + dir

     Looks like we're failing 

             if self._is_gcc(compiler)

     ???

     This is a hack, but it get's us running and alive!

Installing Mercurial
----------------------------------------------------
On NASA Pleiades, we need to install mercurial itself::

     wget http://mercurial.selenic.com/release/mercurial-2.9.tar.gz
     tar xvf mercurial-2.9.tar.gz 
     cd mercurial-2.9
     make install PREFIX=$BUILD_HOME


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



