Install notes for NASA/Discover
***************************************************************************

This installation is fairly straightforward because most of the work has
already been done by the NASA/Discover staff, namely Jules Kouatchou.

First, add the following lines to your ``~/.bashrc`` file and source it:

::
  
  module purge
  module load other/comp/gcc-4.9.1
  module load lib/mkl-15.0.0.090
  module load other/Py3Dist/py-3.4.1_gcc-4.9.1_mkl-15.0.0.090
  module load other/mpi/openmpi/1.8.2-gcc-4.9.1
  
  export BUILD_HOME=$HOME/build
  export PYTHONPATH=$HOME/dedalus2


This loads the gcc compiler, MKL linear algebra package, openmpi version 1.8.2,
and crucially various python3 libraries.  To see the list of python libraries,

::
  
  listPyPackages

We actually have all the python libraries we need for Dedalus.  However, we still need
fftw.  To install fftw,

::
  
  mkdir build
  
  cd $BUILD_HOME
  wget http://www.fftw.org/fftw-3.3.4.tar.gz
  tar -xzf fftw-3.3.4.tar.gz
  cd fftw-3.3.4
  
  ./configure --prefix=$BUILD_HOME \
                       CC=mpicc \
                       CXX=mpicxx \
                       F77=mpif90 \
                       MPICC=mpicc MPICXX=mpicxx \
                       --enable-shared \
                       --enable-mpi --enable-openmp --enable-threads

  make
  make install

All that remains is to pull Dedalus down from Bitbucket and install it.

::
  
  cd $HOME
  hg clone https://bitbucket.org/dedalus-project/dedalus2

  export FFTW_PATH=$BUILD_HOME
  export HDF5_DIR=$BUILD_HOME
  export MPI_DIR=/usr/local/other/SLES11.1/openMpi/1.8.2/gcc-4.9.1
  cd $HOME/dedalus2
  python3 setup.py build_ext --inplace


