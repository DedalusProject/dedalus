Install notes for BRC HPC SAVIO cluster
***************************************************************************

Installing on the SAVIO cluster is pretty straightforward, as many things
can be loaded via modules.  First, load the following modules.

::

  module purge
  module load intel
  module load openmpi
  module load fftw/3.3.4-intel
  module load python/3.2.3
  module load nose
  module load numpy/1.8.1
  module load scipy/0.14.0
  module load mpi4py
  module load pip
  module load virtualenv/1.7.2 
  module load mercurial/2.0.2 
  module load hdf5/1.8.13-intel-p

We next need to make a virtual environment in which to build the rest of
the Dedalus depencencies.

::

  virtualenv python_build
  source python_build/bin/activate

The rest of the depencies will be pip-installed.  However, because we are
using intel compilers, we need to specify the compiler and some how to link
things properly.

::

  export CC=icc
  export LDFLAGS="-lirc -limf"

Now we can use pip to install most of the remaining dependencies.

::

  pip-3.2 install cython
  pip-3.2 install h5py
  pip-3.2 install matplotlib==1.3.1

Dedalus itself can be pulled down from Bitbucket.

::

  hg clone https://bitbucket.org/dedalus-project/dedalus
  cd dedalus
  pip-3.2 install -r requirements.txt

To build Dedalus, you must specify the locations of FFTW and MPI.

::

  export FFTW_PATH=/global/software/sl-6.x86_64/modules/intel/2013_sp1.4.211/fftw/3.3.4-intel
  export MPI_PATH=/global/software/sl-6.x86_64/modules/intel/2013_sp1.2.144/openmpi/1.6.5-intel
  python3 setup.py build_ext --inplace


Using Dedalus
--------------------------

To use Dedalus, put the following in your ``~/.bashrc`` file::

  module purge
  module load intel
  module load openmpi
  module load fftw/3.3.4-intel
  module load python/3.2.3
  module load numpy/1.8.1
  module load scipy/0.14.0
  module load mpi4py
  module load mercurial/2.0.2
  module load hdf5/1.8.13-intel-p
  source python_build/bin/activate
  export PYTHONPATH=$PYTHONPATH:~/dedalus


