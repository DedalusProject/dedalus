Install notes for Compute Canada's Cedar cluster
************************************************

Notes
-----

The following instructions have been provided by `David Goluskin <goluskin@uvic.ca>`_ and `Hannah Swan <hannah.swan.3.14@gmail.com>`_.
Last updated 2020/03/31.

Instructions
------------

In your home directory, load the following modules::

    module purge
    module load intel/2018.3
    module load openmpi/3.1.2
    module load fftw-mpi/3.3.8
    module load python/3.6.3
    module load scipy-stack
    module load mpi4py/3.0.3
    module load hdf5-mpi/1.10.3

Now build and activate the virtual environment for your python build.
This environment is where Dedalus will be built.
You should also update pip as soon as the environment is activated. ::

    virtualenv --no-download python_env
    source python_env/bin/activate
    pip install --no-index --upgrade pip

Now install the remaining few dependencies using pip::

    pip install h5py
    pip install mercurial
    pip install pytest

Pull down Dedalus from bitbucket::

    hg clone https://bitbucket.org/dedalus-project/dedalus
    cd dedalus
    pip install -r requirements.txt

Before building Dedalus, the ``FFTW PATH`` and ``MPI PATH`` need to be set.
At the same time, set the ``FFTW STATIC`` environment variable.
This lets the setup script know to statically link the FFTW build to the Dedalus extensions, preventing MKL from overwriting the FFTW symbols and breaking everything (full FFTW functionality is not implemented in MKL). ::

    export FFTW_PATH=$EBROOTFFTW
    export MPI_PATH=$EBROOTOPENMPI
    export FFTW_STATIC=1

Finally, build Dedalus::

    python3 setup.py build_ext --inplace

Cedar strongly recommends against automatically loading modules using your ``.bashrc``, rather advocating for loading modules manually at the start of each job script (note: you may do this with a module collection).
In light of that, to use Dedalus, add the following at the top of your job script:

    module purge
    module load intel/2018.3
    module load openmpi/3.1.2
    module load fftw-mpi/3.3.8
    module load python/3.6.3
    module load scipy-stack
    module load mpi4py/3.0.3
    module load hdf5-mpi/1.10.3
    source ~/python_env/bin/activate
    export PYTHONPATH=$PYTHONPATH:~/dedalus

