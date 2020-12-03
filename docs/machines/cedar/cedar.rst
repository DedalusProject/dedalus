Install notes for Compute Canada's clusters
************************************************

Notes
-----

The following instructions have been provided by Maxime Boissonneault, staff at Compute Canada. 
Last updated 2020-12-03.

Instructions
------------

Compute Canada already pre-builds Dedalus on demand. You can contact https://docs.computecanada.ca/wiki/Technical_support in order to request a new version. In your home: 

    module purge
    module load StdEnv/2020 fftw-mpi mpi4py hdf5-mpi

Now build and activate the virtual environment for your installation: .
You should also update pip as soon as the environment is activated. ::

    virtualenv --no-download python_env
    source python_env/bin/activate
    pip install --no-index --upgrade pip

Now install dedalus: 

    pip install --no-index dedalus

Compute Canada recommends creating your virtual environment and loading your modules in your job scripts. For more information about doing so, please see
https://docs.computecanada.ca/wiki/Python#Creating_and_using_a_virtual_environment

