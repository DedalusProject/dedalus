Install notes for MIT Engaging Cluster
**************************************

This installation uses the Python, BLAS, and MPI modules available on Engaging, while manually building HDF5 and FFTW.

Modules and paths
-----------------

The following commands should be added to your ``~/.bashrc`` file to setup the correct modules and paths.
Modify the ``HDF5_DIR``, ``FFTW_PATH``, and ``DEDALUS_REPO`` environment variables as desired to change the build locations of these packages.

::

    # Basic modules
    module load gcc
    module load slurm

    # Python from modules
    module load engaging/python/2.7.10
    module load engaging/python/3.6.0
    export PATH=~/.local/bin:${PATH}

    # BLAS from modules
    module load engaging/OpenBLAS/0.2.14
    export BLAS=/cm/shared/engaging/OpenBLAS/0.2.14/lib/libopenblas.so

    # MPI from modules
    module load engaging/openmpi/2.0.3
    export MPI_PATH=/cm/shared/engaging/openmpi/2.0.3

    # HDF5 built from source
    export HDF5_DIR=~/software/hdf5
    export HDF5_VERSION=1.10.1
    export HDF5_MPI="ON"
    export PATH=${HDF5_DIR}/bin:${PATH}
    export LD_LIBRARY_PATH=${HDF5_DIR}/lib:${LD_LIBRARY_PATH}

    # FFTW built from source
    export FFTW_PATH=~/software/fftw
    export FFTW_VERSION=3.3.6-pl2
    export PATH=${FFTW_PATH}/bin:${PATH}
    export LD_LIBRARY_PATH=${FFTW_PATH}/lib:${LD_LIBRARY_PATH}

    # Dedalus from mercurial
    export DEDALUS_REPO=~/software/dedalus
    export PYTHONPATH=${DEDALUS_REPO}:${PYTHONPATH}


Build procedure
---------------

Source your ``~/.bashrc`` to activate the above changes, or re-login to the cluster, before running the following build procedure.

::

    # Python basics
    /cm/shared/engaging/python/2.7.10/bin/pip install --ignore-installed --user pip
    /cm/shared/engaging/python/3.6.0/bin/pip3 install --ignore-installed --user pip
    pip2 install --user --upgrade setuptools
    pip2 install --user mercurial
    pip3 install --user --upgrade setuptools
    pip3 install --user nose cython

    # Python packages
    pip3 install --user --no-use-wheel numpy
    pip3 install --user --no-use-wheel scipy
    pip3 install --user mpi4py

    # HDF5 built from source
    mkdir -p ${HDF5_DIR}
    cd ${HDF5_DIR}
    wget https://support.hdfgroup.org/ftp/HDF5/current/src/hdf5-${HDF5_VERSION}.tar
    tar -xvf hdf5-${HDF5_VERSION}.tar
    cd hdf5-${HDF5_VERSION}
    ./configure --prefix=${HDF5_DIR} \
        CC=mpicc \
        CXX=mpicxx \
        F77=mpif90 \
        MPICC=mpicc \
        MPICXX=mpicxx \
        --enable-shared \
        --enable-parallel
    make
    make install
    pip3 install --user --no-binary=h5py h5py

    # FFTW built from source
    mkdir -p ${FFTW_PATH}
    cd ${FFTW_PATH}
    wget http://www.fftw.org/fftw-${FFTW_VERSION}.tar.gz
    tar -xvzf fftw-${FFTW_VERSION}.tar.gz
    cd fftw-${FFTW_VERSION}
    ./configure --prefix=${FFTW_PATH} \
        CC=mpicc \
        CXX=mpicxx \
        F77=mpif90 \
        MPICC=mpicc \
        MPICXX=mpicxx \
        --enable-shared \
        --enable-mpi \
        --enable-openmp \
        --enable-threads
    make
    make install

    # Dedalus from mercurial
    hg clone https://bitbucket.org/dedalus-project/dedalus ${DEDALUS_REPO}
    cd ${DEDALUS_REPO}
    pip3 install --user -r requirements.txt
    python3 setup.py build_ext --inplace


Notes
-----

Last updated on 2017/09/18 by Keaton Burns.
