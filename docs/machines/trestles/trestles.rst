Install notes for Trestles
==========================

.. note::
   These are a very old set of installation instructions.  They likely no longer work.

Make sure to do 

`$ module purge` 

first.

Modules
-------

This is a minimalist list for now:

* gnu/4.8.2 (this is now the default gnu module)
* openmpi_ib 
* fftw/3.3.3 (make sure to do this one last, as it's compiler/MPI dependent)

Building Python3
----------------

I usually build everything in `~/build`, but you can do it
whereever. Download Python-3.3. Once loading the above modules, in the
Python-3.3 source directory, do

`$ ./configure --prefix=$HOME/build`

followed by the usual `make -j4; make install` (the `-j4` tells make
to use 4 cores). You may get something like this:

::

    Python build finished, but the necessary bits to build these modules were not found:
    _dbm               _gdbm              _lzma           
    _sqlite3                                              
    To find the necessary bits, look in setup.py in detect_modules() for the module's name.

I think this should be totally fine.

**At this point, make sure the python3 you installed is in your path!**

Installing virtualenv
---------------------

In order to test multiple numpys and scipys (and really, their
underlying BLAS libraries), I am using virtualenv. In order install
virtulenv, once Python-3.3 is installed, you first need to install pip
manuall. Follow the steps here
http://www.pip-installer.org/en/latest/installing.html for "Install or
Upgrade Setuptools" and then "Install or Upgrade pip". Briefly, you
need to download and run ez_setup.py and then get-pip.py. This should
run without incident. Once `pip` is installed, do

`$ pip install virtualenv` 


Building OpenBLAS
-----------------

To build openBLAS, first do `$ git clone https://github.com/xianyi/OpenBLAS.git` to get OpenBLAS. Then, with the modules loaded, do
`make -j4`;  and `make PREFIX=path/to/build/dir install`

Building numpy
--------------

First construct a virtualenv to hold all of your python modules. I like to do this right in my home directory. For example,

`$ mkdir ~/venv` (assuming you don't already have `~/venv`)
`$ cd ~/venv`
`$ virtualenv openblas`

will create an `openblas` directory, with a `bin` subdirectory. You "activate" the virtual env by doing `$ source path/to/virtualenv/bin/activate`. This will change all of your environment variables so that the active python will see whatever modules are in that directory. **Note that this messes with modules!** To be safe, I'd recommend `$ module purge; module load gnu openmpi_ib` afterwards. 

* `$ cp site.cfg.example site.cfg`

edit `site.cfg` to uncomment the [openblas] section and fill in the include and library dirs to whereever you installed openblas.

* `$ python setup.py config`

to make sure that the numpy build has FOUND your openblas install. If it did, you should see something like this:

::

    (openblas)trestles-login1:/home/../numpy-1.8.0 [10:15]$ python setup.py config
    Running from numpy source directory.
    F2PY Version 2
    blas_opt_info:
    blas_mkl_info:
      libraries mkl,vml,guide not found in ['/home/joishi/venv/openblas/lib', '/usr/local/lib64', '/usr/local/lib', '/usr/lib64', '/usr/lib', '/usr/lib/']
      NOT AVAILABLE
    
    openblas_info:
      FOUND:
        language = f77
        library_dirs = ['/home/joishi/build/lib']
        libraries = ['openblas', 'openblas']
    
      FOUND:
        language = f77
        library_dirs = ['/home/joishi/build/lib']
        libraries = ['openblas', 'openblas']
    
    non-existing path in 'numpy/lib': 'benchmarks'
    lapack_opt_info:
      FOUND:
        language = f77
        library_dirs = ['/home/joishi/build/lib']
        libraries = ['openblas', 'openblas']
    
    /home/joishi/build/lib/python3.3/distutils/dist.py:257: UserWarning: Unknown distribution option: 'define_macros'
      warnings.warn(msg)
    running config

* `$ python setup.py build`

if successful, 

* `$ python setup.py install`

Installing Scipy
----------------

Scipy is easier, because it just gets its config from numpy.

* `$ python setup.py config`

This notes a lack of UMFPACK...will that make a speed difference? Nevertheless, it works ok.

Do

* `$ python setup.py build`

if successful, 

* `$ python setup.py install`


Installing mpi4py
-----------------

This should just be pip installed, `$ pip install mpi4py`

Installing cython
-----------------

This should just be pip installed, `$ pip install cython`

Installing matplotlib
-----------------

This should just be pip installed, `$ pip install matplotlib`

UMFPACK
-------

Requires AMD (another package by the same group, not processor) and SuiteSparse_config, too.

Dedalus2
--------

With the modules set as above (for NOW), set `$ export
FFTW_PATH=/opt/fftw/3.3.3/gnu/openmpi/ib` and `$ export
MPI_PATH=/opt/openmpi/gnu/ib`. Then do `$ python setup.py build_ext
--inplace`.




